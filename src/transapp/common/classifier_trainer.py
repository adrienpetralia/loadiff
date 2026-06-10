"""base_classifier_trainer.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A reusable **PyTorch** training helper tailored for *single-label* classification
problems.

Example
-------
```python
trainer = BaseClassifierTrainer(
    model,
    train_loader,
    valid_loader,
    patience_es=8,
    metrics=ImbalanceClassifMetrics(),
    save_checkpoint=True,
)
trainer.train(n_epochs=50)
valid_loss, valid_metrics = trainer.evaluate(test_loader)
```
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union, Type, Any

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from optim.early_stopper import EarlyStopper


class BaseClassifierTrainer:
    """Generic trainer for *supervised* single-label classification.

    Parameters
    ----------
    model:
        The neural network to optimise.
    train_loader, valid_loader:
        PyTorch dataloaders yielding ``(inputs, labels)``.
    optimizer_cls:
        Optimiser class (default: :class:`torch.optim.AdamW`).
    optimizer_kwargs:
        Extra keyword arguments forwarded to the optimiser constructor.
    lr_scheduler_cls, lr_scheduler_kwargs:
        Scheduler class and parameters. If *None*, no LR scheduling is used.
        When provided, ``ReduceLROnPlateau`` semantics are expected
        (``scheduler.step(valid_loss)``).
    criterion:
        The loss function. Defaults to :class:`torch.nn.CrossEntropyLoss`.
    patience_es:
        Patience for early-stopping *after* ``n_warmup_epochs``. ``None``
        disables the mechanism.
    device:
        ``"cpu"`` or ``"cuda"`` (default), or an explicit
        :class:`torch.device`.
    use_data_parallel:
        Wrap *model* into :class:`torch.nn.DataParallel` when ``True``.
    n_warmup_epochs:
        Number of initial epochs immune to early-stopping.
    metrics:
        Callable receiving ``(y_true, y_pred[, y_proba])`` and returning a
        *dict* of metric names to scalar values.
    writer:
        A pre-constructed :class:`torch.utils.tensorboard.SummaryWriter`.
        When omitted, a writer is created under ``runs/``.
    save_checkpoint:
        Persist a checkpoint *every time* the validation loss improves and
        once again at the end of training.
    checkpoint_path:
        Target file ("/path/to/file.pt"). Defaults to
        ``cwd / 'model.pt'``.
    verbose:
        Enable ``logging.INFO`` progress messages.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader] = None,
        *,
        optimizer_cls: Type[optim.Optimizer] = optim.AdamW,
        optimizer_kwargs: Optional[Dict[str, Union[int, float]]] = None,
        lr_scheduler_cls: Optional[Callable] = torch.optim.lr_scheduler.ReduceLROnPlateau,
        lr_scheduler_kwargs: Optional[Dict[str, Union[int, float, str]]] = None,
        criterion: Callable = nn.CrossEntropyLoss(),
        patience_es: Optional[int] = None,
        device: Union[str, torch.device] = "cuda",
        use_data_parallel: bool = False,
        n_warmup_epochs: int = 0,
        metrics: Callable = lambda *args, **kwargs: {},
        writer: Optional[SummaryWriter] = None,
        save_checkpoint: bool = False,
        checkpoint_path: Optional[Union[str, os.PathLike[str], Path]] = None,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metrics = metrics
        self.verbose = verbose
        self.n_warmup_epochs = n_warmup_epochs
        self.save_checkpoint = save_checkpoint
        self.best_loss: float = float("inf")
        self.passed_epochs: int = 0
        self.passed_steps: int = 0

        # ------------------------------------------------------------------
        # Optimiser & scheduler
        # ------------------------------------------------------------------
        optimizer_kwargs = optimizer_kwargs or {"lr": 1e-4, "weight_decay": 1e-3}
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)  # type: ignore[arg-type]

        if lr_scheduler_cls is not None:
            lr_scheduler_kwargs = lr_scheduler_kwargs or {"mode": "min", "patience": 5, "eps": 1e-7}
            self.scheduler = lr_scheduler_cls(self.optimizer, **lr_scheduler_kwargs)  # type: ignore[arg-type]
        else:
            self.scheduler = None

        self.criterion = criterion

        # ------------------------------------------------------------------
        # Early-stopping
        # ------------------------------------------------------------------
        self.early_stopper: Optional[EarlyStopper] = (
            EarlyStopper(patience_es) if patience_es is not None else None
        )

        # ------------------------------------------------------------------
        # Device placement (optionally DataParallel)
        # ------------------------------------------------------------------
        if use_data_parallel and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        # ------------------------------------------------------------------
        # Logging & bookkeeping
        # ------------------------------------------------------------------
        runs_dir = Path("runs")
        runs_dir.mkdir(exist_ok=True)
        if writer is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.writer = SummaryWriter(log_dir=runs_dir / f"train_{timestamp}")
        else:
            self.writer = writer

        self.checkpoint_path = Path(checkpoint_path or Path.cwd() / "model.pt")

        self.loss_train_history: List[float] = []
        self.loss_valid_history: List[float] = []
        self.acc_train_history: List[float] = []
        self.acc_valid_history: List[float] = []
        self.log: Dict[str, Union[float, List[float], Dict[str, float]]] = {}

        if self.verbose:
            logging.basicConfig(level=logging.INFO, format="%(message)s")
            logging.info("Trainer initialised - device: %s", self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self, n_epochs: int = 10) -> None:
        """Run the full training loop.

        The method populates :pyattr:`loss_train_history`,
        :pyattr:`loss_valid_history`, :pyattr:`acc_train_history` and
        :pyattr:`acc_valid_history`. A ``.log`` dictionary containing the same
        data plus timing information and (optionally) state dicts is also
        updated.
        """

        tic = time.time()
        for epoch in range(1, n_epochs + 1):
            train_loss, train_acc = self._train_epoch()
            self.loss_train_history.append(train_loss)
            self.acc_train_history.append(train_acc)

            # ---------------------------------------------------------
            # Validation phase (optional)
            # ---------------------------------------------------------
            if self.valid_loader is not None:
                valid_loss, valid_acc = self._eval_epoch(self.valid_loader)
                self.loss_valid_history.append(valid_loss)
                self.acc_valid_history.append(valid_acc)
            else:
                valid_loss, valid_acc = train_loss, train_acc  # type: ignore[assignment]

            # ---------------------------------------------------------
            # Scheduler & early-stopping
            # ---------------------------------------------------------
            if self.scheduler is not None:
                self.scheduler.step(valid_loss)

            if (
                self.early_stopper is not None
                and epoch > self.n_warmup_epochs
                and self.early_stopper(valid_loss)
            ):
                if self.verbose:
                    logging.info("Early stopping triggered at epoch %d", epoch)
                break

            # ---------------------------------------------------------
            # Logging & TensorBoard
            # ---------------------------------------------------------
            if self.verbose:
                msg = (
                    f"Epoch {epoch:>3}/{n_epochs} | "
                    f"Train: loss={train_loss:.4f}, acc={train_acc:.2%} | "
                    f"Valid: loss={valid_loss:.4f}, acc={valid_acc:.2%}"
                )
                logging.info(msg)

            self.writer.add_scalars(
                "MeanEpochLoss", {"train": train_loss, "valid": valid_loss}, epoch
            )
            self.writer.add_scalars(
                "Accuracy", {"train": train_acc, "valid": valid_acc}, epoch
            )

            # ---------------------------------------------------------
            # Checkpointing
            # ---------------------------------------------------------
            if valid_loss < self.best_loss and epoch >= self.n_warmup_epochs:
                self.best_loss = valid_loss
                self._update_log(epoch, valid_acc)
                if self.save_checkpoint:
                    self._save_checkpoint("best")

            self.passed_epochs += 1

        self._finalise_training(time.time() - tic)

    def evaluate(
        self,
        test_loader: DataLoader,
        *,
        log_key: str = "test_metrics",
        return_outputs: bool = False,
    ) -> Union[
        Tuple[float, Dict[str, float]],
        Tuple[float, Dict[str, float], np.ndarray, np.ndarray, np.ndarray],
    ]:
        """Evaluate the current model on a loader.

        Parameters
        ----------
        test_loader:
            DataLoader to iterate over.
        log_key:
            The key under which computed metrics are stored in ``self.log``.
        return_outputs:
            When True, returns:
                (loss, metrics, y_true, y_pred, logits)

        Returns
        -------
        Either:
            (mean_loss, metrics)
        or:
            (mean_loss, metrics, y_true, y_pred, logits)
        """
        self.model.eval()
        tic = time.time()

        losses = []
        y_true, y_pred = [], []
        logits_list = []

        with torch.no_grad():
            for batch in test_loader:
                ts, exogene, labels = self._prepare_batch(batch, require_labels=True)
                assert labels is not None

                logits = self._forward_batch(ts, exogene)
                loss = self.criterion(logits, labels)
                losses.append(loss.item())

                preds = logits.argmax(dim=1)

                y_true.append(labels.cpu())
                y_pred.append(preds.cpu())

                if return_outputs:
                    logits_list.append(logits.cpu())

        y_true_arr = torch.cat(y_true).numpy() if y_true else np.empty((0,), dtype=np.int64)
        y_pred_arr = torch.cat(y_pred).numpy() if y_pred else np.empty((0,), dtype=np.int64)

        metrics = self._apply_metrics(y_true_arr, y_pred_arr)

        self.log[log_key] = metrics
        self.log["eval_time"] = round(time.time() - tic, 3)

        if self.save_checkpoint:
            self._save_checkpoint("eval")

        result = (float(np.mean(losses)), metrics)

        if return_outputs:
            logits_arr = (
                torch.cat(logits_list).numpy()
                if logits_list
                else np.empty((0,), dtype=np.float32)
            )
            result += (y_true_arr, y_pred_arr, logits_arr)

        return result  # type: ignore[misc]

    def reduce_lr(self, new_lr: float) -> None:
        """Manually set a new learning-rate for *all* parameter groups."""
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr

    def restore_best_weights(self, tag: str) -> None:
        """Load weights from the *best* checkpoint (lowest validation loss)."""
        checkpoint_file = self.checkpoint_path.with_stem(f"{self.checkpoint_path.stem}_{tag}")
        try:
            state = torch.load(checkpoint_file)
            self.model.load_state_dict(state["model_state_dict"])
            if self.verbose:
                logging.info("Best model restored from %s", checkpoint_file)
        except (FileNotFoundError, KeyError) as exc:
            logging.warning("Could not restore weights - %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_batch(
        self,
        batch: Dict[str, Any],
        *,
        require_labels: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Normalize a batch dict into (ts, exogene, labels).

        Expected batch format:
            {
                "ts": Tensor,                   # required
                "exogene": Tensor,             # optional
                "labels": Tensor               # optional depending on context
            }

        Parameters
        ----------
        batch:
            Batch returned by the DataLoader.
        require_labels:
            Whether labels must be present.

        Returns
        -------
        ts, exogene, labels
        """
        if not isinstance(batch, dict):
            raise TypeError(
                f"Expected batch to be a dict, got {type(batch).__name__}."
            )

        if "ts" not in batch:
            raise KeyError("Batch is missing required key 'ts'.")

        ts = batch["ts"].to(self.device, dtype=torch.float)

        exogene = batch.get("exogene", None)
        if exogene is not None:
            exogene = exogene.to(self.device, dtype=torch.float)

        labels = batch.get("labels", None)
        if labels is not None:
            labels = labels.to(self.device, dtype=torch.long)

        if require_labels and labels is None:
            raise KeyError(
                "Batch is missing required key 'labels'. "
                "This loader appears to be unsupervised/inference-only."
            )

        return ts, exogene, labels

    def _forward_batch(
        self,
        ts: torch.Tensor,
        exogene: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass helper.

        Best practice is for the model to support:
            forward(ts, exogene=None)

        If your model does not, keep the conditional below.
        """
        if exogene is None:
            return self.model(ts)
        return self.model(ts, exogene)

    def _train_epoch(self) -> Tuple[float, float]:
        self.model.train()

        epoch_loss, correct, total = 0.0, 0, 0

        for batch in self.train_loader:
            ts, exogene, labels = self._prepare_batch(batch, require_labels=True)
            assert labels is not None

            logits = self._forward_batch(ts, exogene)
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            self.writer.add_scalar("step_loss", loss.item(), self.passed_steps)
            self.passed_steps += 1

        return epoch_loss / len(self.train_loader), correct / total

    def _eval_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()

        epoch_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for batch in loader:
                ts, exogene, labels = self._prepare_batch(batch, require_labels=True)
                assert labels is not None

                logits = self._forward_batch(ts, exogene)
                loss = self.criterion(logits, labels)

                epoch_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return epoch_loss / len(loader), correct / total

    def _apply_metrics(self, y: np.ndarray, y_hat: np.ndarray, y_hat_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        return self.metrics(y, y_hat, y_hat_prob) if y_hat_prob is not None else self.metrics(y, y_hat)

    # ------------------------------------------------------------------
    # Checkpointing utilities
    # ------------------------------------------------------------------
    def _update_log(self, epoch: int, valid_acc: float) -> None:
        self.log.update(
            {
                "epoch_best_loss": epoch,
                "value_best_loss": self.best_loss,
                "valid_accuracy": valid_acc,
                "loss_train_history": self.loss_train_history,
                "loss_valid_history": self.loss_valid_history,
                "accuracy_train_history": self.acc_train_history,
                "accuracy_valid_history": self.acc_valid_history,
                "model_state_dict": self._state_dict_cpu(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
        )

    def _save_checkpoint(self, tag: str) -> None:
        checkpoint_file = self.checkpoint_path.with_stem(f"{self.checkpoint_path.stem}_{tag}")
        torch.save(self.log, checkpoint_file)
        if self.verbose:
            logging.info("Checkpoint saved to %s", checkpoint_file)

    def _finalise_training(self, elapsed: float) -> None:
        self.log["training_time"] = round(elapsed, 3)
        if self.save_checkpoint:
            self._save_checkpoint("final")
        self.writer.flush()
        self.writer.close()

    def _state_dict_cpu(self) -> Dict[str, torch.Tensor]:
        """Return *model* state dict on **CPU** (safe for serialization)."""
        if isinstance(self.model, nn.DataParallel):
            return {k: v.cpu() for k, v in self.model.module.state_dict().items()}
        return {k: v.cpu() for k, v in self.model.state_dict().items()}
