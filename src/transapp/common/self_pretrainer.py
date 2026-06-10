from __future__ import annotations

"""Self-supervised pre-trainer.

This class borrows the overall structure and conveniences of
`BaseClassifierTrainer` (logging, checkpointing, optional early-stopping,
TensorBoard, transparent CPU⇄GPU state-dict handling, etc.) while keeping the
reconstruction-style loss pipeline of the original `BasedSelfPretrainer`.

Key improvements
----------------
* **Unified scheduler interface** - you can now pass any scheduler name from
  `SchedulerType` (defined in *schedulers.py* or the snippet you provided) via
  the argument ``scheduler_name``. The helper ``get_scheduler`` builds the
  correct scheduler instance with warm-up and other specialised kwargs.
* **Early-stopping** - opt-in patience-based early-stopping working exactly as
  in `BaseClassifierTrainer` (after an optional warm-up period).
* **TensorBoard logging** - losses, LR and (optional) custom metrics are logged
  out-of-the-box.
* **Cleaner device handling & optional `nn.DataParallel`** - identical to
  `BaseClassifierTrainer`.
* **Gradient clipping** - optional, set ``max_grad_norm>0``.
* **Strict typing + modern Python (3.11+) features.**
"""

from collections.abc import Callable
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import asdict

import torch
from torch import nn, optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
# Scheduler helper imports (assumed to live in the same package)
# ---------------------------------------------------------------------------
from optim.lr_scheduler import SchedulerType, get_scheduler
from optim.early_stopper import EarlyStopper


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class SelfPretrainer:
    """Generic *self-supervised* trainer with flexible scheduler support.

    Parameters
    ----------
    model:
        The neural network to optimise - must return either ``recon`` or
        ``(recon, loss)`` when *loss_in_model=True*.
    train_loader, valid_loader:
        Iterable yielding **only inputs**. Labels are *not* required.
    criterion:
        Reconstruction loss function - defaults to :class:`torch.nn.MSELoss`.
        If your model already computes the loss internally set
        ``loss_in_model=True`` (``model`` should then return
        ``(recon, loss)``).
    mask:
        Optional callable that takes a batch of inputs and returns
        ``(mask_loss, masked_inputs)`` (same contract as the original class).
    optimizer_cls, optimizer_kwargs:
        Custom optimiser and kwargs. Defaults to AdamW with lr=1e‑3.
    scheduler_name, scheduler_kwargs, num_warmup_steps:
        Name from :class:`SchedulerType` (str or enum value) and extra kwargs
        passed to :func:`get_scheduler`. If *None*, no LR scheduling is used.
    patience_es, n_warmup_epochs:
        Early‑stopping patience **after** the warm‑up period. Set either to
        *None* to disable early‑stopping.
    writer:
        A ready‑made :class:`~torch.utils.tensorboard.SummaryWriter`. If left
        *None*, one is created under ``runs/``.
    save_checkpoint, checkpoint_path:
        When *True*, save a checkpoint on every improvement of the validation
        loss and at the very end of training.
    max_grad_norm:
        If >0, perform gradient‑clipping with this norm.
    device, use_data_parallel:
        Same semantics as in ``BaseClassifierTrainer``.
    verbose:
        Print progress every epoch.
    plot_history:
        After training, call :pymeth:`plot_history` (requires Matplotlib).
    """

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader] = None,
        criterion: Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor] = nn.MSELoss(),
        mask: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        loss_in_model: bool = False,
        optimizer_cls: type[optim.Optimizer] = optim.AdamW,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler_name: str | SchedulerType | None = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        patience_es: Optional[int] = None,
        n_warmup_epochs: int = 0,
        max_grad_norm: float = 0.0,
        device: str | torch.device = "cuda",
        use_data_parallel: bool = False,
        writer: Optional[SummaryWriter] = None,
        checkpoint_path: Optional[str | os.PathLike[str]] = None,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        # ------------------------------------------------------------------
        # House‑keeping
        # ------------------------------------------------------------------
        self.device = torch.device(device)
        self.verbose = verbose
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.mask = mask
        self.loss_in_model = loss_in_model
        self.max_grad_norm = max_grad_norm
        self.n_warmup_epochs = n_warmup_epochs
        self.best_loss = float("inf")
        self.passed_epochs = 0
        self.passed_steps = 0

        # ------------------------------------------------------------------
        # Model, data‑parallel & device
        # ------------------------------------------------------------------
        if use_data_parallel and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        self.model = model.to(self.device)

        # ------------------------------------------------------------------
        # Optimiser & scheduler
        # ------------------------------------------------------------------
        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 1e-4, "weight_decay": 0.0}
        self.optimizer: Optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)  # type: ignore[arg-type]

        if scheduler_name is not None:
            self.scheduler = get_scheduler(
                scheduler_name,
                optimizer=self.optimizer,
                num_warmup_steps=len(train_loader) * n_warmup_epochs,
                num_training_steps=len(train_loader) * 100,  # placeholder, can be updated via ``set_training_steps``
                scheduler_specific_kwargs=scheduler_kwargs,
            )
            self._scheduler_requires_plateau_value = (
                SchedulerType(str(scheduler_name)) == SchedulerType.REDUCE_ON_PLATEAU
            )
        else:
            self.scheduler = None
            self._scheduler_requires_plateau_value = False

        # ------------------------------------------------------------------
        # Early‑stopping
        # ------------------------------------------------------------------
        self.early_stopper = EarlyStopper(patience_es) if patience_es is not None else None

        # ------------------------------------------------------------------
        # Logging / TensorBoard
        # ------------------------------------------------------------------
        runs_dir = Path("runs")
        runs_dir.mkdir(exist_ok=True)
        if writer is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            writer = SummaryWriter(log_dir=runs_dir / f"pretrain_{timestamp}")
        self.writer = writer

        self.checkpoint_path = Path(checkpoint_path or Path.cwd() / "pretrained_model.pt")

        self.loss_train_history: List[float] = []
        self.loss_valid_history: List[float] = []

        if self.verbose:
            logging.basicConfig(level=logging.INFO, format="%(message)s")
            logging.info("SelfPretrainer initialised  |  device: %s", self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_max_training_steps_scheduler(self, n_max_epochs: int) -> None:
        """Update the total training steps for schedulers that need it."""
        if self.scheduler is None:
            return
        requires_steps = isinstance(self.scheduler, (  # HuggingFace schedulers
            torch.optim.lr_scheduler.LambdaLR,
            torch.optim.lr_scheduler.CosineAnnealingLR,
        ))
        if requires_steps:
            self.scheduler.total_steps = len(self.train_loader) * n_max_epochs  # type: ignore[attr-defined]

    def train(self, n_epochs: int = 10) -> None:
        tic = time.time()
        for epoch in range(1, n_epochs + 1):
            train_loss = self._train_epoch(epoch)
            self.loss_train_history.append(train_loss)

            # ---------------------------------------------------------
            # Validation (optional)
            # ---------------------------------------------------------
            if self.valid_loader is not None:
                valid_loss = self._eval_epoch(self.valid_loader)
                self.loss_valid_history.append(valid_loss)
            else:
                valid_loss = train_loss

            # ---------------------------------------------------------
            # LR scheduling
            # ---------------------------------------------------------
            if self.scheduler is not None:
                if self._scheduler_requires_plateau_value:
                    self.scheduler.step(valid_loss)  # type: ignore[arg-type]
                # batch‑wise schedulers are stepped inside the training loop
                elif not hasattr(self.scheduler, "batch_step"):
                    self.scheduler.step()

            # ---------------------------------------------------------
            # Early‑stopping
            # ---------------------------------------------------------
            if (
                self.early_stopper is not None
                and epoch > self.n_warmup_epochs
                and self.early_stopper(valid_loss)
            ):
                if self.verbose:
                    logging.info("Early stopping triggered at epoch %d", epoch)
                break

            # ---------------------------------------------------------
            # Logging
            # ---------------------------------------------------------
            if self.verbose:
                logging.info(
                    "Epoch %3d/%d | train=%.6f | valid=%.6f | lr=%.2e",
                    epoch,
                    n_epochs,
                    train_loss,
                    valid_loss,
                    self.optimizer.param_groups[0]["lr"],
                )

            self.writer.add_scalars(
                "MeanLossPerEpoch", {"train": train_loss, "valid": valid_loss}, epoch
            )
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

            # ---------------------------------------------------------
            # Checkpointing
            # ---------------------------------------------------------
            if valid_loss < self.best_loss and epoch >= self.n_warmup_epochs:
                self.best_loss = valid_loss
                self._save_checkpoint(tag="best")

            self.passed_epochs += 1

        elapsed = round(time.time() - tic, 3)
        if self.verbose:
            logging.info("Training completed in %ss", elapsed)

        self._save_checkpoint(tag="final")
        self.writer.flush()
        self.writer.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _unpack_batch(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Supported batch formats from TSDataset:
            {"ts": ts}
            {"ts": ts, "exogene": exogene}
            {"ts": ts, "labels": labels}                    # labels ignored here
            {"ts": ts, "exogene": exogene, "labels": labels}  # labels ignored here
        """
        if not isinstance(batch, dict):
            raise TypeError(f"Expected batch to be a dict, got {type(batch).__name__}.")

        if "ts" not in batch:
            raise KeyError("Batch is missing required key 'ts'.")

        ts = batch["ts"].float()

        exogene = batch.get("exogene", None)
        if exogene is not None:
            exogene = exogene.float()

        return ts, exogene


    def _forward_model(
        self,
        inputs: torch.Tensor,
        exogene: Optional[torch.Tensor],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward helper supporting models with or without exogenous inputs.
        """
        if exogene is None:
            return self.model(inputs)
        return self.model(inputs, exogene)

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        n_batches = len(self.train_loader)

        for step, batch in enumerate(self.train_loader, start=1):
            ts, exogene = self._unpack_batch(batch)

            if self.mask is not None:
                mask_loss, ts_masked = self.mask(ts)
                inputs = ts_masked.to(self.device)
                mask_loss = mask_loss.to(self.device)
            else:
                inputs = ts.to(self.device)
                mask_loss = None

            target = ts.to(self.device)

            if exogene is not None:
                exogene = exogene.to(self.device)

            self.optimizer.zero_grad()

            if self.loss_in_model:
                _, loss_tensor = self._forward_model(inputs, exogene)
                loss = loss_tensor.mean()
            else:
                outputs = self._forward_model(inputs, exogene)
                if mask_loss is not None:
                    loss = self.criterion(outputs, target, mask_loss)
                else:
                    loss = self.criterion(outputs, target)

            loss.backward()

            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            self.writer.add_scalar("loss", loss.item(), self.passed_steps)

            if self.scheduler is not None and not self._scheduler_requires_plateau_value:
                self.scheduler.step()

            running_loss += loss.item()
            self.passed_steps += 1

        return running_loss / n_batches

    def _eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                ts, exogene = self._unpack_batch(batch)

                if self.mask is not None:
                    mask_loss, ts_masked = self.mask(ts)
                    inputs = ts_masked.to(self.device)
                    mask_loss = mask_loss.to(self.device)
                else:
                    inputs = ts.to(self.device)
                    mask_loss = None

                target = ts.to(self.device)

                if exogene is not None:
                    exogene = exogene.to(self.device)

                if self.loss_in_model:
                    _, loss_tensor = self._forward_model(inputs, exogene)
                    loss = loss_tensor.mean()
                else:
                    outputs = self._forward_model(inputs, exogene)
                    if mask_loss is not None:
                        loss = self.criterion(outputs, target, mask_loss)
                    else:
                        loss = self.criterion(outputs, target)

                running_loss += loss.item()

        return running_loss / len(loader)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _save_checkpoint(self, tag: str = "epoch") -> None:
        core_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        state = {
            "model_state_dict": core_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_train_history": self.loss_train_history,
            "loss_valid_history": self.loss_valid_history,
            "config": asdict(core_model.config) if hasattr(core_model, "config") else None,
        }
        file = self.checkpoint_path.with_stem(f"{self.checkpoint_path.stem}_{tag}")
        torch.save(state, file)
        if self.verbose:
            logging.info("Checkpoint saved to %s", file)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def set_lr(self, new_lr: float) -> None:
        for g in self.optimizer.param_groups:
            g["lr"] = new_lr

