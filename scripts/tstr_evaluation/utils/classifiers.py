#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classifier wrappers for the TSTR evaluation (ROCKET and TransApp).

Both wrappers expose a uniform ``fit`` / ``predict`` / ``save`` / ``load`` API and
consume load curves in **Watts** (shape ``[N, L]``); each rescales internally so real
and synthetic data share the same normalisation:

* :class:`RocketClassifier` — the seeded ROCKET feature extractor
  (``src.evaluation.features_extractor.ROCKET``) + a scikit-learn ``RidgeClassifier``.
  Persisted to a single ``model.pkl`` (the random kernels are rebuilt deterministically
  from the stored seed).
* :class:`TransAppClassifier` — ``TransAppV2Classif`` trained with
  ``src.transapp.common.classifier_trainer.BaseClassifierTrainer``. Persisted to a
  self-contained ``checkpoint.pt``. Long yearly curves can be tiled into fixed-length
  subsequences (``subsequence_length``) with soft-voting aggregation at inference.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from scripts.tstr_evaluation.utils.data_loader import SMACH_VALUE_SCALE


# ---------------------------------------------------------------------------
# Transapp uses bare imports (``from module...``, ``from common...``,
# ``from optim...``); add its package root to sys.path before importing.
# ---------------------------------------------------------------------------
def _setup_transapp_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    transapp_root = os.path.join(repo_root, "src", "transapp")
    if transapp_root not in sys.path:
        sys.path.insert(0, transapp_root)
    return transapp_root


def resolve_device(device: str) -> str:
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


# ===========================================================================
# ROCKET
# ===========================================================================
class RocketClassifier:
    """ROCKET features + Ridge classifier over load curves in Watts."""

    def __init__(
        self,
        *,
        n_kernels: int = 10000,
        seed: int = 0,
        value_scale: float = SMACH_VALUE_SCALE,
        balance_classes: bool = True,
        device: str = "cuda",
        batch_size: int = 256,
    ) -> None:
        self.n_kernels = int(n_kernels)
        self.seed = int(seed)
        self.value_scale = float(value_scale)
        self.balance_classes = bool(balance_classes)
        self.device = device
        self.batch_size = int(batch_size)
        self.seq_len: Optional[int] = None
        self._classifier = None  # sklearn RidgeClassifier

    # -- internal -----------------------------------------------------------
    def _build_rocket(self, seq_len: int):
        import torch
        from src.evaluation.features_extractor import ROCKET

        dev = resolve_device(self.device)
        rocket = ROCKET(seq_len=seq_len, n_kernels=self.n_kernels, seed=self.seed, c_in=1)
        return rocket.to(torch.device(dev)), torch.device(dev)

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        import torch

        rocket, dev = self._build_rocket(X.shape[1])
        X_scaled = X.astype(np.float32) / self.value_scale
        feats: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(X_scaled), self.batch_size):
                xb = torch.tensor(X_scaled[start : start + self.batch_size], device=dev).unsqueeze(1)
                feats.append(rocket(xb).cpu().numpy())
        return np.concatenate(feats, axis=0)

    # -- public API ---------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RocketClassifier":
        from sklearn.linear_model import RidgeClassifier

        self.seq_len = int(X.shape[1])
        feats = self._extract_features(X)
        self._classifier = RidgeClassifier(
            class_weight="balanced" if self.balance_classes else None
        )
        self._classifier.fit(feats, y.astype(np.int64).ravel())
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        feats = self._extract_features(X)
        return self._classifier.decision_function(feats)

    def predict(self, X: np.ndarray) -> np.ndarray:
        feats = self._extract_features(X)
        return self._classifier.predict(feats).astype(np.int64)

    def save(self, path: str) -> None:
        import joblib

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(
            {
                "type": "rocket",
                "n_kernels": self.n_kernels,
                "seed": self.seed,
                "seq_len": self.seq_len,
                "value_scale": self.value_scale,
                "balance_classes": self.balance_classes,
                "classifier": self._classifier,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, *, device: str = "cuda") -> "RocketClassifier":
        import joblib

        payload = joblib.load(path)
        obj = cls(
            n_kernels=payload["n_kernels"],
            seed=payload["seed"],
            value_scale=payload["value_scale"],
            balance_classes=payload["balance_classes"],
            device=device,
        )
        obj.seq_len = payload["seq_len"]
        obj._classifier = payload["classifier"]
        return obj


# ===========================================================================
# TransApp (TransAppV2)
# ===========================================================================
def _calendar_exogene(
    start_date: str, length: int, freq: str, exogene_var: List[str]
) -> np.ndarray:
    """Sine-angle calendar features ``[n_exo, length]`` (matches TSDataset)."""
    import pandas as pd

    ts = pd.date_range(start=start_date, periods=length, freq=freq)
    arr = np.zeros((len(exogene_var), length), dtype=np.float32)
    two_pi = 2 * np.pi
    for k, var in enumerate(exogene_var):
        if var == "second":
            arr[k] = two_pi * ts.second.values / 60.0
        elif var == "minute":
            arr[k] = two_pi * ts.minute.values / 60.0
        elif var == "hour":
            arr[k] = two_pi * ts.hour.values / 24.0
        elif var == "dayofweek":
            arr[k] = two_pi * ts.dayofweek.values / 7.0
        elif var == "dayofmonth":
            arr[k] = two_pi * ts.day.values / 31.0
        elif var == "dayofyear":
            arr[k] = two_pi * ts.dayofyear.values / 365.0
        elif var == "month":
            arr[k] = two_pi * ts.month.values / 12.0
        else:
            raise ValueError(f"Unknown exogene variable {var!r}.")
    return arr


def _make_windows(
    X: np.ndarray,
    y: np.ndarray,
    subsequence_length: Optional[int],
    *,
    kept_window_idx: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tile each curve into non-overlapping windows; return (Xw, yw, group_ids).

    Tiling keeps full-year SMACH curves (e.g. 17520 points) tractable for the
    transformer's quadratic attention. Each window inherits its curve's label;
    ``group_ids`` maps windows back to their source curve so per-curve predictions
    can be recovered by soft-voting at inference.

    ``kept_window_idx`` optionally restricts which window positions are kept (same
    set for every curve) — used for the appliance-aware seasonal filter (e.g. keep
    only winter windows for CHAUFF_ELEC). Windows share the calendar exogenous
    features computed from ``start_date`` (a deliberate simplification).
    """
    if subsequence_length is None or subsequence_length >= X.shape[1]:
        return X, y, np.arange(len(X))
    win = int(subsequence_length)
    n_win = X.shape[1] // win
    if n_win < 1:
        return X, y, np.arange(len(X))
    idx = list(range(n_win)) if kept_window_idx is None else list(kept_window_idx)
    if not idx:
        raise ValueError(
            "Seasonal window filter removed all subsequences; check the appliance "
            "season vs. subsequence_length / start_date."
        )
    cube = X[:, : n_win * win].reshape(len(X), n_win, win)[:, idx, :]  # [N, k, win]
    k = len(idx)
    Xw = cube.reshape(-1, win)
    yw = np.repeat(y, k)
    group_ids = np.repeat(np.arange(len(X)), k)
    return Xw.astype(np.float32), yw.astype(np.int64), group_ids


# Appliance-aware seasonal filter on subsequence *start month* (TransApp only):
#   heating (smach CHAUFF_ELEC, cer_bis heater) -> December..February ;
#   AC (smach CLIM) -> July..August ; anything else (ECS, ev, cooker, ...) -> no filter.
# Keyed by label name (unique across datasets: 'heater' only exists in cer_bis).
SEASONAL_START_MONTHS: Dict[str, set] = {
    "CHAUFF_ELEC": {12, 1, 2},
    "heater": {12, 1, 2},
    "CLIM": {7, 8},
}


class _TransAppDataset:
    """Torch dataset yielding ``{"ts", "exogene", "labels"}`` dicts."""

    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        *,
        value_scale: float,
        exogene_var: List[str],
        start_date: str,
        freq: str,
    ) -> None:
        import torch

        self.torch = torch
        self.X = X.astype(np.float32)
        self.y = None if y is None else y.astype(np.int64)
        self.value_scale = float(value_scale)
        self.exogene_var = exogene_var
        self.start_date = start_date
        self.freq = freq
        self._exo = _calendar_exogene(start_date, X.shape[1], freq, exogene_var)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        ts = self.torch.tensor(self.X[idx] / self.value_scale, dtype=self.torch.float32).unsqueeze(0)
        sample = {"ts": ts, "exogene": self.torch.tensor(self._exo, dtype=self.torch.float32)}
        if self.y is not None:
            sample["labels"] = self.torch.tensor(int(self.y[idx]), dtype=self.torch.long)
        return sample


class TransAppClassifier:
    """TransAppV2 deep classifier with optional self-supervised pretrained init."""

    DEFAULT_EXOGENE = ["minute", "hour", "dayofweek", "dayofmonth", "dayofyear", "month"]

    def __init__(
        self,
        *,
        value_scale: float = SMACH_VALUE_SCALE,
        exogene_var: Optional[List[str]] = None,
        start_date: str = "01/01/2021",
        freq: str = "30min",
        subsequence_length: Optional[int] = 1024,
        target_label: Optional[str] = None,
        d_model: int = 128,
        n_encoder_layers: int = 3,
        n_head: int = 8,
        epochs: int = 15,
        batch_size: int = 32,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        patience_es: int = 3,
        n_warmup_epochs: int = 1,
        device: str = "cuda",
        seed: int = 0,
        num_workers: int = 0,
    ) -> None:
        self.value_scale = float(value_scale)
        self.exogene_var = list(exogene_var) if exogene_var else list(self.DEFAULT_EXOGENE)
        self.start_date = start_date
        self.freq = freq
        self.subsequence_length = subsequence_length
        self.target_label = target_label
        self.d_model = int(d_model)
        self.n_encoder_layers = int(n_encoder_layers)
        self.n_head = int(n_head)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.patience_es = int(patience_es)
        self.n_warmup_epochs = int(n_warmup_epochs)
        self.device = device
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        self._model = None
        self._config = None

    # -- model construction -------------------------------------------------
    def _build_model(self, pretrained_path: Optional[str] = None):
        import torch

        _setup_transapp_path()
        from model.backbone.TransAppV2 import TransAppV2Classif, TransAppV2Config

        config = TransAppV2Config(
            c_in=1,
            n_exogene_var=len(self.exogene_var),
            nb_class=2,
            d_model=self.d_model,
            n_encoder_layers=self.n_encoder_layers,
            n_head=self.n_head,
        )
        model = TransAppV2Classif(config)
        if pretrained_path:
            log = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            state = log.get("model_state_dict", log) if isinstance(log, dict) else log
            # Pretrained backbone may not include the classification head -> non-strict.
            model.load_state_dict(state, strict=False)
        self._config = config
        return model

    def _kept_windows(self, seq_len: int) -> Optional[List[int]]:
        """Window indices to keep for the appliance season, or ``None`` (no filter)."""
        if self.subsequence_length is None or self.subsequence_length >= seq_len:
            return None
        keep_months = SEASONAL_START_MONTHS.get(self.target_label) if self.target_label else None
        if not keep_months:
            return None  # ECS / unknown -> no seasonal filter
        import pandas as pd

        win = int(self.subsequence_length)
        n_win = seq_len // win
        ts = pd.date_range(start=self.start_date, periods=seq_len, freq=self.freq)
        return [w for w in range(n_win) if ts[w * win].month in keep_months]

    def _make_loader(self, X, y, *, shuffle: bool):
        import torch

        Xw, yw, _ = _make_windows(
            X, y, self.subsequence_length, kept_window_idx=self._kept_windows(X.shape[1])
        )
        ds = _TransAppDataset(
            Xw, yw, value_scale=self.value_scale,
            exogene_var=self.exogene_var, start_date=self.start_date, freq=self.freq,
        )
        return torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers
        )

    # -- public API ---------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        pretrained_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> "TransAppClassifier":
        import torch
        from torch import nn

        _setup_transapp_path()
        from common.classifier_trainer import BaseClassifierTrainer
        from common.metrics import ImbalancedClassificationMetrics

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = resolve_device(self.device)

        self._model = self._build_model(pretrained_path)
        train_loader = self._make_loader(X, y, shuffle=True)
        val_loader = (
            self._make_loader(X_val, y_val, shuffle=False)
            if X_val is not None and y_val is not None
            else None
        )

        ckpt = checkpoint_path or os.path.join(os.getcwd(), "transapp_model.pt")
        trainer = BaseClassifierTrainer(
            self._model,
            train_loader,
            val_loader,
            optimizer_kwargs={"lr": self.lr, "weight_decay": self.weight_decay},
            criterion=nn.CrossEntropyLoss(),
            patience_es=self.patience_es,
            device=device,
            n_warmup_epochs=self.n_warmup_epochs,
            metrics=ImbalancedClassificationMetrics(),
            save_checkpoint=False,
            checkpoint_path=ckpt,
            verbose=True,
        )
        trainer.train(self.epochs)
        # Keep the best-validation weights when a val loader was provided.
        best_state = trainer.log.get("model_state_dict")
        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.to(device).eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Per-curve positive-class probability (soft-voting over windows)."""
        import torch

        device = resolve_device(self.device)
        self._model.to(device).eval()
        Xw, _yw, group_ids = _make_windows(
            X, np.zeros(len(X), dtype=np.int64), self.subsequence_length,
            kept_window_idx=self._kept_windows(X.shape[1]),
        )
        ds = _TransAppDataset(
            Xw, None, value_scale=self.value_scale,
            exogene_var=self.exogene_var, start_date=self.start_date, freq=self.freq,
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        probs: List[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                ts = batch["ts"].to(device, dtype=torch.float)
                exo = batch["exogene"].to(device, dtype=torch.float)
                logits = self._model(ts, exo)
                probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        win_proba = np.concatenate(probs)
        # Aggregate window probabilities back to per-curve via mean.
        n_curves = int(group_ids.max()) + 1
        out = np.zeros(n_curves, dtype=np.float32)
        for cid in range(n_curves):
            out[cid] = win_proba[group_ids == cid].mean()
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(np.int64)

    def save(self, path: str) -> None:
        import torch
        from dataclasses import asdict

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "type": "transapp",
                "config": asdict(self._config),
                "model_state_dict": {k: v.cpu() for k, v in self._model.state_dict().items()},
                "value_scale": self.value_scale,
                "exogene_var": self.exogene_var,
                "start_date": self.start_date,
                "freq": self.freq,
                "subsequence_length": self.subsequence_length,
                "target_label": self.target_label,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, *, device: str = "cuda") -> "TransAppClassifier":
        import torch

        _setup_transapp_path()
        from model.backbone.TransAppV2 import TransAppV2Classif, TransAppV2Config

        payload = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(
            value_scale=payload["value_scale"],
            exogene_var=payload["exogene_var"],
            start_date=payload["start_date"],
            freq=payload["freq"],
            subsequence_length=payload["subsequence_length"],
            target_label=payload.get("target_label"),
            device=device,
        )
        config = TransAppV2Config(**payload["config"])
        model = TransAppV2Classif(config)
        model.load_state_dict(payload["model_state_dict"])
        obj._model = model.to(resolve_device(device)).eval()
        obj._config = config
        return obj
