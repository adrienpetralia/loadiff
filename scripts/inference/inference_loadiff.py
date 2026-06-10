#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified Hydra-driven inference entrypoint for the Loadiff diffusion model.

Replaces the two legacy scripts (``inference_loadit_with_cond.py`` and
``inference_loadit_no_cond.py``) with a single, modular entrypoint supporting
three inference modes selected via ``inference.mode``:

  - ``unconditional``       : generate without conditioning labels.
  - ``dataset_conditioned`` : copy real exogenous features (calendar [+temperature])
                              and labels from a reference dataset split.
  - ``user_conditioned``    : generate from label values chosen by the user in YAML.

The DiT architecture is rebuilt from the training config stored in the checkpoint
(``ckpt['args']``); the Hydra config only carries inference-time parameters.

Example:
    python -m scripts.inference.inference_loadiff \\
        --config-name inference_loadiff_dataset_conditioned \\
        inference.ckpt_path=/path/to/0216000.pt inference.split=test
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.loadit.models import DiT
from src.loadit.diffusion import create_diffusion
from src.helpers.dataset import BaseParquetDailyDataset
from src.helpers.loadiff_inference import (
    build_calendar_exog,
    build_user_labels,
    get_dataset_class,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading (architecture recovered from ckpt['args'])
# ---------------------------------------------------------------------------
def load_model_from_ckpt(
    ckpt_path: str,
    device: torch.device,
    prefer_ema: bool = True,
) -> Tuple[DiT, DictConfig, str]:
    """Rebuild a DiT from a training checkpoint and load its weights.

    Reads the OmegaConf training config stored under ``ckpt['args']`` to
    reconstruct the exact DiT hyperparameters, then loads EMA weights when
    available (and ``prefer_ema``), otherwise the raw model weights.

    Args:
        ckpt_path: Path to a ``.pt`` checkpoint saved during training.
        device: Torch device to load the model onto.
        prefer_ema: Prefer EMA weights over raw model weights when present.

    Returns:
        ``(model, train_cfg, which_weights)`` where ``train_cfg`` is the recovered
        training config (used downstream for value_scale, bool_col_names, etc.).

    Raises:
        FileNotFoundError: If ``ckpt_path`` does not exist.
        KeyError: If the checkpoint does not embed the training config ('args').
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "args" not in ckpt:
        raise KeyError(
            f"Checkpoint {ckpt_path!r} has no embedded training config under 'args'; "
            "cannot rebuild the model architecture. Use a checkpoint saved by the "
            "current training scripts."
        )
    train_cfg: DictConfig = ckpt["args"]
    dm = train_cfg.ditmodelargs

    bool_col_names = OmegaConf.select(train_cfg, "data.bool_col_names", default=None)
    if bool_col_names:
        num_classes = len(bool_col_names)
    else:
        num_classes = int(OmegaConf.select(train_cfg, "ditmodelargs.num_classes", default=0))

    model = DiT(
        input_size=dm.input_size,
        patch_size=dm.patch_size,
        in_channels=OmegaConf.select(train_cfg, "ditmodelargs.in_channels", default=1),
        depth=dm.depth,
        hidden_size=dm.hidden_size,
        n_exo_var=dm.n_exo_var,
        temperature=bool(OmegaConf.select(train_cfg, "ditmodelargs.temperature", default=False)),
        num_classes=num_classes,
        multilabels=bool(OmegaConf.select(train_cfg, "ditmodelargs.multilabels", default=True)),
    ).to(device)

    if prefer_ema and "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
        which = "ema"
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        which = "model"
    elif "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
        which = "ema"
    else:
        raise KeyError(f"Checkpoint {ckpt_path!r} contains neither 'ema' nor 'model' weights.")

    model.eval()
    return model, train_cfg, which


# ---------------------------------------------------------------------------
# Reference dataset (dataset_conditioned mode + temperature source)
# ---------------------------------------------------------------------------
def build_reference_dataset(
    train_cfg: DictConfig,
    *,
    split: str,
    with_conditioning: bool,
) -> BaseParquetDailyDataset:
    """Build the reference dataset for a given split, mirroring training kwargs.

    Args:
        train_cfg: The training config recovered from the checkpoint.
        split: One of ``"train"``, ``"val"``, ``"test"``.
        with_conditioning: Whether to load metadata/temperature (conditioned model).

    Returns:
        An instantiated dataset restricted to the requested split's clients.
    """
    data = train_cfg.data
    dataset_cls = get_dataset_class(data.dataset)

    path_client_split = data.path_client_split
    if not os.path.exists(path_client_split):
        raise FileNotFoundError(f"Client split pickle not found: {path_client_split}")
    with open(path_client_split, "rb") as f:
        splits = pickle.load(f)
    if split not in splits:
        raise KeyError(
            f"Split {split!r} not found in {path_client_split}. "
            f"Available: {sorted(splits.keys())}."
        )
    clients = splits[split]

    kwargs: Dict[str, Any] = dict(
        path_load_curves=data.data_path,
        list_pdl=clients,
        scale_param2=data.value_scale,
        random_window=False,
    )
    if with_conditioning:
        kwargs.update(
            path_metadata=OmegaConf.select(train_cfg, "data.path_parquet_part_metadata", default=None),
            path_temperature=OmegaConf.select(train_cfg, "data.path_temperature", default=None),
            bool_col_names=list(OmegaConf.select(train_cfg, "data.bool_col_names", default=[]) or []),
            scale_meteo=OmegaConf.select(train_cfg, "data.value_scale_meteo", default=data.value_scale),
        )
    return dataset_cls(**kwargs)


# ---------------------------------------------------------------------------
# Shared batched sampling
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_sampling(
    model: DiT,
    diffusion,
    *,
    exog: torch.Tensor,
    y: Optional[torch.Tensor],
    n_samples: int,
    n_days: int,
    patch_length: int,
    value_scale: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Generate ``n_samples`` series in batches (shared across all modes).

    Args:
        model: The DiT used for sampling.
        diffusion: The diffusion process (must expose ``set_exog`` / ``p_sample_loop``).
        exog: Either ``[L, n_exo]`` (shared, broadcast across the batch) or
            ``[N, L, n_exo]`` (per-sample exog, e.g. dataset_conditioned).
        y: Optional ``[N, K]`` label tensor; ``None`` for unconditional generation.
        n_samples: Total number of samples to generate.
        n_days: Number of days per sample (sequence length in patches).
        patch_length: Points per day (e.g. 48).
        value_scale: Multiplier converting model outputs back to Watts.
        batch_size: Inference batch size.
        device: Torch device.

    Returns:
        ``[n_samples, n_days * patch_length]`` float32 array in Watts.
    """
    per_sample_exog = exog.dim() == 3
    preds: List[np.ndarray] = []
    done = 0
    while done < n_samples:
        b = min(batch_size, n_samples - done)

        exog_b = exog[done : done + b].to(device) if per_sample_exog else exog.to(device)
        diffusion.set_exog(exog_b)

        model_kwargs: Dict[str, torch.Tensor] = {}
        if y is not None:
            model_kwargs["y"] = y[done : done + b].to(device)

        samples = diffusion.p_sample_loop(
            model,
            (b, 1, n_days, patch_length),
            clip_denoised=False,
            progress=False,
            model_kwargs=model_kwargs,
            device=device,
        )  # [b, 1, n_days, patch_length]

        pred = samples.squeeze(1).flatten(start_dim=1)  # [b, T]
        preds.append((pred * value_scale).detach().cpu().numpy())
        done += b

    return np.concatenate(preds, axis=0) if preds else np.zeros((0, 0), dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-mode condition builders -> common (exog, y, true_w, row_meta, n_days, patch_length)
# ---------------------------------------------------------------------------
class ConditionBatch:
    """Container for the per-mode sampling inputs shared by all modes."""

    def __init__(
        self,
        *,
        exog: torch.Tensor,
        y: Optional[torch.Tensor],
        n_days: int,
        patch_length: int,
        n_samples: int,
        true_w: Optional[np.ndarray] = None,
        row_meta: Optional[List[Dict[str, Any]]] = None,
        label_names: Optional[List[str]] = None,
    ) -> None:
        self.exog = exog
        self.y = y
        self.n_days = n_days
        self.patch_length = patch_length
        self.n_samples = n_samples
        self.true_w = true_w
        self.row_meta = row_meta
        self.label_names = label_names or []


def _resolve_gen_window(cfg: DictConfig, train_cfg: DictConfig) -> Tuple[str, int, int]:
    """Resolve (start_date, n_days, patch_length) for synthetic-window modes."""
    start_date = cfg.inference.gen_sample_start_date
    if start_date is None:
        start_date = train_cfg.valid.gen_sample_start_date
    n_days = cfg.inference.gen_sample_days
    if n_days is None:
        n_days = int(train_cfg.valid.gen_sample_days)
    patch_length = int(train_cfg.ditmodelargs.input_size[1])
    return str(start_date), int(n_days), patch_length


def _maybe_temperature_column(cfg: DictConfig, train_cfg: DictConfig, n_days: int) -> Optional[torch.Tensor]:
    """Build a [n_days, 1] temperature column for temperature-enabled models.

    Returns ``None`` when the model was trained with ``temperature=False``. When
    ``temperature=True``, a temperature source dataset must be configured.
    """
    if not bool(OmegaConf.select(train_cfg, "ditmodelargs.temperature", default=False)):
        return None

    ts = cfg.inference.temperature_source
    if ts is None or ts.get("dataset") is None:
        raise ValueError(
            "Model was trained with temperature=True but no temperature source was "
            "provided for this mode. Set inference.temperature_source.dataset (+ paths) "
            "or use a checkpoint trained with temperature=False."
        )
    dataset_cls = get_dataset_class(ts.dataset)
    ref = dataset_cls(
        path_load_curves=ts.data_path,
        path_metadata=ts.get("path_parquet_part_metadata"),
        path_temperature=ts.get("path_temperature"),
        bool_col_names=list(ts.get("bool_col_names", []) or []),
        scale_param2=train_cfg.data.value_scale,
        scale_meteo=ts.get("value_scale_meteo", train_cfg.data.value_scale),
        random_window=False,
    )
    _, exog0, _ = ref[0]
    if exog0.shape[1] < 5:
        raise ValueError(
            "temperature_source dataset did not yield a temperature column "
            "(set its path_temperature)."
        )
    return exog0[:n_days, -1:]


def build_unconditional(cfg: DictConfig, train_cfg: DictConfig, device: torch.device) -> ConditionBatch:
    """Build inputs for ``unconditional`` generation (no labels)."""
    start_date, n_days, patch_length = _resolve_gen_window(cfg, train_cfg)
    temp_col = _maybe_temperature_column(cfg, train_cfg, n_days)
    exog = build_calendar_exog(start_date, n_days, temperature_col=temp_col)
    n_samples = int(cfg.inference.n_samples)
    logger.info("Mode 'unconditional': %d samples, %d days from %s.", n_samples, n_days, start_date)
    return ConditionBatch(exog=exog, y=None, n_days=n_days, patch_length=patch_length, n_samples=n_samples)


def build_user_conditioned(cfg: DictConfig, train_cfg: DictConfig, device: torch.device) -> ConditionBatch:
    """Build inputs for ``user_conditioned`` generation from YAML label values."""
    bool_col_names = list(OmegaConf.select(train_cfg, "data.bool_col_names", default=[]) or [])
    multilabels = bool(OmegaConf.select(train_cfg, "ditmodelargs.multilabels", default=True))

    conditioning = OmegaConf.to_container(cfg.inference.conditioning, resolve=True)
    y, row_meta = build_user_labels(
        conditioning,
        bool_col_names,
        multilabels=multilabels,
        default_num_samples=int(cfg.inference.n_samples) if cfg.inference.n_samples is not None else None,
    )

    start_date, n_days, patch_length = _resolve_gen_window(cfg, train_cfg)
    temp_col = _maybe_temperature_column(cfg, train_cfg, n_days)
    exog = build_calendar_exog(start_date, n_days, temperature_col=temp_col)
    n_samples = int(y.shape[0])
    logger.info(
        "Mode 'user_conditioned': %d samples across %d combination(s), %d days from %s.",
        n_samples, len({m["combination_id"] for m in row_meta}), n_days, start_date,
    )
    return ConditionBatch(
        exog=exog, y=y, n_days=n_days, patch_length=patch_length, n_samples=n_samples,
        row_meta=row_meta, label_names=bool_col_names,
    )


def build_dataset_conditioned(
    cfg: DictConfig, train_cfg: DictConfig, model: DiT, device: torch.device
) -> ConditionBatch:
    """Build inputs for ``dataset_conditioned`` generation (copy real exog + labels).

    When ``inference.n_samples`` exceeds the number of clients in the split, the
    split is re-looped: the real conditions (exog + labels) are reused, each extra
    pass reordered by a different seed (``inference.seed + pass``). The diffusion
    samples independent noise per row, so reused conditions yield distinct curves —
    letting you generate more synthetic samples than the real split contains
    (e.g. to augment rare appliances). When ``n_samples`` fits within the split, the
    behaviour is unchanged (first ``n_samples`` clients, in order).
    """
    n_samples = int(cfg.inference.n_samples)
    bool_col_names = list(OmegaConf.select(train_cfg, "data.bool_col_names", default=[]) or [])
    has_labels = model.num_classes > 0 and len(bool_col_names) > 0

    dataset = build_reference_dataset(train_cfg, split=cfg.inference.split, with_conditioning=True)
    value_scale = float(train_cfg.data.value_scale)
    patch_length = dataset.patch_length
    n_days = dataset.nb_days

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.inference.batch_size),
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # First pass over the split, capped at n_samples (original behaviour).
    exog_chunks: List[torch.Tensor] = []
    y_chunks: List[torch.Tensor] = []
    true_chunks: List[np.ndarray] = []
    collected = 0
    for x, exog, y in loader:
        if collected >= n_samples:
            break
        keep = min(x.shape[0], n_samples - collected)
        exog_chunks.append(exog[:keep])
        true_chunks.append((x[:keep].flatten(start_dim=1) * value_scale).numpy())
        if has_labels:
            y_chunks.append(y[:keep].long())
        collected += keep

    if collected == 0:
        raise RuntimeError(
            f"Reference dataset split {cfg.inference.split!r} yielded no samples."
        )

    exog_all = torch.cat(exog_chunks, dim=0)  # [<=n_samples, L, n_exo]
    true_w = np.concatenate(true_chunks, axis=0)  # [<=n_samples, T] Watts
    y_all = torch.cat(y_chunks, dim=0) if has_labels else None  # [<=n_samples, K] or None
    orig_idx = list(range(collected))  # source client index of each row

    # Oversample by re-looping when the split was exhausted before reaching n_samples.
    n_passes = 1
    if collected < n_samples:
        n_dataset = collected  # the split was fully consumed, so this is its size
        base_seed = int(cfg.inference.seed)
        sel_parts: List[np.ndarray] = []
        produced = n_dataset
        p = 1
        while produced < n_samples:
            take = min(n_dataset, n_samples - produced)
            perm = np.random.default_rng(base_seed + p).permutation(n_dataset)[:take]
            sel_parts.append(perm)
            produced += take
            p += 1
        n_passes = p
        sel = np.concatenate(sel_parts)
        sel_t = torch.as_tensor(sel, dtype=torch.long)

        exog_all = torch.cat([exog_all, exog_all[sel_t]], dim=0)
        true_w = np.concatenate([true_w, true_w[sel]], axis=0)
        if y_all is not None:
            y_all = torch.cat([y_all, y_all[sel_t]], dim=0)
        orig_idx.extend(int(i) for i in sel)

    final_n = exog_all.shape[0]
    row_meta = None
    if y_all is not None:
        y_np = y_all.numpy()
        row_meta = [
            {**{name: int(y_np[i, k]) for k, name in enumerate(bool_col_names)},
             "reference_index": int(orig_idx[i])}
            for i in range(y_np.shape[0])
        ]

    if n_passes > 1:
        logger.info(
            "Mode 'dataset_conditioned': %d samples from split '%s' (size %d) over %d passes "
            "(oversampling; labels=%s).",
            final_n, cfg.inference.split, collected, n_passes, has_labels,
        )
    else:
        logger.info(
            "Mode 'dataset_conditioned': %d samples from split '%s' (labels=%s).",
            final_n, cfg.inference.split, has_labels,
        )
    return ConditionBatch(
        exog=exog_all, y=y_all, n_days=n_days, patch_length=patch_length, n_samples=final_n,
        true_w=true_w, row_meta=row_meta, label_names=bool_col_names,
    )


# ---------------------------------------------------------------------------
# Saving + optional evaluation
# ---------------------------------------------------------------------------
def save_outputs(
    out_dir: str,
    pred_w: np.ndarray,
    batch: ConditionBatch,
    *,
    cfg: DictConfig,
    train_cfg: DictConfig,
    ckpt_path: str,
    which_weights: str,
    device: torch.device,
) -> None:
    """Persist generated series, conditioning metadata, config and provenance."""
    os.makedirs(out_dir, exist_ok=True)

    # Generated series (legacy filename kept for backward compatibility).
    np.save(os.path.join(out_dir, "loadit_samples.npy"), pred_w)

    if batch.true_w is not None:
        np.save(os.path.join(out_dir, "true.npy"), batch.true_w)

    if batch.y is not None:
        y_np = batch.y.detach().cpu().numpy().astype(np.int64)
        np.save(os.path.join(out_dir, "y.npy"), y_np)
        if batch.row_meta is not None:
            pd.DataFrame(batch.row_meta).to_csv(os.path.join(out_dir, "metadata.csv"), index=False)

    # Resolved Hydra config (for reproducibility) + provenance.
    OmegaConf.save(config=cfg, f=os.path.join(out_dir, "resolved_config.yaml"), resolve=True)

    gen_start = cfg.inference.gen_sample_start_date
    if gen_start is None:
        gen_start = OmegaConf.select(train_cfg, "valid.gen_sample_start_date", default=None)

    run_info = {
        "mode": cfg.inference.mode,
        "ckpt_path": str(ckpt_path),
        "which_weights": which_weights,
        "seed": int(cfg.inference.seed),
        "device": str(device),
        "n_samples": int(pred_w.shape[0]),
        "n_days": batch.n_days,
        "patch_length": batch.patch_length,
        "gen_sample_start_date": str(gen_start) if gen_start is not None else None,
        "label_names": batch.label_names,
        "diffusion_steps": int(cfg.inference.diffusion_steps)
        if cfg.inference.diffusion_steps is not None
        else int(train_cfg.training.diffusion_steps),
        "timestep_respacing": cfg.inference.timestep_respacing,
        "model_name": OmegaConf.select(train_cfg, "model_name", default=None),
        "dataset": OmegaConf.select(train_cfg, "data.dataset", default=None),
    }
    with open(os.path.join(out_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, sort_keys=True)

    logger.info("Saved outputs to: %s", out_dir)


def _load_train_reference(train_cfg: DictConfig, max_n: int = 1000) -> Optional[np.ndarray]:
    """Load the first ``max_n`` real curves of the dataset's TRAIN split (Watts).

    Used as the ``real_data_train`` privacy reference when no explicit file is given.
    Returns ``None`` (caller falls back to the test reference) if it cannot be built.
    """
    try:
        data = train_cfg.data
        split_path = data.path_client_split
        if not os.path.exists(split_path):
            return None
        with open(split_path, "rb") as f:
            splits = pickle.load(f)
        if "train" not in splits:
            return None
        clients = list(splits["train"])[: int(max_n)]
        if not clients:
            return None
        dataset_cls = get_dataset_class(OmegaConf.select(train_cfg, "data.dataset", default="smach"))
        ds = dataset_cls(
            path_load_curves=data.data_path,
            list_pdl=clients,
            scale_param2=data.value_scale,
            random_window=False,
        )
        seq = ds.nb_days * ds.patch_length
        # ds.data is the RAW parquet curves already in Watts (no per-getitem scaling here).
        return ds.data[:, :seq].cpu().numpy().astype(np.float32)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never break evaluation
        logger.warning("Could not load TRAIN reference for real_data_train (%s).", exc)
        return None


def _dedup_rows(arr: np.ndarray, name: str) -> np.ndarray:
    """Return ``arr`` with exact-duplicate rows removed (logs how many were dropped)."""
    uniq = np.unique(arr, axis=0)
    removed = arr.shape[0] - uniq.shape[0]
    if removed:
        logger.info(
            "Deduplicated %s: removed %d/%d exact-duplicate curves before metrics.",
            name, removed, arr.shape[0],
        )
    return uniq


def run_evaluation(
    out_dir: str,
    pred_w: np.ndarray,
    batch: ConditionBatch,
    cfg: DictConfig,
    train_cfg: DictConfig,
    device: torch.device,
) -> None:
    """Optional evaluation (opt-in via ``inference.evaluate``).

    Reuses ``compute_report`` + ROCKET and the relocated per-label plots. Skips
    gracefully (with a warning) when required real-data inputs are unavailable.
    """
    from src.evaluation.evaluate import compute_report
    from src.evaluation.features_extractor import ROCKET
    from src.evaluation.utils import to_jsonable

    # Resolve the real reference data.
    real_data = batch.true_w
    if real_data is None:
        real_path = cfg.inference.evaluate_paths.get("real_data")
        if real_path and os.path.exists(real_path):
            real_data = np.load(real_path)
            if real_data.ndim == 3:
                real_data = real_data.reshape(real_data.shape[0], -1)
        else:
            logger.warning(
                "inference.evaluate=true but no real reference data is available "
                "(mode has no ground truth and inference.evaluate_paths.real_data is unset/missing). "
                "Skipping evaluation."
            )
            return

    # Deduplicate exact-duplicate real curves before metrics. dataset_conditioned
    # oversampling (n_samples > split size) repeats the real reference, which biases
    # nearest-neighbour metrics (1-NN / DCR / NNDR) with distance-0 twins. No-op when
    # the reference has no exact duplicates (e.g. a real .npy file).
    real_data = _dedup_rows(real_data, "real reference")

    train_path = cfg.inference.evaluate_paths.get("real_data_train")
    if train_path and os.path.exists(train_path):
        real_data_train = np.load(train_path)[:1000]
        if real_data_train.ndim == 3:
            real_data_train = real_data_train.reshape(real_data_train.shape[0], -1)
        real_data_train = _dedup_rows(real_data_train, "real train reference")
    else:
        # No file given: use the dataset's own TRAIN split (first 1000 curves) as the
        # privacy reference rather than reusing the test reference.
        real_data_train = _load_train_reference(train_cfg, max_n=1000)
        if real_data_train is not None:
            logger.info("Using the dataset TRAIN split as real_data_train (%d curves).", real_data_train.shape[0])
        else:
            logger.warning("TRAIN reference unavailable; falling back to the test reference for real_data_train.")
            real_data_train = real_data  # already deduplicated above

    n = min(real_data.shape[0], pred_w.shape[0])
    features_extractor = ROCKET().to(device)
    start_date_str = cfg.inference.get("eval_plot_start_date", "01/01/2024")

    report = compute_report(
        real_data=real_data[:n],
        synth_data=pred_w[:n],
        real_data_train=real_data_train,
        start_date=start_date_str,
        features_extractor=features_extractor,
        output_dir=out_dir,
        plot_set="full",
        log_metrics=True,
        log_plots=True,
        return_report=True,
    )

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": to_jsonable(report.metrics),
                "plot_paths": to_jsonable(getattr(report, "plot_paths", {})),
                "meta": {"num_samples": int(n), "device": str(device)},
            },
            f,
            indent=2,
            sort_keys=True,
        )

    # Per-label binary plots when labels are available.
    if batch.y is not None and batch.label_names and batch.true_w is not None:
        from src.evaluation.inference_plots import plot_by_binary_per_label

        plot_by_binary_per_label(
            real_data=real_data[:n],
            synth_data=pred_w[:n],
            y_np=batch.y.detach().cpu().numpy()[:n],
            label_names=batch.label_names,
            run_dir=out_dir,
            start_date_str=start_date_str,
            patch_per_day=batch.patch_length,
            features_extractor=features_extractor,
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../../configs", config_name="inference_loadiff")
def main(cfg: DictConfig) -> None:
    mode = cfg.inference.mode
    valid_modes = {"unconditional", "dataset_conditioned", "user_conditioned"}
    if mode not in valid_modes:
        raise ValueError(f"Unknown inference.mode {mode!r}. Valid modes: {sorted(valid_modes)}.")

    seed = int(cfg.inference.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    requested = str(cfg.inference.device)
    device = torch.device(requested if (requested != "cuda" or torch.cuda.is_available()) else "cpu")
    logger.info("Inference mode=%s, device=%s, seed=%d", mode, device, seed)

    model, train_cfg, which = load_model_from_ckpt(
        cfg.inference.ckpt_path, device, prefer_ema=bool(cfg.inference.prefer_ema)
    )
    logger.info("Loaded DiT weights from '%s' (num_classes=%d).", which, model.num_classes)

    diffusion_steps = cfg.inference.diffusion_steps
    if diffusion_steps is None:
        diffusion_steps = int(train_cfg.training.diffusion_steps)
    diffusion = create_diffusion(
        timestep_respacing=cfg.inference.timestep_respacing,
        diffusion_steps=int(diffusion_steps),
    )

    if mode == "unconditional":
        batch = build_unconditional(cfg, train_cfg, device)
    elif mode == "user_conditioned":
        batch = build_user_conditioned(cfg, train_cfg, device)
    else:  # dataset_conditioned
        batch = build_dataset_conditioned(cfg, train_cfg, model, device)

    pred_w = run_sampling(
        model,
        diffusion,
        exog=batch.exog,
        y=batch.y,
        n_samples=batch.n_samples,
        n_days=batch.n_days,
        patch_length=batch.patch_length,
        value_scale=float(train_cfg.data.value_scale),
        batch_size=int(cfg.inference.batch_size),
        device=device,
    )

    out_dir = HydraConfig.get().runtime.output_dir
    save_outputs(
        out_dir, pred_w, batch,
        cfg=cfg, train_cfg=train_cfg, ckpt_path=cfg.inference.ckpt_path,
        which_weights=which, device=device,
    )

    if bool(cfg.inference.evaluate):
        run_evaluation(out_dir, pred_w, batch, cfg, train_cfg, device)

    # Optional post-processing / quality control (disabled by default).
    pp = OmegaConf.select(cfg, "postprocessing", default=None)
    pp_enabled = (
        pp is not None
        and bool(pp.get("enabled", False))
        and bool(pp.get("run_after_inference", False))
    )
    pp_out_dir = (
        _run_postprocessing_after_inference(
            out_dir, pp, dataset=OmegaConf.select(train_cfg, "data.dataset", default=None)
        )
        if pp_enabled
        else None
    )

    # When post-processing ran, also evaluate the cleaned curves (written to a separate
    # evaluation_postprocessed/ dir) so raw vs post-processed metrics can be compared.
    if bool(cfg.inference.evaluate) and pp_out_dir is not None:
        _run_postprocessed_evaluation(out_dir, pp_out_dir, batch, cfg, train_cfg, device)

    logger.info("Done. Generated %d samples (shape=%s).", pred_w.shape[0], pred_w.shape)


def _run_postprocessing_after_inference(run_dir: str, pp: DictConfig, *, dataset: str = None) -> str:
    """Run the QC pipeline on freshly generated curves (opt-in). Returns the QC out dir."""
    from src.postprocessing.batch_io import postprocess_directory

    config_path = pp.get("config_path")
    if config_path in (None, "", "???"):
        raise ValueError(
            "postprocessing.enabled/run_after_inference is true but postprocessing.config_path "
            "is not set. Point it to a configs/postprocessing/postprocess_generated_curves.yaml."
        )
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Post-processing config not found: {config_path}")

    pp_conf = OmegaConf.load(config_path)
    # The curves to clean are the ones we just generated.
    OmegaConf.update(pp_conf, "input.run_dir", run_dir, force_add=True)

    plaus_enabled = bool(OmegaConf.select(pp_conf, "plausibility_filter.enabled", default=False))
    if not plaus_enabled:
        # Avoid a mandatory-missing ('???') envelope_path when plausibility is disabled.
        OmegaConf.update(pp_conf, "plausibility_filter.envelope_path", None, force_add=True)
    elif bool(pp.get("auto_envelope_path", True)) and dataset:
        # Derive the calibrated envelope from the dataset name (one per train split):
        #   <envelope_root>/calibrate_<dataset>_train/plausibility_envelope.json
        envelope_root = str(pp.get("envelope_root", "runs_postprocessing"))
        envelope_filename = str(pp.get("envelope_filename", "plausibility_envelope.json"))
        env_path = os.path.join(envelope_root, f"calibrate_{dataset}_train", envelope_filename)
        if not os.path.exists(env_path):
            raise FileNotFoundError(
                f"Auto-derived plausibility envelope not found for dataset {dataset!r}:\n  {env_path}\n"
                "Calibrate it (scripts.postprocessing.calibrate_plausibility_envelopes) or set "
                "postprocessing.auto_envelope_path=false to use the config's envelope_path."
            )
        OmegaConf.update(pp_conf, "plausibility_filter.envelope_path", env_path, force_add=True)
        logger.info("Auto-derived plausibility envelope for dataset '%s': %s", dataset, env_path)

    pp_cfg = OmegaConf.to_container(pp_conf, resolve=True)
    out_dir = os.path.join(run_dir, "postprocessing")
    report = postprocess_directory(run_dir, pp_cfg, out_dir)
    logger.info(
        "Post-processing: %d total -> keep=%d repair=%d reject=%d (saved to %s)",
        report["n_total"], report["n_keep"], report["n_repair"], report["n_reject"], out_dir,
    )
    return out_dir


def _run_postprocessed_evaluation(
    run_dir: str,
    pp_out_dir: str,
    batch: ConditionBatch,
    cfg: DictConfig,
    train_cfg: DictConfig,
    device: torch.device,
) -> None:
    """Re-run the evaluation on the post-processed (cleaned) curves.

    Loads ``cleaned_curves.npy`` + ``cleaned_metadata.csv`` (whose ``curve_id`` maps each
    kept curve back to its original row), aligns the real reference (``true_w``) and the
    labels to those kept rows, and writes metrics/plots under
    ``<run_dir>/evaluation_postprocessed`` so they can be compared with the raw evaluation
    written at the run-dir root.
    """
    import pandas as pd
    from copy import copy

    cleaned_path = os.path.join(pp_out_dir, "cleaned_curves.npy")
    if not os.path.exists(cleaned_path):
        logger.warning("No cleaned_curves.npy in %s; skipping post-processed evaluation.", pp_out_dir)
        return
    cleaned = np.load(cleaned_path)
    if cleaned.shape[0] == 0:
        logger.warning("Post-processing kept 0 curves; skipping post-processed evaluation.")
        return

    meta_path = os.path.join(pp_out_dir, "cleaned_metadata.csv")
    if os.path.exists(meta_path):
        kept = pd.read_csv(meta_path)["curve_id"].to_numpy().astype(np.int64)
    else:
        kept = np.arange(cleaned.shape[0], dtype=np.int64)

    # Align the real reference + labels to the surviving rows so the comparison stays paired.
    batch_pp = copy(batch)
    if batch.true_w is not None:
        batch_pp.true_w = batch.true_w[kept]
    if batch.y is not None:
        batch_pp.y = batch.y[torch.as_tensor(kept, dtype=torch.long)]

    eval_dir = os.path.join(run_dir, "evaluation_postprocessed")
    os.makedirs(eval_dir, exist_ok=True)
    logger.info(
        "Evaluating post-processed curves (%d kept of %d) -> %s",
        cleaned.shape[0], batch.n_samples, eval_dir,
    )
    run_evaluation(eval_dir, cleaned, batch_pp, cfg, train_cfg, device)


if __name__ == "__main__":
    main()