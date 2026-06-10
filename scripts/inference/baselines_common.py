#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared model construction + sampling for the baselines (no duplicated logic).

A single place to rebuild ``timegan`` / ``timevae`` / ``diffusion_ts`` from a
checkpoint (``ckpt['config']`` + ``ckpt['model']``) and to draw samples, returning
curves as a flat ``[N, T]`` array in Watts — the representation consumed by the TSTR
data loader and written as ``loadit_samples.npy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src.baselines import Diffusion_TS, TimeGAN, TimeVAE


def get_cfg(cfg: Any, *keys: str, default: Any = None) -> Any:
    """Nested lookup tolerant to dict-like and attribute-like (OmegaConf) configs."""
    current = cfg
    for key in keys:
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
        else:
            if not hasattr(current, key):
                return default
            current = getattr(current, key)
    return current


def require_cfg(cfg: Any, *keys: str) -> Any:
    value = get_cfg(cfg, *keys, default=None)
    if value is None:
        raise ValueError(f"Missing required config value: {'.'.join(keys)}")
    return value


@dataclass
class BaselineModel:
    baseline: str
    model: Any
    cfg: Any
    n_days: Optional[int]
    patch_length_day: Optional[int]
    seq_length: int
    feature_size: int
    scale_min: float
    scale_max: float


# ---------------------------------------------------------------------------
# diffusion_ts dimension inference (from the checkpoint state dict)
# ---------------------------------------------------------------------------
def _find_by_suffix(state_dict: Dict[str, torch.Tensor], *suffixes: str):
    for key, tensor in state_dict.items():
        for suffix in suffixes:
            if key.endswith(suffix):
                return key, tensor
    return None


def _infer_feature_size(state_dict) -> Optional[int]:
    emb = _find_by_suffix(state_dict, "emb.sequential.1.weight")
    if emb is not None:
        return int(emb[1].shape[1])
    combine = _find_by_suffix(state_dict, "combine_s.weight")
    if combine is not None:
        return int(combine[1].shape[0])
    return None


def _infer_seq_length(state_dict) -> Optional[int]:
    pos = _find_by_suffix(state_dict, "pos_enc.pe", "pos_dec.pe")
    return int(pos[1].shape[1]) if pos is not None else None


def _resolve_diffusion_ts_dims(
    cfg: Any,
    state_dict,
    seq_length: Optional[int],
    feature_size: Optional[int],
) -> Tuple[int, int, Optional[int], Optional[int]]:
    nb_days = get_cfg(cfg, "data", "nb_days")
    patch = get_cfg(cfg, "data", "patch_length_day")

    fs = feature_size
    if fs is None:
        fs = _infer_feature_size(state_dict)
    if fs is None:
        fs = get_cfg(cfg, "model", "feature_size", default=get_cfg(cfg, "model", "in_channels"))

    sl = seq_length
    if sl is None:
        sl = _infer_seq_length(state_dict)
    if sl is None:
        sl = get_cfg(cfg, "model", "seq_length")
    if sl is None and nb_days is not None and patch is not None:
        sl = int(nb_days) * int(patch)

    if fs is None or sl is None:
        raise ValueError("Unable to resolve diffusion_ts feature_size or seq_length.")
    return int(sl), int(fs), nb_days, patch


_DIFFUSION_TS_OPTIONAL = (
    "n_layer_enc", "n_layer_dec", "d_model", "loss_type", "beta_schedule", "n_heads",
    "mlp_hidden_times", "eta", "attn_pd", "resid_pd", "kernel_size", "padding_size",
    "use_ff", "reg_weight",
)


def build_model(
    baseline: str,
    ckpt_path: str,
    device: torch.device,
    *,
    seq_length: Optional[int] = None,
    feature_size: Optional[int] = None,
    sampling_timesteps: Optional[int] = None,
) -> BaselineModel:
    """Rebuild a baseline from its checkpoint and load its weights."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    nb_days = get_cfg(cfg, "data", "nb_days")
    patch = get_cfg(cfg, "data", "patch_length_day")
    scale_min = require_cfg(cfg, "data", "value_scale_min")
    scale_max = require_cfg(cfg, "data", "value_scale_max")

    if baseline == "timegan":
        seq = int(nb_days) * int(patch)
        model = TimeGAN(
            input_length=seq,
            input_dim=get_cfg(cfg, "model", "input_dim", default=1),
            hidden_dim=require_cfg(cfg, "model", "hidden_dim"),
            num_layers=require_cfg(cfg, "model", "num_layers"),
            z_dim=require_cfg(cfg, "model", "z_dim"),
            dropout=get_cfg(cfg, "model", "dropout", default=0.0),
        )
        model.load_state_dict(ckpt["model"])
        fs = int(get_cfg(cfg, "model", "input_dim", default=1))
    elif baseline == "timevae":
        seq = int(nb_days) * int(patch)
        model = TimeVAE(
            input_length=seq,
            latent_dim=require_cfg(cfg, "model", "latent_dim"),
            hidden_channels=require_cfg(cfg, "model", "hidden_channels"),
            kernel_size=require_cfg(cfg, "model", "kernel_size"),
            dropout=get_cfg(cfg, "model", "dropout", default=0.0),
        )
        model.load_state_dict(ckpt["model"])
        fs = 1
    elif baseline == "diffusion_ts":
        state_dict = ckpt["model"]
        seq, fs, nb_days, patch = _resolve_diffusion_ts_dims(cfg, state_dict, seq_length, feature_size)
        timesteps = get_cfg(cfg, "model", "timesteps")
        if timesteps is None:
            timesteps = require_cfg(cfg, "training", "diffusion_steps")
        model_kwargs: Dict[str, Any] = {"seq_length": seq, "feature_size": fs, "timesteps": timesteps}
        for key in _DIFFUSION_TS_OPTIONAL:
            value = get_cfg(cfg, "model", key, default=None)
            if value is not None:
                model_kwargs[key] = value
        st = get_cfg(cfg, "model", "sampling_timesteps")
        if sampling_timesteps is not None:
            st = sampling_timesteps
        if st is not None:
            model_kwargs["sampling_timesteps"] = st
        model = Diffusion_TS(**model_kwargs)
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unknown baseline {baseline!r}.")

    model = model.to(device).eval()
    return BaselineModel(
        baseline=baseline, model=model, cfg=cfg,
        n_days=nb_days, patch_length_day=patch,
        seq_length=int(seq), feature_size=int(fs),
        scale_min=float(scale_min), scale_max=float(scale_max),
    )


@torch.no_grad()
def sample_curves(bm: BaselineModel, n_samples: int, batch_size: int, device: torch.device) -> np.ndarray:
    """Draw ``n_samples`` curves and return a flat ``[N, T]`` array in Watts."""
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0.")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    parts = []
    done = 0
    while done < n_samples:
        b = min(batch_size, n_samples - done)
        if bm.baseline in ("timegan", "timevae"):
            s = bm.model.sample(b, device=device)
        else:  # diffusion_ts
            s = bm.model.generate_mts(batch_size=b)
        parts.append(s.detach().cpu().numpy())
        done += b
    arr = np.concatenate(parts, axis=0)
    arr = arr * (bm.scale_max - bm.scale_min) + bm.scale_min
    return arr.reshape(arr.shape[0], -1).astype(np.float32)
