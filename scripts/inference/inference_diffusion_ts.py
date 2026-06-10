#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Any
import json

import numpy as np
import torch

from src.baselines import Diffusion_TS
from src.evaluation.evaluate import compute_report
from src.evaluation.features_extractor import ROCKET
from src.evaluation.utils import to_jsonable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate samples with Diffusion-TS baseline.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to diffusion_ts checkpoint.")
    parser.add_argument("--dataset", type=str, default="cer", help="Dataset used")
    parser.add_argument("--num-samples", type=int, default=32, help="Number of samples to generate.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for sampling.",
    )
    parser.add_argument("--out-dir", type=str, default="outputs/diffusion_ts", help="Output directory.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    parser.add_argument(
        "--seq-length",
        type=int,
        default=None,
        help="Override sequence length (defaults to checkpoint/config-derived length).",
    )
    parser.add_argument(
        "--feature-size",
        type=int,
        default=None,
        help="Override feature size (defaults to checkpoint/config-derived size).",
    )
    parser.add_argument(
        "--sampling-timesteps",
        type=int,
        default=None,
        help="Override sampling timesteps (enables fast sampling when < timesteps).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling.")
    return parser.parse_args()


def get_cfg(cfg: Any, *keys: str, default: Any = None) -> Any:
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
        dotted = ".".join(keys)
        raise ValueError(f"Missing required config value: {dotted}")
    return value


def find_state_dict_tensor_by_suffix(
    state_dict: dict[str, torch.Tensor],
    *suffixes: str,
) -> tuple[str, torch.Tensor] | None:
    for key, tensor in state_dict.items():
        for suffix in suffixes:
            if key.endswith(suffix):
                return key, tensor
    return None


def infer_feature_size(state_dict: dict[str, torch.Tensor]) -> tuple[int | None, str | None]:
    emb_match = find_state_dict_tensor_by_suffix(state_dict, "emb.sequential.1.weight")
    if emb_match is not None:
        key, weight = emb_match
        return int(weight.shape[1]), key
    combine_match = find_state_dict_tensor_by_suffix(state_dict, "combine_s.weight")
    if combine_match is not None:
        key, weight = combine_match
        return int(weight.shape[0]), key
    return None, None


def infer_seq_length(state_dict: dict[str, torch.Tensor]) -> tuple[int | None, str | None]:
    pos_match = find_state_dict_tensor_by_suffix(state_dict, "pos_enc.pe", "pos_dec.pe")
    if pos_match is None:
        return None, None
    key, pos_enc = pos_match
    return int(pos_enc.shape[1]), key


def resolve_dimensions(
    args: argparse.Namespace,
    cfg: Any,
    state_dict: dict[str, torch.Tensor],
) -> tuple[int, int, int, int]:
    nb_days = get_cfg(cfg, "data", "nb_days")
    patch_length_day = get_cfg(cfg, "data", "patch_length_day")

    inferred_feature_size, feature_key = infer_feature_size(state_dict)
    inferred_seq_length, seq_key = infer_seq_length(state_dict)

    feature_size = args.feature_size
    if feature_size is None:
        feature_size = inferred_feature_size
    if feature_size is None:
        feature_size = get_cfg(cfg, "model", "feature_size", default=None)
    if feature_size is None:
        feature_size = get_cfg(cfg, "model", "in_channels", default=None)

    seq_length = args.seq_length
    if seq_length is None:
        seq_length = inferred_seq_length
    if seq_length is None:
        seq_length = get_cfg(cfg, "model", "seq_length", default=None)
    if seq_length is None and nb_days is not None and patch_length_day is not None:
        seq_length = nb_days * patch_length_day

    if feature_size is None or seq_length is None:
        raise ValueError("Unable to resolve feature_size or seq_length from inputs/checkpoint/config.")

    if inferred_feature_size is not None and feature_size != inferred_feature_size:
        print(
            "Warning: resolved feature_size does not match checkpoint. "
            f"resolved={feature_size}, checkpoint={inferred_feature_size} ({feature_key})."
        )
    if inferred_seq_length is not None and seq_length != inferred_seq_length:
        print(
            "Warning: resolved seq_length does not match checkpoint. "
            f"resolved={seq_length}, checkpoint={inferred_seq_length} ({seq_key})."
        )

    return int(seq_length), int(feature_size), nb_days, patch_length_day


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    state_dict = ckpt["model"]
    seq_length, feature_size, nb_days, patch_length_day = resolve_dimensions(args, cfg, state_dict)

    timesteps = get_cfg(cfg, "model", "timesteps")
    if timesteps is None:
        timesteps = require_cfg(cfg, "training", "diffusion_steps")

    sampling_timesteps = get_cfg(cfg, "model", "sampling_timesteps")
    if args.sampling_timesteps is not None:
        sampling_timesteps = args.sampling_timesteps

    model_kwargs = {
        "seq_length": seq_length,
        "feature_size": feature_size,
        "timesteps": timesteps,
    }

    print(model_kwargs)

    optional_keys = {
        "n_layer_enc": ("model", "n_layer_enc"),
        "n_layer_dec": ("model", "n_layer_dec"),
        "d_model": ("model", "d_model"),
        "loss_type": ("model", "loss_type"),
        "beta_schedule": ("model", "beta_schedule"),
        "n_heads": ("model", "n_heads"),
        "mlp_hidden_times": ("model", "mlp_hidden_times"),
        "eta": ("model", "eta"),
        "attn_pd": ("model", "attn_pd"),
        "resid_pd": ("model", "resid_pd"),
        "kernel_size": ("model", "kernel_size"),
        "padding_size": ("model", "padding_size"),
        "use_ff": ("model", "use_ff"),
        "reg_weight": ("model", "reg_weight"),
    }

    for key, path in optional_keys.items():
        value = get_cfg(cfg, *path, default=None)
        if value is not None:
            model_kwargs[key] = value

    if sampling_timesteps is not None:
        model_kwargs["sampling_timesteps"] = sampling_timesteps

    model = Diffusion_TS(**model_kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    batch_size = args.batch_size or args.num_samples
    if batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if args.num_samples <= 0:
        raise ValueError("num-samples must be positive.")

    sample_batches = []
    with torch.no_grad():
        for start in range(0, args.num_samples, batch_size):
            current_batch = min(batch_size, args.num_samples - start)
            sample_batches.append(model.generate_mts(batch_size=current_batch))

    samples = torch.cat(sample_batches, dim=0).cpu().numpy()

    scale_min = require_cfg(cfg, "data", "value_scale_min")
    scale_max = require_cfg(cfg, "data", "value_scale_max")
    samples = samples * (scale_max - scale_min) + scale_min

    if nb_days is not None and patch_length_day is not None and seq_length == nb_days * patch_length_day:
        samples = samples.reshape(args.num_samples, nb_days, patch_length_day, feature_size)
        if feature_size == 1:
            samples = samples.squeeze(-1)
    else:
        print(
            "Skipping day-based reshape: "
            f"seq_length={seq_length}, nb_days={nb_days}, patch_length_day={patch_length_day}."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "diffusion_ts_samples.npy", samples)

    REAL_DATA_PATH = f"data/{args.dataset}/test_data.npy"
    REAL_DATA_TRAIN_PATH = f"data/{args.dataset}/train_data.npy"

    real_data = np.load(REAL_DATA_PATH)
    real_data_train = np.load(REAL_DATA_TRAIN_PATH)[:1000]

    if len(real_data.shape)==3:
        real_data = real_data.reshape(real_data.shape[0], -1)

    if len(real_data_train.shape)==3:
        real_data_train = real_data_train.reshape(real_data_train.shape[0], -1)

    report = compute_report(
        real_data=real_data,          # can be np.ndarray or torch.Tensor
        synth_data=samples.reshape(samples.shape[0], -1),
        real_data_train=real_data_train,
        start_date="01/01/2024",
        features_extractor=ROCKET().to(device),         # or pass a BaseFeaturesExtractor (e.g., ROCKET(...).to(device))
        output_dir=out_dir,             # saves plots here
        plot_set="full",             # "none" | "summary" | "full" | iterable of plot names
        log_metrics=True,              # no TensorBoard writer in this test
        log_plots=True,
        return_report=True,
    )

    # ----------------------------
    # Save metrics to JSON (no printing)
    # ----------------------------
    metrics_path = out_dir / "metrics.json"

    payload = {
        "metrics": to_jsonable(report.metrics),  # dict of dicts
        "plot_paths": to_jsonable(getattr(report, "plot_paths", {})),
        "meta": {
            "ckpt": str(args.ckpt),
            "num_samples": int(args.num_samples),
            "device": str(device),
            "real_data_path": REAL_DATA_PATH,
            "real_data_train_path": REAL_DATA_TRAIN_PATH,
        },
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
