#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.baselines import TimeVAE
from src.evaluation.evaluate import compute_report
from src.evaluation.features_extractor import ROCKET
from src.evaluation.utils import to_jsonable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate samples with TimeVAE baseline.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to timevae checkpoint.")
    parser.add_argument("--dataset", type=str, default="cer", help="Dataset used")
    parser.add_argument("--num-samples", type=int, default=32, help="Number of samples to generate.")
    parser.add_argument("--out-dir", type=str, default="outputs/timevae", help="Output directory.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    input_length = cfg.data.nb_days * cfg.data.patch_length_day
    model = TimeVAE(
        input_length=input_length,
        latent_dim=cfg.model.latent_dim,
        hidden_channels=cfg.model.hidden_channels,
        kernel_size=cfg.model.kernel_size,
        dropout=cfg.model.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    samples = model.sample(args.num_samples, device=device).cpu().numpy()

    scale_min = cfg.data.value_scale_min
    scale_max = cfg.data.value_scale_max
    samples = samples * (scale_max - scale_min) + scale_min
    samples = samples.reshape(args.num_samples, cfg.data.nb_days, cfg.data.patch_length_day)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "timevae_samples.npy", samples)

    REAL_DATA_PATH = f"data/{args.dataset}/test_data.npy"
    REAL_DATA_TRAIN_PATH = f"data/{args.dataset}/train_data.npy"

    real_data = np.load(REAL_DATA_PATH)
    real_data_train = np.load(REAL_DATA_TRAIN_PATH)[:1000]

    if len(real_data.shape)==3:
        real_data = real_data.reshape(real_data.shape[0], -1)

    if len(real_data_train.shape)==3:
        real_data_train = real_data_train.reshape(real_data_train.shape[0], -1)

    report = compute_report(
        real_data=real_data,
        synth_data=samples.reshape(samples.shape[0], -1),
        real_data_train=real_data_train,
        start_date="01/01/2024",
        features_extractor=ROCKET().to(device),
        output_dir=out_dir,
        plot_set="full",
        log_metrics=True,
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
