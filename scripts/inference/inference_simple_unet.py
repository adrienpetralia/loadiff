#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import torch

from src.baselines import SimpleUnet
from src.loadit.diffusion import create_diffusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate samples with a simple Unet baseline.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to diffusion_ts checkpoint.")
    parser.add_argument("--num-samples", type=int, default=32, help="Number of samples to generate.")
    parser.add_argument("--out-dir", type=str, default="outputs/diffusion_ts", help="Output directory.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    parser.add_argument("--use-ddim", action="store_true", help="Use DDIM sampling instead of ancestral sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    input_length = cfg.data.nb_days * cfg.data.patch_length_day
    model = SimpleUnet(
        input_length=input_length,
        in_channels=cfg.model.in_channels,
        base_channels=cfg.model.base_channels,
        channel_mults=cfg.model.channel_mults,
        num_res_blocks=cfg.model.num_res_blocks,
        time_emb_dim=cfg.model.time_emb_dim,
        dropout=cfg.model.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = create_diffusion(
        timestep_respacing="",
        diffusion_steps=cfg.training.diffusion_steps,
        learn_sigma=False,
    )

    shape = (args.num_samples, cfg.model.in_channels, input_length)
    if args.use_ddim:
        samples = diffusion.ddim_sample_loop(model, shape=shape, device=device)
    else:
        samples = diffusion.p_sample_loop(model, shape=shape, device=device)

    samples = samples.squeeze(1).cpu().numpy()

    scale_min = cfg.data.value_scale_min
    scale_max = cfg.data.value_scale_max
    samples = samples * (scale_max - scale_min) + scale_min

    samples = samples.reshape(args.num_samples, cfg.data.nb_days, cfg.data.patch_length_day)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "unet_samples.npy", samples)


if __name__ == "__main__":
    main()