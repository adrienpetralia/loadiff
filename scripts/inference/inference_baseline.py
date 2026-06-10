#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified inference entrypoint for the baselines (timegan / timevae / diffusion_ts).

Mirrors the loadiff inference interface with two modes:

  - ``unconditional``    : loads ``<baseline>_unconditional`` and generates curves.
  - ``user_conditioned`` : for each conditioning combination ``{appliance: value}``,
    loads the specialised ``<baseline>_<appliance>_label<value>`` checkpoint and
    generates ``num_samples`` curves, then concatenates them into one labelled
    population. Each combination must reference exactly one appliance (the baselines
    are single-appliance specialised models; multilabel specs are rejected).

No temperature conditioning and no post-processing are applied (not supported by these
baselines). Outputs match the loadiff / TSTR layout: ``loadit_samples.npy`` (+ ``y.npy``
and ``run_info.json`` with ``label_names`` in user_conditioned mode).

Example (mirrors loadiff overrides):
    python -m scripts.inference.inference_baseline \\
        inference.baseline=timegan inference.dataset=smach \\
        inference.mode=user_conditioned \\
        'inference.conditioning.combinations=[{values:{CHAUFF_ELEC:0},num_samples:512},{values:{CHAUFF_ELEC:1},num_samples:512}]'
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from scripts.inference.baseline_conditioning import (
    build_label_array,
    normalize_combinations,
    parse_single_appliance,
    resolve_checkpoint,
    validate_baseline,
)
from scripts.inference.baselines_common import build_model, sample_curves

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _save_outputs(
    out_dir: str,
    curves: np.ndarray,
    *,
    label_names: List[str],
    y: np.ndarray = None,
    run_info: Dict[str, Any],
    cfg: DictConfig,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "loadit_samples.npy"), curves.astype(np.float32))
    if y is not None:
        np.save(os.path.join(out_dir, "y.npy"), y.astype(np.int64))
    with open(os.path.join(out_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, sort_keys=True)
    OmegaConf.save(config=cfg, f=os.path.join(out_dir, "resolved_config.yaml"), resolve=True)
    logger.info("Saved %d curves to %s (label_names=%s).", curves.shape[0], out_dir, label_names)


@hydra.main(version_base=None, config_path="../../configs", config_name="inference_baseline")
def main(cfg: DictConfig) -> None:
    inf = cfg.inference
    baseline = str(inf.baseline)
    validate_baseline(baseline)
    mode = str(inf.mode)
    if mode not in {"unconditional", "user_conditioned"}:
        raise ValueError(f"inference.mode must be unconditional|user_conditioned, got {mode!r}.")

    seed = int(inf.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(str(inf.device) if (str(inf.device) != "cuda" or torch.cuda.is_available()) else "cpu")
    runs_root = str(inf.runs_root)
    ckpt_filename = OmegaConf.select(cfg, "inference.checkpoint_filename", default=None)
    sampling_timesteps = OmegaConf.select(cfg, "inference.sampling_timesteps", default=None)
    batch_size = int(inf.batch_size)
    out_dir = HydraConfig.get().runtime.output_dir

    logger.info("Baseline=%s dataset=%s mode=%s device=%s", baseline, inf.dataset, mode, device)

    common_info = {
        "baseline": baseline,
        "dataset": str(inf.dataset),
        "mode": mode,
        "seed": seed,
        "runs_root": runs_root,
        "temperature": False,
        "postprocessing": False,
    }

    if mode == "unconditional":
        ckpt = resolve_checkpoint(runs_root, baseline, ckpt_filename=ckpt_filename)
        bm = build_model(baseline, ckpt, device, sampling_timesteps=sampling_timesteps)
        curves = sample_curves(bm, int(inf.n_samples), batch_size, device)
        run_info = {
            **common_info,
            "ckpt_path": ckpt,
            "n_samples": int(curves.shape[0]),
            "n_days": bm.n_days,
            "patch_length": bm.patch_length_day,
            "label_names": [],
        }
        _save_outputs(out_dir, curves, label_names=[], y=None, run_info=run_info, cfg=cfg)
        return

    # ---- user_conditioned -------------------------------------------------
    conditioning = OmegaConf.to_container(inf.conditioning, resolve=True) or {}
    default_n = conditioning.get("num_samples_per_combination")
    if default_n is None and OmegaConf.select(cfg, "inference.n_samples", default=None) is not None:
        default_n = int(inf.n_samples)
    combos = normalize_combinations(conditioning, default_n)

    curves_list: List[np.ndarray] = []
    per_combo: List[Dict[str, Any]] = []
    combo_meta: List[Dict[str, Any]] = []
    bm0 = None
    for j, combo in enumerate(combos):
        appliance, value = parse_single_appliance(combo["values"], j)
        num = combo["num_samples"]
        if num is None:
            raise ValueError(
                f"Combination #{j} ({appliance}=label{value}) has no num_samples. Set it "
                "per-combination, or via inference.conditioning.num_samples_per_combination / "
                "inference.n_samples."
            )
        num = int(num)
        ckpt = resolve_checkpoint(runs_root, baseline, appliance=appliance, label_value=value, ckpt_filename=ckpt_filename)
        logger.info("Combination #%d: %s=label%d -> %d samples (%s)", j, appliance, value, num, ckpt)
        bm = build_model(baseline, ckpt, device, sampling_timesteps=sampling_timesteps)
        bm0 = bm0 or bm
        curves_list.append(sample_curves(bm, num, batch_size, device))
        per_combo.append({"appliance": appliance, "value": value, "num": num})
        combo_meta.append({"appliance": appliance, "label_value": value, "num_samples": num, "ckpt_path": ckpt})

    label_names = sorted({pc["appliance"] for pc in per_combo})
    y = build_label_array(per_combo, label_names)
    curves = np.concatenate(curves_list, axis=0)
    run_info = {
        **common_info,
        "n_samples": int(curves.shape[0]),
        "n_days": bm0.n_days,
        "patch_length": bm0.patch_length_day,
        "label_names": label_names,
        "combinations": combo_meta,
    }
    _save_outputs(out_dir, curves, label_names=label_names, y=y, run_info=run_info, cfg=cfg)


if __name__ == "__main__":
    main()
