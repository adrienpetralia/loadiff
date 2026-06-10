#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone post-processing / quality control of generated Loadiff curves.

Reads an inference output directory (loadit_samples.npy + optional metadata.csv +
run_info.json), runs the QC pipeline, and writes cleaned/rejected splits, a quality
report, per-curve diagnostics and compact repair masks.

Example:
    python -m scripts.postprocessing.postprocess_generated_curves \\
        input.run_dir=/path/to/inference_run \\
        plausibility_filter.envelope_path=/path/to/plausibility_envelope.json
"""

from __future__ import annotations

import logging
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.postprocessing.batch_io import postprocess_directory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs/postprocessing",
            config_name="postprocess_generated_curves")
def main(cfg: DictConfig) -> None:
    out_dir = HydraConfig.get().runtime.output_dir
    run_dir = cfg.input.run_dir
    if run_dir in (None, "", "???"):
        raise ValueError("input.run_dir is required (the inference output directory to clean).")

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    report = postprocess_directory(run_dir, cfg_dict, out_dir)

    if bool(OmegaConf.select(cfg, "output.save_resolved_config", default=True)):
        OmegaConf.save(config=cfg, f=os.path.join(out_dir, "resolved_config.yaml"), resolve=True)

    logger.info(
        "QC done: %d total -> keep=%d repair=%d reject=%d (saved to %s)",
        report["n_total"], report["n_keep"], report["n_repair"], report["n_reject"], out_dir,
    )


if __name__ == "__main__":
    main()