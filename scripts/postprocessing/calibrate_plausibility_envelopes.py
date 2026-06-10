#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calibrate plausibility envelopes from REAL load curves (train split only).

Loads the configured dataset split (train by default), computes plausibility
features over the real curves, and serialises robust per-feature envelopes
(optionally per metadata group). Envelopes are never calibrated on test or
synthetic data.

Example:
    python -m scripts.postprocessing.calibrate_plausibility_envelopes \\
        dataset.name=cer_bis calibration_split=train
"""

from __future__ import annotations

import logging
import os
import pickle

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.helpers.loadiff_inference import get_dataset_class
from src.postprocessing.plausibility_envelopes import calibrate_envelope
from src.postprocessing.plausibility_features import FeatureConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs/postprocessing",
            config_name="calibrate_plausibility")
def main(cfg: DictConfig) -> None:
    out_dir = HydraConfig.get().runtime.output_dir
    split = cfg.calibration_split
    if split == "test":
        raise ValueError(
            "Refusing to calibrate plausibility envelopes on the 'test' split "
            "(data leakage). Use 'train' (recommended) or 'val'."
        )

    groupby = list(OmegaConf.select(cfg, "plausibility_filter.groupby_metadata", default=[]) or [])

    with open(cfg.dataset.path_client_split, "rb") as f:
        splits = pickle.load(f)
    if split not in splits:
        raise KeyError(f"Split {split!r} not in client split. Available: {sorted(splits)}.")
    clients = splits[split]

    ds_cls = get_dataset_class(cfg.dataset.name)
    kwargs = dict(
        path_load_curves=cfg.dataset.data_path,
        list_pdl=clients,
        scale_param2=cfg.dataset.value_scale,
        random_window=False,
    )
    if groupby:
        kwargs.update(
            path_metadata=OmegaConf.select(cfg, "dataset.path_metadata", default=None),
            bool_col_names=list(OmegaConf.select(cfg, "dataset.bool_col_names", default=[]) or []),
        )
    dataset = ds_cls(**kwargs)

    seq_len = dataset.nb_days * dataset.patch_length
    curves = dataset.data[:, :seq_len].cpu().numpy()  # real curves in Watts
    dt_minutes = 1440.0 / dataset.patch_length

    metadata = None
    if groupby:
        cols = list(dataset.bool_col_names)
        metadata = pd.DataFrame(dataset.data_pop.cpu().numpy().astype(int), columns=cols)

    feature_config = FeatureConfig(
        near_zero_w=float(OmegaConf.select(cfg, "features.near_zero_w", default=10.0)),
        feature_names=OmegaConf.select(cfg, "features.feature_names", default=None),
    )

    envelope = calibrate_envelope(
        curves,
        dt_minutes,
        dataset=cfg.dataset.name,
        split=split,
        lower_quantile=float(cfg.plausibility_filter.lower_quantile),
        upper_quantile=float(cfg.plausibility_filter.upper_quantile),
        hard_min=float(cfg.physical_filter.hard_min),
        hard_max=float(cfg.physical_filter.hard_max),
        feature_config=feature_config,
        metadata=metadata,
        groupby_metadata=groupby,
        min_group_size=int(OmegaConf.select(cfg, "plausibility_filter.min_group_size", default=100)),
    )

    env_path = os.path.join(out_dir, cfg.envelope_filename)
    envelope.save(env_path)
    OmegaConf.save(config=cfg, f=os.path.join(out_dir, "resolved_config.yaml"), resolve=True)
    logger.info(
        "Calibrated envelope on %d real curves (%s/%s), %d features, %d group(s) -> %s",
        envelope.n_curves, cfg.dataset.name, split, len(envelope.feature_names),
        len(envelope.groups), env_path,
    )


if __name__ == "__main__":
    main()