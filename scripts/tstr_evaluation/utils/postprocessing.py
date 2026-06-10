#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply the generated-curve quality-control pipeline to in-memory arrays.

This is a thin adapter around ``src.postprocessing.batch_io.postprocess_directory``
so the TSTR data loader can run the *same* QC pipeline that the standalone
post-processing script uses. ``postprocess_directory`` works on a directory of
inference outputs, so we materialise the curves to a temporary ``loadit_samples.npy``
(+ a minimal ``run_info.json``), run the pipeline with the default physical filter
(plausibility disabled by default), then read back the cleaned curves and the
indices that were kept.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.postprocessing.batch_io import postprocess_directory


# Default post-processing config: physical filter only (no calibrated plausibility
# envelope required), mirroring configs/postprocessing/postprocess_generated_curves.yaml
# with the plausibility stage disabled so it runs without external artifacts.
DEFAULT_PP_CONFIG: Dict[str, Any] = {
    "input": {
        "samples_file": "loadit_samples.npy",
        "metadata_file": "metadata.csv",
        "run_info_file": "run_info.json",
    },
    "physical_filter": {
        "hard_min": 0.0,
        "hard_max": 60000.0,
        "reject_fraction": 0.02,
        "max_repair_gap_hours": 2.0,
        "numerical_tolerance": 1.0e-6,
        "soft_margin": 500.0,
    },
    "features": {"near_zero_w": 10.0, "feature_names": None},
    "plausibility_filter": {"enabled": False, "envelope_path": None},
    "output": {
        "save_rejected_curves": False,
        "save_repair_masks": False,
        "save_diagnostics": False,
    },
}


def apply_postprocessing(
    X: np.ndarray,
    *,
    patch_length: int = 48,
    config: Optional[Dict[str, Any]] = None,
    return_kept_indices: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the QC pipeline on generated curves and return the cleaned subset.

    Args:
        X: Generated load curves in Watts, shape ``[N, L]``.
        patch_length: Points per day (used to derive ``dt_minutes = 1440 / patch_length``).
        config: Optional override of :data:`DEFAULT_PP_CONFIG`.
        return_kept_indices: When ``True`` (default), also return the original row
            indices that survived the filter so aligned labels can be subset.

    Returns:
        ``(X_clean, kept_indices)`` where ``X_clean`` has shape ``[M, L]`` (``M <= N``)
        and ``kept_indices`` (shape ``[M]``) maps cleaned rows back to ``X``.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D curves [N, L], got shape {X.shape}.")

    cfg = config if config is not None else DEFAULT_PP_CONFIG

    with tempfile.TemporaryDirectory(prefix="tstr_pp_") as run_dir:
        np.save(os.path.join(run_dir, "loadit_samples.npy"), X.astype(np.float32))
        run_info = {"patch_length": int(patch_length), "n_days": int(X.shape[1] // patch_length)}
        with open(os.path.join(run_dir, "run_info.json"), "w", encoding="utf-8") as f:
            json.dump(run_info, f)

        out_dir = os.path.join(run_dir, "postprocessing")
        postprocess_directory(run_dir, cfg, out_dir)

        X_clean = np.load(os.path.join(out_dir, "cleaned_curves.npy"))
        cleaned_meta = pd.read_csv(os.path.join(out_dir, "cleaned_metadata.csv"))
        kept_indices = cleaned_meta["curve_id"].to_numpy().astype(np.int64)

    if return_kept_indices:
        return X_clean, kept_indices
    return X_clean, kept_indices
