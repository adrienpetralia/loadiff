"""Plausibility features for synthetic load-curve quality control.

Features summarise the temporal and statistical plausibility of a single load
curve (1-D array, business unit = Watts). They are intentionally simple, robust
and interpretable so that reference envelopes calibrated on real data remain
auditable.

All thresholds are expressed in the curve's business unit (Watts) or in hours;
nothing here changes units. The time granularity is provided explicitly via
``dt_minutes`` (derived from the dataset's ``patch_length``: ``1440 / patch_length``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Canonical, always-computable feature set.
BASE_FEATURES: List[str] = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "q01",
    "q05",
    "q25",
    "q75",
    "q95",
    "q99",
    "fraction_at_lower_bound",
    "fraction_at_upper_bound",
    "mean_absolute_first_difference",
    "max_absolute_first_difference",
    "q95_absolute_first_difference",
    "mean_absolute_second_difference",
    "max_absolute_second_difference",
    "fraction_zero_or_near_zero",
    "longest_zero_or_near_zero_run_hours",
]

# Daily aggregates, only computable when the series length is a whole number of days.
DAILY_FEATURES: List[str] = [
    "daily_energy_mean",
    "daily_energy_std",
    "daily_peak_mean",
    "daily_peak_max",
    "load_factor",
]


@dataclass
class FeatureConfig:
    """Configuration for plausibility feature computation.

    Attributes:
        near_zero_w: Threshold (Watts) below which a point is considered near-zero.
        feature_names: Optional explicit subset of features to compute/keep. When
            ``None``, all computable features are returned.
        bound_atol: Absolute tolerance (Watts) used for the at-bound fractions.
    """

    near_zero_w: float = 10.0
    feature_names: Optional[List[str]] = None
    bound_atol: float = 1e-9


def longest_true_run(mask: np.ndarray) -> int:
    """Return the length (in points) of the longest contiguous run of ``True``."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return 0
    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    run_lengths = changes[1::2] - changes[::2]
    return int(run_lengths.max(initial=0))


def compute_curve_features(
    x: np.ndarray,
    dt_minutes: float,
    *,
    hard_min: float = 0.0,
    hard_max: float = 60_000.0,
    config: Optional[FeatureConfig] = None,
) -> Dict[str, float]:
    """Compute plausibility features for one finite 1-D load curve.

    Args:
        x: Finite 1-D array (Watts). Must not contain NaN/inf (compute features
            on the *repaired* curve in the QC pipeline, or on clean real data).
        dt_minutes: Sampling step in minutes (e.g. 30 for half-hourly data).
        hard_min: Lower physical bound (Watts), used for the at-lower-bound fraction.
        hard_max: Upper physical bound (Watts), used for the at-upper-bound fraction.
        config: Optional :class:`FeatureConfig`.

    Returns:
        Mapping ``feature_name -> value``. Daily features are included only when
        the series length is a whole number of days. If ``config.feature_names``
        is given, only those features are returned (missing ones are omitted).

    Raises:
        ValueError: If ``x`` is not 1-D, is empty, or contains non-finite values.
    """
    config = config or FeatureConfig()
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Expected a 1-D curve, got shape {x.shape}.")
    if x.size == 0:
        raise ValueError("Cannot compute features on an empty curve.")
    if not np.isfinite(x).all():
        raise ValueError("compute_curve_features requires a fully finite curve.")
    if dt_minutes <= 0:
        raise ValueError(f"dt_minutes must be > 0, got {dt_minutes}.")

    dt_hours = dt_minutes / 60.0
    feats: Dict[str, float] = {}

    feats["mean"] = float(np.mean(x))
    feats["std"] = float(np.std(x))
    feats["min"] = float(np.min(x))
    feats["max"] = float(np.max(x))
    feats["median"] = float(np.median(x))
    for q, name in [(0.01, "q01"), (0.05, "q05"), (0.25, "q25"),
                    (0.75, "q75"), (0.95, "q95"), (0.99, "q99")]:
        feats[name] = float(np.quantile(x, q))

    feats["fraction_at_lower_bound"] = float(np.mean(x <= hard_min + config.bound_atol))
    feats["fraction_at_upper_bound"] = float(np.mean(x >= hard_max - config.bound_atol))

    if x.size > 1:
        d1 = np.abs(np.diff(x))
        feats["mean_absolute_first_difference"] = float(np.mean(d1))
        feats["max_absolute_first_difference"] = float(np.max(d1))
        feats["q95_absolute_first_difference"] = float(np.quantile(d1, 0.95))
    else:
        feats["mean_absolute_first_difference"] = 0.0
        feats["max_absolute_first_difference"] = 0.0
        feats["q95_absolute_first_difference"] = 0.0

    if x.size > 2:
        d2 = np.abs(np.diff(x, n=2))
        feats["mean_absolute_second_difference"] = float(np.mean(d2))
        feats["max_absolute_second_difference"] = float(np.max(d2))
    else:
        feats["mean_absolute_second_difference"] = 0.0
        feats["max_absolute_second_difference"] = 0.0

    near_zero = x <= config.near_zero_w
    feats["fraction_zero_or_near_zero"] = float(np.mean(near_zero))
    feats["longest_zero_or_near_zero_run_hours"] = float(longest_true_run(near_zero) * dt_hours)

    # Daily aggregates (only when the series spans whole days).
    points_per_day = int(round(1440.0 / dt_minutes))
    if points_per_day > 0 and x.size % points_per_day == 0:
        days = x.reshape(-1, points_per_day)
        daily_energy = days.sum(axis=1) * dt_hours  # Watt-hours per day
        daily_peak = days.max(axis=1)
        feats["daily_energy_mean"] = float(np.mean(daily_energy))
        feats["daily_energy_std"] = float(np.std(daily_energy))
        feats["daily_peak_mean"] = float(np.mean(daily_peak))
        feats["daily_peak_max"] = float(np.max(daily_peak))
        peak = float(np.max(x))
        feats["load_factor"] = float(np.mean(x) / peak) if peak > 0 else 0.0

    if config.feature_names is not None:
        feats = {k: feats[k] for k in config.feature_names if k in feats}

    return feats