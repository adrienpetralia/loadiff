"""Learned plausibility envelopes calibrated on real load curves.

An envelope stores, for each plausibility feature, robust lower/upper bounds
derived from quantiles of the *real* curves of a given split (train by default).
Envelopes are never calibrated on test or synthetic data, to avoid leakage.

The artifact is a self-describing JSON file (see :meth:`PlausibilityEnvelope.to_dict`)
recording the dataset, split, units, time granularity, feature list, bounds,
calibration quantiles, optional metadata groups, number of real curves used,
creation date and a schema version.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.postprocessing.plausibility_features import (
    FeatureConfig,
    compute_curve_features,
)

SCHEMA_VERSION = "1.0"


@dataclass
class FeatureViolation:
    """A single feature found outside its reference envelope."""

    feature_name: str
    observed_value: float
    lower_bound: float
    upper_bound: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "feature_name": self.feature_name,
            "observed_value": float(self.observed_value),
            "lower_bound": float(self.lower_bound),
            "upper_bound": float(self.upper_bound),
        }


@dataclass
class PlausibilityEnvelope:
    """Robust per-feature plausibility bounds learned on real data.

    Attributes:
        dataset: Dataset name the envelope was calibrated on.
        split: Split used for calibration (e.g. ``"train"``).
        units: Business unit of the curves (e.g. ``"W"``).
        dt_minutes: Time granularity in minutes.
        feature_names: Ordered list of calibrated features.
        bounds: Global ``feature -> (lower, upper)`` bounds.
        lower_quantile: Lower calibration quantile.
        upper_quantile: Upper calibration quantile.
        groupby_metadata: Metadata columns used for group envelopes (may be empty).
        groups: Optional ``group_key -> {feature: (lower, upper)}`` group bounds.
        min_group_size: Minimum real curves required to build a group envelope.
        n_curves: Number of real curves used for the global calibration.
        created_at: ISO timestamp of creation.
        schema_version: Artifact schema version.
    """

    dataset: str
    split: str
    units: str
    dt_minutes: float
    feature_names: List[str]
    bounds: Dict[str, Tuple[float, float]]
    lower_quantile: float
    upper_quantile: float
    groupby_metadata: List[str] = field(default_factory=list)
    groups: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=dict)
    min_group_size: int = 100
    n_curves: int = 0
    created_at: str = ""
    schema_version: str = SCHEMA_VERSION

    # -- group key helpers -------------------------------------------------
    @staticmethod
    def make_group_key(values: Sequence) -> str:
        """Build a stable string group key from a sequence of metadata values."""
        return "|".join(str(v) for v in values)

    def _bounds_for(self, group_key: Optional[str]) -> Dict[str, Tuple[float, float]]:
        if group_key is not None and group_key in self.groups:
            return self.groups[group_key]
        return self.bounds

    # -- checking ----------------------------------------------------------
    def check(
        self,
        features: Dict[str, float],
        group_key: Optional[str] = None,
    ) -> List[FeatureViolation]:
        """Return the features of ``features`` lying outside the envelope.

        Args:
            features: Mapping ``feature_name -> observed value``.
            group_key: Optional group key; falls back to the global envelope when
                the group is unknown (e.g. too small / absent at calibration).

        Returns:
            A list of :class:`FeatureViolation` (empty when all features are within bounds).
        """
        bounds = self._bounds_for(group_key)
        violations: List[FeatureViolation] = []
        for name in self.feature_names:
            if name not in bounds or name not in features:
                continue
            lo, hi = bounds[name]
            obs = float(features[name])
            if obs < lo or obs > hi:
                violations.append(FeatureViolation(name, obs, lo, hi))
        return violations

    # -- (de)serialization -------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "split": self.split,
            "units": self.units,
            "dt_minutes": self.dt_minutes,
            "feature_names": list(self.feature_names),
            "bounds": {k: [float(v[0]), float(v[1])] for k, v in self.bounds.items()},
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "groupby_metadata": list(self.groupby_metadata),
            "groups": {
                gk: {k: [float(v[0]), float(v[1])] for k, v in gb.items()}
                for gk, gb in self.groups.items()
            },
            "min_group_size": self.min_group_size,
            "n_curves": self.n_curves,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlausibilityEnvelope":
        return cls(
            dataset=d["dataset"],
            split=d["split"],
            units=d["units"],
            dt_minutes=float(d["dt_minutes"]),
            feature_names=list(d["feature_names"]),
            bounds={k: (float(v[0]), float(v[1])) for k, v in d["bounds"].items()},
            lower_quantile=float(d["lower_quantile"]),
            upper_quantile=float(d["upper_quantile"]),
            groupby_metadata=list(d.get("groupby_metadata", [])),
            groups={
                gk: {k: (float(v[0]), float(v[1])) for k, v in gb.items()}
                for gk, gb in d.get("groups", {}).items()
            },
            min_group_size=int(d.get("min_group_size", 100)),
            n_curves=int(d.get("n_curves", 0)),
            created_at=d.get("created_at", ""),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )

    def save(self, path: str) -> None:
        """Serialize the envelope to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "PlausibilityEnvelope":
        """Load an envelope from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def _feature_matrix(
    curves: np.ndarray,
    dt_minutes: float,
    *,
    hard_min: float,
    hard_max: float,
    feature_config: FeatureConfig,
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute a [n_valid, n_features] DataFrame of features over finite curves."""
    rows: List[Dict[str, float]] = []
    for i in range(curves.shape[0]):
        x = np.asarray(curves[i], dtype=float)
        if not np.isfinite(x).all():
            # Skip non-finite real curves rather than poison the calibration.
            continue
        rows.append(compute_curve_features(
            x, dt_minutes, hard_min=hard_min, hard_max=hard_max, config=feature_config
        ))
    if not rows:
        raise ValueError("No finite curves available to calibrate the envelope.")
    df = pd.DataFrame(rows)
    # Keep only features that are fully defined across all curves.
    feature_names = [c for c in df.columns if df[c].notna().all()]
    return df[feature_names], feature_names


def calibrate_envelope(
    curves: np.ndarray,
    dt_minutes: float,
    *,
    dataset: str,
    split: str,
    lower_quantile: float = 0.001,
    upper_quantile: float = 0.999,
    hard_min: float = 0.0,
    hard_max: float = 60_000.0,
    units: str = "W",
    feature_config: Optional[FeatureConfig] = None,
    metadata: Optional[pd.DataFrame] = None,
    groupby_metadata: Optional[Sequence[str]] = None,
    min_group_size: int = 100,
) -> PlausibilityEnvelope:
    """Calibrate a :class:`PlausibilityEnvelope` from real curves.

    Args:
        curves: Real curves ``[M, T]`` in business unit (Watts), from the
            calibration split only.
        dt_minutes: Time granularity in minutes.
        dataset: Dataset name (recorded in the artifact).
        split: Calibration split name (recorded in the artifact).
        lower_quantile: Lower quantile for the robust bounds.
        upper_quantile: Upper quantile for the robust bounds.
        hard_min: Lower physical bound (for at-bound features).
        hard_max: Upper physical bound (for at-bound features).
        units: Business unit string.
        feature_config: Optional feature configuration.
        metadata: Optional per-curve metadata aligned with ``curves`` rows.
        groupby_metadata: Optional metadata columns to build group envelopes.
        min_group_size: Minimum real curves required to keep a group envelope.

    Returns:
        A calibrated :class:`PlausibilityEnvelope`.
    """
    feature_config = feature_config or FeatureConfig()
    curves = np.asarray(curves, dtype=float)
    if curves.ndim != 2:
        raise ValueError(f"curves must be 2-D [M, T], got shape {curves.shape}.")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Require 0 <= lower_quantile < upper_quantile <= 1.")

    df, feature_names = _feature_matrix(
        curves, dt_minutes, hard_min=hard_min, hard_max=hard_max, feature_config=feature_config
    )

    def _bounds(frame: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        lo = frame.quantile(lower_quantile)
        hi = frame.quantile(upper_quantile)
        return {name: (float(lo[name]), float(hi[name])) for name in feature_names}

    bounds = _bounds(df)

    groups: Dict[str, Dict[str, Tuple[float, float]]] = {}
    groupby_metadata = list(groupby_metadata or [])
    if groupby_metadata:
        if metadata is None:
            raise ValueError("groupby_metadata given but no metadata provided.")
        missing = [c for c in groupby_metadata if c not in metadata.columns]
        if missing:
            raise ValueError(f"groupby_metadata columns not in metadata: {missing}.")
        # Align metadata to the finite-curve subset used for df (same row order, finite filter).
        finite_mask = np.array([np.isfinite(curves[i]).all() for i in range(curves.shape[0])])
        meta_valid = metadata.loc[finite_mask].reset_index(drop=True)
        df_idx = df.reset_index(drop=True)
        for keys, idx in meta_valid.groupby(groupby_metadata).groups.items():
            if len(idx) < min_group_size:
                continue  # fall back to the global envelope at check time
            key_values = keys if isinstance(keys, tuple) else (keys,)
            group_key = PlausibilityEnvelope.make_group_key(key_values)
            groups[group_key] = _bounds(df_idx.loc[idx])

    return PlausibilityEnvelope(
        dataset=dataset,
        split=split,
        units=units,
        dt_minutes=float(dt_minutes),
        feature_names=feature_names,
        bounds=bounds,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        groupby_metadata=groupby_metadata,
        groups=groups,
        min_group_size=min_group_size,
        n_curves=int(df.shape[0]),
        created_at=_dt.datetime.now().isoformat(timespec="seconds"),
        schema_version=SCHEMA_VERSION,
    )