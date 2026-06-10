"""Quality-control pipeline for synthetic Loadiff load curves.

Given a single generated curve (1-D, business unit = Watts), the pipeline:

  1. validates the structure (non-empty, 1-D, expected length, timestamps);
  2. detects physical violations against ``[hard_min, hard_max]`` and non-finite values;
  3. computes violation diagnostics;
  4. decides conservatively: keep / numerical clip / local interpolation / reject;
  5. computes plausibility features on the repaired curve;
  6. compares them to a learned reference envelope (real-data) and decides keep/reject.

Design principles:
  * never extrapolate silently at the curve edges (reject instead);
  * never modify healthy values during a local repair;
  * a local physical repair must NOT hide a statistically aberrant curve: the
    plausibility check runs on the repaired curve and can still reject it;
  * units are never changed; all thresholds are in Watts or hours.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.postprocessing.plausibility_envelopes import PlausibilityEnvelope
from src.postprocessing.plausibility_features import (
    FeatureConfig,
    compute_curve_features,
    longest_true_run,
)

Status = Literal["keep", "repair", "reject"]


# --------------------------------------------------------------------------- #
# Quality flags
# --------------------------------------------------------------------------- #
class QualityFlag:
    """String constants for the quality flags attached to a curve."""

    EMPTY_CURVE = "empty_curve"
    WRONG_DIMENSION = "wrong_dimension"
    WRONG_LENGTH = "wrong_length"
    NON_FINITE_VALUES = "non_finite_values"
    UNORDERED_TIMESTAMPS = "unordered_timestamps"
    DUPLICATE_TIMESTAMPS = "duplicate_timestamps"
    IRREGULAR_TIMESTAMPS = "irregular_timestamps"
    PHYSICAL_VIOLATION = "physical_violation"
    NUMERICAL_PROJECTION = "numerical_projection"
    LOCAL_INTERPOLATION = "local_interpolation"
    UNSAFE_EDGE_EXTRAPOLATION = "unsafe_edge_extrapolation"
    TOO_MANY_INVALID_POINTS = "too_many_invalid_points"
    INVALID_BLOCK_TOO_LONG = "invalid_block_too_long"
    SEVERE_PHYSICAL_VIOLATION = "severe_physical_violation"
    FEATURE_OUTSIDE_REFERENCE_ENVELOPE = "feature_outside_reference_envelope"
    TOO_MANY_PLAUSIBILITY_VIOLATIONS = "too_many_plausibility_violations"


# --------------------------------------------------------------------------- #
# Configuration dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class PhysicalFilterConfig:
    """Thresholds for the physical filter (all bounds/margins in Watts, gaps in hours)."""

    hard_min: float = 0.0
    hard_max: float = 60_000.0
    reject_fraction: float = 0.02
    max_repair_gap_hours: float = 2.0
    numerical_tolerance: float = 1e-6
    soft_margin: float = 500.0


@dataclass
class PlausibilityDecisionConfig:
    """Decision parameters for the statistical plausibility check."""

    enabled: bool = True
    max_outside_features: int = 0


# --------------------------------------------------------------------------- #
# Result dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class CurveDiagnostics:
    """Detailed diagnostics for one curve."""

    status: Status
    reason: str
    n_points: int
    n_bad: int
    bad_fraction: float
    max_bad_run_points: int
    max_bad_run_hours: float
    max_violation: float
    n_below_min: int
    n_above_max: int
    n_non_finite: int
    dt_minutes: Optional[float] = None
    repair_kind: Optional[str] = None  # None | "numerical_projection" | "local_interpolation"
    n_repaired_points: int = 0
    fraction_at_lower_bound_after_repair: Optional[float] = None
    fraction_at_upper_bound_after_repair: Optional[float] = None
    plausibility_violations: List[Dict[str, float]] = field(default_factory=list)
    n_plausibility_violations: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CurvePostprocessingResult:
    """Outcome of the QC pipeline for one curve."""

    status: Status
    curve_repaired: Optional[np.ndarray]
    quality_flags: List[str]
    diagnostics: CurveDiagnostics
    repair_mask: np.ndarray


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _changed_mask(original: np.ndarray, repaired: np.ndarray) -> np.ndarray:
    """Boolean mask of points actually modified (handles non-finite originals)."""
    orig_finite = np.isfinite(original)
    changed = np.zeros(original.shape, dtype=bool)
    changed[~orig_finite] = True  # non-finite values were necessarily replaced
    of = orig_finite
    changed[of] = original[of] != repaired[of]
    return changed


def _resolve_dt_minutes(
    timestamps: Optional[Sequence],
    dt_minutes: Optional[float],
) -> Tuple[Optional[float], List[str]]:
    """Resolve dt from timestamps if needed and validate timestamp regularity.

    Returns ``(dt_minutes, structural_flags)``. ``dt_minutes`` is ``None`` only
    when it cannot be resolved (caller turns that into a reject/raise).
    """
    flags: List[str] = []
    if timestamps is None:
        return dt_minutes, flags

    idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(timestamps)))
    if len(idx) >= 2:
        diffs = np.diff(idx.asi8)  # nanoseconds
        if (diffs <= 0).any():
            # not strictly increasing -> unordered and/or duplicate
            if (diffs == 0).any():
                flags.append(QualityFlag.DUPLICATE_TIMESTAMPS)
            if (diffs < 0).any():
                flags.append(QualityFlag.UNORDERED_TIMESTAMPS)
        elif not np.all(diffs == diffs[0]):
            flags.append(QualityFlag.IRREGULAR_TIMESTAMPS)
        resolved = float(np.median(diffs) / 1e9 / 60.0)
    else:
        resolved = dt_minutes
    if dt_minutes is None:
        dt_minutes = resolved
    return dt_minutes, flags


# --------------------------------------------------------------------------- #
# Core pipeline
# --------------------------------------------------------------------------- #
def postprocess_load_curve(
    values: np.ndarray,
    *,
    timestamps: Optional[Sequence] = None,
    dt_minutes: Optional[float] = None,
    config: PhysicalFilterConfig,
    expected_length: Optional[int] = None,
    plausibility_envelope: Optional[PlausibilityEnvelope] = None,
    plausibility_config: Optional[PlausibilityDecisionConfig] = None,
    feature_config: Optional[FeatureConfig] = None,
    group_key: Optional[str] = None,
) -> CurvePostprocessingResult:
    """Run the full QC pipeline on a single load curve.

    Args:
        values: The 1-D curve (Watts).
        timestamps: Optional timestamps aligned with ``values`` (used to derive/validate dt).
        dt_minutes: Time step in minutes; required if ``timestamps`` is not given.
        config: Physical filter configuration.
        expected_length: Optional expected number of points for the dataset.
        plausibility_envelope: Optional learned envelope for the plausibility check.
        plausibility_config: Plausibility decision parameters.
        feature_config: Feature computation configuration.
        group_key: Optional group key for group-specific envelope bounds.

    Returns:
        A :class:`CurvePostprocessingResult`.

    Raises:
        ValueError: If neither ``timestamps`` nor ``dt_minutes`` can resolve a time step.
    """
    plausibility_config = plausibility_config or PlausibilityDecisionConfig(enabled=False)
    x = np.asarray(values, dtype=float)
    flags: List[str] = []

    def _reject(reason: str, diag: CurveDiagnostics, mask: np.ndarray) -> CurvePostprocessingResult:
        diag.status = "reject"
        diag.reason = reason
        return CurvePostprocessingResult("reject", None, list(flags), diag, mask)

    # --- Step 1: structural validation ------------------------------------
    if x.size == 0:
        flags.append(QualityFlag.EMPTY_CURVE)
        diag = CurveDiagnostics("reject", "empty_curve", 0, 0, 1.0, 0, 0.0,
                                float("nan"), 0, 0, 0, dt_minutes)
        return _reject("empty_curve", diag, np.zeros(0, dtype=bool))

    if x.ndim != 1:
        flags.append(QualityFlag.WRONG_DIMENSION)
        diag = CurveDiagnostics("reject", "wrong_dimension", int(x.size), int(x.size), 1.0,
                                int(x.size), 0.0, float("nan"), 0, 0, 0, dt_minutes)
        return _reject("wrong_dimension", diag, np.zeros(x.shape, dtype=bool))

    dt_minutes, ts_flags = _resolve_dt_minutes(timestamps, dt_minutes)
    flags.extend(ts_flags)
    if dt_minutes is None or dt_minutes <= 0:
        raise ValueError(
            "Cannot resolve a positive time step: provide dt_minutes or regular timestamps."
        )

    n = int(x.size)
    base_diag = lambda status, reason, **kw: CurveDiagnostics(  # noqa: E731
        status=status, reason=reason, n_points=n, dt_minutes=dt_minutes,
        n_bad=kw.get("n_bad", 0), bad_fraction=kw.get("bad_fraction", 0.0),
        max_bad_run_points=kw.get("max_bad_run_points", 0),
        max_bad_run_hours=kw.get("max_bad_run_hours", 0.0),
        max_violation=kw.get("max_violation", 0.0),
        n_below_min=kw.get("n_below_min", 0), n_above_max=kw.get("n_above_max", 0),
        n_non_finite=kw.get("n_non_finite", 0),
    )

    if expected_length is not None and n != int(expected_length):
        flags.append(QualityFlag.WRONG_LENGTH)
        return _reject("wrong_length", base_diag("reject", "wrong_length"),
                       np.zeros(x.shape, dtype=bool))

    # Reject curves whose timestamps are structurally broken.
    for ts_flag in (QualityFlag.DUPLICATE_TIMESTAMPS, QualityFlag.UNORDERED_TIMESTAMPS,
                    QualityFlag.IRREGULAR_TIMESTAMPS):
        if ts_flag in ts_flags:
            return _reject(ts_flag, base_diag("reject", ts_flag), np.zeros(x.shape, dtype=bool))

    # --- Step 2 & 3: physical violations and diagnostics ------------------
    finite = np.isfinite(x)
    below = finite & (x < config.hard_min)
    above = finite & (x > config.hard_max)
    non_finite = ~finite
    hard_bad = below | above | non_finite

    n_bad = int(hard_bad.sum())
    bad_fraction = float(n_bad / n)
    max_bad_run = longest_true_run(hard_bad)
    max_bad_run_hours = float(max_bad_run * dt_minutes / 60.0)
    lower_violation = float(np.max(config.hard_min - x[below])) if below.any() else 0.0
    upper_violation = float(np.max(x[above] - config.hard_max)) if above.any() else 0.0
    max_violation = max(lower_violation, upper_violation)
    n_non_finite = int(non_finite.sum())

    if n_non_finite:
        flags.append(QualityFlag.NON_FINITE_VALUES)
    if n_bad:
        flags.append(QualityFlag.PHYSICAL_VIOLATION)

    def diag_for(status: Status, reason: str) -> CurveDiagnostics:
        return CurveDiagnostics(
            status=status, reason=reason, n_points=n, n_bad=n_bad, bad_fraction=bad_fraction,
            max_bad_run_points=max_bad_run, max_bad_run_hours=max_bad_run_hours,
            max_violation=max_violation, n_below_min=int(below.sum()),
            n_above_max=int(above.sum()), n_non_finite=n_non_finite, dt_minutes=dt_minutes,
        )

    repaired: Optional[np.ndarray] = None
    repair_kind: Optional[str] = None

    # --- Step 4: conservative physical decision ---------------------------
    if n_bad == 0:
        repaired = x.copy()
    elif bad_fraction > config.reject_fraction:
        flags.append(QualityFlag.TOO_MANY_INVALID_POINTS)
        return _reject("too_many_invalid_points", diag_for("reject", "too_many_invalid_points"),
                       hard_bad)
    elif max_bad_run_hours > config.max_repair_gap_hours:
        flags.append(QualityFlag.INVALID_BLOCK_TOO_LONG)
        return _reject("invalid_block_too_long", diag_for("reject", "invalid_block_too_long"),
                       hard_bad)
    else:
        severe = finite & ((x < config.hard_min - config.soft_margin)
                           | (x > config.hard_max + config.soft_margin))
        if severe.any():
            flags.append(QualityFlag.SEVERE_PHYSICAL_VIOLATION)
            return _reject("severe_physical_violation",
                           diag_for("reject", "severe_physical_violation"), hard_bad)

        tiny = finite & (
            ((x < config.hard_min) & (x >= config.hard_min - config.numerical_tolerance))
            | ((x > config.hard_max) & (x <= config.hard_max + config.numerical_tolerance))
        )
        if np.array_equal(tiny, hard_bad):
            # Pure numerical projection (no non-finite, no real anomaly).
            repaired = np.clip(x, config.hard_min, config.hard_max)
            repair_kind = QualityFlag.NUMERICAL_PROJECTION
            flags.append(QualityFlag.NUMERICAL_PROJECTION)
        else:
            # Local, constrained interpolation; never extrapolate at the edges.
            series = pd.Series(x)
            series[hard_bad] = np.nan
            series = series.interpolate(method="linear", limit_area="inside")
            if series.isna().any():
                flags.append(QualityFlag.UNSAFE_EDGE_EXTRAPOLATION)
                return _reject("unsafe_edge_extrapolation",
                               diag_for("reject", "unsafe_edge_extrapolation"), hard_bad)
            repaired = series.to_numpy()
            repair_kind = QualityFlag.LOCAL_INTERPOLATION
            flags.append(QualityFlag.LOCAL_INTERPOLATION)

    # --- Final clip towards [hard_min, hard_max] --------------------------
    repaired = np.clip(repaired, config.hard_min, config.hard_max)
    repair_mask = _changed_mask(x, repaired)

    diag = diag_for("keep", "no_physical_violation" if n_bad == 0 else (repair_kind or "repair"))
    diag.repair_kind = repair_kind
    diag.n_repaired_points = int(repair_mask.sum())
    diag.fraction_at_lower_bound_after_repair = float(np.mean(repaired <= config.hard_min))
    diag.fraction_at_upper_bound_after_repair = float(np.mean(repaired >= config.hard_max))

    physical_repaired = bool(repair_kind is not None)

    # --- Step 5-7: plausibility on the REPAIRED curve ---------------------
    if plausibility_config.enabled and plausibility_envelope is not None:
        feats = compute_curve_features(
            repaired, dt_minutes, hard_min=config.hard_min, hard_max=config.hard_max,
            config=feature_config,
        )
        violations = plausibility_envelope.check(feats, group_key=group_key)
        diag.plausibility_violations = [v.to_dict() for v in violations]
        diag.n_plausibility_violations = len(violations)
        if violations:
            flags.append(QualityFlag.FEATURE_OUTSIDE_REFERENCE_ENVELOPE)
        if len(violations) > plausibility_config.max_outside_features:
            flags.append(QualityFlag.TOO_MANY_PLAUSIBILITY_VIOLATIONS)
            diag.status = "reject"
            diag.reason = "too_many_plausibility_violations"
            return CurvePostprocessingResult("reject", None, list(flags), diag, repair_mask)

    status: Status = "repair" if physical_repaired else "keep"
    diag.status = status
    diag.reason = (repair_kind or "no_physical_violation") if status != "keep" else "kept"
    return CurvePostprocessingResult(status, repaired, list(flags), diag, repair_mask)


# --------------------------------------------------------------------------- #
# Batch orchestration
# --------------------------------------------------------------------------- #
@dataclass
class BatchPostprocessingResult:
    """Aggregated outcome of a batch run (memory-friendly: no per-curve arrays kept)."""

    cleaned_curves: np.ndarray            # [K, T] kept + repaired curves
    cleaned_rows: List[int]               # input row index for each cleaned curve
    cleaned_status: List[str]             # "keep" | "repair"
    rejected_rows: List[int]
    rejected_reason: List[str]
    diagnostics: List[dict]               # len N (per input row)
    quality_flags: List[List[str]]        # len N
    repair_masks: Dict[int, List[int]]    # row index -> list of modified point indices
    report: dict


def postprocess_curves_batch(
    curves: np.ndarray,
    *,
    dt_minutes: float,
    config: PhysicalFilterConfig,
    expected_length: Optional[int] = None,
    plausibility_envelope: Optional[PlausibilityEnvelope] = None,
    plausibility_config: Optional[PlausibilityDecisionConfig] = None,
    feature_config: Optional[FeatureConfig] = None,
    group_keys: Optional[Sequence[Optional[str]]] = None,
) -> BatchPostprocessingResult:
    """Run :func:`postprocess_load_curve` over a batch of curves.

    Args:
        curves: ``[N, T]`` array of generated curves (Watts).
        dt_minutes: Time step in minutes.
        config: Physical filter configuration.
        expected_length: Optional expected length per curve.
        plausibility_envelope: Optional learned envelope.
        plausibility_config: Plausibility decision parameters.
        feature_config: Feature computation configuration.
        group_keys: Optional per-curve group keys (aligned with rows).

    Returns:
        A :class:`BatchPostprocessingResult` with cleaned/rejected splits, per-curve
        diagnostics/flags, compact repair masks and an aggregate report.
    """
    curves = np.asarray(curves, dtype=float)
    if curves.ndim != 2:
        raise ValueError(f"curves must be 2-D [N, T], got shape {curves.shape}.")
    n_total = curves.shape[0]

    cleaned: List[np.ndarray] = []
    cleaned_rows: List[int] = []
    cleaned_status: List[str] = []
    rejected_rows: List[int] = []
    rejected_reason: List[str] = []
    diagnostics: List[dict] = []
    quality_flags: List[List[str]] = []
    repair_masks: Dict[int, List[int]] = {}

    counts_by_reason: Dict[str, int] = {}
    counts_by_flag: Dict[str, int] = {}
    n_keep = n_repair = n_reject = 0

    for i in range(n_total):
        gk = group_keys[i] if group_keys is not None else None
        res = postprocess_load_curve(
            curves[i],
            dt_minutes=dt_minutes,
            config=config,
            expected_length=expected_length,
            plausibility_envelope=plausibility_envelope,
            plausibility_config=plausibility_config,
            feature_config=feature_config,
            group_key=gk,
        )
        diagnostics.append(res.diagnostics.to_dict())
        quality_flags.append(res.quality_flags)
        counts_by_reason[res.diagnostics.reason] = counts_by_reason.get(res.diagnostics.reason, 0) + 1
        for fl in res.quality_flags:
            counts_by_flag[fl] = counts_by_flag.get(fl, 0) + 1

        if res.status == "reject":
            n_reject += 1
            rejected_rows.append(i)
            rejected_reason.append(res.diagnostics.reason)
            continue

        if res.status == "repair":
            n_repair += 1
            repair_masks[i] = np.flatnonzero(res.repair_mask).astype(int).tolist()
        else:
            n_keep += 1
        cleaned.append(res.curve_repaired)
        cleaned_rows.append(i)
        cleaned_status.append(res.status)

    cleaned_arr = (
        np.stack(cleaned, axis=0) if cleaned else np.zeros((0, curves.shape[1]), dtype=float)
    )
    report = {
        "n_total": n_total,
        "n_keep": n_keep,
        "n_repair": n_repair,
        "n_reject": n_reject,
        "keep_fraction": n_keep / n_total if n_total else 0.0,
        "repair_fraction": n_repair / n_total if n_total else 0.0,
        "reject_fraction": n_reject / n_total if n_total else 0.0,
        "counts_by_reason": counts_by_reason,
        "counts_by_quality_flag": counts_by_flag,
    }
    return BatchPostprocessingResult(
        cleaned_curves=cleaned_arr,
        cleaned_rows=cleaned_rows,
        cleaned_status=cleaned_status,
        rejected_rows=rejected_rows,
        rejected_reason=rejected_reason,
        diagnostics=diagnostics,
        quality_flags=quality_flags,
        repair_masks=repair_masks,
        report=report,
    )