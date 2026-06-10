"""I/O and orchestration helpers to post-process a directory of generated curves.

These helpers read the outputs produced by the unified inference entrypoint
(``loadit_samples.npy`` + optional ``metadata.csv`` + ``run_info.json``), run the
QC batch pipeline, and write the cleaned/rejected splits, diagnostics, compact
repair masks and a quality report. They are shared by the standalone Hydra script
and by the optional post-inference integration so the logic lives in one place.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.postprocessing.load_curve_quality import (
    BatchPostprocessingResult,
    PhysicalFilterConfig,
    PlausibilityDecisionConfig,
    postprocess_curves_batch,
)
from src.postprocessing.plausibility_envelopes import PlausibilityEnvelope
from src.postprocessing.plausibility_features import FeatureConfig


def load_inference_outputs(
    run_dir: str,
    *,
    samples_file: str = "loadit_samples.npy",
    metadata_file: str = "metadata.csv",
    run_info_file: str = "run_info.json",
) -> tuple[np.ndarray, Optional[pd.DataFrame], Dict[str, Any]]:
    """Load curves, optional metadata and run_info from an inference output dir."""
    samples_path = os.path.join(run_dir, samples_file)
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"Generated curves not found: {samples_path}")
    samples = np.load(samples_path)
    if samples.ndim != 2:
        raise ValueError(f"Expected 2-D samples [N, T] in {samples_path}, got {samples.shape}.")

    metadata: Optional[pd.DataFrame] = None
    meta_path = os.path.join(run_dir, metadata_file)
    if os.path.exists(meta_path):
        metadata = pd.read_csv(meta_path)
        if len(metadata) != samples.shape[0]:
            raise ValueError(
                f"metadata rows ({len(metadata)}) != number of curves ({samples.shape[0]})."
            )

    run_info: Dict[str, Any] = {}
    info_path = os.path.join(run_dir, run_info_file)
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            run_info = json.load(f)

    return samples, metadata, run_info


def _resolve_dt_and_length(
    cfg_input: Dict[str, Any],
    run_info: Dict[str, Any],
) -> tuple[float, Optional[int]]:
    """Resolve dt_minutes and expected_length from config overrides or run_info."""
    dt_minutes = cfg_input.get("dt_minutes")
    patch_length = run_info.get("patch_length")
    if dt_minutes is None:
        if patch_length:
            dt_minutes = 1440.0 / float(patch_length)
        else:
            raise ValueError(
                "Cannot resolve dt_minutes: set input.dt_minutes or ensure run_info.json "
                "provides 'patch_length'."
            )

    expected_length = cfg_input.get("expected_length")
    if expected_length is None and run_info.get("n_days") and patch_length:
        expected_length = int(run_info["n_days"]) * int(patch_length)
    return float(dt_minutes), expected_length


def _build_group_keys(
    metadata: Optional[pd.DataFrame],
    envelope: Optional[PlausibilityEnvelope],
    groupby_metadata: List[str],
) -> Optional[List[Optional[str]]]:
    """Build per-curve group keys from metadata columns (or None)."""
    cols = list(groupby_metadata or (envelope.groupby_metadata if envelope else []))
    if not cols or metadata is None:
        return None
    missing = [c for c in cols if c not in metadata.columns]
    if missing:
        raise ValueError(f"groupby_metadata columns not in metadata: {missing}.")
    return [
        PlausibilityEnvelope.make_group_key([row[c] for c in cols])
        for _, row in metadata[cols].iterrows()
    ]


def save_batch_outputs(
    out_dir: str,
    batch: BatchPostprocessingResult,
    *,
    samples: np.ndarray,
    metadata: Optional[pd.DataFrame],
    save_rejected_curves: bool = True,
    save_repair_masks: bool = True,
    save_diagnostics: bool = True,
) -> None:
    """Persist the cleaned/rejected splits, report, diagnostics and repair masks."""
    os.makedirs(out_dir, exist_ok=True)
    n_total = samples.shape[0]

    # Base metadata table with a stable curve_id (input row index).
    if metadata is not None:
        base = metadata.copy()
        base.insert(0, "curve_id", np.arange(n_total))
    else:
        base = pd.DataFrame({"curve_id": np.arange(n_total)})

    flags_joined = [";".join(f) for f in batch.quality_flags]
    reasons = [d["reason"] for d in batch.diagnostics]
    statuses = [d["status"] for d in batch.diagnostics]

    # Cleaned (kept + repaired).
    np.save(os.path.join(out_dir, "cleaned_curves.npy"), batch.cleaned_curves)
    cleaned_meta = base.iloc[batch.cleaned_rows].copy()
    cleaned_meta["status"] = batch.cleaned_status
    cleaned_meta["quality_flags"] = [flags_joined[i] for i in batch.cleaned_rows]
    cleaned_meta.to_csv(os.path.join(out_dir, "cleaned_metadata.csv"), index=False)

    # Rejected (kept separately for auditing, with original curves).
    if save_rejected_curves:
        rejected_curves = (
            samples[batch.rejected_rows] if batch.rejected_rows
            else np.zeros((0, samples.shape[1]), dtype=samples.dtype)
        )
        np.save(os.path.join(out_dir, "rejected_curves.npy"), rejected_curves)
    rejected_meta = base.iloc[batch.rejected_rows].copy()
    rejected_meta["reason"] = batch.rejected_reason
    rejected_meta["quality_flags"] = [flags_joined[i] for i in batch.rejected_rows]
    rejected_meta.to_csv(os.path.join(out_dir, "rejected_metadata.csv"), index=False)

    # Quality report.
    with open(os.path.join(out_dir, "quality_report.json"), "w", encoding="utf-8") as f:
        json.dump(batch.report, f, indent=2, sort_keys=True)

    # Per-curve diagnostics (one JSON object per line).
    if save_diagnostics:
        with open(os.path.join(out_dir, "diagnostics.jsonl"), "w", encoding="utf-8") as f:
            for i in range(n_total):
                rec = {"curve_id": int(i), "status": statuses[i], "reason": reasons[i],
                       "quality_flags": batch.quality_flags[i], **batch.diagnostics[i]}
                f.write(json.dumps(rec) + "\n")

    # Compact repair masks: {curve_id: [modified point indices]} for repaired curves.
    if save_repair_masks:
        with open(os.path.join(out_dir, "repair_masks.json"), "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in batch.repair_masks.items()}, f)


def postprocess_directory(run_dir: str, cfg: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    """Post-process all curves in ``run_dir`` according to ``cfg`` and save to ``out_dir``.

    Args:
        run_dir: Inference output directory (with ``loadit_samples.npy`` etc.).
        cfg: Plain-dict post-processing config (see configs/postprocessing/*).
        out_dir: Destination directory for the QC artifacts.

    Returns:
        The aggregate quality report (also written to ``out_dir/quality_report.json``).
    """
    cfg_input = dict(cfg.get("input", {}) or {})
    samples, metadata, run_info = load_inference_outputs(
        run_dir,
        samples_file=cfg_input.get("samples_file", "loadit_samples.npy"),
        metadata_file=cfg_input.get("metadata_file", "metadata.csv"),
        run_info_file=cfg_input.get("run_info_file", "run_info.json"),
    )
    dt_minutes, expected_length = _resolve_dt_and_length(cfg_input, run_info)

    pf = dict(cfg.get("physical_filter", {}) or {})
    phys_cfg = PhysicalFilterConfig(
        hard_min=float(pf.get("hard_min", 0.0)),
        hard_max=float(pf.get("hard_max", 60_000.0)),
        reject_fraction=float(pf.get("reject_fraction", 0.02)),
        max_repair_gap_hours=float(pf.get("max_repair_gap_hours", 2.0)),
        numerical_tolerance=float(pf.get("numerical_tolerance", 1e-6)),
        soft_margin=float(pf.get("soft_margin", 500.0)),
    )

    plf = dict(cfg.get("plausibility_filter", {}) or {})
    envelope: Optional[PlausibilityEnvelope] = None
    envelope_path = plf.get("envelope_path")
    plaus_enabled = bool(plf.get("enabled", False)) and envelope_path not in (None, "", "???")
    if plaus_enabled:
        if not os.path.exists(envelope_path):
            raise FileNotFoundError(f"Plausibility envelope not found: {envelope_path}")
        envelope = PlausibilityEnvelope.load(envelope_path)

    feat = dict(cfg.get("features", {}) or {})
    feature_config = FeatureConfig(
        near_zero_w=float(feat.get("near_zero_w", 10.0)),
        feature_names=(list(envelope.feature_names) if envelope is not None
                       else feat.get("feature_names")),
    )
    plaus_cfg = PlausibilityDecisionConfig(
        enabled=envelope is not None,
        max_outside_features=int(plf.get("max_outside_features", 0)),
    )
    group_keys = _build_group_keys(metadata, envelope, plf.get("groupby_metadata", []))

    batch = postprocess_curves_batch(
        samples,
        dt_minutes=dt_minutes,
        config=phys_cfg,
        expected_length=expected_length,
        plausibility_envelope=envelope,
        plausibility_config=plaus_cfg,
        feature_config=feature_config,
        group_keys=group_keys,
    )

    out = dict(cfg.get("output", {}) or {})
    save_batch_outputs(
        out_dir, batch, samples=samples, metadata=metadata,
        save_rejected_curves=bool(out.get("save_rejected_curves", True)),
        save_repair_masks=bool(out.get("save_repair_masks", True)),
        save_diagnostics=bool(out.get("save_diagnostics", True)),
    )
    return batch.report