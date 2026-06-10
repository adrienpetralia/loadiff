#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpoint resolution + conditioning logic for the baseline generators.

These baselines (``timegan`` / ``timevae`` / ``diffusion_ts``) are **not** conditional
at runtime: each ``(appliance, label_value)`` pair is a separately trained, specialised
model. This module (intentionally torch-free, so it can be unit-tested cheaply) maps a
loadiff-style conditioning spec onto the right specialised checkpoints and builds the
``y`` label array.

Checkpoint layout (per dataset, under ``runs_<dataset>/<baseline>/``):
    <baseline>_unconditional/checkpoints/<baseline>_best.pt
    <baseline>_<appliance>_label<value>/checkpoints/<baseline>_best.pt
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

BASELINES = ("timegan", "timevae", "diffusion_ts")


def validate_baseline(baseline: str) -> None:
    if baseline not in BASELINES:
        raise ValueError(f"Unknown baseline {baseline!r}. Valid: {list(BASELINES)}.")


def resolve_checkpoint(
    runs_root: str,
    baseline: str,
    *,
    appliance: Optional[str] = None,
    label_value: Optional[int] = None,
    ckpt_filename: Optional[str] = None,
) -> str:
    """Resolve the checkpoint path, with an explicit error when it is missing.

    Args:
        runs_root: e.g. ``/scratch/.../runs_smach``.
        baseline: ``timegan`` / ``timevae`` / ``diffusion_ts``.
        appliance: ``None`` -> the unconditional model; otherwise the specialised
            ``<baseline>_<appliance>_label<value>`` model.
        label_value: required when ``appliance`` is given (0 or 1).
        ckpt_filename: defaults to ``<baseline>_best.pt``.
    """
    validate_baseline(baseline)
    fname = ckpt_filename or f"{baseline}_best.pt"
    if appliance is None:
        sub = f"{baseline}_unconditional"
        desc = "unconditional model"
    else:
        if label_value is None:
            raise ValueError("label_value is required when appliance is set.")
        sub = f"{baseline}_{appliance}_label{int(label_value)}"
        desc = f"specialised model for {appliance}=label{int(label_value)}"
    # Checkpoints live under a per-baseline subdirectory: runs_<dataset>/<baseline>/<sub>/...
    path = os.path.join(runs_root, baseline, sub, "checkpoints", fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected {baseline} checkpoint ({desc}) not found:\n  {path}\n"
            f"Check inference.runs_root={runs_root!r} and that the model "
            f"'{baseline}/{sub}' was trained (directory + checkpoints/{fname})."
        )
    return path


def normalize_combinations(
    conditioning: Mapping[str, Any],
    default_num_samples: Optional[int],
) -> List[Dict[str, Any]]:
    """Normalise a loadiff-style conditioning spec into a list of combinations.

    Accepts ``conditioning.values`` (single combination shorthand) XOR
    ``conditioning.combinations`` (list of ``{values, num_samples}``).
    """
    values = conditioning.get("values")
    combinations = conditioning.get("combinations")
    if (values is None) == (combinations is None):
        raise ValueError(
            "Provide exactly one of 'conditioning.values' (single combination) or "
            "'conditioning.combinations' (list of combinations)."
        )
    if values is not None:
        return [{"values": dict(values), "num_samples": default_num_samples}]

    out: List[Dict[str, Any]] = []
    for j, combo in enumerate(combinations):
        if "values" not in combo:
            raise ValueError(
                f"conditioning.combinations[{j}] must contain a 'values' mapping, "
                f"got keys {sorted(combo)}."
            )
        out.append(
            {"values": dict(combo["values"]), "num_samples": combo.get("num_samples", default_num_samples)}
        )
    return out


def parse_single_appliance(values: Mapping[str, Any], combo_index: int) -> Tuple[str, int]:
    """Validate that a combination targets exactly one appliance with value 0/1."""
    if len(values) != 1:
        raise ValueError(
            f"Combination #{combo_index}: the baselines are single-appliance specialised "
            f"models, so each combination must reference exactly one appliance — got "
            f"{dict(values)}. Multilabel specs like {{cooker:0, dishwasher:1}} are not "
            f"supported for baselines."
        )
    appliance, raw_value = next(iter(values.items()))
    value = int(raw_value)
    if value not in (0, 1):
        raise ValueError(
            f"Combination #{combo_index}: label value for {appliance!r} must be 0 or 1 "
            f"(specialised checkpoints exist for label0/label1), got {raw_value!r}."
        )
    return str(appliance), value


def build_label_array(per_combo: Sequence[Dict[str, Any]], label_names: Sequence[str]) -> np.ndarray:
    """Build a ``[N, K]`` int64 label array (target column set, others -1)."""
    name_to_idx = {name: i for i, name in enumerate(label_names)}
    k = len(label_names)
    rows: List[List[int]] = []
    for combo in per_combo:
        row = [-1] * k
        row[name_to_idx[combo["appliance"]]] = int(combo["value"])
        rows.extend([list(row) for _ in range(int(combo["num"]))])
    return np.asarray(rows, dtype=np.int64)