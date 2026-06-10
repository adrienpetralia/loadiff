"""Reusable helpers for the unified Loadiff inference entrypoint.

This module groups the *pure* inference building blocks so that they can be
imported and unit-tested without pulling in the heavy model / plotting stack:

  - :func:`get_dataset_class`  -- formalises the historical ``CdCDataset`` alias
    (``smach``/``cer``/``cer_bis`` -> dataset class).
  - :func:`build_calendar_exog` -- builds the calendar exogenous tensor with the
    *exact* fixed divisors used during training.
  - :func:`build_user_labels`   -- expands a user-defined conditioning spec into a
    ``y`` tensor + per-sample metadata, with strict validation.

The model-loading logic lives in ``scripts/inference/inference_loadiff.py`` to
avoid importing ``src.loadit.models`` (and matplotlib) from this module.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import numpy as np
import pandas as pd
import torch

from src.helpers.dataset import (
    BaseParquetDailyDataset,
    CERBisDataset,
    CERDataset,
    SmachDataset,
)

# Historical alias: training/inference scripts used a local variable named
# ``CdCDataset`` selected from ``data.dataset``. There is no ``CdCDataset`` class
# in ``dataset.py`` (only a legacy one in ``dataset_old.py``). This registry
# formalises that mapping in a single place.
DATASET_REGISTRY: Dict[str, Type[BaseParquetDailyDataset]] = {
    "smach": SmachDataset,
    "cer": CERDataset,
    "cer_bis": CERBisDataset,
}

# Allowed ternary states for a multilabel slot: -1 unknown, 0 absent, 1 present.
VALID_LABEL_STATES = (-1, 0, 1)


def get_dataset_class(name: str) -> Type[BaseParquetDailyDataset]:
    """Return the dataset class associated with a dataset ``name``.

    Args:
        name: One of ``"smach"``, ``"cer"`` or ``"cer_bis"``.

    Returns:
        The matching :class:`BaseParquetDailyDataset` subclass.

    Raises:
        ValueError: If ``name`` is not a known dataset type.
    """
    try:
        return DATASET_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset type {name!r}. "
            f"Valid dataset types: {sorted(DATASET_REGISTRY)}."
        ) from None


def build_calendar_exog(
    start_date: str,
    n_days: int,
    temperature_col: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build the calendar exogenous tensor as done during training.

    Uses the *fixed* divisors from ``BaseParquetDailyDataset._create_exogene``
    (``2*pi/6``, ``2*pi/31``, ``2*pi/365``, ``2*pi/12``) so the encoding matches
    exactly what the model saw at training time. This intentionally differs from
    the per-window ``max()`` divisors of the old ``inference_loadit_no_cond``
    script, which produced a subtly different encoding for short windows.

    Args:
        start_date: Generation start date. Parsed with :func:`pandas.Timestamp`
            (ISO ``YYYY-MM-DD`` recommended, matching ``valid.gen_sample_start_date``).
        n_days: Number of days to generate.
        temperature_col: Optional ``[n_days, 1]`` temperature column. When given,
            it is concatenated as a 5th column (for ``temperature=True`` models).

    Returns:
        A ``[n_days, 4]`` (or ``[n_days, 5]`` with temperature) float32 tensor.
    """
    if n_days <= 0:
        raise ValueError(f"n_days must be > 0, got {n_days}.")

    start = pd.Timestamp(start_date)
    end = start + timedelta(days=int(n_days))
    extra = pd.date_range(start=start, end=end - timedelta(days=1), freq="D")

    exogene_array = np.vstack(
        [
            extra.weekday.values * (2 * np.pi / 6),
            extra.day.values * (2 * np.pi / 31),
            extra.dayofyear.values * (2 * np.pi / 365),
            extra.month.values * (2 * np.pi / 12),
        ]
    )
    exog = torch.tensor(exogene_array, dtype=torch.float32).permute(1, 0)  # [L, 4]

    if temperature_col is not None:
        temp = temperature_col.to(dtype=torch.float32)
        if temp.dim() == 1:
            temp = temp.unsqueeze(-1)
        if temp.shape[0] != exog.shape[0]:
            raise ValueError(
                f"temperature_col has {temp.shape[0]} days but n_days={n_days}."
            )
        exog = torch.cat([exog, temp], dim=1)  # [L, 5]

    return exog


def _normalize_combinations(
    conditioning: Mapping[str, Any],
    default_num_samples: Optional[int],
) -> List[Dict[str, Any]]:
    """Normalise the conditioning spec into a list of combinations.

    Accepts either ``conditioning.values`` (a single ``{field: state}`` mapping,
    a shorthand for one combination) or ``conditioning.combinations`` (a list of
    ``{values: {...}, num_samples: int}`` objects). Exactly one must be provided.
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

    normalized: List[Dict[str, Any]] = []
    for j, combo in enumerate(combinations):
        if "values" not in combo:
            raise ValueError(
                f"conditioning.combinations[{j}] must contain a 'values' mapping, "
                f"got keys {sorted(combo)}."
            )
        normalized.append(
            {
                "values": dict(combo["values"]),
                "num_samples": combo.get("num_samples", default_num_samples),
            }
        )
    return normalized


def build_user_labels(
    conditioning: Mapping[str, Any],
    bool_col_names: Sequence[str],
    *,
    multilabels: bool = True,
    default_num_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Expand a user-defined conditioning spec into labels and metadata.

    Args:
        conditioning: Mapping with keys ``values`` xor ``combinations`` (see
            :func:`_normalize_combinations`), plus optional ``missing_field_policy``
            (``"unknown"`` (default) or ``"require_all"``) and
            ``num_samples_per_combination``.
        bool_col_names: Ordered list of valid label names (from the checkpoint).
        multilabels: Whether the checkpoint model is a multilabel model. The
            ``user_conditioned`` mode only supports multilabel models.
        default_num_samples: Fallback ``num_samples`` per combination when neither
            the combination nor ``num_samples_per_combination`` specifies one.

    Returns:
        A tuple ``(y, row_meta)`` where ``y`` is a ``LongTensor`` of shape
        ``[N, K]`` (``K == len(bool_col_names)``) with states in ``{-1, 0, 1}``,
        and ``row_meta`` is a list of ``N`` dicts mapping label names to states
        plus a ``combination_id`` key.

    Raises:
        ValueError: On non-multilabel models, empty ``bool_col_names``, unknown
            field names, invalid states, missing fields under ``require_all``, or
            an unresolved ``num_samples``.
    """
    if not multilabels:
        raise ValueError(
            "Mode 'user_conditioned' requires a multilabel checkpoint "
            "(ditmodelargs.multilabels=True). This checkpoint uses a single-label "
            "embedder; use 'dataset_conditioned' or 'unconditional' instead."
        )

    bool_col_names = list(bool_col_names)
    if not bool_col_names:
        raise ValueError(
            "Mode 'user_conditioned' requires the checkpoint to define "
            "data.bool_col_names (the set of conditionable labels). None found."
        )

    policy = conditioning.get("missing_field_policy", "unknown")
    if policy not in {"unknown", "require_all"}:
        raise ValueError(
            f"missing_field_policy must be 'unknown' or 'require_all', got {policy!r}."
        )

    per_combo_default = conditioning.get("num_samples_per_combination")
    if per_combo_default is None:
        per_combo_default = default_num_samples

    combos = _normalize_combinations(conditioning, per_combo_default)
    name_to_idx = {name: i for i, name in enumerate(bool_col_names)}
    K = len(bool_col_names)

    rows: List[List[int]] = []
    row_meta: List[Dict[str, Any]] = []

    for j, combo in enumerate(combos):
        vals: Dict[str, Any] = combo["values"]
        num_samples = combo["num_samples"]
        if num_samples is None:
            raise ValueError(
                f"Combination #{j} has no 'num_samples'. Set it per-combination, or "
                "set conditioning.num_samples_per_combination / inference.n_samples."
            )
        num_samples = int(num_samples)
        if num_samples <= 0:
            raise ValueError(f"Combination #{j}: num_samples must be > 0, got {num_samples}.")

        # -1 (unknown) is the default state for unspecified fields.
        row = [-1] * K
        for field, state in vals.items():
            if field not in name_to_idx:
                raise ValueError(
                    f"Unknown label {field!r} in combination #{j}. "
                    f"Valid labels: {bool_col_names}."
                )
            state_int = int(state)
            if state_int not in VALID_LABEL_STATES:
                raise ValueError(
                    f"Invalid value {state!r} for label {field!r} in combination #{j}. "
                    f"Allowed states: {{-1 (unknown), 0 (absent), 1 (present)}}."
                )
            row[name_to_idx[field]] = state_int

        if policy == "require_all":
            missing = [name for name in bool_col_names if name not in vals]
            if missing:
                raise ValueError(
                    f"missing_field_policy='require_all' but combination #{j} omits "
                    f"fields {missing}. Provide all of {bool_col_names}."
                )

        meta_row = {name: row[name_to_idx[name]] for name in bool_col_names}
        meta_row["combination_id"] = j
        for _ in range(num_samples):
            rows.append(list(row))
            row_meta.append(dict(meta_row))

    y = torch.tensor(rows, dtype=torch.long)  # [N, K]
    return y, row_meta