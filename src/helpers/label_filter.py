"""Filter a parquet daily dataset to clients owning a given appliance.

Used to train per-appliance baselines (one dedicated, *unconditioned* model per
appliance) on the subset of clients whose metadata label matches a target value
(e.g. ``heater == 1``). The filtering mirrors the in-place subsetting already used
by ``CERDataset`` (drop policy): it slices the underlying tensors so every dataset
attribute (``nb_days``, ``patch_length``, ``data``, ``user_start_date`` ...) stays
valid and ``__getitem__`` keeps returning ``(values, exog, y)`` consistently.

Label states follow the project convention: 1 (present), 0 (absent), -1 (unknown).
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch


def select_label_indices(
    data_pop,
    bool_col_names: Sequence[str],
    filter_by_label: Mapping[str, int],
) -> List[int]:
    """Return the row indices whose labels satisfy every ``col == value`` condition.

    Args:
        data_pop: ``[N, K]`` label matrix (tensor or array) with states in {-1, 0, 1}.
        bool_col_names: Ordered label names matching the columns of ``data_pop``.
        filter_by_label: Mapping ``label_name -> target_state`` (conditions are ANDed).

    Returns:
        Sorted list of matching row indices.

    Raises:
        ValueError: On shape mismatch, unknown label name, or invalid target state.
    """
    arr = data_pop.cpu().numpy() if isinstance(data_pop, torch.Tensor) else np.asarray(data_pop)
    bool_col_names = list(bool_col_names)
    if arr.ndim != 2 or arr.shape[1] != len(bool_col_names):
        raise ValueError(
            f"data_pop shape {arr.shape} is incompatible with "
            f"{len(bool_col_names)} label columns {bool_col_names}."
        )

    name_to_idx = {name: i for i, name in enumerate(bool_col_names)}
    mask = np.ones(arr.shape[0], dtype=bool)
    for col, value in dict(filter_by_label).items():
        if col not in name_to_idx:
            raise ValueError(
                f"Unknown label {col!r} in filter_by_label. Valid labels: {bool_col_names}."
            )
        value = int(value)
        if value not in (-1, 0, 1):
            raise ValueError(
                f"Invalid target state {value!r} for label {col!r}. Allowed: {{-1, 0, 1}}."
            )
        mask &= arr[:, name_to_idx[col]] == float(value)
    return np.flatnonzero(mask).tolist()


def filter_dataset_by_label(dataset, filter_by_label: Mapping[str, int]):
    """Filter ``dataset`` in place to the clients matching ``filter_by_label``.

    Requires the dataset to have loaded metadata (non-empty ``bool_col_names`` and
    ``data_pop``). Slices ``data``, ``id_clients``, ``data_pop`` and, when present,
    per-id ``temps_full`` so the dataset remains internally consistent.

    Args:
        dataset: A :class:`BaseParquetDailyDataset` instance with metadata loaded.
        filter_by_label: Mapping ``label_name -> target_state`` (conditions ANDed).

    Returns:
        The same dataset object, filtered in place.

    Raises:
        ValueError: If metadata is unavailable or no client matches the filter.
    """
    if not filter_by_label:
        return dataset

    bool_col_names = list(getattr(dataset, "bool_col_names", []) or [])
    data_pop = getattr(dataset, "data_pop", None)
    if not bool_col_names or data_pop is None or data_pop.numel() == 0:
        raise ValueError(
            "filter_by_label requires the dataset to load metadata. Set "
            "data.bool_col_names and data.path_parquet_part_metadata in the config."
        )

    idx = select_label_indices(data_pop, bool_col_names, filter_by_label)
    if not idx:
        raise ValueError(
            f"No client matches filter_by_label={dict(filter_by_label)} "
            f"(labels={bool_col_names}). Check the requested appliance/value."
        )

    dataset.data = dataset.data[idx]
    dataset.id_clients = [dataset.id_clients[i] for i in idx]
    dataset.data_pop = dataset.data_pop[idx]
    dataset.num_ids = dataset.data.shape[0]
    if getattr(dataset, "temps_full", None) is not None and getattr(dataset, "_temp_is_per_id", False):
        dataset.temps_full = dataset.temps_full[idx]
    return dataset