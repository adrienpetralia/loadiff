#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified data loading for the TSTR evaluation.

Two data sources are handled transparently and returned in the *same* representation
(load curves in Watts ``X`` of shape ``[N, L]`` and binary labels ``y`` of shape
``[N]``) so that the downstream classifiers never need to know where the data came
from:

* **Real SMACH data** (``data/smach/``): loaded with
  :class:`src.helpers.dataset.SmachDataset` restricted to a ``train`` / ``val`` /
  ``test`` split (``train_valid_test_id_split.pkl``). The binary label is whether the
  client owns ``target_label`` (e.g. ``CHAUFF_ELEC``).
* **Synthetic / generated data** (``runs_inference/...`` or a mixed-dataset dir):
  loaded from ``loadit_samples.npy`` + ``y.npy`` (with ``run_info.json`` giving the
  ``label_names`` order) or from a self-describing ``X.npy`` + ``y.npy`` pair.

The post-processing QC pipeline is applied to *generated* data only.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Appliance label columns supported by the SMACH experiments.
SMACH_APPLIANCES = ("CHAUFF_ELEC", "ECS", "CLIM")

# Supported appliance label columns per dataset (Train Synthetic, Test Real).
DATASET_APPLIANCES = {
    "smach": SMACH_APPLIANCES,
    "cer": ("cooker", "dishwasher", "water_heater"),
    "cer_bis": ("ev", "heater", "water_heater"),
}


def validate_dataset_label(dataset: str, target_label: str) -> None:
    """Explicit validation of the (dataset, appliance) pair."""
    if dataset not in DATASET_APPLIANCES:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Valid datasets: {sorted(DATASET_APPLIANCES)}."
        )
    if target_label not in DATASET_APPLIANCES[dataset]:
        raise ValueError(
            f"target_label {target_label!r} is not a {dataset} appliance "
            f"{DATASET_APPLIANCES[dataset]}."
        )

# Default amplitude scale for SMACH curves (Watts). Curves are kept in Watts here
# and scaled to [0, 1] inside each classifier so real and synthetic data share the
# exact same normalisation (matches configs/rocket_binary_classifier.yaml:scale_param2
# and src/transapp/config/TransAppV2.yaml:scale_param2).
SMACH_VALUE_SCALE = 10000.0

# Default geometry of a SMACH yearly curve.
SMACH_NB_DAYS = 365
SMACH_PATCH_LENGTH_DAY = 48


def is_synthetic_dir(data_dir: str) -> bool:
    """Heuristic: a directory holds *generated* data (vs. real SMACH parquet data)."""
    if "runs_inference" in str(data_dir).replace("\\", "/"):
        return True
    return (
        os.path.exists(os.path.join(data_dir, "loadit_samples.npy"))
        or os.path.exists(os.path.join(data_dir, "X.npy"))
    )


# ---------------------------------------------------------------------------
# Balancing / sub-sampling helpers
# ---------------------------------------------------------------------------
def balance_xy(
    X: np.ndarray, y: np.ndarray, *, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Undersample the majority class to a 50/50 positive/negative split."""
    y = y.astype(np.int64).ravel()
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n = min(len(pos), len(neg))
    if n == 0:
        raise ValueError(
            f"Cannot balance: one class is empty (n_pos={len(pos)}, n_neg={len(neg)})."
        )
    rng = np.random.default_rng(seed)
    sel = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
    rng.shuffle(sel)
    return X[sel], y[sel]


def cap_n_samples(
    X: np.ndarray, y: np.ndarray, n_samples: Optional[int], *, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly cap the dataset to at most ``n_samples`` rows (preserving order)."""
    if n_samples is None or n_samples >= len(y):
        return X, y
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(len(y), int(n_samples), replace=False))
    return X[sel], y[sel]


# ---------------------------------------------------------------------------
# Real (SMACH parquet) loading
# Candidate file names for the per-client appliance metadata (the repo standard is
# ``labels.parquet``; older configs used ``metadata.parquet``).
METADATA_FILE_CANDIDATES = ("labels.parquet", "label.parquet", "metadata.parquet")


def _resolve_metadata_path(data_dir: str, metadata_file: Optional[str]) -> str:
    """Resolve the metadata parquet, auto-detecting the file name when unspecified."""
    candidates = [metadata_file] if metadata_file else list(METADATA_FILE_CANDIDATES)
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No appliance metadata parquet found in {data_dir} "
        f"(looked for: {', '.join(candidates)})."
    )


# ---------------------------------------------------------------------------
def load_real_data(
    data_dir: str,
    target_label: str,
    split: str,
    *,
    dataset: str = "smach",
    load_curve_file: str = "load_curve.parquet",
    metadata_file: Optional[str] = None,
    split_file: str = "train_valid_test_id_split.pkl",
    nb_days: int = SMACH_NB_DAYS,
    patch_length_day: int = SMACH_PATCH_LENGTH_DAY,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load real curves (Watts) and the binary ``target_label`` for one split.

    ``dataset`` selects the dataset class (``smach`` -> SmachDataset, ``cer`` ->
    CERDataset, ``cer_bis`` -> CERBisDataset); each uses its own id/time-column and
    date defaults. Curves are returned in raw Watts (classifiers rescale internally).
    """
    from src.helpers.loadiff_inference import get_dataset_class

    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be train/val/test, got {split!r}.")
    dataset_cls = get_dataset_class(dataset)

    split_path = os.path.join(data_dir, split_file)
    with open(split_path, "rb") as f:
        splits = pickle.load(f)
    if split not in splits:
        raise KeyError(f"Split {split!r} not in {split_path}. Available: {sorted(splits)}.")
    clients = splits[split]

    dataset_obj = dataset_cls(
        path_load_curves=os.path.join(data_dir, load_curve_file),
        path_metadata=_resolve_metadata_path(data_dir, metadata_file),
        list_pdl=clients,
        bool_col_names=[target_label],
        nb_days=nb_days,
        patch_length_day=patch_length_day,
        scale_param1=0.0,
        scale_param2=1.0,  # keep raw Watts; classifiers rescale by the value scale
        random_window=False,
    )

    X = np.empty((len(dataset_obj), nb_days * patch_length_day), dtype=np.float32)
    y = np.empty((len(dataset_obj),), dtype=np.int64)
    for i in range(len(dataset_obj)):
        values_win, _exo, label = dataset_obj[i]
        X[i] = values_win.reshape(-1).numpy()
        if label.numel() < 1:
            raise ValueError(
                f"No label '{target_label}' returned; check metadata columns."
            )
        y[i] = int(label[0].item() > 0.5)
    return X, y


# ---------------------------------------------------------------------------
# Synthetic / generated loading
# ---------------------------------------------------------------------------
def _resolve_label_index(run_info_path: str, target_label: str) -> int:
    with open(run_info_path, "r", encoding="utf-8") as f:
        run_info = json.load(f)
    label_names: List[str] = run_info.get("label_names", []) or []
    if not label_names:
        raise ValueError(f"{run_info_path} has no 'label_names'; cannot resolve labels.")
    if target_label not in label_names:
        raise ValueError(
            f"target_label {target_label!r} not in run_info label_names {label_names}."
        )
    return label_names.index(target_label)


def load_synthetic_data(
    data_dir: str,
    target_label: str,
    *,
    postprocess: bool = True,
    patch_length_day: int = SMACH_PATCH_LENGTH_DAY,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load generated curves (Watts) + binary ``target_label``, optionally QC-cleaned.

    Supports two on-disk layouts:
      * inference output: ``loadit_samples.npy`` [N, L] + ``y.npy`` [N, K] +
        ``run_info.json`` (``label_names`` gives the column order of ``y``);
      * self-describing pair: ``X.npy`` [N, L] + ``y.npy`` [N] (already binary for
        ``target_label``; produced by ``create_mixed_dataset``).
    """
    x_npy = os.path.join(data_dir, "X.npy")
    if os.path.exists(x_npy):
        X = np.load(x_npy).astype(np.float32)
        y = np.load(os.path.join(data_dir, "y.npy")).astype(np.int64).ravel()
    else:
        samples_path = os.path.join(data_dir, "loadit_samples.npy")
        if not os.path.exists(samples_path):
            raise FileNotFoundError(f"No loadit_samples.npy or X.npy in {data_dir}.")
        X = np.load(samples_path).astype(np.float32)
        labels = np.load(os.path.join(data_dir, "y.npy"))
        if labels.ndim == 1:
            y = (labels > 0.5).astype(np.int64)
        else:
            idx = _resolve_label_index(os.path.join(data_dir, "run_info.json"), target_label)
            y = (labels[:, idx] > 0.5).astype(np.int64)

    if X.ndim != 2:
        raise ValueError(f"Generated curves must be 2-D [N, L]; got {X.shape}.")
    if len(X) != len(y):
        raise ValueError(f"Curves ({len(X)}) and labels ({len(y)}) length mismatch.")

    if postprocess:
        from scripts.tstr_evaluation.utils.postprocessing import apply_postprocessing

        X, kept = apply_postprocessing(X, patch_length=patch_length_day)
        y = y[kept]
    return X, y


# ---------------------------------------------------------------------------
# Pre-generated baseline loading (timevqvae / energydiff)
#
# Some generative baselines ship their synthetic populations as ready-made ``.npy``
# files — one per ``(appliance, label_value)`` pair — instead of being run at inference
# time. On-disk layout (one curve per row, Watts)::
#
#   <runs_root>/<baseline>/<baseline>_<appliance>/label<value>.npy
#   e.g. .../runs_smach/energydiff/energydiff_CHAUFF_ELEC/label0.npy
#
# No post-processing is applied; at most ``max_per_file`` curves are kept per file,
# with a reproducible (seeded) sub-selection when the file holds more.
# ---------------------------------------------------------------------------
DEFAULT_MAX_PER_FILE = 2048


def pregenerated_baseline_dir(runs_root: str, baseline: str, target_label: str) -> str:
    """Directory holding a baseline's per-label ``.npy`` files for one appliance."""
    return os.path.join(runs_root, baseline, f"{baseline}_{target_label}")


def _read_npy_header(fp):
    """Parse a ``.npy`` header, returning ``(shape, fortran_order, dtype, data_offset)``."""
    from numpy.lib import format as npformat

    version = npformat.read_magic(fp)
    if version == (1, 0):
        shape, fortran_order, dtype = npformat.read_array_header_1_0(fp)
    elif version == (2, 0):
        shape, fortran_order, dtype = npformat.read_array_header_2_0(fp)
    else:  # pragma: no cover - very rare future header versions
        shape, fortran_order, dtype = npformat._read_array_header(fp, version)
    return shape, fortran_order, dtype, fp.tell()


def _curve_length(shape, path: str) -> int:
    """Validate the on-disk shape and return the per-curve length ``L``.

    Accepts ``[N, L]`` and a singleton channel axis (``[N, 1, L]`` / ``[N, L, 1]``,
    e.g. timevqvae). Anything else (1-D, genuinely multi-channel) raises an error.
    """
    if len(shape) == 2:
        return int(shape[1])
    if len(shape) == 3 and 1 in shape[1:]:
        return int(shape[2] if shape[1] == 1 else shape[1])
    raise ValueError(
        f"Pre-generated baseline file {path} must be 2-D [N, L] "
        f"(or [N, 1, L] / [N, L, 1]); got shape {tuple(shape)}."
    )


def _open_curve_meta(fp, path: str):
    """Parse a curve ``.npy`` and return its streaming metadata.

    Returns ``(L, usable_rows, dtype, data_offset, row_elems, row_bytes)``. ``L`` is the
    per-curve length (a singleton channel axis is squeezed). ``usable_rows`` clamps the
    declared row count to the number of rows actually written, so truncated files are
    tolerated. Raises on a missing file, an unreadable shape, or Fortran order.
    """
    shape, fortran_order, dtype, data_offset = _read_npy_header(fp)
    L = _curve_length(shape, path)
    if fortran_order and len(shape) > 1:
        raise ValueError(
            f"{path}: Fortran-ordered .npy is not supported by the streaming reader; "
            f"re-save in C order."
        )
    n_rows = int(shape[0])
    row_elems = int(np.prod(shape[1:])) if len(shape) > 1 else 1  # == L (singleton axis)
    row_bytes = row_elems * dtype.itemsize

    file_size = os.path.getsize(path)
    avail_rows = (file_size - data_offset) // row_bytes if row_bytes else 0
    usable_rows = min(n_rows, avail_rows)
    if usable_rows <= 0:
        raise ValueError(
            f"{path}: no complete rows available (header declares {n_rows} rows, "
            f"file holds {avail_rows})."
        )
    if usable_rows < n_rows:
        logger.warning(
            "%s appears truncated: header declares %d rows but only %d are fully "
            "written; using those %d.", path, n_rows, usable_rows, usable_rows,
        )
    return L, usable_rows, dtype, data_offset, row_elems, row_bytes


def _subsample_indices(
    candidates: np.ndarray, max_n: Optional[int], seed: int
) -> np.ndarray:
    """Reproducibly pick at most ``max_n`` of ``candidates`` (sorted ascending)."""
    candidates = np.asarray(candidates)
    if max_n is not None and len(candidates) > int(max_n):
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(candidates), int(max_n), replace=False)
        candidates = candidates[sel]
    return np.sort(candidates)


def _read_rows(fp, idx, dtype, data_offset, row_elems, row_bytes, L, path):
    """Read the sorted row indices ``idx`` from an open curve ``.npy`` as float32 ``[k, L]``.

    Only the requested rows are read (contiguous runs coalesced into single reads); the
    whole array is never materialised in memory.
    """
    out = np.empty((len(idx), row_elems), dtype=dtype)
    i, n = 0, len(idx)
    while i < n:
        j = i
        while j + 1 < n and idx[j + 1] == idx[j] + 1:
            j += 1
        count = j - i + 1
        fp.seek(data_offset + int(idx[i]) * row_bytes)
        buf = fp.read(row_bytes * count)
        if len(buf) != row_bytes * count:
            raise ValueError(f"{path}: short read at row {int(idx[i])} (truncated file).")
        out[i : j + 1] = np.frombuffer(buf, dtype=dtype, count=row_elems * count).reshape(
            count, row_elems
        )
        i = j + 1
    return out.astype(np.float32, copy=False).reshape(len(idx), L)


def _load_capped_npy(
    path: str, max_per_file: Optional[int], seed: int
) -> np.ndarray:
    """Load up to ``max_per_file`` rows of a ``[N, L]`` curve file *without* reading it whole.

    The ``.npy`` header is parsed to locate the data; then only the selected rows are read
    via ``seek`` (the whole array is never materialised in memory). This also tolerates a
    **truncated** file — only fully-written rows are considered, so a cap of e.g. 1024
    still succeeds on a partially-written file that holds at least that many complete rows.

    A singleton channel axis is squeezed (``[N, 1, L]`` / ``[N, L, 1]`` -> ``[N, L]``).
    Raises an explicit error if the file is missing or cannot be reduced to a 2-D curve
    array, so the TSTR pipeline fails loudly on a format mismatch.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pre-generated baseline file not found: {path}")
    with open(path, "rb") as fp:
        L, usable_rows, dtype, data_offset, row_elems, row_bytes = _open_curve_meta(fp, path)
        # Reproducible sub-selection among the usable rows (identical to the previous
        # behaviour when the file is intact, since usable_rows == n_rows then).
        idx = _subsample_indices(np.arange(usable_rows), max_per_file, seed)
        return _read_rows(fp, idx, dtype, data_offset, row_elems, row_bytes, L, path)


def load_pregenerated_baseline_data(
    runs_root: str,
    baseline: str,
    target_label: str,
    *,
    labels=(0, 1),
    max_per_file: Optional[int] = DEFAULT_MAX_PER_FILE,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a pre-generated baseline population for one appliance as ``(X, y)``.

    Reads ``label<value>.npy`` for each value in ``labels`` from
    ``<runs_root>/<baseline>/<baseline>_<target_label>/``. Each file is capped to at
    most ``max_per_file`` curves (reproducible, seeded sub-selection per file). No
    post-processing is applied. Every curve's label is its source file's value.

    Returns ``(X, y)`` with ``X`` of shape ``[N, L]`` (Watts) and ``y`` of shape
    ``[N]`` (binary), identical to the other TSTR loaders.
    """
    base = pregenerated_baseline_dir(runs_root, baseline, target_label)
    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for offset, value in enumerate(labels):
        path = os.path.join(base, f"label{int(value)}.npy")
        # Per-file seed offset keeps each file's sub-selection reproducible & independent.
        arr = _load_capped_npy(path, max_per_file, seed + offset)
        Xs.append(arr)
        ys.append(np.full(len(arr), int(value), dtype=np.int64))
    lengths = {a.shape[1] for a in Xs}
    if len(lengths) != 1:
        raise ValueError(
            f"Inconsistent curve length across {base} label files: {sorted(lengths)}."
        )
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Pre-generated baseline loading (timeweaver)
#
# timeweaver is multi-conditional: a single directory holds *one* generated population
# whose curves carry several appliance labels at once. On-disk layout::
#
#   <runs_root>/timeweaver/samples.npy        # [N, L] (or [N, 1, L]) curves
#   <runs_root>/timeweaver/y.npy              # [N, n_labels] multi-label matrix
#   <runs_root>/timeweaver/logs_summary.json  # meta.label_names gives the y column order
#
# For one appliance we read ``meta.label_names``, locate the appliance column (never
# hard-coding the order), and split ``samples`` into the label0 / label1 sub-populations
# by that column. Each class is capped to at most ``max_per_class`` curves (reproducible,
# seeded). No post-processing is applied.
# ---------------------------------------------------------------------------
TIMEWEAVER_BASELINES = ("timeweaver",)  # baselines using the single-dir multilabel layout


def timeweaver_baseline_dir(runs_root: str, baseline: str = "timeweaver") -> str:
    """Directory holding timeweaver's ``samples.npy`` / ``y.npy`` / ``logs_summary.json``."""
    return os.path.join(runs_root, baseline)


def _read_timeweaver_label_names(summary_path: str) -> List[str]:
    """Read ``meta.label_names`` from a timeweaver ``logs_summary.json``."""
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    label_names = ((summary.get("meta") or {}).get("label_names")) or []
    if not label_names:
        raise ValueError(
            f"{summary_path} has no 'meta.label_names'; cannot resolve labels."
        )
    return list(label_names)


def load_timeweaver_baseline_data(
    runs_root: str,
    target_label: str,
    *,
    baseline: str = "timeweaver",
    max_per_class: Optional[int] = DEFAULT_MAX_PER_FILE,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the timeweaver population for one appliance as ``(X, y)``.

    Reads ``meta.label_names`` from ``logs_summary.json``, finds ``target_label``'s column
    in ``y.npy``, and splits ``samples.npy`` into label0 / label1 by that column. Each
    class is capped to at most ``max_per_class`` curves (reproducible, seeded). Only the
    selected curve rows are read from ``samples.npy`` (never the whole array). No
    post-processing is applied. The label order is read from the metadata, never assumed.
    """
    base = timeweaver_baseline_dir(runs_root, baseline)
    summary_path = os.path.join(base, "logs_summary.json")
    samples_path = os.path.join(base, "samples.npy")
    y_path = os.path.join(base, "y.npy")
    for p in (summary_path, samples_path, y_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"timeweaver baseline file not found: {p}")

    label_names = _read_timeweaver_label_names(summary_path)
    if target_label not in label_names:
        raise ValueError(
            f"target_label {target_label!r} not in timeweaver label_names {label_names}."
        )
    col = label_names.index(target_label)

    y_all = np.load(y_path)  # [N, n_labels] — small relative to the curves
    if y_all.ndim == 1:
        y_all = y_all[:, None]
    if y_all.ndim != 2:
        raise ValueError(f"{y_path} must be 2-D [N, n_labels]; got shape {y_all.shape}.")
    if y_all.shape[1] != len(label_names):
        raise ValueError(
            f"{y_path} has {y_all.shape[1]} label columns but logs_summary.json lists "
            f"{len(label_names)} label_names {label_names}."
        )

    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    with open(samples_path, "rb") as fp:
        L, usable_rows, dtype, data_offset, row_elems, row_bytes = _open_curve_meta(
            fp, samples_path
        )
        n = min(usable_rows, len(y_all))
        if n <= 0:
            raise ValueError(
                f"timeweaver: no aligned rows between {samples_path} and {y_path}."
            )
        if usable_rows != len(y_all):
            logger.warning(
                "timeweaver dimension mismatch: samples has %d usable rows but y.npy has "
                "%d; using the first %d aligned rows.", usable_rows, len(y_all), n,
            )
        col_vals = (y_all[:n, col] > 0.5).astype(np.int64)
        for offset, cls in enumerate((0, 1)):
            candidates = np.where(col_vals == cls)[0]
            if len(candidates) == 0:
                raise ValueError(
                    f"timeweaver: no examples with {target_label}={cls} "
                    f"(column {col}) among the first {n} rows."
                )
            # Per-class seed offset keeps each class's sub-selection reproducible.
            idx = _subsample_indices(candidates, max_per_class, seed + offset)
            Xs.append(_read_rows(fp, idx, dtype, data_offset, row_elems, row_bytes, L, samples_path))
            ys.append(np.full(len(idx), cls, dtype=np.int64))

    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Dispatcher across the pre-generated baseline layouts
# ---------------------------------------------------------------------------
def baseline_population_dir(runs_root: str, baseline: str, target_label: str) -> str:
    """Source directory for a pre-generated baseline (layout depends on the baseline)."""
    if baseline in TIMEWEAVER_BASELINES:
        return timeweaver_baseline_dir(runs_root, baseline)
    return pregenerated_baseline_dir(runs_root, baseline, target_label)


def load_baseline_population(
    runs_root: str,
    baseline: str,
    target_label: str,
    *,
    max_per_class: Optional[int] = DEFAULT_MAX_PER_FILE,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a pre-generated baseline population for one appliance, dispatching on layout.

    * ``timeweaver`` -> single-dir multilabel layout (``samples.npy`` + ``y.npy`` +
      ``logs_summary.json``), split by the appliance column.
    * everything else -> per-label files (``<baseline>_<appliance>/label<value>.npy``).

    Both cap each class to at most ``max_per_class`` curves (reproducible, seeded) and
    apply no post-processing.
    """
    if baseline in TIMEWEAVER_BASELINES:
        return load_timeweaver_baseline_data(
            runs_root, target_label, baseline=baseline, max_per_class=max_per_class, seed=seed
        )
    return load_pregenerated_baseline_data(
        runs_root, baseline, target_label, max_per_file=max_per_class, seed=seed
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def load_data(
    data_dir: str,
    target_label: str,
    *,
    split: Optional[str] = None,
    dataset: str = "smach",
    balanced: bool = True,
    postprocess: bool = True,
    n_samples: Optional[int] = None,
    seed: int = 0,
    patch_length_day: int = SMACH_PATCH_LENGTH_DAY,
    metadata_file: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load real or generated data as ``(X, y)`` with optional balancing/capping.

    Args:
        data_dir: ``data/smach`` for real data, or an inference / mixed-dataset
            directory for generated data.
        target_label: Appliance to classify (``CHAUFF_ELEC`` / ``ECS`` / ``CLIM``).
        split: ``train`` / ``val`` / ``test`` — required for real data, ignored for
            generated data.
        balanced: When ``True``, undersample to a 50/50 class split.
        postprocess: Apply the QC pipeline (generated data only).
        n_samples: Optional cap on the number of returned samples.
        seed: RNG seed for balancing / capping.
        patch_length_day: Points per day (QC ``dt`` derivation, curve reshaping).
        metadata_file: Optional explicit appliance-metadata file name for real data
            (default: auto-detect ``labels.parquet`` / ``label.parquet`` / ``metadata.parquet``).

    Returns:
        ``(X, y)`` with ``X`` of shape ``[N, L]`` (Watts) and ``y`` of shape ``[N]``.
    """
    validate_dataset_label(dataset, target_label)

    if is_synthetic_dir(data_dir):
        X, y = load_synthetic_data(
            data_dir, target_label, postprocess=postprocess, patch_length_day=patch_length_day
        )
    else:
        if split is None:
            raise ValueError("split is required when loading real data.")
        X, y = load_real_data(
            data_dir, target_label, split, dataset=dataset,
            metadata_file=metadata_file, patch_length_day=patch_length_day,
        )

    if balanced:
        X, y = balance_xy(X, y, seed=seed)
    if n_samples is not None:
        X, y = cap_n_samples(X, y, n_samples, seed=seed)
    return X, y
