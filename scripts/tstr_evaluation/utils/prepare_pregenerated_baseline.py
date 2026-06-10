#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Materialise a pre-generated baseline population into a TSTR-ready directory.

Some generative baselines ship their synthetic curves as ready-made files instead of
being run at inference time. Two on-disk layouts are supported:

* **per-label** (``timevqvae`` / ``energydiff``) — one file per ``(appliance, label)``::

      <runs_root>/<baseline>/<baseline>_<appliance>/label<value>.npy

* **single-dir multilabel** (``timeweaver``) — one population carrying every label::

      <runs_root>/timeweaver/{samples.npy, y.npy, logs_summary.json}

This script loads the requested appliance (capping each class to at most
``--max_per_file`` curves with a reproducible, seeded sub-selection, and **no
post-processing**) and writes a self-describing ``X.npy`` / ``y.npy`` pair plus a
``metadata.json``. The existing TSTR pipeline (``train_classifiers`` / ``evaluate_tstr``)
then consumes that directory unchanged — exactly like the mixed datasets produced by
``create_mixed_dataset`` — so no part of the pipeline is duplicated.

Example::

    python -m scripts.tstr_evaluation.utils.prepare_pregenerated_baseline \\
        --runs_root /scratch/users/i50280/2026_generation_cdc/runs_smach \\
        --baseline energydiff --target_label CHAUFF_ELEC --dataset smach \\
        --output_dir runs_inference/tstr_baselines_pregenerated/energydiff_smach_CHAUFF_ELEC
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np

from scripts.tstr_evaluation.utils.data_loader import (
    DEFAULT_MAX_PER_FILE,
    baseline_population_dir,
    load_baseline_population,
    validate_dataset_label,
)


def prepare_pregenerated_baseline(
    runs_root: str,
    baseline: str,
    target_label: str,
    output_dir: str,
    *,
    dataset: str = "smach",
    max_per_file: int = DEFAULT_MAX_PER_FILE,
    seed: int = 0,
) -> Dict[str, object]:
    """Load a pre-generated baseline population and persist it as ``X.npy`` / ``y.npy``.

    Routes automatically to the right on-disk layout (per-label vs. timeweaver). Returns
    the metadata dict describing the produced dataset (also written to disk).
    """
    validate_dataset_label(dataset, target_label)

    X, y = load_baseline_population(
        runs_root,
        baseline,
        target_label,
        max_per_class=max_per_file,
        seed=seed,
    )

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X.npy"), X.astype(np.float32))
    np.save(os.path.join(output_dir, "y.npy"), y.astype(np.int64))

    metadata: Dict[str, object] = {
        "dataset": dataset,
        "baseline": baseline,
        "target_label": target_label,
        "source_dir": baseline_population_dir(runs_root, baseline, target_label),
        "max_per_file": int(max_per_file),
        "seed": int(seed),
        "postprocess": False,
        "n_total": int(len(y)),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "sequence_length": int(X.shape[1]),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Materialise a pre-generated baseline population for TSTR."
    )
    p.add_argument("--runs_root", required=True, help="Root dir holding <baseline>/...")
    p.add_argument("--baseline", required=True, help="e.g. timevqvae | energydiff | timeweaver")
    p.add_argument("--target_label", required=True, help="Appliance to classify.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset", default="smach", choices=["smach", "cer", "cer_bis"])
    p.add_argument(
        "--max_per_file",
        type=int,
        default=DEFAULT_MAX_PER_FILE,
        help="Max curves kept per class (reproducible sub-selection above this).",
    )
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)
    meta = prepare_pregenerated_baseline(
        runs_root=args.runs_root,
        baseline=args.baseline,
        target_label=args.target_label,
        output_dir=args.output_dir,
        dataset=args.dataset,
        max_per_file=args.max_per_file,
        seed=args.seed,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
