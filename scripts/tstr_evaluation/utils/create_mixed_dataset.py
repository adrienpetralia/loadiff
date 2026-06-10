#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a mixed Synthetic + Real training dataset for TSTR Experiment 2.

The synthetic curves (post-processed) are combined with a configurable fraction of
the *real* ``train`` split. The percentage refers to the fraction of the total real
training data that is injected. The combined set is class-balanced (50/50
appliance-present / appliance-absent) so the classifier never sees a skewed prior.

The result is written as a self-describing ``X.npy`` / ``y.npy`` pair (Watts, binary
labels) plus a ``metadata.json`` recording the composition of the mix.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np

from scripts.tstr_evaluation.utils.data_loader import (
    SMACH_PATCH_LENGTH_DAY,
    balance_xy,
    load_real_data,
    load_synthetic_data,
)


def _take_fraction(
    X: np.ndarray, y: np.ndarray, fraction: float, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Take a random ``fraction`` (0..1) of ``(X, y)`` preserving class proportions."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}.")
    if fraction >= 1.0:
        return X, y
    rng = np.random.default_rng(seed)
    keep = []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        n = int(round(len(idx) * fraction))
        if n > 0:
            keep.append(rng.choice(idx, n, replace=False))
    if not keep:
        return X[:0], y[:0]
    sel = np.sort(np.concatenate(keep))
    return X[sel], y[sel]


def create_mixed_dataset(
    synthetic_dir: str,
    real_dir: str,
    output_dir: str,
    percentage: float,
    target_label: str,
    *,
    dataset: str = "smach",
    balanced: bool = True,
    postprocess: bool = True,
    seed: int = 0,
    patch_length_day: int = SMACH_PATCH_LENGTH_DAY,
) -> Dict[str, object]:
    """Create and persist a mixed Synthetic + Real dataset.

    Args:
        synthetic_dir: Directory with generated curves (``loadit_samples.npy`` ...).
        real_dir: Real SMACH data directory (``data/smach``).
        output_dir: Where to write ``X.npy`` / ``y.npy`` / ``metadata.json``.
        percentage: Percentage (0..100) of the real ``train`` split to inject.
        target_label: Appliance to classify.
        balanced: Balance the final mix to 50/50 classes.
        postprocess: Apply QC to the synthetic curves.
        seed: RNG seed for sub-sampling / balancing.
        patch_length_day: Points per day (used by QC and for reshaping).

    Returns:
        The metadata dict describing the produced mix (also written to disk).
    """
    fraction = float(percentage) / 100.0

    X_syn, y_syn = load_synthetic_data(
        synthetic_dir, target_label, postprocess=postprocess, patch_length_day=patch_length_day
    )
    X_real, y_real = load_real_data(
        real_dir, target_label, split="train", dataset=dataset, patch_length_day=patch_length_day
    )
    X_real_sel, y_real_sel = _take_fraction(X_real, y_real, fraction, seed=seed)

    if X_syn.shape[1] != X_real_sel.shape[1] and len(X_real_sel) > 0:
        raise ValueError(
            f"Sequence length mismatch synthetic={X_syn.shape[1]} vs real={X_real_sel.shape[1]}."
        )

    X_mix = np.concatenate([X_syn, X_real_sel], axis=0) if len(X_real_sel) else X_syn
    y_mix = np.concatenate([y_syn, y_real_sel], axis=0) if len(y_real_sel) else y_syn

    n_before_balance = len(y_mix)
    if balanced:
        X_mix, y_mix = balance_xy(X_mix, y_mix, seed=seed)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X.npy"), X_mix.astype(np.float32))
    np.save(os.path.join(output_dir, "y.npy"), y_mix.astype(np.int64))

    metadata: Dict[str, object] = {
        "dataset": dataset,
        "target_label": target_label,
        "percentage_real": float(percentage),
        "balanced": bool(balanced),
        "postprocess": bool(postprocess),
        "seed": int(seed),
        "n_synthetic": int(len(y_syn)),
        "n_real_available": int(len(y_real)),
        "n_real_used": int(len(y_real_sel)),
        "n_before_balance": int(n_before_balance),
        "n_final": int(len(y_mix)),
        "n_final_positive": int((y_mix == 1).sum()),
        "n_final_negative": int((y_mix == 0).sum()),
        "synthetic_dir": str(synthetic_dir),
        "real_dir": str(real_dir),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(description="Create a mixed Synthetic + Real dataset.")
    p.add_argument("--synthetic_dir", required=True)
    p.add_argument("--real_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--percentage", type=float, required=True, help="0..100 of real train.")
    p.add_argument("--target_label", required=True)
    p.add_argument("--dataset", default="smach", choices=["smach", "cer", "cer_bis"])
    p.add_argument("--balanced", action="store_true", default=True)
    p.add_argument("--no_postprocess", dest="postprocess", action="store_false", default=True)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    meta = create_mixed_dataset(
        synthetic_dir=args.synthetic_dir,
        real_dir=args.real_dir,
        output_dir=args.output_dir,
        percentage=args.percentage,
        target_label=args.target_label,
        dataset=args.dataset,
        balanced=args.balanced,
        postprocess=args.postprocess,
        seed=args.seed,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
