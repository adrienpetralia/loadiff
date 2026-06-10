#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a trained TSTR classifier on the **real test split** (SMACH).

This is the "Test Real" half of Train-Synthetic-Test-Real: regardless of what the
model was trained on (synthetic / real / mixed), it is always evaluated against the
held-out real ``test`` split defined in ``data/cer/train_valid_test_id_split.pkl``.
The primary metric written to ``metrics.json`` is ``BALANCED_ACCURACY``.

Example:
    python -m scripts.tstr_evaluation.evaluate_tstr \\
        --classifier_type rocket --target_label cooker \\
        --model_path results/tstr_experiments/exp1_pure_synthetic/rocket/cooker/model.pkl \\
        --test_data_path data/cer \\
        --output_dir results/tstr_experiments/exp1_pure_synthetic/rocket/cooker
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Optional

from scripts.tstr_evaluation.utils.data_loader import load_data
from scripts.tstr_evaluation.utils.metrics import compute_classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_classifier(classifier_type: str, model_path: str, *, device: str):
    if classifier_type == "rocket":
        from scripts.tstr_evaluation.utils.classifiers import RocketClassifier

        return RocketClassifier.load(model_path, device=device)
    if classifier_type == "transapp":
        from scripts.tstr_evaluation.utils.classifiers import TransAppClassifier

        return TransAppClassifier.load(model_path, device=device)
    raise ValueError(f"Unknown classifier_type {classifier_type!r} (rocket|transapp).")


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a TSTR classifier on real test data.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_data_path", required=True, help="Real data dir (uses the test split).")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_label", required=True)
    parser.add_argument("--classifier_type", required=True, choices=["rocket", "transapp"])
    parser.add_argument("--dataset", default="smach", choices=["smach", "cer", "cer_bis"])
    parser.add_argument("--test_split", default="test")
    parser.add_argument(
        "--balanced", action="store_true", default=False,
        help="Balance the test set. Default off: report on the natural real test distribution.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading real test data from %s (split=%s).", args.test_data_path, args.test_split)
    X_test, y_test = load_data(
        args.test_data_path,
        args.target_label,
        split=args.test_split,
        dataset=args.dataset,
        balanced=args.balanced,
        postprocess=False,  # real data is never post-processed
        seed=args.seed,
    )
    logger.info("Test set: X=%s, positives=%d/%d", X_test.shape, int((y_test == 1).sum()), len(y_test))

    clf = load_classifier(args.classifier_type, args.model_path, device=args.device)
    y_pred = clf.predict(X_test)
    metrics = compute_classification_metrics(y_test, y_pred)

    payload = {
        **metrics,
        "meta": {
            "classifier_type": args.classifier_type,
            "target_label": args.target_label,
            "test_data_path": args.test_data_path,
            "test_split": args.test_split,
            "model_path": args.model_path,
            "balanced_test": args.balanced,
        },
    }
    out_path = os.path.join(args.output_dir, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(
        "BALANCED_ACCURACY=%.4f (ACCURACY=%.4f) -> %s",
        metrics["BALANCED_ACCURACY"], metrics["ACCURACY"], out_path,
    )


if __name__ == "__main__":
    main()
