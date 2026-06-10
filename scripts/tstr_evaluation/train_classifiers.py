#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a TSTR classifier (ROCKET or TransApp) on real / synthetic / mixed data.

The training data can be:
  * the real cer ``train`` split (baseline);
  * a directory of generated curves (Experiment 1, pure synthetic);
  * a mixed Synthetic + Real directory (Experiment 2).

Validation always uses the *real* ``val`` split (early stopping for TransApp). The
trained model is saved as ``model.pkl`` (ROCKET) or ``checkpoint.pt`` (TransApp), and
train/val metrics are written to ``metrics.json``.

Example:
    python -m scripts.tstr_evaluation.train_classifiers \\
        --classifier_type rocket --target_label cooker \\
        --train_data_path runs_inference/inference_user_conditioned_latest \\
        --val_data_path data/smach --val_split val \\
        --output_dir results/tstr_experiments/exp1_pure_synthetic/rocket/cooker
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, Optional

from scripts.tstr_evaluation.utils.data_loader import load_data
from scripts.tstr_evaluation.utils.metrics import compute_classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}


def build_classifier(
    classifier_type: str, cfg: Dict[str, Any], *, balanced: bool, device: str, target_label: str
):
    """Instantiate a ROCKET or TransApp classifier from a (possibly empty) config."""
    if classifier_type == "rocket":
        from scripts.tstr_evaluation.utils.classifiers import RocketClassifier

        params = dict(cfg.get("rocket", {}))
        return RocketClassifier(
            n_kernels=int(params.get("n_kernels", 10000)),
            seed=int(params.get("seed", 0)),
            value_scale=float(params.get("value_scale", 10000.0)),
            balance_classes=balanced,
            device=device,
            batch_size=int(params.get("batch_size", 256)),
        )
    if classifier_type == "transapp":
        from scripts.tstr_evaluation.utils.classifiers import TransAppClassifier

        params = dict(cfg.get("transapp", {}))
        return TransAppClassifier(
            value_scale=float(params.get("value_scale", 10000.0)),
            exogene_var=params.get("exogene_var"),
            start_date=str(params.get("start_date", "01/01/2021")),
            freq=str(params.get("freq", "30min")),
            subsequence_length=params.get("subsequence_length", 1024),
            target_label=target_label,
            d_model=int(params.get("d_model", 128)),
            n_encoder_layers=int(params.get("n_encoder_layers", 3)),
            n_head=int(params.get("n_head", 8)),
            epochs=int(params.get("epochs", 15)),
            batch_size=int(params.get("batch_size", 32)),
            lr=float(params.get("lr", 1e-4)),
            weight_decay=float(params.get("weight_decay", 1e-3)),
            patience_es=int(params.get("patience_es", 3)),
            n_warmup_epochs=int(params.get("n_warmup_epochs", 1)),
            device=device,
            seed=int(params.get("seed", 0)),
            num_workers=int(params.get("num_workers", 0)),
        )
    raise ValueError(f"Unknown classifier_type {classifier_type!r} (rocket|transapp).")


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Train a TSTR classifier.")
    parser.add_argument("--classifier_type", required=True, choices=["rocket", "transapp"])
    parser.add_argument("--train_data_path", required=True)
    parser.add_argument("--val_data_path", default=None, help="Real data dir for validation.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_label", required=True)
    parser.add_argument("--dataset", default="smach", choices=["smach", "cer", "cer_bis"])
    parser.add_argument("--train_split", default=None, help="Split for real train data (train).")
    parser.add_argument("--val_split", default="val", help="Real validation split (val).")
    parser.add_argument("--balanced", action="store_true", default=True)
    parser.add_argument("--no_balanced", dest="balanced", action="store_false")
    parser.add_argument("--no_postprocess", dest="postprocess", action="store_false", default=True)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--pretrained_path", default=None, help="TransApp pretrained init.")
    parser.add_argument("--config", default=None, help="YAML with classifier hyper-params.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = _load_config(args.config)

    logger.info("Loading training data from %s (label=%s).", args.train_data_path, args.target_label)
    X_train, y_train = load_data(
        args.train_data_path,
        args.target_label,
        split=args.train_split,
        dataset=args.dataset,
        balanced=args.balanced,
        postprocess=args.postprocess,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    logger.info("Train set: X=%s, positives=%d/%d", X_train.shape, int((y_train == 1).sum()), len(y_train))

    X_val = y_val = None
    if args.val_data_path:
        logger.info("Loading validation data from %s (split=%s).", args.val_data_path, args.val_split)
        X_val, y_val = load_data(
            args.val_data_path,
            args.target_label,
            split=args.val_split,
            dataset=args.dataset,
            balanced=args.balanced,
            postprocess=args.postprocess,
            seed=args.seed,
        )

    clf = build_classifier(
        args.classifier_type, cfg, balanced=args.balanced, device=args.device,
        target_label=args.target_label,
    )

    if args.classifier_type == "transapp":
        clf.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            pretrained_path=args.pretrained_path,
            checkpoint_path=os.path.join(args.output_dir, "_transapp_tmp.pt"),
        )
        model_path = os.path.join(args.output_dir, "checkpoint.pt")
    else:
        clf.fit(X_train, y_train)
        model_path = os.path.join(args.output_dir, "model.pkl")
    clf.save(model_path)
    logger.info("Saved model to %s", model_path)

    metrics: Dict[str, Any] = {
        "train": compute_classification_metrics(y_train, clf.predict(X_train)),
    }
    if X_val is not None:
        metrics["val"] = compute_classification_metrics(y_val, clf.predict(X_val))
        logger.info("Val BALANCED_ACCURACY=%.4f", metrics["val"]["BALANCED_ACCURACY"])

    meta = {
        "classifier_type": args.classifier_type,
        "target_label": args.target_label,
        "train_data_path": args.train_data_path,
        "val_data_path": args.val_data_path,
        "balanced": args.balanced,
        "postprocess": args.postprocess,
        "n_train": int(len(y_train)),
        "model_path": model_path,
    }
    with open(os.path.join(args.output_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "meta": meta}, f, indent=2)
    logger.info("Done training %s for %s.", args.classifier_type, args.target_label)


if __name__ == "__main__":
    main()
