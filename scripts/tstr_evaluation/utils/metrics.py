#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classification metrics for the TSTR (Train Synthetic, Test Real) evaluation.

The primary metric is ``BALANCED_ACCURACY`` (mean of per-class recalls), which is
robust to class imbalance on the real test split. All metrics are computed with
``sklearn.metrics`` so they match the rest of the codebase (see
``src/transapp/common/metrics.py``).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    pos_label: int = 1,
) -> Dict[str, object]:
    """Compute the standard binary-classification metric bundle.

    Args:
        y_true: Ground-truth binary labels, shape ``[N]``.
        y_pred: Predicted binary labels, shape ``[N]``.
        pos_label: Positive class for PRECISION / RECALL / F1 (the appliance is
            present == 1 by convention).

    Returns:
        A JSON-serialisable dict with ``BALANCED_ACCURACY`` (primary), ``ACCURACY``,
        ``PRECISION``, ``RECALL``, ``F1``, ``CONFUSION_MATRIX`` and ``SUPPORT``.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true {y_true.shape} and y_pred {y_pred.shape} must have the same shape."
        )
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    metrics: Dict[str, object] = {
        "BALANCED_ACCURACY": float(balanced_accuracy_score(y_true, y_pred)),
        "ACCURACY": float(accuracy_score(y_true, y_pred)),
        "PRECISION": float(
            precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        ),
        "RECALL": float(
            recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        ),
        "F1": float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "CONFUSION_MATRIX": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "SUPPORT": {
            "n_total": int(y_true.size),
            "n_positive": int((y_true == 1).sum()),
            "n_negative": int((y_true == 0).sum()),
        },
    }
    return metrics


def metrics_from_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: Optional[np.ndarray] = None,
    *,
    pos_label: int = 1,
) -> Dict[str, object]:
    """Like :func:`compute_classification_metrics` but optionally adds ROC-AUC.

    ``scores`` are continuous decision-function / probability values; when ``None``
    or degenerate (single class in ``y_true``) ROC-AUC is silently skipped.
    """
    metrics = compute_classification_metrics(y_true, y_pred, pos_label=pos_label)
    if scores is not None:
        try:
            from sklearn.metrics import roc_auc_score

            metrics["ROC_AUC"] = float(roc_auc_score(y_true, scores))
        except ValueError:
            metrics["ROC_AUC"] = None
    return metrics
