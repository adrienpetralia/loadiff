"""metrics.py
~~~~~~~~~~~~~~~~~~~~~~~~

Modules in this file
--------------------
* :class:`ImbalancedClassificationMetrics` — quick-fire metric bundle that
  automatically focuses on the *minority* class.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from numpy.typing import ArrayLike

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

__all__ = [
    "ImbalancedClassificationMetrics",
]

# ---------------------------------------------------------------------------
# Logging config (overwrite only once)
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

class ImbalancedClassificationMetrics:
    """Compute a standard set of metrics for *binary* **or** *multi-class* tasks.

    For binary problems the class with the *fewest* samples in ``y`` is
    considered the *minority* (positive) class unless *minority_class* is
    supplied explicitly.

    The callable returns a dict containing at least the following keys:

    * ``ACCURACY``
    * ``PRECISION``, ``RECALL``, ``F1`` (minority-focused for binary)
    * ``*_MACRO`` and ``*_WEIGHTED`` variants
    * ``CONFUSION_MATRIX``

    If *y_hat_prob* is provided **and** shape is compatible, ROC-AUC statistics are
    included as:

    * ``ROC_AUC`` (binary)  **or** ``ROC_AUC_OVO`` / ``ROC_AUC_OVR`` (multi-class)
    * ``ROC_AUC_MACRO``
    * ``ROC_AUC_WEIGHTED``

    Parameters
    ----------
    minority_class:
        Index of the minority class for *binary* metrics. Ignored for multi-class.
    """

    def __init__(self, minority_class: Optional[int] = None) -> None:  # noqa: D401 – simple docstring
        self.minority_class = minority_class
        self.minority_class_: Optional[int] = None  # set during __call__

    # ------------------------------------------------------------------
    # Call interface
    # ------------------------------------------------------------------
    def __call__(
        self,
        y: ArrayLike,
        y_hat: ArrayLike,
        y_hat_prob: Optional[ArrayLike] = None,
    ) -> Dict[str, Any]:
        y_arr = np.asarray(y)
        y_hat_arr = np.asarray(y_hat)

        # ----------------------- minority (binary only) ------------------ #
        unique, counts = np.unique(y_arr, return_counts=True)
        n_classes = unique.size
        if n_classes == 2:  # binary semantics apply
            if self.minority_class is not None:
                minority = self.minority_class
            else:
                minority = unique[np.argmin(counts)]
            self.minority_class_ = int(minority)
        else:
            minority = None  # not used

        # ------------------------------- core ---------------------------- #
        metrics: Dict[str, Any] = {
            "ACCURACY": accuracy_score(y_arr, y_hat_arr),
            "BALANCED_ACCURACY": balanced_accuracy_score(y_arr, y_hat_arr),
            "PRECISION": precision_score(
                y_arr, y_hat_arr, average="binary" if n_classes == 2 else "macro", pos_label=minority
            ),
            "RECALL": recall_score(
                y_arr, y_hat_arr, average="binary" if n_classes == 2 else "macro", pos_label=minority
            ),
            "F1": f1_score(
                y_arr, y_hat_arr, average="binary" if n_classes == 2 else "macro", pos_label=minority
            ),
            "PRECISION_MACRO": precision_score(y_arr, y_hat_arr, average="macro"),
            "PRECISION_WEIGHTED": precision_score(y_arr, y_hat_arr, average="weighted"),
            "RECALL_MACRO": recall_score(y_arr, y_hat_arr, average="macro"),
            "RECALL_WEIGHTED": recall_score(y_arr, y_hat_arr, average="weighted"),
            "F1_MACRO": f1_score(y_arr, y_hat_arr, average="macro"),
            "F1_WEIGHTED": f1_score(y_arr, y_hat_arr, average="weighted"),
            "CONFUSION_MATRIX": confusion_matrix(y_arr, y_hat_arr),
        }

        # ------------------------------ ROC‑AUC -------------------------- #
        if y_hat_prob is not None:
            try:
                metrics.update(
                    {
                        "ROC_AUC": roc_auc_score(
                            y_arr, y_hat_prob, average="macro" if n_classes > 2 else "weighted", multi_class="ovo"
                        ),
                        "ROC_AUC_MACRO": roc_auc_score(
                            y_arr, y_hat_prob, average="macro", multi_class="ovo" if n_classes > 2 else "raise"
                        ),
                        "ROC_AUC_WEIGHTED": roc_auc_score(
                            y_arr, y_hat_prob, average="weighted", multi_class="ovo" if n_classes > 2 else "raise"
                        ),
                    }
                )
            except ValueError:
                # Incompatible shapes (e.g. proba missing one column) – silently skip.
                logging.debug("roc_auc_score skipped due to incompatible shapes.")

        return metrics