from __future__ import annotations

import logging
import warnings
from typing import Callable, Optional, Union, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common.datasets import TSDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ADF:
    """
    Appliance Detection Framework (ADF)

    A thin wrapper around a binary PyTorch classifier that calibrates
    an aggregation rule over per-window probabilities.
    """

    def __init__(
        self,
        model: nn.Module,
        average_mode: str = "quantile",
        quantile: Optional[float] = None,
        classif_metrics: Callable = lambda *args, **kwargs: {},
        dataset_kwargs: Optional[Dict[str, Union[int, float, str]]] = None,
        device: str = "cuda",
        batch_size_voter: int = 1,
    ) -> None:
        if average_mode not in {"mean", "quantile"}:
            raise ValueError("average_mode must be either 'mean' or 'quantile'.")

        if device not in {"cuda", "cpu"}:
            raise ValueError("device must be 'cuda' or 'cpu'.")

        if dataset_kwargs is None:
            raise ValueError("dataset_kwargs must be provided.")

        self.model = model.to(device)
        self.device = device

        self.classif_metrics = classif_metrics
        self.dataset_kwargs = dataset_kwargs

        self.average_mode = average_mode
        self.quantile = 0.5 if quantile is None else float(quantile)
        self.batch_size_voter = batch_size_voter

        self.is_fitted = average_mode == "mean"
        self.dict_res: Dict[str, Dict[str, float]] = {}

        super().__init__()

    def __repr__(self) -> str:
        return (
            f"ADF(model={self.model.__class__.__name__}, "
            f"avg_mode='{self.average_mode}', "
            f"quantile={self.quantile:.2f}, "
            f"device='{self.device}', "
            f"fitted={self.is_fitted})"
        )

    def train(
        self,
        calibration_dataset: pd.DataFrame,
        monitoring_metric: str = "F1_MACRO",
    ) -> Union[Dict[str, float], Tuple[float, Dict[str, float]]]:
        """
        Calibrate the optimal aggregation quantile on `calibration_dataset`.

        Returns
        -------
        dict
            Empty dict if `average_mode == "mean"`.
        (float, dict)
            Best quantile and associated metrics otherwise.
        """
        if self.average_mode == "mean":
            logger.info(
                "average_mode='mean' selected -> no calibration required. "
                "Skipping training."
            )
            self.is_fitted = True
            return {}

        if not isinstance(calibration_dataset, pd.DataFrame):
            raise ValueError(
                "calibration_dataset must be a pandas.DataFrame where rows "
                "are time-series instances and the last column holds labels."
            )

        best_score = float("-inf")
        best_quantile = self.quantile
        best_metrics: Optional[Dict[str, float]] = None

        for q in np.arange(0.1, 1.0, 0.1):
            y, y_hat, y_hat_prob = self._voter(calibration_dataset, quantile=float(q))
            metrics = self.classif_metrics(y, y_hat, y_hat_prob)

            if monitoring_metric not in metrics:
                raise KeyError(
                    f"monitoring_metric='{monitoring_metric}' not found in metrics. "
                    f"Available keys: {list(metrics.keys())}"
                )

            self.dict_res[f"{q:.2f}"] = metrics
            logger.info("q=%.2f | %s=%.4f", q, monitoring_metric, metrics[monitoring_metric])

            if metrics[monitoring_metric] > best_score:
                best_score = metrics[monitoring_metric]
                best_quantile = float(q)
                best_metrics = metrics

        self.quantile = best_quantile
        self.is_fitted = True

        logger.info(
            "Best q=%.2f -> %s=%.4f",
            best_quantile,
            monitoring_metric,
            best_score,
        )

        return (best_quantile, best_metrics) if best_metrics is not None else {}

    def test(
        self,
        data: pd.DataFrame,
        return_output: bool = False,
    ) -> Union[
        Dict[str, float],
        Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray],
    ]:
        """
        Predict on new data and return evaluation metrics.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                "data must be a pandas.DataFrame where rows correspond to "
                "time-series instances and the last column holds labels."
            )

        if self.average_mode == "quantile" and not self.is_fitted:
            warnings.warn(
                "average_mode='quantile' requested, but the framework is not "
                "fitted. Training with a calibration dataset is strongly "
                "recommended; falling back to q=%.2f."
                % self.quantile
            )

        y, y_hat, y_hat_prob = self._voter(data, quantile=self.quantile)
        metrics = self.classif_metrics(y, y_hat, y_hat_prob)

        return (metrics, y, y_hat, y_hat_prob) if return_output else metrics

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _prepare_batch(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Normalize a batch dict into (ts, exogene, labels).

        Expected keys:
            - 'ts'       : required
            - 'exogene'  : optional
            - 'labels'   : optional
        """
        if not isinstance(batch, dict):
            raise TypeError(f"Expected batch to be a dict, got {type(batch).__name__}.")

        if "ts" not in batch:
            raise KeyError("Batch is missing required key 'ts'.")

        ts = batch["ts"].to(self.device, dtype=torch.float)

        exogene = batch.get("exogene", None)
        if exogene is not None:
            exogene = exogene.to(self.device, dtype=torch.float)

        labels = batch.get("labels", None)
        if labels is not None:
            labels = labels.to(self.device, dtype=torch.long)

        return ts, exogene, labels

    def _forward_batch(
        self,
        ts: torch.Tensor,
        exogene: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward helper for models with optional exogenous inputs.

        Preferred model signature:
            forward(ts, exogene=None)
        """
        if exogene is None:
            return self.model(ts)
        return self.model(ts, exogene)

    def _predict_client_windows(self, tmp_data: pd.DataFrame) -> np.ndarray:
        """
        Run the model on all windows for one client and return P(class=1)
        for each window.
        """
        dl = torch.utils.data.DataLoader(
            TSDataset(df=tmp_data, **self.dataset_kwargs),
            batch_size=self.batch_size_voter,
            shuffle=False,
        )

        probs: List[float] = []

        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            for batch in dl:
                ts, exogene, _ = self._prepare_batch(batch)

                logits = self._forward_batch(ts, exogene)
                batch_probs = torch.softmax(logits, dim=1)[:, 1]

                probs.extend(batch_probs.cpu().numpy().tolist())

        return np.asarray(probs, dtype=np.float32)

    def _voter(
        self,
        data: pd.DataFrame,
        quantile: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aggregate window probabilities with the mean or a given quantile.
        """
        id_clients_col = self.dataset_kwargs["id_clients"]
        id_label_col = self.dataset_kwargs["id_label"]

        y: List[int] = []
        y_hat: List[int] = []
        y_hat_prob: List[float] = []

        list_id_clients = data[id_clients_col].unique()

        for client_id in list_id_clients:
            tmp_data = data.loc[data[id_clients_col] == client_id]
            client_label = int(tmp_data[id_label_col].iloc[0])

            y.append(client_label)

            probs = self._predict_client_windows(tmp_data)

            if probs.size == 0:
                raise ValueError(f"No window probabilities produced for client_id={client_id!r}.")

            if self.average_mode == "mean":
                proba_inst = float(np.mean(probs))
            else:
                proba_inst = float(np.quantile(probs, q=quantile))

            y_hat_prob.append(proba_inst)
            y_hat.append(int(np.rint(proba_inst)))

        return (
            np.asarray(y, dtype=np.int64),
            np.asarray(y_hat, dtype=np.int64),
            np.asarray(y_hat_prob, dtype=np.float32),
        )