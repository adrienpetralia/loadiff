import torch

import numpy as np
import pandas as pd

from typing import Dict, List, Any, Optional


class TSDataset(torch.utils.data.Dataset):
    VALID_EXOGENE_VARS = {
        "second",
        "minute",
        "hour",
        "dayofweek",
        "dayofmonth",
        "dayofyear",
        "month"
    }

    VALID_SCALING_METHODS = {"minmax", "znorm", "none"}

    def __init__(
        self,
        df: pd.DataFrame,
        id_clients: str = "id_pdl",
        id_start_subseq_dates: str = "start_date",
        id_label: Optional[str] = None,
        freq: str = "30min",
        exogene_var: Optional[List[str]] = None,
        scaling_method: str = "minmax",
        scale_param1: float = 0.0,
        scale_param2: float = 1000.0
    ):
        if exogene_var is not None:
            self._validate_exogene_vars(exogene_var)
            self.exogene_var = exogene_var
        else:
            self.exogene_var = []

        self._validate_scaling_method(scaling_method)
        self.scaling_method = scaling_method
        self.scale_param1 = scale_param1
        self.scale_param2 = scale_param2

        self.id_clients = df[id_clients].values
        self.start_subseq_dates = df[id_start_subseq_dates].values

        df = df.drop(columns=[id_clients, id_start_subseq_dates])

        if id_label is not None:
            self.labels = torch.tensor(df[id_label].values.ravel(), dtype=torch.float32)
            df = df.drop(columns=[id_label])
        else:
            self.labels = None

        self.data = torch.tensor(df.to_numpy(), dtype=torch.float32).unsqueeze(1)

        self.subseq_length = self.data.shape[-1]
        self.freq = freq

    def __len__(self) -> int:
        return len(self.id_clients)

    def _validate_exogene_vars(self, exo_vars: List[str]) -> None:
        invalid_vars = [var for var in exo_vars if var not in self.VALID_EXOGENE_VARS]
        if invalid_vars:
            raise ValueError(
                f"Invalid exogenous variables: {invalid_vars}. "
                f"Valid options are: {sorted(self.VALID_EXOGENE_VARS)}."
            )

    def _validate_scaling_method(self, scaling_method: str) -> None:
        if scaling_method not in self.VALID_SCALING_METHODS:
            raise ValueError(
                f"Invalid scaling_method: {scaling_method}. "
                f"Valid options are: {sorted(self.VALID_SCALING_METHODS)}."
            )

    def _scale_data(self, values: torch.Tensor) -> torch.Tensor:
        if self.scaling_method == "none":
            return values

        if self.scaling_method == "minmax":
            denom = self.scale_param2 - self.scale_param1
            if denom == 0:
                raise ValueError("For 'minmax' scaling, scale_param2 - scale_param1 must be non-zero.")
            return (values - self.scale_param1) / denom

        if self.scaling_method == "znorm":
            mean = values.mean(dim=-1, keepdim=True)
            std = values.std(dim=-1, keepdim=True, unbiased=False)
            eps = 1e-8
            return (values - mean) / (std + eps)

        raise RuntimeError(f"Unsupported scaling_method: {self.scaling_method}")

    def _create_exogene(self, idx: int) -> torch.Tensor:
        exogene_array = np.zeros((len(self.exogene_var), self.subseq_length), dtype=np.float32)
        timestamp = pd.date_range(
            start=self.start_subseq_dates[idx],
            periods=self.subseq_length,
            freq=self.freq
        )

        for k, var in enumerate(self.exogene_var):
            if var == "second":
                exogene_array[k, :] = 2 * np.pi * timestamp.second.values / 60.0
            elif var == "minute":
                exogene_array[k, :] = 2 * np.pi * timestamp.minute.values / 60.0
            elif var == "hour":
                exogene_array[k, :] = 2 * np.pi * timestamp.hour.values / 24.0
            elif var == "dayofweek":
                exogene_array[k, :] = 2 * np.pi * timestamp.dayofweek.values / 7.0
            elif var == "dayofmonth":
                exogene_array[k, :] = 2 * np.pi * timestamp.day.values / 31.0
            elif var == "dayofyear":
                exogene_array[k, :] = 2 * np.pi * timestamp.dayofyear.values / 365.0
            elif var == "month":
                exogene_array[k, :] = 2 * np.pi * timestamp.month.values / 12.0

        return torch.tensor(exogene_array, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        values = self._scale_data(self.data[idx])

        sample: Dict[str, Any] = {
            "ts": values
        }

        if self.exogene_var:
            sample["exogene"] = self._create_exogene(idx)

        if self.labels is not None:
            sample["labels"] = self.labels[idx]

        return sample

