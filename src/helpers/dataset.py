import os
import warnings
from typing import Union, Tuple, List, Optional, Sequence, Set

import torch
import numpy as np
import pandas as pd
import polars as pl


# ----------------------------
# Base (mother) class
# ----------------------------
class BaseParquetDailyDataset(torch.utils.data.Dataset):
    """
    Base class for datasets stored as:
      - parquet with one time column + many ID columns (each ID = a client/house/PDL)
      - half-hourly / hourly points with fixed patch_length per day
    Provides:
      - parquet loading + column filtering
      - patchify into (days, patch_length)
      - random/fixed day window
      - scaling
      - exogenous calendar features
      - optional temperature feature (global or per-id)
      - optional metadata labels (per-id)
    """

    def __init__(
        self,
        path_load_curves: str,
        nb_days: int,
        patch_length_day: int,
        start_date: str,
        end_date: Optional[str],
        col_time_mask: str,
        list_ids: Optional[Union[list, pl.Series]] = None,
        scale_param1: float = 0.0,
        scale_param2: float = 1.0,
        random_window: bool = False,
        scale_meteo: float = 1.0,
    ):
        if not os.path.exists(path_load_curves):
            raise FileNotFoundError(f"File {path_load_curves} does not exist.")

        self.patch_length = int(patch_length_day)
        self.nb_days = int(nb_days)
        self.random_window = bool(random_window)

        self.scale_param1 = float(scale_param1)
        self.scale_param2 = float(scale_param2)
        if self.scale_param2 == self.scale_param1:
            raise ValueError("scale_param2 must be different from scale_param1.")

        self.scale_meteo = float(scale_meteo)
        if self.scale_meteo == 0:
            raise ValueError("scale_meteo must be non-zero.")

        # --- Load parquet curves ---
        df = pl.read_parquet(path_load_curves)

        if col_time_mask not in df.columns:
            raise ValueError(f"Column '{col_time_mask}' not found in parquet: {path_load_curves}")

        self.dates = df[col_time_mask]
        df = df.drop(col_time_mask)

        if list_ids is not None:
            df = self._select_id_columns(df, list_ids)

        # Curves tensor: (num_ids, num_timepoints)
        data_np = df.to_numpy().T
        self.data = torch.tensor(data_np, dtype=torch.float32)
        self.id_clients = [str(c) for c in df.columns]

        self.num_ids = self.data.shape[0]
        self.num_timepoints = self.data.shape[1]

        self.available_days = self.num_timepoints // self.patch_length
        if self.available_days <= 0:
            raise ValueError("No full days available; check patch_length_day vs. timepoints.")
        if self.nb_days > self.available_days:
            raise ValueError(
                f"Requested nb_days={self.nb_days}, but only {self.available_days} full days available."
            )

        # --- Build day index (extra_dates) robustly ---
        # Your original code used start/end, but that can silently mismatch parquet length.
        # Here: we try start/end; if mismatch, we fall back to 'periods=available_days'.
        start_dt = pd.to_datetime(start_date, format="%d/%m/%Y", errors="raise")
        self.user_start_date = start_date
        self.user_end_date = end_date

        if end_date is not None:
            end_dt = pd.to_datetime(end_date, format="%d/%m/%Y", errors="raise")
            candidate = pd.date_range(start=start_dt, end=end_dt, freq="D")
            if len(candidate) != self.available_days:
                warnings.warn(
                    f"Date range ({start_date} -> {end_date}) gives {len(candidate)} days "
                    f"but parquet provides {self.available_days} full days. "
                    f"Falling back to periods={self.available_days} from start_date."
                )
                self.extra_dates = pd.date_range(start=start_dt, periods=self.available_days, freq="D")
            else:
                self.extra_dates = candidate
        else:
            self.extra_dates = pd.date_range(start=start_dt, periods=self.available_days, freq="D")

        self.exogene_full = self._create_exogene(self.extra_dates)  # (num_days_total, 4)

        # --- Optional metadata (per-id) ---
        self.meta_cols: List[str] = []
        self.data_pop = torch.zeros((self.num_ids, 0), dtype=torch.float32)
        self._load_metadata()  # child can override

        # --- Optional temperature ---
        # temps_full stored as:
        #   - (num_ids, num_days_total) for per-id temperatures
        #   - (1, num_days_total) for global temperature shared by all ids
        self.temps_full: Optional[torch.Tensor] = None
        self._temp_is_per_id: bool = False
        self._load_temperature()  # child can override

    # ----------------------------
    # Helpers
    # ----------------------------
    @staticmethod
    def _normalize_id_list(list_ids: Union[list, pl.Series]) -> List[str]:
        if isinstance(list_ids, pl.Series):
            list_ids = list_ids.to_list()
        list_ids = [str(c) for c in list_ids]

        seen: Set[str] = set()
        unique = [c for c in list_ids if not (c in seen or seen.add(c))]
        return unique

    def _select_id_columns(self, df: pl.DataFrame, list_ids: Union[list, pl.Series]) -> pl.DataFrame:
        cols = self._normalize_id_list(list_ids)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{len(missing)} requested columns not in df: {missing}")
        return df.select(pl.col(cols))

    @staticmethod
    def _align_meta_to_ids(df_meta: pl.DataFrame, col_id: str, ids: Sequence[str]) -> pl.DataFrame:
        """
        Filter metadata to ids and reorder to match ids exactly (right join on ids).
        """
        if col_id not in df_meta.columns:
            raise ValueError(f"Column '{col_id}' not found in metadata.")

        df_meta = df_meta.with_columns(pl.col(col_id).cast(pl.Utf8))
        ids = [str(x) for x in ids]

        df_meta = df_meta.filter(pl.col(col_id).is_in(ids))
        df_meta = df_meta.join(pl.DataFrame({col_id: ids}), on=col_id, how="right")
        return df_meta

    def _scale_data(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.scale_param1) / (self.scale_param2 - self.scale_param1)

    @staticmethod
    def _create_exogene(extra_dates: pd.DatetimeIndex) -> torch.Tensor:
        """
        Calendar exogenous features:
        weekday, day, day_of_year, month, mapped to angles (your original mapping kept as-is).
        """
        exogene_array = np.vstack([
            extra_dates.weekday.values      * (2 * np.pi / 6),
            extra_dates.day.values          * (2 * np.pi / 31),
            extra_dates.dayofyear.values    * (2 * np.pi / 365),
            extra_dates.month.values        * (2 * np.pi / 12),
        ])
        return torch.tensor(exogene_array, dtype=torch.float32).permute(1, 0)  # (D, 4)

    def _get_temp_series(self, idx: int) -> Optional[torch.Tensor]:
        """
        Returns 1D tensor (num_days_total,) for a given client index.
        Handles per-id temps (num_ids, D) or global temps (1, D).
        """
        if self.temps_full is None:
            return None

        if self._temp_is_per_id:
            return self.temps_full[idx]
        else:
            # global shared temperature
            return self.temps_full[0]

    # ----------------------------
    # Hooks for child classes
    # ----------------------------
    def _load_metadata(self) -> None:
        """Override in child classes if needed."""
        return

    def _load_temperature(self) -> None:
        """Override in child classes if needed."""
        return

    # ----------------------------
    # Dataset protocol
    # ----------------------------
    def __len__(self) -> int:
        return len(self.id_clients)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Always returns:
          values_win:  (nb_days, patch_length)
          exogene_win: (nb_days, 4 [+1 if temp])
          y:           (F,) possibly empty
        """
        # (T,)
        series = self.data[idx]

        # (available_days, patch_length)
        values = series.unfold(dimension=0, size=self.patch_length, step=self.patch_length)
        values = self._scale_data(values)

        if self.random_window:
            max_start = self.available_days - self.nb_days
            start_day = int(torch.randint(0, max_start + 1, (1,)).item())
        else:
            start_day = 0

        end_day = start_day + self.nb_days

        values_win = values[start_day:end_day]
        exogene_win = self.exogene_full[start_day:end_day]

        # optional temperature concat
        temp_series = self._get_temp_series(idx)
        if temp_series is not None:
            temp_win = temp_series[start_day:end_day].unsqueeze(-1)  # (nb_days, 1)
            exogene_win = torch.cat([exogene_win, temp_win], dim=1)

        # y always returned (possibly empty)
        if self.data_pop.numel() == 0 or self.data_pop.shape[1] == 0:
            y = torch.zeros((0,), dtype=torch.float32)
        else:
            y = self.data_pop[idx]

        return values_win, exogene_win, y


# ----------------------------
# Child 1: CdC
# ----------------------------
class SmachDataset(BaseParquetDailyDataset):
    def __init__(
        self,
        path_load_curves: str,
        nb_days: int = 365,
        patch_length_day: int = 48,
        start_date: str = "01/01/2021",
        end_date: str = "31/12/2022",
        list_pdl: Optional[Union[list, pl.Series]] = None,
        col_time_mask: str = "HORODATAGE",
        scale_param1: float = 0.0,
        scale_param2: float = 1.0,
        scale_meteo: float = 1.0,
        random_window: bool = False,
        path_metadata: Optional[str] = None,
        col_id: str = "ID_PDL",
        bool_col_names: Optional[List[str]] = None,
        path_temperature: Optional[str] = None,
    ):
        self.path_metadata = path_metadata
        self.col_id = col_id
        self.bool_col_names = bool_col_names or []
        self.path_temperature = path_temperature

        # meteo city per client
        self.meteo_city: List[Optional[str]] = []

        if self.path_metadata is not None and not os.path.exists(self.path_metadata):
            raise FileNotFoundError(f"Metadata file {self.path_metadata} does not exist.")
        if self.path_temperature is not None and not os.path.exists(self.path_temperature):
            raise FileNotFoundError(f"Meteo file {self.path_temperature} does not exist.")

        super().__init__(
            path_load_curves=path_load_curves,
            nb_days=nb_days,
            patch_length_day=patch_length_day,
            start_date=start_date,
            end_date=end_date,
            col_time_mask=col_time_mask,
            list_ids=list_pdl,
            scale_param1=scale_param1,
            scale_param2=scale_param2,
            random_window=random_window,
            scale_meteo=scale_meteo,
        )

    def _load_metadata(self) -> None:
        if self.path_metadata is None:
            self.meteo_city = [None] * self.num_ids
            return

        df_meta = pl.read_parquet(self.path_metadata)

        # ECS labels derivation
        if "ECS_ELEC" in df_meta.columns:
            df_meta = df_meta.with_columns(
                pl.col("ECS_ELEC")
                .cast(pl.Utf8)
                .str.strip_chars()
                .str.to_lowercase()
                .is_in(["joule", "pac"])
                .fill_null(False)
                .alias("ECS")
            )

        df_meta = self._align_meta_to_ids(df_meta, self.col_id, self.id_clients)

        # VILLE_METEO
        if "VILLE_METEO" in df_meta.columns:
            self.meteo_city = df_meta.select(pl.col("VILLE_METEO").cast(pl.Utf8))["VILLE_METEO"].to_list()
        else:
            self.meteo_city = [None] * self.num_ids

        # boolean labels
        if len(self.bool_col_names) == 0:
            self.data_pop = torch.zeros((self.num_ids, 0), dtype=torch.float32)
            self.meta_cols = []
            return

        missing = [c for c in self.bool_col_names if c not in df_meta.columns]
        if missing:
            raise ValueError(f"Missing metadata columns: {missing}")

        meta_np = (
            df_meta.select([pl.col(c).cast(pl.Int8).alias(c) for c in self.bool_col_names])
            .to_numpy()
        )
        self.data_pop = torch.tensor(meta_np, dtype=torch.float32)
        self.meta_cols = list(self.bool_col_names)

    def _load_temperature(self) -> None:
        if self.path_temperature is None:
            return

        if len(self.meteo_city) == 0:
            # metadata not provided or missing VILLE_METEO
            raise ValueError("path_temperature provided but meteo city mapping is unavailable (VILLE_METEO missing).")

        if all(city is None for city in self.meteo_city):
            raise ValueError("path_temperature provided but all VILLE_METEO are None.")

        meteo_df = pd.read_parquet(self.path_temperature)
        meteo_df = meteo_df.reindex(self.extra_dates)

        if meteo_df.isna().any().any():
            raise ValueError("NaNs found in meteo data after reindexing to dataset date range.")

        used_cities = {c for c in self.meteo_city if c is not None}
        missing_cities = used_cities - set(meteo_df.columns)
        if missing_cities:
            raise ValueError(f"Meteo cities not found in meteo CSV columns: {missing_cities}")

        temp_mat = []
        for city in self.meteo_city:
            if city is None:
                raise ValueError("A client has no VILLE_METEO mapping; cannot build per-id temperature.")
            temp_mat.append(meteo_df[city].to_numpy(dtype=np.float32))

        temp_mat = np.stack(temp_mat, axis=0)  # (num_ids, D)
        self.temps_full = torch.tensor(temp_mat, dtype=torch.float32) / self.scale_meteo
        self._temp_is_per_id = True


# ----------------------------
# Child 2: CER
# ----------------------------
class CERDataset(BaseParquetDailyDataset):
    def __init__(
        self,
        path_load_curves: str,
        nb_days: int = 365,
        patch_length_day: int = 48,
        start_date: str = "15/07/2009",
        end_date: str = "01/01/2011",
        list_pdl: Optional[Union[list, pl.Series]] = None,
        col_time_mask: str = "time",
        scale_param1: float = 0.0,
        scale_param2: float = 1.0,
        scale_meteo: float = 1.0,
        random_window: bool = False,
        path_metadata: Optional[str] = None,
        col_id: str = "id_house",
        bool_col_names: Optional[List[str]] = None,
        missing_label_policy: str = "fill",  # "fill" or "drop"
        missing_label_value: int = -1,
        path_temperature: Optional[str] = None,
    ):
        self.path_metadata = path_metadata
        self.col_id = col_id
        self.bool_col_names = bool_col_names or []
        self.missing_label_policy = missing_label_policy
        self.missing_label_value = missing_label_value
        self.path_temperature = path_temperature

        if self.path_metadata is not None and not os.path.exists(self.path_metadata):
            raise FileNotFoundError(f"Metadata file {self.path_metadata} does not exist.")
        if self.missing_label_policy not in {"fill", "drop"}:
            raise ValueError("missing_label_policy must be 'fill' or 'drop'")
        if self.path_temperature is not None and not os.path.exists(self.path_temperature):
            raise FileNotFoundError(f"Meteo file {self.path_temperature} does not exist.")

        super().__init__(
            path_load_curves=path_load_curves,
            nb_days=nb_days,
            patch_length_day=patch_length_day,
            start_date=start_date,
            end_date=end_date,
            col_time_mask=col_time_mask,
            list_ids=list_pdl,
            scale_param1=scale_param1,
            scale_param2=scale_param2,
            random_window=random_window,
            scale_meteo=scale_meteo,
        )

    def _load_metadata(self) -> None:
        if self.path_metadata is None or len(self.bool_col_names) == 0:
            return

        df_meta = pl.read_parquet(self.path_metadata)

        if self.col_id not in df_meta.columns:
            raise ValueError(f"Column '{self.col_id}' not found in metadata file.")

        df_meta = df_meta.with_columns(pl.col(self.col_id).cast(pl.Utf8)).unique(subset=[self.col_id], keep="first")

        missing = [c for c in self.bool_col_names if c not in df_meta.columns]
        if missing:
            raise ValueError(f"Missing metadata columns: {missing}")

        # keep only ids that exist in parquet
        df_meta = df_meta.filter(pl.col(self.col_id).is_in(self.id_clients))

        # cast labels (non strict -> null if parsing fails)
        df_meta = df_meta.with_columns([
            pl.col(c).cast(pl.Int8, strict=False).alias(c) for c in self.bool_col_names
        ])

        if self.missing_label_policy == "fill":
            df_meta = self._align_meta_to_ids(df_meta, self.col_id, self.id_clients)

            df_meta = df_meta.with_columns([
                pl.col(c)
                .fill_null(self.missing_label_value)
                .fill_nan(self.missing_label_value)
                .alias(c)
                for c in self.bool_col_names
            ])

            meta_np = df_meta.select(self.bool_col_names).to_numpy()
            self.data_pop = torch.tensor(meta_np, dtype=torch.float32)
            self.meta_cols = list(self.bool_col_names)

        else:  # drop
            df_meta_valid = (
                df_meta
                .with_columns([pl.col(c).fill_nan(None).alias(c) for c in self.bool_col_names])
                .drop_nulls(subset=self.bool_col_names)
            )

            valid_ids = df_meta_valid.select(self.col_id).to_series().to_list()
            valid_set = set(valid_ids)

            keep_idx = [i for i, cid in enumerate(self.id_clients) if cid in valid_set]
            if len(keep_idx) == 0:
                raise ValueError("After dropping missing-label rows, no IDs remain.")

            # filter curves
            self.data = self.data[keep_idx]
            self.id_clients = [self.id_clients[i] for i in keep_idx]
            self.num_ids = self.data.shape[0]

            # reorder metadata to filtered ids
            df_meta_valid = df_meta_valid.join(
                pl.DataFrame({self.col_id: self.id_clients}),
                on=self.col_id,
                how="inner",
            )

            meta_np = df_meta_valid.select(self.bool_col_names).to_numpy()
            self.data_pop = torch.tensor(meta_np, dtype=torch.float32)
            self.meta_cols = list(self.bool_col_names)

    def _load_temperature(self) -> None:
        if self.path_temperature is None:
            return

        meteo_df = pd.read_parquet(self.path_temperature)
        meteo_df = meteo_df.sort_values("date").set_index("date")

        # daily mean temperature
        daily_mean = meteo_df["temperature"].resample("D").mean()

        # align to dataset day index
        daily_mean = daily_mean.reindex(self.extra_dates)
        if daily_mean.isna().any():
            raise ValueError("NaNs found in temperature after aligning to dataset date range.")

        temp = daily_mean.to_numpy(dtype=np.float32) / self.scale_meteo  # (D,)
        self.temps_full = torch.tensor(temp, dtype=torch.float32).unsqueeze(0)  # (1, D)
        self._temp_is_per_id = False


# ----------------------------
# Child 3: CERBis (very close to CER)
# ----------------------------
class CERBisDataset(CERDataset):
    """
    Same behavior as CERDataset but defaults differ and metadata is parquet instead of CSV.
    Temperature is not supported here (keeps your original behavior).
    """

    def __init__(
        self,
        path_load_curves: str,
        nb_days: int = 365,
        patch_length_day: int = 48,
        start_date: str = "01/01/2025",
        end_date: str = "31/12/2025",
        list_pdl: Optional[Union[list, pl.Series]] = None,
        col_time_mask: str = "time",
        scale_param1: float = 0.0,
        scale_param2: float = 1.0,
        scale_meteo: float = 1.0,
        random_window: bool = False,
        path_metadata: Optional[str] = None,   # parquet here
        col_id: str = "ID_PDL",
        bool_col_names: Optional[List[str]] = None,
        missing_label_policy: str = "fill",
        missing_label_value: int = -1,
        path_temperature: Optional[str] = None
    ):

        super().__init__(
            path_load_curves=path_load_curves,
            nb_days=nb_days,
            patch_length_day=patch_length_day,
            start_date=start_date,
            end_date=end_date,
            list_pdl=list_pdl,
            col_time_mask=col_time_mask,
            scale_param1=scale_param1,
            scale_param2=scale_param2,
            scale_meteo=scale_meteo,
            random_window=random_window,
            path_metadata=path_metadata,
            col_id=col_id,
            bool_col_names=bool_col_names,
            missing_label_policy=missing_label_policy,
            missing_label_value=missing_label_value,
            path_temperature=path_temperature,
        )

    def _load_metadata(self) -> None:
        if self.path_metadata is None or len(self.bool_col_names) == 0:
            return

        df_meta = pl.read_parquet(self.path_metadata)

        if self.col_id not in df_meta.columns:
            raise ValueError(f"Column '{self.col_id}' not found in metadata file.")

        df_meta = df_meta.with_columns(pl.col(self.col_id).cast(pl.Utf8)).unique(subset=[self.col_id], keep="first")

        missing = [c for c in self.bool_col_names if c not in df_meta.columns]
        if missing:
            raise ValueError(f"Missing metadata columns: {missing}")

        df_meta = df_meta.filter(pl.col(self.col_id).is_in(self.id_clients))
        df_meta = df_meta.with_columns([
            pl.col(c).cast(pl.Int8, strict=False).alias(c) for c in self.bool_col_names
        ])

        if self.missing_label_policy == "fill":
            df_meta = self._align_meta_to_ids(df_meta, self.col_id, self.id_clients)
            df_meta = df_meta.with_columns([
                pl.col(c)
                .fill_null(self.missing_label_value)
                .fill_nan(self.missing_label_value)
                .alias(c)
                for c in self.bool_col_names
            ])

            meta_np = df_meta.select(self.bool_col_names).to_numpy()
            self.data_pop = torch.tensor(meta_np, dtype=torch.float32)
            self.meta_cols = list(self.bool_col_names)

        else:
            df_meta_valid = (
                df_meta
                .with_columns([pl.col(c).fill_nan(None).alias(c) for c in self.bool_col_names])
                .drop_nulls(subset=self.bool_col_names)
            )
            valid_ids = df_meta_valid.select(self.col_id).to_series().to_list()
            valid_set = set(valid_ids)

            keep_idx = [i for i, cid in enumerate(self.id_clients) if cid in valid_set]
            if len(keep_idx) == 0:
                raise ValueError("After dropping missing-label rows, no IDs remain.")

            self.data = self.data[keep_idx]
            self.id_clients = [self.id_clients[i] for i in keep_idx]
            self.num_ids = self.data.shape[0]

            df_meta_valid = df_meta_valid.join(
                pl.DataFrame({self.col_id: self.id_clients}),
                on=self.col_id,
                how="inner",
            )

            meta_np = df_meta_valid.select(self.bool_col_names).to_numpy()
            self.data_pop = torch.tensor(meta_np, dtype=torch.float32)
            self.meta_cols = list(self.bool_col_names)