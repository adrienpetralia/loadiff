import os
import torch
import numpy as np
import pandas as pd
import polars as pl

from typing import Union, Tuple, List, Optional

class CdCDataset(torch.utils.data.Dataset): 
    """
    PyTorch dataset for smach data.
    """

    def __init__(
        self, 
        path_parquet_part: str,
        nb_days: int = 365,
        patch_length_day: int = 48,
        start_date: str = '01/01/2021',
        end_date: str = '12/31/2022',
        list_pdl: Union[list, None] = None,
        col_time_mask: str = "HORODATAGE",
        scale_param1: int = 0,
        scale_param2: int = 1,
        scale_meteo: int = 1,
        random_window: bool = False,
        path_parquet_part_metadata: Optional[str] = None,
        col_id: str = "ID_PDL",
        bool_col_names: Optional[List[str]] = None,   # e.g. ["CLIM", "CHAUFF_ELEC", "ECS_ASSERVI"]
        path_meteo_daily: Optional[str] = None,       # CSV with daily meteo, one col per city
    ):
        assert os.path.exists(path_parquet_part), f'File {path_parquet_part} does not exist.'
        
        if path_parquet_part_metadata is not None:
            assert os.path.exists(path_parquet_part_metadata), (
                f'Metadata file {path_parquet_part_metadata} does not exist.'
            )
        if path_meteo_daily is not None:
            assert os.path.exists(path_meteo_daily), (
                f'Meteo file {path_meteo_daily} does not exist.'
            )

        self.patch_length = patch_length_day

        self.start_date = start_date
        self.end_date   = end_date
        self.nb_days    = nb_days
        self.scale_param1 = scale_param1
        self.scale_param2 = scale_param2
        self.scale_meteo   = scale_meteo
        self.random_window = random_window
        self.bool_col_names = bool_col_names

        # --- Load data from parquet once --- #
        df = pl.read_parquet(path_parquet_part)

        # Keep track of the 'date' index
        self.dates = df[col_time_mask]
        
        # Remove the date column, now only have columns for each id
        df = df.drop(col_time_mask)

        if list_pdl is not None:
            if isinstance(list_pdl, pl.Series):
                list_pdl = list_pdl.to_list()
            else:
                list_pdl = [str(c) for c in list_pdl]
                
            seen = set()
            list_pdl = [c for c in list_pdl if not (c in seen or seen.add(c))]

            missing = [c for c in list_pdl if c not in df.columns]
            if missing:
                raise ValueError(f"{len(missing)} requested columns not in df: {missing}.")

            cols = [c for c in list_pdl if c in df.columns]

            df = df.select(pl.col(cols))
        
        # Convert everything to a (rows x columns) torch.Tensor 
        #   shape = (#rows, #ids_in_this_file)
        data_np = df.to_numpy().T
        self.data = torch.tensor(data_np, dtype=torch.float32)
        
        # Save the name of each ID (the columns in the original DF)
        self.id_clients = [str(c) for c in df.columns]

        # Geometry
        self.num_ids        = self.data.shape[0]
        self.num_timepoints = self.data.shape[1]

        self.available_days = self.num_timepoints // self.patch_length
        if self.available_days <= 0:
            raise ValueError("No full days available; check patch_length vs. timepoints.")
        if self.nb_days > self.available_days:
            raise ValueError(
                f"Requested nb_days={self.nb_days}, but only {self.available_days} full days available."
            )

        # --- Metadata / population data --- #
        self.meta_cols = []
        self.data_pop = torch.zeros((self.num_ids, 0), dtype=torch.float32)

        # Will store, per client, the meteo city name from VILLE_METEO (aligned with id_clients)
        self.meteo_city: List[Optional[str]] = [None] * self.num_ids

        if path_parquet_part_metadata is not None:
            df_meta = pl.read_parquet(path_parquet_part_metadata)
            df_meta = df_meta.with_columns(
                pl.col("ECS_ELEC").cast(pl.Utf8).str.strip_chars().str.to_lowercase()
                .is_in(["joule", "pac"]).fill_null(False).alias("ECS")
            )
            
            if col_id not in df_meta.columns:
                raise ValueError(f"Column '{col_id}' not found in metadata parquet.")

            # Ensure both sides are comparable (Utf8)
            df_meta = df_meta.with_columns(pl.col(col_id).cast(pl.Utf8))
            self.id_clients = [str(c) for c in self.id_clients]

            # filter to our clients then reorder to match self.id_clients
            df_meta = df_meta.filter(pl.col(col_id).is_in(self.id_clients))
            df_meta = df_meta.join(
                pl.DataFrame({col_id: self.id_clients}),
                on=col_id,
                how="right",
            )

            # --- VILLE_METEO mapping (per client) --- #
            if "VILLE_METEO" in df_meta.columns:
                df_meteo_city = df_meta.select(
                    [pl.col(col_id), pl.col("VILLE_METEO").cast(pl.Utf8)]
                )
                # order already aligned with self.id_clients thanks to the join above
                self.meteo_city = df_meteo_city["VILLE_METEO"].to_list()
            else:
                # If you *require* VILLE_METEO, you can raise instead:
                # raise ValueError("Column 'VILLE_METEO' not found in metadata parquet.")
                self.meteo_city = [None] * self.num_ids

            # --- Boolean population features (as before) --- #
            if bool_col_names is None or len(bool_col_names) == 0:
                self.data_pop = torch.zeros((self.num_ids, 0), dtype=torch.float32)
                self.meta_cols = []
            else:
                missing_meta = [c for c in bool_col_names if c not in df_meta.columns]
                if missing_meta:
                    raise ValueError(f"Missing metadata columns: {missing_meta}")

                df_meta_sel = df_meta.select(
                    [pl.col(col_id)] + [pl.col(c).cast(pl.Int8).alias(c) for c in bool_col_names]
                )
                meta_np = df_meta_sel.select(bool_col_names).to_numpy()  # (N, F)
                self.data_pop = torch.tensor(meta_np, dtype=torch.float32)
                self.meta_cols = list(bool_col_names)

        # --- Pre‐build exogene for the entire date range once (unchanged behaviour) --- #
        # Keep the actual date range as attribute so we can align meteo with it.
        self.extra_dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        self.exogene_full = self._create_exogene()   # (num_days_total, 4)

        # --- Meteo (daily temperature) per client --- #
        self.temps_full: Optional[torch.Tensor] = None  # shape: (num_ids, num_days_total)

        if path_meteo_daily is not None:
            if all(city is None for city in self.meteo_city):
                raise ValueError(
                    "path_meteo_daily was provided but VILLE_METEOis missing in metadata."
                )

            # Load daily meteo: index = date, columns = city names (abbeville, baleMulhouse, ...)
            meteo_df = pd.read_csv(path_meteo_daily, index_col=0, parse_dates=True)

            # Align meteo_df to our date range (self.extra_dates)
            meteo_df = meteo_df.reindex(self.extra_dates)

            if meteo_df.isna().any().any():
                # You can choose to fillna instead if needed:
                # meteo_df = meteo_df.interpolate().ffill().bfill()
                # For now, be strict:
                raise ValueError("NaNs found in meteo data after reindexing to date range.")

            # Check that all used cities exist in meteo_df columns
            used_cities = {c for c in self.meteo_city if c is not None}
            missing_cities = used_cities - set(meteo_df.columns)
            if missing_cities:
                raise ValueError(
                    f"Meteo cities in metadata not found in meteo file columns: {missing_cities}"
                )

            # Build temp matrix: (num_ids, num_days_total)
            temp_mat = []
            for city in self.meteo_city:
                if city is None:
                    # No mapping: you could put zeros, or NaNs.
                    raise ValueError(f"{city} has no mapping with temperature !")
                    # temp_mat.append(np.zeros(len(self.extra_dates), dtype=np.float32))
                else:
                    temp_mat.append(meteo_df[city].to_numpy(dtype=np.float32))

            temp_mat = np.stack(temp_mat, axis=0)  # (num_ids, num_days_total)
            self.temps_full = torch.tensor(temp_mat, dtype=torch.float32) / self.scale_meteo

    def __len__(self) -> int:
        """
        Number of "samples" here is the number of ID columns in this file.
        """
        return len(self.id_clients)

    def _scale_data(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.scale_param1) / (self.scale_param2 - self.scale_param1)

    def _create_exogene(self) -> torch.Tensor:
        # Precompute exogene data once for the entire date range.
        extra = self.extra_dates

        exogene_array = np.vstack([
            extra.weekday.values       * (2 * np.pi / 6),
            extra.day.values           * (2 * np.pi / 31),
            extra.day_of_year.values   * (2 * np.pi / 365),
            extra.month.values         * (2 * np.pi / 12)
        ])

        # shape = (4, num_days) -> transpose -> (num_days, 4)
        return torch.tensor(exogene_array, dtype=torch.float32).permute(1, 0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (values, exogene, y) for client idx.
        - Patchify into daily chunks (available_days, patch_length).
        - Choose window:
            * if random_window: random start in [0, available_days - nb_days]
            * else: start=0 (original behavior)
        Shapes:
            values:   (nb_days, patch_length)
            exogene:  (nb_days, 4) or (nb_days, 5) if temperature is available
            y:        (F,) metadata vector (possibly empty if no metadata provided)
        """
        # Client series (T,)
        series = self.data[idx]

        # Patchify into daily chunks: (available_days, patch_length)
        values = series.unfold(dimension=0, size=self.patch_length, step=self.patch_length)

        # Scale
        values = self._scale_data(values)

        # Choose start
        if self.random_window:
            max_start = self.available_days - self.nb_days
            start_day = int(torch.randint(0, max_start + 1, (1,)).item())
        else:
            start_day = 0

        end_day = start_day + self.nb_days

        values_win  = values[start_day:end_day]            # (nb_days, patch_length)
        exogene_win = self.exogene_full[start_day:end_day] # (nb_days, 4)

        # --- Add temperature as an extra exogenous feature if available --- #
        if self.temps_full is not None:
            temp_series = self.temps_full[idx]             # (num_days_total,)
            temp_win = temp_series[start_day:end_day].unsqueeze(-1)  # (nb_days, 1)
            exogene_win = torch.cat([exogene_win, temp_win], dim=1)  # (nb_days, 5)

        if self.meta_cols:
            y = self.data_pop[idx]
            return values_win, exogene_win, y
        else:
            return values_win, exogene_win


class CERDataset(torch.utils.data.Dataset): 
    """
    PyTorch dataset for cer data.
    """

    def __init__(
        self, 
        path_parquet_part: str,
        nb_days: int = 365,
        patch_length_day: int = 48,
        start_date: str = '15/07/2009',
        end_date: str = '01/01/2011',
        list_pdl: Union[list, None] = None,
        col_time_mask: str = "time",
        scale_param1: int = 0,
        scale_param2: int = 1,
        scale_meteo: int = 1,
        random_window: bool = False,
        path_metadata: Optional[str] = None,
        col_id: str = "id_house",
        bool_col_names: Optional[List[str]] = None, 
        missing_label_policy: str = "fill",  # "fill" or "drop"
        missing_label_value: int = -1,
        path_temperature: Optional[str] = None, 
    ):
        assert os.path.exists(path_parquet_part), f'File {path_parquet_part} does not exist.'
        
        if path_metadata is not None:
            assert os.path.exists(path_metadata), (
                f'Metadata file {path_metadata} does not exist.'
            )
            if missing_label_policy not in {"fill", "drop"}:
                raise ValueError("missing_label_policy must be 'fill' or 'drop'")
            
        if path_temperature is not None:
            assert os.path.exists(path_temperature), (
                f'Meteo file {path_temperature} does not exist.'
            )

        self.patch_length = patch_length_day

        self.start_date = start_date
        self.end_date   = end_date
        self.nb_days    = nb_days
        self.scale_param1 = scale_param1
        self.scale_param2 = scale_param2
        self.scale_meteo   = scale_meteo
        self.random_window = random_window
        self.bool_col_names = bool_col_names

        self.missing_label_policy = missing_label_policy
        self.missing_label_value = missing_label_value

        # --- Load data from parquet once --- #
        df = pl.read_parquet(path_parquet_part)

        # Keep track of the 'time' index
        self.dates = df[col_time_mask]
        
        # Remove the date column, now only have columns for each id
        df = df.drop(col_time_mask)

        if list_pdl is not None:
            
            if isinstance(list_pdl, pl.Series):
                list_pdl = list_pdl.to_list()
            else:
                list_pdl = [str(c) for c in list_pdl]
                
            seen = set()
            list_pdl = [c for c in list_pdl if not (c in seen or seen.add(c))]

            missing = [c for c in list_pdl if c not in df.columns]
            if missing:
                raise ValueError(f"{len(missing)} requested columns not in df: {missing}.")

            cols = [c for c in list_pdl if c in df.columns]

            df = df.select(pl.col(cols))
        
        # Convert everything to a (rows x columns) torch.Tensor 
        #   shape = (#rows, #ids_in_this_file)
        data_np = df.to_numpy().T
        
        self.data = torch.tensor(data_np, dtype=torch.float32)
        
        # Save the name of each ID (the columns in the original DF)
        self.id_clients = [str(c) for c in df.columns]
        
        # Geometry
        self.num_ids        = self.data.shape[0]
        self.num_timepoints = self.data.shape[1]

        self.available_days = self.num_timepoints // self.patch_length
        if self.available_days <= 0:
            raise ValueError("No full days available; check patch_length vs. timepoints.")
        if self.nb_days > self.available_days:
            raise ValueError(
                f"Requested nb_days={self.nb_days}, but only {self.available_days} full days available."
            )

        # --- Metadata / population data --- #
        self.meta_cols = []
        self.data_pop = torch.zeros((self.num_ids, 0), dtype=torch.float32)


        if path_metadata is not None:
            df_meta = pl.read_csv(path_metadata)

            if col_id not in df_meta.columns:
                raise ValueError(f"Column '{col_id}' not found in metadata file.")

            df_meta = df_meta.with_columns(pl.col(col_id).cast(pl.Utf8))
            self.id_clients = [str(c) for c in self.id_clients]

            # Optional: if duplicate IDs exist in metadata, keep first
            df_meta = df_meta.unique(subset=[col_id], keep="first")

            if bool_col_names is None or len(bool_col_names) == 0:
                # No labels requested → keep everything
                self.data_pop = torch.zeros((self.num_ids, 0), dtype=torch.float32)
                self.meta_cols = []
            else:
                missing_meta = [c for c in bool_col_names if c not in df_meta.columns]
                if missing_meta:
                    raise ValueError(f"Missing metadata columns: {missing_meta}")

                # Keep only ids that exist in the parquet (curve ids)
                df_meta = df_meta.filter(pl.col(col_id).is_in(self.id_clients))

                # Cast label columns, allow failures -> null
                df_meta = df_meta.with_columns([
                    pl.col(c).cast(pl.Int8, strict=False).alias(c) for c in bool_col_names
                ])

                if self.missing_label_policy == "fill":
                    # Reindex to ALL parquet ids (keeps all curves)
                    df_meta = df_meta.join(
                        pl.DataFrame({col_id: self.id_clients}),
                        on=col_id,
                        how="right",
                    )

                    # Fill missing labels with -1 (and NaNs if present)
                    df_meta = df_meta.with_columns([
                        pl.col(c)
                        .fill_null(self.missing_label_value)
                        .fill_nan(self.missing_label_value)
                        .alias(c)
                        for c in bool_col_names
                    ])

                    meta_np = df_meta.select(bool_col_names).to_numpy()
                    self.data_pop = torch.tensor(meta_np, dtype=torch.float32)
                    self.meta_cols = list(bool_col_names)

                else:  # "drop"
                    # Drop any id with any missing label (also drops ids absent from metadata)
                    df_meta_valid = (
                        df_meta
                        .with_columns([pl.col(c).fill_nan(None).alias(c) for c in bool_col_names])
                        .drop_nulls(subset=bool_col_names)
                    )

                    valid_ids = df_meta_valid.select(col_id).to_series().to_list()
                    valid_set = set(valid_ids)

                    # Filter curves to those valid ids
                    keep_idx = [i for i, cid in enumerate(self.id_clients) if cid in valid_set]
                    if len(keep_idx) == 0:
                        raise ValueError("After dropping missing-label rows, no IDs remain.")

                    self.data = self.data[keep_idx]
                    self.id_clients = [self.id_clients[i] for i in keep_idx]

                    # Update geometry dependent on num_ids
                    self.num_ids = self.data.shape[0]

                    # Reorder metadata to match filtered id_clients
                    df_meta_valid = df_meta_valid.join(
                        pl.DataFrame({col_id: self.id_clients}),
                        on=col_id,
                        how="inner",
                    )

                    meta_np = df_meta_valid.select(bool_col_names).to_numpy()
                    self.data_pop = torch.tensor(meta_np, dtype=torch.float32)
                    self.meta_cols = list(bool_col_names)

        # --- Pre‐build exogene for the entire date range once (unchanged behaviour) --- #
        # Keep the actual date range as attribute so we can align meteo with it.
        self.extra_dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        self.exogene_full = self._create_exogene()   # (num_days_total, 4)

        # --- Meteo (daily temperature) per client --- #
        self.temps_full: Optional[torch.Tensor] = None  # shape: (num_ids, num_days_total)

        if path_temperature is not None:
            
            meteo_df = pd.read_csv(
                path_temperature,
                parse_dates=["date"],          # parse the timestamp
            )

            # make sure it's sorted + use datetime index (handy for resampling)
            meteo_df = meteo_df.sort_values("date").set_index("date")

            # daily temperature (pick the aggregation you want)
            daily_mean = meteo_df["temperature"].resample("D").mean()   # average per day

            temp_full = daily_mean.values
            self.temps_full = torch.tensor(temp_full, dtype=torch.float32) / self.scale_meteo

    def __len__(self) -> int:
        """
        Number of "samples" here is the number of ID columns in this file.
        """
        return len(self.id_clients)

    def _scale_data(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.scale_param1) / (self.scale_param2 - self.scale_param1)

    def _create_exogene(self) -> torch.Tensor:
        # Precompute exogene data once for the entire date range.
        extra = self.extra_dates

        exogene_array = np.vstack([
            extra.weekday.values       * (2 * np.pi / 6),
            extra.day.values           * (2 * np.pi / 31),
            extra.day_of_year.values   * (2 * np.pi / 365),
            extra.month.values         * (2 * np.pi / 12)
        ])

        # shape = (4, num_days) -> transpose -> (num_days, 4)
        return torch.tensor(exogene_array, dtype=torch.float32).permute(1, 0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (values, exogene, y) for client idx.
        - Patchify into daily chunks (available_days, patch_length).
        - Choose window:
            * if random_window: random start in [0, available_days - nb_days]
            * else: start=0 (original behavior)
        Shapes:
            values:   (nb_days, patch_length)
            exogene:  (nb_days, 4) or (nb_days, 5) if temperature is available
            y:        (F,) metadata vector (possibly empty if no metadata provided)
        """
        # Client series (T,)
        series = self.data[idx]

        # Patchify into daily chunks: (available_days, patch_length)
        values = series.unfold(dimension=0, size=self.patch_length, step=self.patch_length)

        # Scale
        values = self._scale_data(values)

        # Choose start
        if self.random_window:
            max_start = self.available_days - self.nb_days
            start_day = int(torch.randint(0, max_start + 1, (1,)).item())
        else:
            start_day = 0

        end_day = start_day + self.nb_days

        values_win  = values[start_day:end_day]            # (nb_days, patch_length)
        exogene_win = self.exogene_full[start_day:end_day] # (nb_days, 4)

        # --- Add temperature as an extra exogenous feature if available --- #
        if self.temps_full is not None:
            temp_win = self.temps_full[start_day:end_day].unsqueeze(-1)  # (nb_days, 1)
            exogene_win = torch.cat([exogene_win, temp_win], dim=1)  # (nb_days, 5)

        if self.meta_cols:
            y = self.data_pop[idx]
            return values_win, exogene_win, y
        else:
            return values_win, exogene_win


class CERBisDataset(torch.utils.data.Dataset): 
    """
    PyTorch dataset for cer data.
    """

    def __init__(
        self, 
        path_parquet_part: str,
        nb_days: int = 365,
        patch_length_day: int = 48,
        start_date: str = '01/01/2025',
        end_date: str = '01/01/2026',
        list_pdl: Union[list, None] = None,
        col_time_mask: str = "time",
        scale_param1: int = 0,
        scale_param2: int = 1,
        scale_meteo: int = 1,
        random_window: bool = False,
        path_metadata: Optional[str] = None,
        col_id: str = "ID_PDL",
        bool_col_names: Optional[List[str]] = None, 
        missing_label_policy: str = "fill",  # "fill" or "drop"
        missing_label_value: int = -1,
        path_temperature: Optional[str] = None, 
    ):
        assert os.path.exists(path_parquet_part), f'File {path_parquet_part} does not exist.'
        
        if path_metadata is not None:
            assert os.path.exists(path_metadata), (
                f'Metadata file {path_metadata} does not exist.'
            )
            if missing_label_policy not in {"fill", "drop"}:
                raise ValueError("missing_label_policy must be 'fill' or 'drop'")
            
        if path_temperature is not None:
            assert os.path.exists(path_temperature), (
                f'Meteo file {path_temperature} does not exist.'
            )

        self.patch_length = patch_length_day

        self.start_date = start_date
        self.end_date   = end_date
        self.nb_days    = nb_days
        self.scale_param1 = scale_param1
        self.scale_param2 = scale_param2
        self.scale_meteo   = scale_meteo
        self.random_window = random_window
        self.bool_col_names = bool_col_names

        self.missing_label_policy = missing_label_policy
        self.missing_label_value = missing_label_value

        # --- Load data from parquet once --- #
        df = pl.read_parquet(path_parquet_part)

        # Keep track of the 'time' index
        self.dates = df[col_time_mask]
        
        # Remove the date column, now only have columns for each id
        # if col_time_mask in df.columns:
        df = df.drop(col_time_mask)

        if list_pdl is not None:
            
            if isinstance(list_pdl, pl.Series):
                list_pdl = list_pdl.to_list()
            else:
                list_pdl = [str(c) for c in list_pdl]
                
            seen = set()
            list_pdl = [c for c in list_pdl if not (c in seen or seen.add(c))]

            missing = [c for c in list_pdl if c not in df.columns]
            if missing:
                raise ValueError(f"{len(missing)} requested columns not in df: {missing}.")

            cols = [c for c in list_pdl if c in df.columns]

            df = df.select(pl.col(cols))
        
        # Convert everything to a (rows x columns) torch.Tensor 
        #   shape = (#rows, #ids_in_this_file)
        data_np = df.to_numpy().T
        
        self.data = torch.tensor(data_np, dtype=torch.float32)
        
        # Save the name of each ID (the columns in the original DF)
        self.id_clients = [str(c) for c in df.columns]
        
        # Geometry
        self.num_ids        = self.data.shape[0]
        self.num_timepoints = self.data.shape[1]

        self.available_days = self.num_timepoints // self.patch_length
        if self.available_days <= 0:
            raise ValueError("No full days available; check patch_length vs. timepoints.")
        if self.nb_days > self.available_days:
            raise ValueError(
                f"Requested nb_days={self.nb_days}, but only {self.available_days} full days available."
            )

        # --- Metadata / population data --- #
        self.meta_cols = []
        self.data_pop = torch.zeros((self.num_ids, 0), dtype=torch.float32)


        if path_metadata is not None:
            df_meta = pl.read_parquet(path_metadata)

            if col_id not in df_meta.columns:
                raise ValueError(f"Column '{col_id}' not found in metadata file.")

            df_meta = df_meta.with_columns(pl.col(col_id).cast(pl.Utf8))
            self.id_clients = [str(c) for c in self.id_clients]

            # Optional: if duplicate IDs exist in metadata, keep first
            df_meta = df_meta.unique(subset=[col_id], keep="first")

            if bool_col_names is None or len(bool_col_names) == 0:
                # No labels requested → keep everything
                self.data_pop = torch.zeros((self.num_ids, 0), dtype=torch.float32)
                self.meta_cols = []
            else:
                missing_meta = [c for c in bool_col_names if c not in df_meta.columns]
                if missing_meta:
                    raise ValueError(f"Missing metadata columns: {missing_meta}")

                # Keep only ids that exist in the parquet (curve ids)
                df_meta = df_meta.filter(pl.col(col_id).is_in(self.id_clients))

                # Cast label columns, allow failures -> null
                df_meta = df_meta.with_columns([
                    pl.col(c).cast(pl.Int8, strict=False).alias(c) for c in bool_col_names
                ])

                if self.missing_label_policy == "fill":
                    # Reindex to ALL parquet ids (keeps all curves)
                    df_meta = df_meta.join(
                        pl.DataFrame({col_id: self.id_clients}),
                        on=col_id,
                        how="right",
                    )

                    # Fill missing labels with -1 (and NaNs if present)
                    df_meta = df_meta.with_columns([
                        pl.col(c)
                        .fill_null(self.missing_label_value)
                        .fill_nan(self.missing_label_value)
                        .alias(c)
                        for c in bool_col_names
                    ])

                    meta_np = df_meta.select(bool_col_names).to_numpy()
                    self.data_pop = torch.tensor(meta_np, dtype=torch.float32)
                    self.meta_cols = list(bool_col_names)

                else:  # "drop"
                    # Drop any id with any missing label (also drops ids absent from metadata)
                    df_meta_valid = (
                        df_meta
                        .with_columns([pl.col(c).fill_nan(None).alias(c) for c in bool_col_names])
                        .drop_nulls(subset=bool_col_names)
                    )

                    valid_ids = df_meta_valid.select(col_id).to_series().to_list()
                    valid_set = set(valid_ids)

                    # Filter curves to those valid ids
                    keep_idx = [i for i, cid in enumerate(self.id_clients) if cid in valid_set]
                    if len(keep_idx) == 0:
                        raise ValueError("After dropping missing-label rows, no IDs remain.")

                    self.data = self.data[keep_idx]
                    self.id_clients = [self.id_clients[i] for i in keep_idx]

                    # Update geometry dependent on num_ids
                    self.num_ids = self.data.shape[0]

                    # Reorder metadata to match filtered id_clients
                    df_meta_valid = df_meta_valid.join(
                        pl.DataFrame({col_id: self.id_clients}),
                        on=col_id,
                        how="inner",
                    )

                    meta_np = df_meta_valid.select(bool_col_names).to_numpy()
                    self.data_pop = torch.tensor(meta_np, dtype=torch.float32)
                    self.meta_cols = list(bool_col_names)

        # --- Pre‐build exogene for the entire date range once (unchanged behaviour) --- #
        # Keep the actual date range as attribute so we can align meteo with it.
        self.extra_dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        self.exogene_full = self._create_exogene()   # (num_days_total, 4)

        # --- Meteo (daily temperature) per client --- #
        self.temps_full: Optional[torch.Tensor] = None  # shape: (num_ids, num_days_total)

        if path_temperature is not None:
            raise ValueError("No meteo link")

    def __len__(self) -> int:
        """
        Number of "samples" here is the number of ID columns in this file.
        """
        return len(self.id_clients)

    def _scale_data(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.scale_param1) / (self.scale_param2 - self.scale_param1)

    def _create_exogene(self) -> torch.Tensor:
        # Precompute exogene data once for the entire date range.
        extra = self.extra_dates

        exogene_array = np.vstack([
            extra.weekday.values       * (2 * np.pi / 6),
            extra.day.values           * (2 * np.pi / 31),
            extra.day_of_year.values   * (2 * np.pi / 365),
            extra.month.values         * (2 * np.pi / 12)
        ])

        # shape = (4, num_days) -> transpose -> (num_days, 4)
        return torch.tensor(exogene_array, dtype=torch.float32).permute(1, 0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (values, exogene, y) for client idx.
        - Patchify into daily chunks (available_days, patch_length).
        - Choose window:
            * if random_window: random start in [0, available_days - nb_days]
            * else: start=0 (original behavior)
        Shapes:
            values:   (nb_days, patch_length)
            exogene:  (nb_days, 4) or (nb_days, 5) if temperature is available
            y:        (F,) metadata vector (possibly empty if no metadata provided)
        """
        # Client series (T,)
        series = self.data[idx]

        # Patchify into daily chunks: (available_days, patch_length)
        values = series.unfold(dimension=0, size=self.patch_length, step=self.patch_length)

        # Scale
        values = self._scale_data(values)

        # Choose start
        if self.random_window:
            max_start = self.available_days - self.nb_days
            start_day = int(torch.randint(0, max_start + 1, (1,)).item())
        else:
            start_day = 0

        end_day = start_day + self.nb_days

        values_win  = values[start_day:end_day]            # (nb_days, patch_length)
        exogene_win = self.exogene_full[start_day:end_day] # (nb_days, 4)

        # --- Add temperature as an extra exogenous feature if available --- #
        if self.temps_full is not None:
            temp_win = self.temps_full[start_day:end_day].unsqueeze(-1)  # (nb_days, 1)
            exogene_win = torch.cat([exogene_win, temp_win], dim=1)  # (nb_days, 5)

        if self.meta_cols:
            y = self.data_pop[idx]
            return values_win, exogene_win, y
        else:
            return values_win, exogene_win