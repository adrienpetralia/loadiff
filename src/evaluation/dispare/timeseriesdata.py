import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
import seaborn as sns
import statsmodels.api as sm
import torch
from typing import Optional

from functools import partial
from itertools import product
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from scipy.signal import periodogram
from sklearn.cluster import KMeans
from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.datatypes import convert
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import MiniRocket
from typing import Dict, Union, List, Tuple, Callable

sns.set_theme()


class TimeSeriesData:
    """Class for managing time series data"""

    def __init__(self, dict_df):
        """Initialize TimeSeriesData object with given data.

        Args:
            dict_df(Dict[str, DataFrame]): Dict[name, dataframe]
        """
        self.dict_df = dict_df
        self.validate_input()

    @staticmethod
    def array_to_df(
        ts: NDArray,
        col_value="value",
        col_time="timestamp",
        col_id="id",
        min_time=datetime.datetime(2023, 1, 1, 0, 0, 0, 0),
        timedelta=datetime.timedelta(0, 30 * 60),
        ids: NDArray | None = None,
    ):
        """Transform a time series in array format into a DataFrame

        Args:
            ts(NDArray): Time series with shape(N, T)
            col_value(str, optional): Name of the value column in the
                dataframe. Default to "value"
            col_time(str, optional): Name of the time column in the dataframe.
                Default to "Timestamp"
            col_time(str, optional): Name of the id column in the dataframe.
                Default to "id"
            min_time(datetime, optional): first timestamp. Defaults to
                2023 january 1st, 00: 00
            timedelta(timedelta, optional): time between two consecutive
                sample. Defaults to 30 min

        Returns:
            DataFrame: Corresponding dataframe
        """
        times = [min_time + i * timedelta for i in range(ts.shape[1])]
        if ids is None:
            ids = np.arange(ts.shape[0]).repeat(ts.shape[1])
        else:
            ids = ids.repeat(ts.shape[1])
        values = np.concatenate([ts[i] for i in range(ts.shape[0])], 0)
        times = times * ts.shape[0]
        df = pd.DataFrame({col_id: ids, col_value: values, col_time: times})
        return df

    def validate_input(self):
        if not all(df is not None for df in self.dict_df.values()):
            raise ValueError(
                "The dictionary must contain non-null DataFrames"
            )  # nopep8 # Noqa

    def validate_column(self, col_name: Union[str, list]):
        """Check if column is present in each dataframe"""
        if isinstance(col_name, list):
            for name in col_name:
                for key, df in self.dict_df.items():
                    if name not in df.columns:
                        raise ValueError(
                            f"'{name}' column not found in DataFrame {key}"
                        )  # noqa
        else:
            for key, df in self.dict_df.items():
                if col_name not in df.columns:
                    raise ValueError(
                        f"'{col_name}' column not found in DataFrame {key}"
                    )  # noqa

    def compute_daily_profile(
        self, col_value: str = "value", col_time: str = "timestamp"
    ):
        """Calculate the daily profile by grouping data based on timestamps."""
        self.validate_column(col_value)
        self.validate_column(col_time)

        merged_daily = pd.DataFrame(columns=[col_time])

        for key, value in self.dict_df.items():
            # value[col_time] = pd.to_datetime(value[col_time])
            daily_profile = (
                # value.groupby(value[col_time].dt.strftime('%H:%M'))[col_value]
                value[[col_value]].groupby(value[col_time].dt.time).mean().reset_index()
            )
            daily_profile.columns = [col_time, f"Value {key}"]
            merged_daily = pd.merge(
                merged_daily, daily_profile, on=col_time, how="outer"
            )

        # merged_daily[col_time] = pd.to_datetime(merged_daily[col_time])
        # merged_daily[col_time] = merged_daily[col_time].dt.strftime('%H:%M')
        merged_daily = merged_daily.set_index(col_time)

        return merged_daily

    def graph_daily_profile(
        self, col_value: str = "value", col_time: str = "timestamp"
    ):
        """Plot the daily profile

        Args:
            col_value (str): Name of the column containing the values of
                interest. Default to 'value'.
            col_time (str): Name of the column containing the temporal values.
                Default to 'timestamp'.
        """
        self.daily_profile = self.compute_daily_profile(col_value, col_time)

        colors = sns.color_palette("colorblind", len(self.dict_df))

        fig = plt.figure(figsize=(12, 6))

        for idx, (key, color) in enumerate(zip(self.dict_df.keys(), colors)):
            sns.lineplot(
                x=self.daily_profile.index.astype(str),
                y=f"Value {key}",
                data=self.daily_profile,
                label=f"Value {key}",
                color=color,
                linewidth=2,
            )

        plt.xlabel(col_time)
        plt.ylabel("Value")
        plt.title("Comparison - Daily Profile")
        plt.legend()

        sns.set_style("whitegrid")
        sns.despine()

        plt.tight_layout()
        plt.xticks(rotation=45, ha="right")
        plt.show()
        return fig

    def compute_weekly_profile(
        self, col_value: str = "value", col_time: str = "timestamp", merge_days=True
    ):
        """Calculate the weekly profile by grouping data based on
        timestamps."""
        self.validate_column(col_value)
        self.validate_column(col_time)

        merged_weekly = pd.DataFrame(columns=[col_time])

        for key, value in self.dict_df.items():
            if merge_days:
                by = value[col_time].dt.day_of_week
            else:
                by = value[col_time].dt.strftime("%u %A %H:%M")
            weekly_profile = (
                value[col_value].groupby(by).mean().reset_index().sort_values(col_time)
            )
            weekly_profile.columns = [col_time, f"{col_value} {key}"]
            merged_weekly = pd.merge(
                merged_weekly, weekly_profile, on=col_time, how="outer"
            )

        merged_weekly = merged_weekly.set_index(col_time)
        return merged_weekly

    def graph_weekly_profile(
        self, col_value: str = "value", col_time: str = "timestamp", merge_days=True
    ):
        """Plot the weekly profile

        Args:
            col_value (str): Name of the column containing the values of
                interest. Default to 'value'.
            col_time (str): Name of the column containing the temporal values.
                Default to 'timestamp'.
        """
        self.weekly_profile = self.compute_weekly_profile(
            col_value, col_time, merge_days
        )

        colors = sns.color_palette("colorblind", len(self.dict_df))

        fig = plt.figure(figsize=(12, 6))

        for idx, (key, color) in enumerate(zip(self.dict_df.keys(), colors)):
            sns.lineplot(
                x=col_time,
                y=f"{col_value} {key}",
                data=self.weekly_profile.reset_index(),
                label=f"{col_value} {key}",
                color=color,
                linewidth=2,
            )

        plt.xlabel(col_time)
        plt.ylabel(col_value)
        plt.title("Comparison - Weekly Profile")
        plt.legend()

        sns.set_style("whitegrid")
        sns.despine()

        plt.tight_layout()
        plt.xticks(rotation=45, ha="right")
        t0 = np.array(plt.xticks()[0])
        if len(t0) > 10:
            t0 = t0[np.linspace(0, len(t0) - 1, 7).astype(np.int_)]
        else:
            t0 = list(range(7))
        plt.xticks(
            t0,
            ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"],
        )
        plt.show()
        return fig

    def compute_monthly_profile(
        self, col_value: str = "value", col_time: str = "timestamp", merge_days=True
    ):
        """Calculate the monthly profile by grouping data based on
        timestamps."""
        self.validate_column(col_value)
        self.validate_column(col_time)

        merged_monthly = pd.DataFrame(columns=[col_time])

        for key, value in self.dict_df.items():
            if merge_days:
                by = value[col_time].dt.day
            else:
                by = value[col_time].dt.strftime("%d %H:%M")
            monthly_profile = (
                value[col_value].groupby(by).mean().reset_index().sort_values(col_time)
            )
            monthly_profile.columns = [col_time, f"{col_value} {key}"]
            merged_monthly = pd.merge(
                merged_monthly, monthly_profile, on=col_time, how="outer"
            )

        merged_monthly = merged_monthly.set_index(col_time)
        return merged_monthly

    def graph_monthly_profile(
        self, col_value: str = "value", col_time: str = "timestamp", merge_days=True
    ):
        """Plot the monthly profile

        Args:
            col_value (str): Name of the column containing the values of
                interest. Default to 'value'.
            col_time (str): Name of the column containing the temporal values.
                Default to 'timestamp'.
        """
        self.monthly_profile = self.compute_monthly_profile(
            col_value, col_time, merge_days
        )

        colors = sns.color_palette("colorblind", len(self.dict_df))

        fig = plt.figure(figsize=(12, 6))

        for idx, (key, color) in enumerate(zip(self.dict_df.keys(), colors)):
            sns.lineplot(
                x=col_time,
                y=f"{col_value} {key}",
                data=self.monthly_profile.reset_index(),
                label=f"{col_value} {key}",
                color=color,
                linewidth=2,
            )

        plt.xlabel(col_time)
        plt.ylabel(col_value)
        plt.title("Comparison - Monthly Profile")
        plt.legend()

        sns.set_style("whitegrid")
        sns.despine()

        plt.tight_layout()
        plt.xticks(rotation=45, ha="right")
        t0 = np.array(plt.xticks()[0])
        if len(t0) > 35:
            t0 = t0[np.linspace(0, len(t0) - 1, 31).astype(np.int_)]
        else:
            t0 = list(range(1, 32))
        plt.xticks(t0, list(map(str, range(1, 32))))
        plt.show()
        return fig

    def compute_yearly_profile(
        self, col_value: str = "value", col_time: str = "timestamp", merge_days=True
    ):
        """Calculate the yearly profile by grouping data based on
        timestamps."""
        self.validate_column(col_value)
        self.validate_column(col_time)

        merged_yearly = pd.DataFrame(columns=[col_time])

        for key, value in self.dict_df.items():
            if merge_days:
                by = value[col_time].dt.month
            else:
                by = value[col_time].dt.strftime("%m %B %d %H:%M")
            yearly_profile = (
                value[col_value].groupby(by).mean().reset_index().sort_values(col_time)
            )
            yearly_profile.columns = [col_time, f"{col_value} {key}"]
            merged_yearly = pd.merge(
                merged_yearly, yearly_profile, on=col_time, how="outer"
            )

        merged_yearly = merged_yearly.set_index(col_time)
        return merged_yearly

    def graph_yearly_profile(
        self, col_value: str = "value", col_time: str = "timestamp", merge_days=True
    ):
        """Plot the yearly profile

        Args:
            col_value (str): Name of the column containing the values of
                interest. Default to 'value'.
            col_time (str): Name of the column containing the temporal values.
                Default to 'timestamp'.
        """
        self.yearly_profile = self.compute_yearly_profile(
            col_value, col_time, merge_days
        )

        colors = sns.color_palette("colorblind", len(self.dict_df))

        fig = plt.figure(figsize=(12, 6))

        for idx, (key, color) in enumerate(zip(self.dict_df.keys(), colors)):
            sns.lineplot(
                x=self.yearly_profile.index.astype(str),
                y=f"{col_value} {key}",
                data=self.yearly_profile,
                label=f"{col_value} {key}",
                color=color,
                linewidth=2,
            )

        plt.xlabel(col_time)
        plt.ylabel(col_value)
        plt.title("Comparison - Yearly Profile")
        plt.legend()

        sns.set_style("whitegrid")
        sns.despine()

        plt.tight_layout()
        plt.xticks(rotation=45, ha="right")
        t0 = np.array(plt.xticks()[0])
        if len(t0) > 15:
            t0 = t0[np.linspace(0, len(t0) - 1, 12).astype(np.int_)]
        else:
            t0 = list(range(12))
        plt.xticks(
            t0,
            [
                "Janvier",
                "Février",
                "Mars",
                "Avril",
                "Mai",
                "Juin",
                "Juillet",
                "Août",
                "Septembre",
                "Octobre",
                "Novembre",
                "Décembre",
            ],
        )
        plt.show()
        return fig

    def compute_profile(
        self, col_value: str = "value", col_time: str = "timestamp"
    ) -> pd.DataFrame:
        """Calculate the profile by grouping data based on timestamps

        Args:
            col_value (str): Name of the column containing the values of
                interest. Default to 'value'.
            col_time (str): Name of the column containing the temporal values.
                Default to 'timestamp'.

        Returns:
            pd.DataFrame: DataFrame containing the computed profile.
        """
        self.validate_input()
        self.validate_column(col_value)
        self.validate_column(col_time)
        self.validate_column(col_value)
        self.validate_column(col_time)

        merged_profile = pd.DataFrame(columns=[col_time])
        merged_profile = pd.DataFrame(columns=[col_time])

        for key, value in self.dict_df.items():
            # value[col_time] = pd.to_datetime(value[col_time])
            # profile = (value.groupby(value[col_time])[col_value]
            profile = (
                value[[col_time, col_value]].groupby(col_time).mean().reset_index()
            )
            profile.columns = [col_time, f"Value {key}"]
            merged_profile = pd.merge(merged_profile, profile, on=col_time, how="outer")

        # merged_profile[col_time] = pd.to_datetime(
        #     merged_profile[col_time], format='"%Y-%m-%d')
        merged_profile = merged_profile.reset_index(drop=True)

        return merged_profile

    def graph_profile(self, col_value: str = "value", col_time: str = "timestamp"):
        """Plot the profile

        Args:
            col_value (str): Name of the column containing the values of
                interest. Default to 'value'.
            col_time (str): Name of the column containing the temporal values.
                Default to 'timestamp'.
        """
        self.value_profile = self.compute_profile(col_value, col_time)

        # Define a custom color palette with orange first, then blue,
        # then the rest
        colors = sns.color_palette("colorblind", len(self.dict_df))

        sns.set_style("whitegrid")

        fig = plt.figure(figsize=(12, 6))

        for key, color in zip(self.dict_df.keys(), colors):
            sns.lineplot(
                x=col_time,
                y=f"Value {key}",
                data=self.value_profile,
                label=f"Value {key}",
                color=color,
                alpha=0.8,  # Adjusted alpha for better visibility
            )

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xlabel(col_time)
        plt.xlabel(col_time)
        plt.ylabel("Value")
        plt.title("Comparison - Profile")
        plt.legend()

        plt.grid(True)  # Ensure grids are shown

        sns.despine()

        plt.tight_layout()
        plt.xticks(rotation=45, ha="right")
        plt.xticks(rotation=45, ha="right")
        plt.show()
        return fig

    def compute_monthly_average(
        self,
        col_value: str = "value",
        col_time: str = "timestamp",
        col_id: Union[str, List[str]] = "id",
    ):
        """Calculate the monthly value average by grouping data on id"""
        self.validate_column(col_value)
        self.validate_column(col_time)

        merged_monthly = pd.DataFrame()

        for key, value in self.dict_df.items():
            by = value[col_time].dt.strftime("%m %B")
            monthly_profile = (
                value[col_value]
                .groupby([by, value[col_id]])
                .mean()
                .reset_index()
                .sort_values(col_time)
            )
            monthly_profile.columns = [col_time, col_id, col_value]
            monthly_profile = monthly_profile.assign(Category=key)
            merged_monthly = pd.concat([merged_monthly, monthly_profile])

        merged_monthly = merged_monthly.set_index(col_time)
        return merged_monthly

    def compute_energy_profile_by_day(
        self, col_value: str = "value", col_time: str = "timestamp"
    ):
        """
        Calcule le profil journalier moyen de consommation d'énergie :
        1. Regroupe les données par date et somme les valeurs, puis divise par 2.
        2. Moyenne ces profils sur la population (plusieurs clients).
        """
        self.validate_column(col_value)
        self.validate_column(col_time)

        merged_energy = pd.DataFrame(columns=["date"])

        for key, value in self.dict_df.items():
            # Assurer que la colonne timestamp est bien en datetime
            value[col_time] = pd.to_datetime(value[col_time])

            # Extraire uniquement la date (sans l'heure)
            value["date"] = value[col_time].dt.date

            # Somme quotidienne, puis division par 2
            daily_energy = value.groupby("date")[[col_value]].sum().div(2).reset_index()

            daily_energy.columns = ["date", f"Energy {key}"]

            # Fusion avec le tableau final
            merged_energy = pd.merge(
                merged_energy, daily_energy, on="date", how="outer"
            )

        # Calcul du profil moyen sur tous les clients (colonnes)
        merged_energy["mean_energy"] = merged_energy.drop(columns=["date"]).mean(axis=1)

        # Facultatif : remettre l'index sur la date
        merged_energy = merged_energy.set_index("date")

        return merged_energy

    # def graph_daily_energy_profile(
    #     self, col_value: str = "value", col_time: str = "timestamp"
    # ):
    #     """
    #     Trace le profil moyen de consommation énergétique par jour
    #     (moyenne sur la population).

    #     Args:
    #         col_value (str): Nom de la colonne contenant les valeurs d'énergie.
    #         col_time (str): Nom de la colonne contenant les timestamps.
    #     """
    #     self.daily_profile = self.compute_energy_profile_by_day(col_value, col_time)

    #     colors = sns.color_palette("colorblind", len(self.dict_df))

    #     fig = plt.figure(figsize=(12, 6))

    #     # Tracer les profils individuels (optionnel)
    #     for idx, (key, color) in enumerate(zip(self.dict_df.keys(), colors)):
    #         col_name = f"Energy {key}"
    #         if col_name in self.daily_profile.columns:
    #             sns.lineplot(
    #                 x=self.daily_profile.index.astype(str),
    #                 y=col_name,
    #                 data=self.daily_profile,
    #                 label=col_name,
    #                 color=color,
    #                 linewidth=1,
    #                 alpha=1,
    #             )

    #     plt.xlabel("Date")
    #     plt.ylabel("Daily Energy")
    #     plt.title("Average Daily Energy Profile")
    #     plt.legend()

    #     sns.set_style("whitegrid")
    #     sns.despine()

    #     plt.tight_layout()
    #     plt.xticks(rotation=45, ha="right")

    #     return fig

    def graph_daily_energy_profile(
        self, col_value: str = "value", col_time: str = "timestamp"
    ):
        """
        Trace le profil moyen de consommation énergétique par jour
        (moyenne sur la population).

        Args:
            col_value (str): Nom de la colonne contenant les valeurs d'énergie.
            col_time (str): Nom de la colonne contenant les timestamps.
        """
        self.daily_profile = self.compute_energy_profile_by_day(col_value, col_time)

        colors = sns.color_palette("colorblind", len(self.dict_df))

        fig, ax = plt.subplots(figsize=(12, 6))

        # Tracer les profils individuels
        for idx, (key, color) in enumerate(zip(self.dict_df.keys(), colors)):
            col_name = f"Energy {key}"
            if col_name in self.daily_profile.columns:
                sns.lineplot(
                    x=self.daily_profile.index,
                    y=col_name,
                    data=self.daily_profile,
                    label=col_name,
                    color=color,
                    linewidth=1,
                    alpha=1,
                    ax=ax,
                )

        ax.set_xlabel("Date")
        ax.set_ylabel("Daily Energy")
        ax.set_title("Average Daily Energy Profile")

        sns.set_style("whitegrid")
        sns.despine()

        # Only show 1st of each month
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        fig.autofmt_xdate(rotation=45, ha="right")

        plt.legend()
        plt.tight_layout()

        return fig

    def graph_monthly_hist(
        self,
        col_value: str = "value",
        col_time: str = "timestamp",
        col_id: Union[str, List[str]] = "id",
    ):
        """Plot the histogram of monthly average values for different ids

        Args:
            col_value (str): Name of the column containing the values of
                interest. Default to 'value'.
            col_time (str): Name of the column containing the temporal values.
                Default to 'timestamp'.
            col_id(str | list[str]): Name of the column containing the ids.
                Default to 'id'.
        """
        self.monthly_average = self.compute_monthly_average(
            col_value,
            col_time,
            col_id,
        )

        fig = plt.figure(figsize=(12, 6))

        sns.boxplot(x=col_time, y=col_value, hue="Category", data=self.monthly_average)

        plt.title("Boxplots de la consommation mensuelle par classe")
        plt.xlabel("Mois")
        plt.ylabel("Consommation moyenne")

        sns.set_style("whitegrid")
        sns.despine()

        plt.tight_layout()
        plt.xticks(rotation=45, ha="right")
        t0 = np.array(plt.xticks()[0])
        if len(t0) > 15:
            t0 = t0[np.linspace(0, len(t0) - 1, 12).astype(np.int_)]
        else:
            t0 = list(range(12))
        plt.xticks(
            t0,
            [
                "Janvier",
                "Février",
                "Mars",
                "Avril",
                "Mai",
                "Juin",
                "Juillet",
                "Août",
                "Septembre",
                "Octobre",
                "Novembre",
                "Décembre",
            ],
        )
        plt.show()
        return fig

    def compute_distribution(self, col_value: str = "value") -> None:
        """Plot the distribution of values

        Args:
            col_value(str): Name of the column containing the values of
                interest. Default to 'value'.
            col_time(str): Name of the column containing the temporal values.
                Default to 'timestamp'.
        """
        plt.figure(figsize=(10, 6))

        self.validate_input()
        self.validate_column(col_value)

        if col_value not in self.dict_df[next(iter(self.dict_df))].columns:
            raise ValueError(f"'{col_value}' column not found in DataFrame")

        for key, value in self.dict_df.items():
            sns.kdeplot(
                data=value,
                x=col_value,
                fill=True,
                common_norm=False,
                label=key,
                bw_method=0.1,
            )

        plt.title("Distribution")
        plt.xlabel("Value")
        plt.ylabel("Density")

        plt.legend()
        plt.show()

    def compute_autocorrelation(
        self, col_value: str = "value", col_time: str = "timestamp", nlags: int = 336
    ) -> None:
        """
        Compute autocorrelation for each dataset in self.dict_df and plot the
        autocorrelation function.

        Parameters:
            col_value (str): Name of the column containing the values of
                interest. Default to 'value'.
            col_time (str): Name of the column containing the temporal values.
                Default to 'timestamp'.
            nlags (int): Number of lags to include in autocorrelation
                computation. Default to 336.

        """
        self.validate_input()
        self.validate_column(col_value)
        self.validate_column(col_time)

        autocorr_df = pd.DataFrame({"Lag": range(nlags + 1)})
        merged_daily = pd.DataFrame(columns=[col_value])

        for key, value in self.dict_df.items():
            value[col_time] = pd.to_datetime(value[col_time])
            daily_profile = (
                value.groupby(value[col_time].dt.strftime("%m-%d %H:%M:%S"))[col_value]
                .mean()
                .reset_index()
            )
            daily_profile.columns = [col_value, f"Value {key}"]
            merged_daily = pd.merge(
                merged_daily, daily_profile, on=col_value, how="outer"
            )

        for key in self.dict_df.keys():
            autocorr_values = sm.tsa.acf(
                merged_daily[f"Value {key}"].dropna(), nlags=nlags
            )
            autocorr_df[f"Autocorr {key}"] = autocorr_values[: nlags + 1]

        fig, ax = plt.subplots(figsize=(10, 6))
        for key in self.dict_df.keys():
            sm.graphics.tsa.plot_acf(
                merged_daily[f"Value {key}"].dropna(),
                lags=nlags,
                ax=ax,
                marker="o",
                markersize=2.5,
                use_vlines=False,
                label=f"{key}",
            )

        plt.title("Autocorrelation of Time Series")
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.grid(True, linewidth=1)
        plt.legend()
        plt.show()

    def graph_power_spectral_density(self, col_value="value", col_time="timestamp"):
        """Plots the power spectral density of the time series.

        It is estimated using scipy.signal.periodogram

        Args:
            col_value(str): Name of the column containing the values of
                interest. Default to 'value'.
            col_time(str): Name of the column containing the temporal values.
                Default to 'timestamp'.
        """
        for key, cdc in self.dict_df.items():
            cdc = cdc[[col_time, col_value]]
            cdc = cdc.groupby(col_time).mean().reset_index()
            cdc.sort_values(by=col_time, inplace=True)
            ts = cdc[col_value].values
            psd = periodogram(ts, fs=1, scaling="density")
            plt.plot(psd[0], psd[1], alpha=0.4, label=key)
            plt.title("Power Spectral Density")
            plt.xticks(
                ticks=[1 / (48 * 7), 1 / 48, 2 / 48, 4 / 48, 1 / 6, 1 / 4, 1 / 2],
                labels=["1w", "1d", "12h", "6h", "3h", "2h", "1h"],
            )
            plt.xlabel("Frequency")
            # plt.ylim(1e-1, 1e12)
            plt.yscale("log")
            plt.ylabel("Power")
        plt.legend()

    def compute_ft(
        self, norm="forward", col_value="value", col_time="timestamp"
    ) -> "Dict[str, NDArray]":
        """Computes the Fourier transform of the average time-serie
        of each Time Series data.

        Args:
            norm(str): Normalisation mode to pass to torch.fft.rfft
                Default to "forward"
            col_value(str): Name of the column containing the values of
                interest. Default to 'value'.
            col_time(str): Name of the column containing the temporal values.
                Default to 'timestamp'.

        Returns:
            Dict[str, NDArray]: Dictionary df name -> Fourier coefs
        """
        self.validate_column(col_value)
        self.validate_column(col_time)
        dict_ft = dict()
        for key, cdc in self.dict_df.items():
            cdc = cdc[[col_time, col_value]]
            cdc = cdc.groupby(col_time).mean().reset_index()
            cdc.sort_values(by=col_time, inplace=True)
            t_val = torch.tensor(cdc[col_value].values, dtype=torch.float32)
            ft = torch.fft.rfft(t_val, norm=norm)
            dict_ft[key] = ft.numpy()
        return dict_ft

    def graph_ft(self, norm="forward", col_value="value", col_time="timestamp"):
        """Plots the Fourier transform of the average time-serie
        of each Time Series data.

        Args:
            norm(str): Normalisation mode to pass to torch.fft.rfft
                Default to "forward"
            col_value(str): Name of the column containing the values of
                interest. Default to 'value'.
            col_time(str): Name of the column containing the temporal values.
                Default to 'timestamp'.
        """
        plt.subplots(len(self.dict_df), 1, sharex=True, figsize=(8, 5))

        # Fourier transform
        dict_ft = self.compute_ft(norm, col_value, col_time)

        # Plots
        for k, (key, ft) in enumerate(dict_ft.items()):
            plt.subplot(len(self.dict_df), 1, k + 1)
            plt.title(key)
            plt.plot(np.arange(ft.shape[0]), ft)
            plt.yscale("log")
            if k == len(self.dict_df) - 1:
                plt.xlabel("Fourier coefs idx")
            if k == len(self.dict_df) // 2:
                plt.ylabel("Fourier coefs value")
        plt.show()

    def compute_stft(
        self,
        n_fft: int = 48,
        win_length: int = 48,
        hop_length: int = 12,
        normalized: bool = False,
        window=torch.hann_window,
        col_value="value",
        col_time="timestamp",
    ) -> "Tuple[Dict[str, NDArray], Dict[str, NDArray]]":
        """Computes the Short Time Fourier transform and the inverse
        of the average time-serie of each Time Series data.

        Args:
            see torch.fft.stft

            col_value(str): Name of the column containing the values of
                interest. Default to 'value'.
            col_time(str): Name of the column containing the temporal values.
                Default to 'timestamp'.

        Returns:
            Dict[str, NDArray]: Dictionary df name -> STFT coefs
            Dict[str, NDArray]: Dictionary df name -> iSTFT
        """
        self.validate_column(col_value)
        self.validate_column(col_time)
        window = window(win_length)
        spec_kwargs = {
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
            "normalized": normalized,
        }
        spec_kwargs["window"] = torch.hann_window(spec_kwargs["win_length"])

        dict_stft = {}
        dict_istft = {}
        for key, cdc in self.dict_df.items():
            cdc = cdc[[col_time, col_value]]
            cdc = cdc.groupby(col_time).mean().reset_index()
            t_val = torch.tensor(cdc[col_value].values, dtype=torch.float32).unsqueeze(
                0
            )
            spec = torch.stft(t_val, return_complex=True, **spec_kwargs)[0].real

            i_cdc = torch.istft(spec.to(torch.cfloat).unsqueeze(0), **spec_kwargs)[0]
            dict_stft[key] = spec.numpy()
            dict_istft[key] = i_cdc.numpy()
        return dict_stft, dict_istft
    
    def graph_tsne(
        self,
        features: Union[str, Callable] = "raw",
        col_value="value",
        col_time="timestamp",
        col_id: Union[str, List[str]] = "id",
    ):
        """Train a T-SNE representation and plot every data points
        colored by their dataset.

        Args:
            features(str | function, optional): How to extract
                features from the time series. Defaults to raw time series.
            col_value(str): Name of the column containing the values of
                interest. Default to 'value'.
            col_time(str): Name of the column containing the temporal values.
                Default to 'timestamp'.
            col_id(str | list[str]): Name of the column containing the ids.
                Default to 'id'.
        """
        self.validate_column(col_value)
        self.validate_column(col_time)
        ts = []
        keys = []
        for key, df in self.dict_df.items():
            keys.append(key)
            df_ts = (
                df.groupby(col_id)
                .apply(
                    func=partial(
                        self.__collect_timeseries,
                        col_value=col_value,
                        col_time=col_time,
                    )
                )
                .values
            )
            ts.append(np.stack(df_ts, 0))
        X = np.concatenate(ts, 0)


        # TSNE
        tsne = TSNE(n_components=2, random_state=0)
        y = tsne.fit_transform(X)

        # Plot TSNE
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        length = np.array([a.shape[0] for a in ts])
        for i in range(len(ts)):
            axes[0][0].scatter(
                y[sum(length[:i]) : sum(length[: i + 1]), 0],
                y[sum(length[:i]) : sum(length[: i + 1]), 1],
                s=5,
                alpha=0.8,
                label=keys[i],
            )
            sns.kdeplot(x=y[sum(length[:i]) : sum(length[: i + 1]), 0], ax=axes[1][0])
            sns.kdeplot(y=y[sum(length[:i]) : sum(length[: i + 1]), 1], ax=axes[0][1])
            axes[1][1].scatter([], [], label=keys[i])
            sns.kdeplot(
                x=y[sum(length[:i]) : sum(length[: i + 1]), 0],
                y=y[sum(length[:i]) : sum(length[: i + 1]), 1],
                label=keys[i],
                ax=axes[1][1],
            )
        plt.axis("off")
        axes[0][0].set_title("TSNE transformation")
        axes[1][0].set_title("Density x-axis")
        axes[0][1].set_title("Density y-axis")
        axes[1][1].legend(loc="upper left")
        plt.show()
        return fig

    def graph_stft(
        self,
        n_fft: int = 48,
        win_length: int = 48,
        hop_length: int = 6,
        normalized: bool = False,
        window=torch.hann_window,
        col_value="value",
        col_time="timestamp",
    ):
        """Plots the STFT of the average time-serie of each Time Series data.

        Args:
            see torch.fft.stft

            col_value(str): Name of the column containing the values of
                interest. Default to 'value'.
            col_time(str): Name of the column containing the temporal values.
                Default to 'timestamp'.
        """
        self.validate_column(col_value)
        self.validate_column(col_time)

        # STFT
        spec_kwargs = {
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
            "normalized": normalized,
            "window": window,
            "col_value": col_value,
            "col_time": col_time,
        }
        dict_stft, dict_istft = self.compute_stft(**spec_kwargs)

        # STFT Plots
        nrows = len(self.dict_df)
        _, axs = plt.subplots(nrows, 1, sharex=True, figsize=(8, 5))
        for k, (key, stft) in enumerate(dict_stft.items()):
            plt.subplot(nrows, 1, k + 1)
            plt.title(key)
            plt.imshow(
                stft[::-1],
                cmap="cividis",
                interpolation="nearest",
                extent=(0, 10, 0, 2),
                norm="log",
            )
            plt.grid(False)
            if k == len(self.dict_df) - 1:
                plt.xlabel("Time")
                ticks = np.linspace(0, 10, 5)
                labels = self.dict_df[key][col_time].sort_values().values
                idx = (np.linspace(0, 0.99, 5) * labels.shape[0]).astype(np.int64)
                labels = labels[idx]
                labels = [pd.to_datetime(lab) for lab in labels]
                plt.xticks(ticks=ticks, labels=labels, rotation=90)
            if k == len(self.dict_df) // 2:
                plt.ylabel("Frequency")
            plt.yticks(
                ticks=np.linspace(0, 2, 5),
                labels=np.linspace(0, win_length - 1, 5).astype(np.int64),
            )
        plt.colorbar(ax=axs)

        # iSTFT Plots
        plt.figure(figsize=(12, 6))
        for k, (key, istft) in enumerate(dict_istft.items()):
            print(istft.shape)
            sns.lineplot(
                x=self.dict_df[key][col_time].sort_values().unique()[: istft.shape[0]],
                y=istft,
                label=f"Value {key}",
                alpha=0.4,
            )
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xlabel(col_time)
        plt.ylabel(col_value)
        plt.title("Reconstitution after STFT")
        plt.legend()

        plt.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.xticks(rotation=45, ha="right")
        plt.show()

    

    def __collect_timeseries(
        self, group: pd.DataFrame, col_value="value", col_time="timestamp"
    ):
        group = group[[col_time, col_value]]
        group.sort_values(col_time, inplace=True)
        return group[col_value].values

    def get_timeseries(
        self,
        col_value="value",
        col_time="timestamp",
        col_id: Union[str, List[str]] = "id",
    ) -> Dict[str, NDArray]:
        """Return time series as arrays of shape(N, T) containing the values
        of interest s.t. N is the number of different id and T is the number
        of different timestamp.

        Args:
            col_value(str): Name of the column containing the values of
                interest. Default to 'value'.
            col_time(str): Name of the column containing the temporal values.
                Default to 'timestamp'.
            col_id(str | list[str]): Name of the column containing the ids.
                Default to 'id'.
        """
        self.validate_column(col_value)
        self.validate_column(col_time)
        ts = {}
        for key, df in self.dict_df.items():
            df_ts = (
                df.groupby(col_id)
                .apply(
                    func=partial(
                        self.__collect_timeseries,
                        col_value=col_value,
                        col_time=col_time,
                    )
                )
                .values
            )
            ts[key] = np.stack(df_ts, 0)
        return ts

   
    # def graph_feature_distribution(
    #     self,
    #     feature_fn: Callable,
    #     feature_name=None,
    #     col_value="value",
    #     col_time="timestamp",
    #     col_id: Union[str, List[str]] = "id",
    #     use_log_scale: Optional[bool] = True,
    # ):
    #     """Plot the distribution (hist + KDE) of a feature extracted from each dataset.

    #     Args:
    #         feature_fn (Callable): Function to extract the feature from time series.
    #         feature_name (str, optional): Name of the feature.
    #         col_value (str): Column name with time series values.
    #         col_time (str): Column name with timestamps.
    #         col_id (str | list[str]): Column(s) identifying the time series.
    #         use_log_scale (bool): Whether to log-transform the feature before KDE.
    #     """
    #     self.validate_column(col_value)
    #     self.validate_column(col_time)

    #     time_series = self.get_timeseries(
    #         col_value=col_value, col_time=col_time, col_id=col_id
    #     )

    #     fig = plt.figure(figsize=(9, 7))
    #     if feature_name is not None:
    #         plt.title(f"Densité de {feature_name} par courbe")

    #     colors = sns.color_palette("colorblind", len(self.dict_df))

    #     for c, (key, ts) in zip(colors, time_series.items()):
    #         # Feature extraction
    #         feature = feature_fn(ts).reshape(-1)

    #         # Keep only strictly positive values
    #         feature = feature[feature > 0]

    #         if len(feature) == 0:
    #             print(f"Warning: No positive values for {key}, skipping.")
    #             continue
    #         if use_log_scale:
    #             feature_transformed = np.log(feature)
    #             # print(feature_transformed)
    #             feature_transformed = feature_transformed[
    #                 ~np.isnan(feature_transformed)
    #             ]
    #             print("Shape: ", feature_transformed.shape)
    #         else:
    #             feature_transformed = feature

    #         # KDE
    #         bandwidth = feature_transformed.std() / 5
    #         if np.isnan(bandwidth) or bandwidth <= 0:
    #             raise ValueError(f"Invalid bandwidth: {bandwidth}")
    #         kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    #         kde.fit(feature_transformed.reshape(-1, 1))

    #         # Evaluation grid
    #         x_eval = np.linspace(
    #             feature_transformed.min(), feature_transformed.max(), 1000
    #         ).reshape(-1, 1)
    #         log_density = kde.score_samples(x_eval)

    #         if use_log_scale:
    #             x_plot = np.exp(x_eval[:, 0])
    #             density = (
    #                 np.exp(log_density) * x_plot
    #             )  # correction for change of variable
    #         else:
    #             x_plot = x_eval[:, 0]
    #             density = np.exp(log_density)

    #         # Plot
    #         # plt.hist(feature, bins=60, density=True, alpha=0.3, color=c)
    #         hist_vals, bin_edges = np.histogram(
    #             feature_transformed, bins=60, density=True
    #         )

    #         # Get bin centers and convert back to original scale
    #         bin_centers_log = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    #         bin_centers = np.exp(bin_centers_log)

    #         # Apply change-of-variable correction
    #         hist_vals_corrected = hist_vals / bin_centers  # d/dx = 1/x for exp

    #         # Plot corrected histogram in original scale
    #         plt.bar(
    #             bin_centers,
    #             hist_vals_corrected,
    #             width=np.diff(np.exp(bin_edges)),
    #             alpha=0.3,
    #             color=c,
    #             align="center",
    #             edgecolor="none",
    #         )
    #         plt.plot(x_plot, density, label=key, color=c)

    #     plt.xlabel("Valeur (log-transformée)" if use_log_scale else "Valeur")
    #     plt.ylabel("Densité")
    #     plt.legend()
    #     plt.xlim(left=0)

    #     plt.show()
    #     return fig

    def graph_feature_distribution(
        self,
        feature_fn: Callable,
        feature_name=None,
        col_value="value",
        col_time="timestamp",
        col_id: Union[str, List[str]] = "id",
        use_log_scale: Optional[bool] = False,
        clip_percentiles: Tuple[int, int] = (1, 99),
    ):
        """Plot the distribution (hist + KDE) of a feature extracted from each dataset.

        Args:
            feature_fn (Callable): Function to extract the feature from time series.
            feature_name (str, optional): Name of the feature.
            col_value (str): Column name with time series values.
            col_time (str): Column name with timestamps.
            col_id (str | list[str]): Column(s) identifying the time series.
            use_log_scale (bool): Whether to log-transform the feature before KDE.
            clip_percentiles (tuple[int, int]): Lower and upper percentiles for clipping outliers.
        """
        self.validate_column(col_value)
        self.validate_column(col_time)

        time_series = self.get_timeseries(
            col_value=col_value, col_time=col_time, col_id=col_id
        )

        fig = plt.figure(figsize=(9, 7))
        if feature_name is not None:
            plt.title(f"Densité {feature_name} par courbe")

        colors = sns.color_palette("colorblind", len(self.dict_df))

        for c, (key, ts) in zip(colors, time_series.items()):
            # Feature extraction
            feature = feature_fn(ts).reshape(-1)

            # Keep only strictly positive values
            feature = feature[feature > 0]
            if len(feature) == 0:
                print(f"Warning: No positive values for {key}, skipping.")
                continue

            # Log-transform if requested
            feature_transformed = np.log(feature) if use_log_scale else feature
            feature_transformed = feature_transformed[~np.isnan(feature_transformed)]

            # Apply clipping based on percentiles
            lower, upper = np.percentile(feature_transformed, clip_percentiles)
            feature_clipped = feature_transformed[
                (feature_transformed >= lower) & (feature_transformed <= upper)
            ]
            if len(feature_clipped) == 0:
                print(f"Warning: No data left after clipping for {key}, skipping.")
                continue

            # KDE
            bandwidth = feature_clipped.std() / 5
            if np.isnan(bandwidth) or bandwidth <= 0:
                print(f"Warning: Invalid bandwidth for {key}, skipping.")
                continue

            kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
            kde.fit(feature_clipped.reshape(-1, 1))

            # KDE Evaluation Grid (on clipped range)
            x_eval = np.linspace(lower, upper, 1000).reshape(-1, 1)
            log_density = kde.score_samples(x_eval)

            # Transform back to original scale if log was used
            if use_log_scale:
                x_plot = np.exp(x_eval[:, 0])
                density = np.exp(log_density) * x_plot  # correction d/dx = 1/x
            else:
                x_plot = x_eval[:, 0]
                density = np.exp(log_density)

            # Histogram (also on clipped data)
            hist_vals, bin_edges = np.histogram(feature_clipped, bins=60, density=True)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

            if use_log_scale:
                # Convert back to original scale
                bin_centers_orig = np.exp(bin_centers)
                hist_vals_corrected = hist_vals / bin_centers_orig  # d/dx = 1/x
                bar_widths = np.diff(np.exp(bin_edges))
                plt.bar(
                    bin_centers_orig,
                    hist_vals_corrected,
                    width=bar_widths,
                    alpha=0.3,
                    color=c,
                    align="center",
                    edgecolor="none",
                )
            else:
                plt.bar(
                    bin_centers,
                    hist_vals,
                    width=np.diff(bin_edges),
                    alpha=0.3,
                    color=c,
                    align="center",
                    edgecolor="none",
                )

            plt.plot(x_plot, density, label=key, color=c)

        plt.xlabel("Valeur (log-transformée)" if use_log_scale else "Valeur")
        plt.ylabel("Densité")
        plt.legend()
        plt.xlim(left=0)
        plt.tight_layout()
        plt.show()
        return fig

    def graph_mean_distribution(
        self,
        col_value="value",
        col_time="timestamp",
        col_id: Union[str, List[str]] = "id",
    ):
        """Compares distribution of the time series average of each dataset."""

        def feature_fn(x):
            return np.mean(x, 1)

        feature_name = "la moyenne par courbe"
        return self.graph_feature_distribution(
            feature_fn=feature_fn,
            feature_name=feature_name,
            col_value=col_value,
            col_time=col_time,
            col_id=col_id,
        )

    def graph_std_distribution(
        self,
        col_value="value",
        col_time="timestamp",
        col_id: Union[str, List[str]] = "id",
    ):
        """Compares distribution of the time series std of each dataset."""

        def feature_fn(x):
            return np.std(x, 1)

        feature_name = "l'écart-type par courbe"
        return self.graph_feature_distribution(
            feature_fn=feature_fn,
            feature_name=feature_name,
            col_value=col_value,
            col_time=col_time,
            col_id=col_id,
        )

    def graph_quantile_distribution(
        self,
        q: float = 0.5,
        col_value="value",
        col_time="timestamp",
        col_id: Union[str, List[str]] = "id",
    ):
        """Compares distribution of the time series quantile for each dataset.

        Args:
            q (float): quantile to compute (e.g., 0.5 for median).
        """

        def feature_fn(x):
            return np.quantile(x, q, axis=1)

        feature_name = f"le quantile q={q}"
        return self.graph_feature_distribution(
            feature_fn=feature_fn,
            feature_name=feature_name,
            col_value=col_value,
            col_time=col_time,
            col_id=col_id,
        )

    def graph_variations_distribution(
        self,
        col_value="value",
        col_time="timestamp",
        col_id: Union[str, List[str]] = "id",
    ):
        """Compares distribution of the time series mean variations
        of each dataset."""

        def feature_fn(x):
            return np.mean(np.abs(x[:, :-1] - x[:, 1:]), 1)

        feature_name = "la variation moyenne par courbe"
        return self.graph_feature_distribution(
            feature_fn=feature_fn,
            feature_name=feature_name,
            col_value=col_value,
            col_time=col_time,
            col_id=col_id,
        )

    def graph_autocorrelation_distribution(
        self,
        tau: int = 1,
        col_value="value",
        col_time="timestamp",
        col_id: Union[str, List[str]] = "id",
        abs_corr: bool = False,
    ):
        """Compares distribution of (absolute) autocorrelations across datasets.

        Args:
            tau (int): time lag for autocorrelation.
            abs_corr (bool): if True, use absolute autocorrelation.
        """

        def feature_fn(x):
            if x.shape[1] <= tau:
                return np.full(x.shape[0], np.nan)
            ac = x[:, tau:] * x[:, :-tau]
            ac_mean = np.mean(np.abs(ac), axis=1) if abs_corr else np.mean(ac, axis=1)
            return ac_mean

        name = "l'autocorrélation" if not abs_corr else "l'autocorrélation absolue"
        feature_name = f"{name} (tau={tau})"

        return self.graph_feature_distribution(
            feature_fn=feature_fn,
            feature_name=feature_name,
            col_value=col_value,
            col_time=col_time,
            col_id=col_id,
        )

    # R-Clustering
    def __get_data_array_from_df_day(
        self,
        df: pd.DataFrame,
        col_id: str = "id",
        col_value: str = "value",
        col_time: str = "timestamp",
    ) -> np.ndarray:
        """Convert DataFrame to array."""

        # Groupby par période
        df["hour"] = df[col_time].dt.strftime("%H:%M")
        df_groupBy = df.groupby([col_id, "hour"], as_index=False)[col_value].mean()

        # Dataframe with long format to array with wide format
        ts = df_groupBy.groupby(col_id).apply(
            lambda x: x[col_value].values
        )  # long to wide format
        # reset the index and extracts the values
        ts_v = ts.reset_index(name="values")["values"]
        # numpy array with array value for each id
        list_numpy_array = [ts_v[i] for i in range(len(ts_v))]
        X = np.vstack(list_numpy_array)  # numpy matrix

        return X

    def __get_data_array_from_df_week(
        self,
        df: pd.DataFrame,
        col_id: str = "id",
        col_value: str = "value",
        col_time: str = "timestamp",
    ) -> np.ndarray:
        """Convert DataFrame to array."""

        # Ajouter les colonnes 'weekday_num' et 'weekday' basées sur
        # l'horodatage
        # Chiffre entre 1 et 7 pour chaque jour de la semaine
        df["weekday_num"] = df[col_time].dt.dayofweek + 1
        df["weekday"] = df[col_time].dt.strftime(
            "%A"
        )  # Nom complet du jour de la semaine

        # Groupby par période (numéro du jour de la semaine, heure et minute)
        df_groupBy = df.groupby([col_id, "weekday_num"], as_index=False)[
            col_value
        ].mean()

        # Dataframe avec format long vers array avec format large
        ts = df_groupBy.groupby(col_id).apply(
            lambda x: x[col_value].values
        )  # long to wide format
        # Reset de l'index et extraction des valeurs
        ts_v = ts.reset_index(name="values")["values"]
        # numpy array avec array value pour chaque id
        list_numpy_array = [ts_v[i] for i in range(len(ts_v))]
        X = np.vstack(list_numpy_array)  # numpy matrix

        return X

    def __get_data_array_from_df_month(
        self,
        df: pd.DataFrame,
        col_id: str = "id",
        col_value: str = "value",
        col_time: str = "timestamp",
    ) -> np.ndarray:
        """Convert DataFrame to array."""

        # Ajouter la colonne 'month' basée sur l'horodatage
        df["month"] = df[col_time].dt.month  # Numéro du mois

        # Groupby par période (mois)
        df_groupBy = df.groupby([col_id, "month"], as_index=False)[col_value].mean()

        # Dataframe avec format long vers array avec format large
        ts = df_groupBy.groupby(col_id).apply(
            lambda x: x[col_value].values
        )  # long to wide format
        # Reset de l'index et extraction des valeurs
        ts_v = ts.reset_index(name="values")["values"]
        # numpy array avec array value pour chaque id
        list_numpy_array = [ts_v[i] for i in range(len(ts_v))]
        X = np.vstack(list_numpy_array)  # numpy matrix

        return X

    def __get_data_array_from_df_period(
        self,
        df: pd.DataFrame,
        col_id: str = "id",
        col_value: str = "value",
        col_time: str = "timestamp",
    ) -> np.ndarray:
        """Convert DataFrame to array."""

        # Groupby par période
        df["day"] = df[col_time].dt.strftime("%Y-%m-%d")  # Grouper par jour
        df_groupBy = df.groupby([col_id, "day"], as_index=False)[col_value].mean()

        # Dataframe with long format to array with wide format
        ts = df_groupBy.groupby(col_id).apply(
            lambda x: x[col_value].values
        )  # long to wide format
        # reset the index and extracts the values
        ts_v = ts.reset_index(name="values")["values"]
        # numpy array with array value for each id
        list_numpy_array = [ts_v[i] for i in range(len(ts_v))]
        X = np.vstack(list_numpy_array)  # numpy matrix

        return X

    def __get_data_array_from_df(
        self, df: pd.DataFrame, col_id: str = "id", col_value: str = "value"
    ) -> np.ndarray:
        """Convert DataFrame to array."""

        # Dataframe with long format to array with wide format
        ts = df.groupby(col_id).apply(
            lambda x: x[col_value].values
        )  # long to wide format
        # reset the index and extracts the values
        ts_v = ts.reset_index(name=col_value)[col_value]
        # numpy array with array value for each id
        list_numpy_array = [ts_v[i] for i in range(len(ts_v))]
        X = np.vstack(list_numpy_array)  # numpy matrix

        return X

    def __get_compute_center(self, df: pd.DataFrame) -> np.ndarray:
        """Calculates and converts in sktime format, the cluster center at
        each time step."""

        return convert(df, from_type="nested_univ", to_type="numpy3D").mean(axis=0)[0]

    def __get_cluster_centers(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Obtain the centroids of each cluster at each given time (at time step)."""

        X_nested = convert(
            np.expand_dims(data, axis=1), from_type="numpy3D", to_type="nested_univ"
        )
        df_X_labels = X_nested.merge(
            pd.DataFrame(labels, columns=["labels"]), left_index=True, right_index=True
        )  # Merge data with labels
        # Group the data by labelsand  calculate the center of each cluster
        df_center = df_X_labels.groupby("labels").apply(self.__get_compute_center)
        cluster_centers = np.expand_dims(
            np.stack([c for c in df_center.values]), axis=1
        )

        return cluster_centers

    def __get_norm_data(self, X: np.ndarray) -> np.ndarray:
        """Standardization"""

        mean_values = X.mean(axis=1)
        mean_values = mean_values[:, np.newaxis]
        epsilon = 1e-9  # Pas de division par zéro
        X_norm = X / (mean_values + epsilon)

        return X_norm

    def __get_methode_elbow(
        self, X_acp: np.ndarray, k_choices: List[int]
    ) -> Tuple[List[float], List[int], List[float]]:
        """Compute the elbow method"""

        inertia_list = []
        iter_list = []
        silhouette_list = []

        for k in k_choices:
            k_means_opti = TimeSeriesKMeans(
                n_clusters=k,
                init_algorithm="kmeans++",
                n_init=15,
                max_iter=500,
                metric="euclidean",
                averaging_method="mean",
                random_state=1,
            )
            k_means_opti.fit(X_acp)
            inertia_list.append(k_means_opti.inertia_)
            iter_list.append(k_means_opti.n_iter_)
            silhouette_list.append(silhouette_score(X_acp, k_means_opti.labels_))

        return inertia_list, iter_list, silhouette_list

    def __show_k_method_clustering(
        self, X_acp: np.ndarray, k_choices: List[int]
    ) -> None:
        """Plot the elbow method"""

        # Get the results of the method
        inertia_list, iter_list, silhouette_list = self.__get_methode_elbow(
            X_acp, k_choices
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=k_choices, y=inertia_list, mode="lines+markers"))
        fig.update_layout(
            title="Elbow Method",
            xaxis_title="Number of clusters (k)",
            yaxis_title="Inertia",
            template="plotly_white",
        )

        fig.show()

    def __get_silhouette_score(self, X: np.ndarray, kmeans_labels: np.ndarray) -> float:
        """Compute silhouette score."""

        score = silhouette_score(X, kmeans_labels)
        print(f"Silhouette score : {score}")

        return score

    def __get_features(
        self,
        col_id: str = "id",
        col_value: str = "value",
        col_time: str = "timestamp",
        clustering_day_called: bool = False,
        clustering_week_called: bool = False,
        clustering_month_called: bool = False,
        clustering_period_called: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Feature extraction and dimensionality reduction."""
        # Store the results in dictionaries
        value_array = {}
        data_reduced = {}
        value_data = {}
        value_data_norm = {}

        for key, value in self.dict_df.items():
            if col_time not in value.columns:
                raise ValueError(
                    f"Column '{col_time}' not found in DataFrame for key '{key}'"
                )

            if clustering_day_called:
                value_array[key] = self.__get_data_array_from_df_day(
                    value, col_id, col_value, col_time
                )
            elif clustering_week_called:
                value_array[key] = self.__get_data_array_from_df_week(
                    value, col_id, col_value, col_time
                )
            elif clustering_month_called:
                value_array[key] = self.__get_data_array_from_df_month(
                    value, col_id, col_value, col_time
                )
            elif clustering_period_called:
                value_array[key] = self.__get_data_array_from_df_period(
                    value, col_id, col_value, col_time
                )

            value_data[key] = self.__get_data_array_from_df(value, col_id, col_value)

            # Feature normalization
            value_data_norm[key] = self.__get_norm_data(value_data[key])

            data_nested = convert(
                np.expand_dims(value_data[key], axis=1),
                from_type="numpy3D",
                to_type="nested_univ",
            )

            # Feature extraction using 500 convolution kernels
            minirocket = MiniRocket(num_kernels=500)
            minirocket.fit(data_nested)
            data_extract = minirocket.transform(data_nested)

            # Standardization
            scaler = StandardScaler()
            X_std = scaler.fit_transform(data_extract)

            # Principal component analysis
            pca = PCA().fit(X_std)
            optimal_dimensions = np.argmax(pca.explained_variance_ratio_ < 0.01)
            pca_optimal = PCA(n_components=optimal_dimensions)
            data_reduced[key] = pca_optimal.fit_transform(X_std)

        return value_array, data_reduced, value_data_norm

    def __get_size_clusters(
        self, cluster_labels: np.ndarray, nb_clusters: int
    ) -> pd.DataFrame:
        """Cluster sizes from cluster labels."""
        # Size of each cluster
        size_clusters = (
            pd.DataFrame(cluster_labels, columns=["labels"]).groupby("labels").size()
        )
        n_cluster = size_clusters.shape[0]  # number of clusters

        df_cluster_size = pd.DataFrame(data={"size": size_clusters.values})
        list_title_cluster = ["Cluster " + str(c + 1) for c in range(n_cluster)]
        df_cluster_size["cluster_name"] = list_title_cluster

        return df_cluster_size

    def __get_perform_clustering(
        self,
        n_clusters: Union[int, str],
        col_id: str = "id",
        col_value: str = "value",
        col_time: str = "timestamp",
        clustering_day_called: bool = False,
        clustering_week_called: bool = False,
        clustering_month_called: bool = False,
        clustering_period_called: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Clustering par Kmeans"""

        value_array, data_reduced, value_data_norm = self.__get_features(
            col_id,
            col_value,
            col_time,
            clustering_day_called,
            clustering_week_called,
            clustering_month_called,
            clustering_period_called,
        )
        graph_number = 1

        clusters_centers_all = {}
        cluster_sizes_all = {}

        for (
            (key_array, value_array),
            (key_data_reduced, value_data_reduced),
            (key_value, value_data_all),
        ) in zip(value_array.items(), data_reduced.items(), value_data_norm.items()):
            k_means = TimeSeriesKMeans(
                n_clusters=n_clusters,
                init_algorithm="kmeans++",
                n_init=15,
                max_iter=500,
                metric="euclidean",
                averaging_method="mean",
                random_state=1,
            )

            k_means.fit(value_data_reduced)

            silhouette = self.__get_silhouette_score(value_data_all, k_means.labels_)

            df_cluster_size = self.__get_size_clusters(k_means.labels_, n_clusters)

            value_df_data_day = self.__get_norm_data(value_array)

            clusters_centers_all[key_array] = self.__get_cluster_centers(
                value_df_data_day, k_means.labels_
            )

            # Stocker les tailles des clusters
            cluster_sizes_all[key_array] = df_cluster_size

            graph_number += 1

        if len(clusters_centers_all) > 1:
            self.__get_pairwise_distance(
                clusters_centers_all,
                cluster_sizes_all,
                k_means,
                0,
                value_df_data_day,
                graph_number,
                clustering_day_called,
                clustering_week_called,
                clustering_month_called,
                clustering_period_called,
            )
        else:
            self.__get_layout_div_cluster(
                k_means,
                0,
                value_df_data_day,
                clusters_centers_all[key_array],
                df_cluster_size,
                graph_number,
                clustering_day_called,
                clustering_week_called,
                clustering_month_called,
                clustering_period_called,
            )

            graph_number += 1

        return clusters_centers_all

    def __get_pairwise_distance(
        self,
        clusters_centers: Dict[str, np.ndarray],
        cluster_sizes: Dict[str, pd.DataFrame],
        kmeans_model: KMeans,
        nb_samples: int,
        cdc_data: np.ndarray,
        graph_number: int,
        clustering_day_called: bool = False,
        clustering_week_called: bool = False,
        clustering_month_called: bool = False,
        clustering_period_called: bool = False,
    ) -> None:
        min_scores = {}
        centre_clusters_all = []

        centers_names = list(clusters_centers.keys())
        centers_values = list(clusters_centers.values())

        for length in range(len(clusters_centers) - 1):
            num_rows = len(centers_values[0])
            num_cols = len(centers_values[length + 1])

            # Initialize matrix to store distances
            distance_matrix = np.zeros((num_rows, num_cols))
            # Initial global minimum distance
            min_global_distance = float("inf")
            distances = []  # Store distances in a list for constructing the DataFrame

            for i, center1 in enumerate(centers_values[0]):
                min_distance = float("inf")
                min_center = None
                min_cluster_pair = None
                row_distances = []  # Store distances for this row `i`

                for j, center2 in enumerate(centers_values[length + 1]):
                    min_pair_distance = float("inf")

                    # Cartesian product of two sets: every possible combination
                    for point1, point2 in product(center1, center2):
                        distance_scalar = pairwise_distances([point1], [point2])[0][0]

                    row_distances.append(distance_scalar)

                distances.append(row_distances)

            # Dataframe : distance dim(j,i)
            df_distance = pd.DataFrame(
                distances, index=range(num_rows), columns=range(num_cols)
            )
            df_distance = df_distance.T

            orig_row_indices = list(range(num_rows))
            orig_col_indices = list(range(num_cols))

            # Lists
            deleted_clusters = []
            size_clusters = []  # Initialize a list to store sizes

            # Tant que la matrice transposée n'est pas de dim(1,1)
            while df_distance.shape[0] > 1 and df_distance.shape[1] > 1:
                # Valeur minimale de la matrice : on sélectionne l'indice
                min_indices = np.unravel_index(
                    np.argmin(df_distance.values), df_distance.shape
                )

                # Ligne et colonne de la valeur minimale de la matrice
                min_row_name = orig_row_indices[df_distance.index[min_indices[0]]]
                min_col_name = orig_col_indices[df_distance.columns[min_indices[1]]]

                # Store the corresponding cluster centers
                deleted_clusters.append(
                    (
                        centers_values[0][min_col_name].ravel(),
                        centers_values[length + 1][min_row_name].ravel(),
                    )
                )

                # Print the selected pairs of clusters with their sizes
                size_cluster1 = cluster_sizes[centers_names[0]].iloc[min_col_name][
                    "size"
                ]
                size_cluster2 = cluster_sizes[centers_names[length + 1]].iloc[
                    min_row_name
                ]["size"]

                # Store the corresponding cluster sizes
                size_clusters.append((size_cluster1, size_cluster2))

                # Remove the row and column corresponding to the minimum value
                df_distance = df_distance.drop(index=min_row_name, columns=min_col_name)

            if df_distance.shape[0] == 1 and df_distance.shape[1] == 1:
                min_row_name = orig_row_indices[df_distance.index[0]]
                min_col_name = orig_col_indices[df_distance.columns[0]]

                deleted_clusters.append(
                    (
                        centers_values[0][min_col_name].ravel(),
                        centers_values[length + 1][min_row_name].ravel(),
                    )
                )

                # Print the selected pairs of clusters with their sizes
                size_cluster1 = cluster_sizes[centers_names[0]].iloc[min_col_name][
                    "size"
                ]
                size_cluster2 = cluster_sizes[centers_names[length + 1]].iloc[
                    min_row_name
                ]["size"]

                # Store the corresponding cluster sizes
                size_clusters.append((size_cluster1, size_cluster2))

            self.__get_pairwise_layout_div_cluster(
                kmeans_model,
                centers_names,
                nb_samples,
                cdc_data,
                deleted_clusters,
                size_clusters,
                graph_number,
                clustering_day_called,
                clustering_week_called,
                clustering_month_called,
                clustering_period_called,
            )

    def __get_pairwise_layout_div_cluster(
        self,
        kmeans_model: KMeans,
        centers_names: List[str],
        nb_samples: int,
        cdc_data: np.ndarray,
        cluster_centers: List[Tuple[np.ndarray, np.ndarray]],
        size_clusters: List[Tuple[int, int]],
        graph_number: int,
        clustering_day_called: bool = False,
        clustering_week_called: bool = False,
        clustering_month_called: bool = False,
        clustering_period_called: bool = False,
    ) -> None:
        """
        Visualization of clustering generated by a K-means clustering model
        """

        # Number of clusters
        nb_clusters = kmeans_model.cluster_centers_.shape[0]

        # Generate subplot titles with cluster sizes
        subplot_titles = []
        for c, (cluster_pair, (size_cluster1, size_cluster2)) in enumerate(
            zip(cluster_centers, size_clusters)
        ):
            title = f"<span style='font-size:140%'>Cluster {c + 1} - Size :</span><br><span style='font-size:90%'>   {size_cluster1} curves of {centers_names[0]}</span><br><span style='font-size:90%'>   {size_cluster2} curves of {centers_names[1]}</span>"
            subplot_titles.append(title)

        fig_clusters = make_subplots(
            # Increased vertical spacing for better separation
            rows=nb_clusters,
            cols=1,
            vertical_spacing=0.06,
            subplot_titles=subplot_titles,
        )

        # Define different color schemes for each sub-cluster
        colors = ["#ff7f0e", "#1f77b4"]

        # Create an empty list to store unique cluster center names
        cluster_names = []

        # Determine the title based on the aggregation type called
        if clustering_day_called:
            title_text = ""
        elif clustering_week_called:
            title_text = ""
        elif clustering_month_called:
            title_text = ""
        elif clustering_period_called:
            title_text = ""
        else:
            title_text = "R-Clustering"

        # Loop to generate the plots
        for c, cluster_pair in enumerate(cluster_centers):
            # Plot cluster centers for each pair
            for i, cluster_center in enumerate(cluster_pair):
                cluster_name = f"Cluster centers : {centers_names[i]}"
                if cluster_name not in cluster_names:
                    cluster_names.append(cluster_name)
                    showlegend = True
                else:
                    showlegend = False
                fig_clusters.add_trace(
                    go.Scatter(
                        y=cluster_center,
                        line={"color": colors[i % len(colors)], "width": 2},
                        name=cluster_name,
                        showlegend=showlegend,
                    ),
                    row=c + 1,
                    col=1,
                )

            # Add the average curve
            if c == 0:  # Only show legend for the first subplot
                fig_clusters.add_trace(
                    go.Scatter(
                        y=cdc_data.mean(axis=0),
                        line={"color": "black", "width": 2, "dash": "dot"},
                        name="Average of all cluster centers",
                        showlegend=True,
                    ),
                    row=c + 1,
                    col=1,
                )
            else:
                fig_clusters.add_trace(
                    go.Scatter(
                        y=cdc_data.mean(axis=0),
                        line={"color": "black", "width": 2, "dash": "dot"},
                        name="Average of all cluster centers",
                        showlegend=False,  # Hide legend for other subplots
                    ),
                    row=c + 1,
                    col=1,
                )

        # Update layout
        fig_clusters.update_layout(
            height=1800,
            width=1200,
            title={
                "text": title_text,
                "y": 0.99,  # Adjust y position of the title for more spacing
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": {
                    "size": 50  # Increase the font size of the title
                    # Change title color to make it more prominent
                },
            },
            # Add top margin to create space between the title and subplots
            margin={"t": 150},
            legend={
                "x": 1.02,  # Place the legend to the right of the graph
                "y": 1,
                "traceorder": "normal",
                "font": {"size": 20},  # Set the font size of the legend
            },
            showlegend=True,
        )

        # Add gridlines to the subplots
        for i in range(0, nb_clusters + 1):
            fig_clusters.update_xaxes(showgrid=True, row=i, col=1)
            fig_clusters.update_yaxes(showgrid=True, row=i, col=1)

        fig_clusters.show()

    def __get_layout_div_cluster(
        self,
        kmeans_model: KMeans,
        nb_samples: int,
        cdc_data: np.ndarray,
        cluster_centers: np.ndarray,
        df_cluster_size: pd.DataFrame,
        graph_number: int,
        clustering_day_called: bool = False,
        clustering_week_called: bool = False,
        clustering_month_called: bool = False,
        clustering_period_called: bool = False,
    ) -> None:
        """Plot the clusters."""

        # number of clusters
        nb_clusters = kmeans_model.cluster_centers_.shape[0]
        fig_clusters = go.Figure()

        colors = px.colors.qualitative.G10

        # Plot the average curve of all clusters
        fig_clusters.add_trace(
            go.Scatter(
                y=cdc_data.mean(axis=0),
                line={"color": "black", "width": 2, "dash": "dot"},
                name="Average of all cluster centers",
            )
        )

        # Cross each cluster
        for c in range(nb_clusters):
            cluster_size = df_cluster_size.loc[
                df_cluster_size["cluster_name"] == f"Cluster {c + 1}", "size"
            ].iloc[0]

            # Plot the cluster center
            fig_clusters.add_trace(
                go.Scatter(
                    y=cluster_centers[c][0],
                    line={"color": colors[c % len(colors)], "width": 2},
                    name=f"Cluster Center {c + 1} - Size: {cluster_size}",
                )
            )

            # Select a random sample of points in the cluster
            index_c = list(np.where(kmeans_model.labels_ == c)[0])
            list_samples = random.sample(index_c, min(nb_samples, len(index_c)))

            # Plot the samples
            for index_sample in list_samples:
                fig_clusters.add_trace(
                    go.Scatter(
                        y=cdc_data[index_sample],
                        line={"color": colors[c % len(colors)]},
                        opacity=0.4,
                        showlegend=False,
                    )
                )

        fig_clusters.update_layout(
            title="R-Clustering",
            xaxis_title="Temporal Index",
            yaxis_title="Value",
            height=600,
            width=1000,
        )

        fig_clusters.show()

    def clustering_day(
        self,
        n_clusters: Union[int, str],
        col_id: str = "id",
        col_value: str = "value",
        col_time: str = "timestamp",
    ) -> None:
        """The clustering method is designed to facilitate the evaluation of
        synthetic time series data through clustering.

        The combines the use of convolutional kernels to extract features from time series
        with Principal Component Analysis (PCA) to reduce the dimensionality of the data.
        By employing the KMeans algorithm, it performs clustering of the time series to
        enable in-depth analysis and interpretation of the data.

        Parameters
        ----------
        n_clusters : int, 'elbow'
            The number of centroids to initialize or return the
            results of the elbow method to aid in determining the optimal
            number of clusters.

        Returns
        -------
        Graphs representing the clusters are generated, allowing for intuitive
        visualization of clustering results.
        Optionally, the method can also return the results of the elbow method
        to aid in determining the optimal number of clusters.
        """
        self.validate_column(col_value)
        self.validate_column(col_time)
        self.__clustering_common(
            n_clusters, col_id, col_value, col_time, clustering_day_called=True
        )

    def clustering_week(
        self,
        n_clusters: Union[int, str],
        col_id: str = "id",
        col_value: str = "value",
        col_time: str = "timestamp",
    ) -> None:
        """The clustering method is designed to facilitate the evaluation of
        synthetic time series data through clustering.

        The combines the use of convolutional kernels to extract features from time series
        with Principal Component Analysis (PCA) to reduce the dimensionality of the data.
        By employing the KMeans algorithm, it performs clustering of the time series to
        enable in-depth analysis and
        retation of the data.

        Parameters
        ----------
        n_clusters : int, 'elbow'
            The number of centroids to initialize or return the
            results of the elbow method to aid in determining the optimal number of clusters.

        Returns
        -------
        Graphs representing the clusters are generated, allowing for intuitive visualization of clustering results.
        Optionally, the method can also return the results of the elbow method
        to aid in determining the optimal number of clusters.
        """
        self.validate_column(col_value)
        self.validate_column(col_time)
        self.__clustering_common(
            n_clusters, col_id, col_value, col_time, clustering_week_called=True
        )

    def clustering_month(
        self,
        n_clusters: Union[int, str],
        col_id: str = "id",
        col_value: str = "value",
        col_time: str = "timestamp",
    ) -> None:
        """The clustering method is designed to facilitate the evaluation of
        synthetic time series data through clustering.

        The combines the use of convolutional kernels to extract features from time series
        with Principal Component Analysis (PCA) to reduce the dimensionality of the data.
        By employing the KMeans algorithm, it performs clustering of the time series to
        enable in-depth analysis and interpretation of the data.

        Parameters
        ----------
        n_clusters : int, 'elbow'
            The number of centroids to initialize or return the
            results of the elbow method to aid in determining the optimal
            number of clusters.

        Returns
        -------
        Graphs representing the clusters are generated, allowing for intuitive
        visualization of clustering results.
        Optionally, the method can also return the results of the elbow method
        to aid in determining the optimal number of clusters.
        """
        self.validate_column(col_value)
        self.validate_column(col_time)
        self.__clustering_common(
            n_clusters, col_id, col_value, col_time, clustering_month_called=True
        )

    def clustering_period(
        self,
        n_clusters: Union[int, str],
        col_id: str = "id",
        col_value: str = "value",
        col_time: str = "timestamp",
    ) -> None:
        """The clustering method is designed to facilitate the evaluation of
        synthetic time series data through clustering.

        The combines the use of convolutional kernels to extract features from time series
        with Principal Component Analysis (PCA) to reduce the dimensionality of the data.
        By employing the KMeans algorithm, it performs clustering of the time series to
        enable in-depth analysis and interpretation of the data.

        Parameters
        ----------
        n_clusters : int, 'elbow'
            The number of centroids to initialize or return the
            results of the elbow method to aid in determining the optimal
            number of clusters.

        Returns
        -------
        Graphs representing the clusters are generated, allowing for intuitive
        visualization of clustering results.
        Optionally, the method can also return the results of the elbow method
        to aid in determining the optimal number of clusters.
        """
        self.validate_column(col_value)
        self.validate_column(col_time)
        self.__clustering_common(
            n_clusters, col_id, col_value, col_time, clustering_period_called=True
        )

    def __clustering_common(
        self,
        n_clusters: Union[int, str],
        col_id: str,
        col_value: str,
        col_time: str,
        clustering_day_called: bool = False,
        clustering_week_called: bool = False,
        clustering_month_called: bool = False,
        clustering_period_called: bool = False,
    ) -> None:
        """Common logic for clustering methods."""

        if isinstance(n_clusters, int):
            self.__get_perform_clustering(
                n_clusters,
                col_id,
                col_value,
                col_time,
                clustering_day_called=clustering_day_called,
                clustering_week_called=clustering_week_called,
                clustering_month_called=clustering_month_called,
                clustering_period_called=clustering_period_called,
            )
        elif n_clusters == "elbow":
            # Pass the column arguments to __get_features
            value_array, data_reduced = self.__get_features(
                col_id,
                col_value,
                col_time,
                clustering_day_called=clustering_day_called,
                clustering_week_called=clustering_week_called,
                clustering_month_called=clustering_month_called,
                clustering_period_called=clustering_period_called,
            )
            for key, value in data_reduced.items():
                self.__show_k_method_clustering(value, k_choices=list(range(2, 9)))
        else:
            raise ValueError(
                "Invalid value for nb_clusters. Please provide an integer or 'elbow'."
            )
