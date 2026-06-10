from __future__ import annotations

import datetime
import os
import time
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

from .discriminative_metrics import compute_discriminative_metrics
from .fidelity_metrics import compute_fidelity_metrics
from .privacy_metrics import compute_privacy_metrics
from .dispare.timeseriesdata import TimeSeriesData
from .features_extractor import BaseFeaturesExtractor, ROCKET


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    metrics: dict[str, Any]
    plot_paths: dict[str, str]


def _to_numpy_2d(data: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    array = np.asarray(data)
    if array.ndim == 3 and array.shape[1] == 1:
        array = array[:, 0, :]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array of shape (B, L). Got {array.shape}.")
    return array


def _flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> Iterable[tuple[str, float]]:
    for key, value in metrics.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            yield from _flatten_metrics(value, name)
        else:
            try:
                yield name, float(value)
            except (TypeError, ValueError):
                continue


class ReportBuilder:
    _PLOT_REGISTRY: dict[str, Callable[[TimeSeriesData], plt.Figure]] = {
        "tsne": lambda tsd: tsd.graph_tsne(),
        "mean_distribution": lambda tsd: tsd.graph_mean_distribution(),
        "quantile_0.2": lambda tsd: tsd.graph_quantile_distribution(q=0.2),
        "quantile_0.5": lambda tsd: tsd.graph_quantile_distribution(q=0.5),
        "quantile_0.8": lambda tsd: tsd.graph_quantile_distribution(q=0.8),
        "profile": lambda tsd: tsd.graph_profile(),
        "daily_profile": lambda tsd: tsd.graph_daily_profile(),
        "daily_energy_profile": lambda tsd: tsd.graph_daily_energy_profile(),
        "weekly_profile": lambda tsd: tsd.graph_weekly_profile(),
        "monthly_profile": lambda tsd: tsd.graph_monthly_profile(),
        "acd_lag1": lambda tsd: tsd.graph_autocorrelation_distribution(tau=1),
        "acd_lag48": lambda tsd: tsd.graph_autocorrelation_distribution(tau=48),
    }

    _PLOT_SETS: dict[str, tuple[str, ...]] = {
        "none": tuple(),
        "summary": ("mean_distribution", "profile", "daily_profile"),
        "full": tuple(_PLOT_REGISTRY.keys()),
    }

    def __init__(self, start_date: str = "01/01/2024") -> None:
        self.start_date = datetime.datetime.strptime(start_date, "%d/%m/%Y")

    def compute_metrics(
        self,
        real_data: np.ndarray,
        synth_data: np.ndarray,
        real_data_train: np.ndarray,
        features_extractor: torch.nn.Module | None,
    ) -> dict[str, Any]:
        logger.info(
            "compute_metrics: real_test=%s synth=%s real_train=%s",
            tuple(real_data.shape),
            tuple(synth_data.shape),
            tuple(real_data_train.shape),
        )

        # -------------------------
        # privacy
        # -------------------------
        t0 = time.perf_counter()
        logger.info("Starting compute_privacy_metrics ...")
        privacy_metrics = compute_privacy_metrics(
            real_data_train=real_data_train,
            real_data_test=real_data,
            synthetic_data=synth_data,
            metrics=["ims", "dcr", "nndr", "authenticity", "neighbors_privacy"],
        )
        logger.info("Finished compute_privacy_metrics in %.3fs", time.perf_counter() - t0)

        # -------------------------
        # discriminative
        # -------------------------
        t0 = time.perf_counter()
        logger.info("Starting compute_discriminative_metrics ...")
        discriminative_metrics = compute_discriminative_metrics(
            real_data=real_data,
            synth_data=synth_data,
        )
        logger.info("Finished compute_discriminative_metrics in %.3fs", time.perf_counter() - t0)

        # -------------------------
        # fidelity
        # -------------------------
        fx_name = type(features_extractor).__name__ if features_extractor is not None else "None"
        logger.info("Fidelity features_extractor=%s", fx_name)

        t0 = time.perf_counter()
        logger.info("Starting compute_fidelity_metrics ...")
        fidelity_metrics = compute_fidelity_metrics(
            real_data=real_data,
            synth_data=synth_data,
            features_extractor=features_extractor,
        )
        logger.info("Finished compute_fidelity_metrics in %.3fs", time.perf_counter() - t0)

        return {
            "privacy_metrics": privacy_metrics,
            "discriminative_metrics": discriminative_metrics,
            "fidelity_metrics": fidelity_metrics,
        }

    def _build_tsd(self, real_data: np.ndarray, synth_data: np.ndarray) -> TimeSeriesData:
        gen_data_df = TimeSeriesData.array_to_df(synth_data, min_time=self.start_date)
        true_data_df = TimeSeriesData.array_to_df(real_data, min_time=self.start_date)
        return TimeSeriesData({"real": true_data_df, "fake": gen_data_df})

    def generate_plots(
        self,
        real_data: np.ndarray,
        synth_data: np.ndarray,
        plot_set: str | Iterable[str] = "full",
        output_dir: str | None = None,
        writer: Any | None = None,
        global_step: int | None = None,
        log_prefix: str = "evaluation",
    ) -> dict[str, str]:
        if isinstance(plot_set, str):
            plot_names = self._PLOT_SETS.get(plot_set, (plot_set,))
        else:
            plot_names = tuple(plot_set)

        if not plot_names:
            return {}

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        tsd = self._build_tsd(real_data, synth_data)
        saved_paths: dict[str, str] = {}

        logger.info("Generating plots: %s", ", ".join(plot_names))
        for name in plot_names:
            plot_fn = self._PLOT_REGISTRY.get(name)
            if plot_fn is None:
                logger.warning("Unknown plot name '%s' (skipping)", name)
                continue

            t0 = time.perf_counter()
            logger.info("Starting plot '%s' ...", name)
            fig = plot_fn(tsd)
            logger.info("Finished plot '%s' in %.3fs", name, time.perf_counter() - t0)

            if output_dir:
                file_path = os.path.join(output_dir, f"{name}.png")
                fig.savefig(file_path)
                saved_paths[name] = file_path

            if writer is not None:
                writer.add_figure(f"{log_prefix}/{name}", fig, global_step=global_step)

            plt.close(fig)

        return saved_paths

    def log_metrics(
        self,
        metrics: dict[str, Any],
        writer: Any,
        global_step: int | None = None,
        prefix: str = "metrics",
    ) -> None:
        for name, value in _flatten_metrics(metrics, prefix):
            writer.add_scalar(name, value, global_step=global_step)


def _jsonable(x: Any) -> Any:
    """Best-effort conversion to JSON-serializable Python types."""
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _save_metrics_json(metrics: dict[str, Any], output_dir: str, filename: str = "metrics.json") -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(metrics), f, indent=2)
    return out_path


def compute_report(
    real_data: np.ndarray | torch.Tensor,
    synth_data: np.ndarray | torch.Tensor,
    real_data_train: np.ndarray | torch.Tensor,
    start_date: str = "01/01/2024",
    features_extractor: torch.nn.Module | BaseFeaturesExtractor | None = None,
    output_dir: str | None = None,
    writer: Any | None = None,
    global_step: int | None = None,
    plot_set: str | Iterable[str] = "full",
    log_metrics: bool = True,
    log_plots: bool = True,
    return_report: bool = False,
) -> dict[str, Any] | EvaluationReport:
    """
    Master switches:
      - log_metrics: if False => skip computing metrics entirely (no TB logging, no file saving, return {}).
      - log_plots  : if False => skip generating plots entirely (no TB logging, no png saving, return {} plot_paths).

    Persistence:
      - If writer is provided, metrics/plots are logged to TensorBoard (depending on switches).
      - If output_dir (or path) is provided, plots are saved as PNGs (depending on log_plots),
        and metrics are saved as JSON (depending on log_metrics).
    """
    logger.info("Starting compute_report ...")

    real_data_np = _to_numpy_2d(real_data, "real_data")
    synth_data_np = _to_numpy_2d(synth_data, "synth_data")
    real_data_train_np = _to_numpy_2d(real_data_train, "real_data_train")

    report_builder = ReportBuilder(start_date=start_date)

    metrics: dict[str, Any] = {}
    metrics_output_dir = output_dir

    # ----------------------------
    # Metrics: compute + persist only if log_metrics=True
    # ----------------------------
    if log_metrics:
        t0 = time.perf_counter()
        logger.info("Computing metrics block ...")
        metrics = report_builder.compute_metrics(
            real_data=real_data_np,
            synth_data=synth_data_np,
            real_data_train=real_data_train_np,
            features_extractor=features_extractor,
        )
        logger.info("Finished metrics block in %.3fs", time.perf_counter() - t0)

        # TensorBoard scalars
        if writer is not None:
            report_builder.log_metrics(metrics, writer, global_step=global_step)

        # Optional JSON save
        if metrics_output_dir is not None:
            metrics_path = _save_metrics_json(metrics, metrics_output_dir, filename="metrics.json")
            logger.info("Saved metrics JSON to %s", metrics_path)
    else:
        logger.info("Skipping metrics computation (log_metrics=False).")

    # ----------------------------
    # Plots: generate + persist only if log_plots=True
    # ----------------------------
    plot_paths: dict[str, str] = {}
    plot_output_dir = output_dir

    if log_plots and (plot_output_dir is not None or writer is not None):
        t0 = time.perf_counter()
        logger.info("Generating plots block ...")
        plot_paths = report_builder.generate_plots(
            real_data=real_data_np,
            synth_data=synth_data_np,
            plot_set=plot_set,
            output_dir=plot_output_dir,   # saves PNGs if not None
            writer=writer,                # logs figures if not None
            global_step=global_step,
        )
        logger.info("Finished plots block in %.3fs", time.perf_counter() - t0)
    else:
        if not log_plots:
            logger.info("Skipping plot generation (log_plots=False).")
        else:
            logger.info("Skipping plot generation (no writer and no output_dir/path).")

    logger.info("compute_report done.")

    if return_report:
        return EvaluationReport(metrics=metrics, plot_paths=plot_paths)
    return metrics



if __name__ == "__main__":
    import numpy as np
    import torch

    # ----------------------------
    # Minimal synthetic test data
    # ----------------------------
    rng = np.random.default_rng(0)
    B, L = 64, 17520  # batch size, sequence length

    # "Train real" data
    real_train = rng.normal(loc=0.0, scale=1.0, size=(B, L)).astype(np.float32)

    # "Test real" data (slightly shifted)
    real_test = rng.normal(loc=0.1, scale=1.0, size=(B, L)).astype(np.float32)

    # "Synthetic" data (different distribution)
    synth = rng.normal(loc=0.2, scale=1.2, size=(B, L)).astype(np.float32)

    # Optionally test torch inputs (your compute_report accepts both)
    real_train_t = torch.from_numpy(real_train)
    real_test_t = torch.from_numpy(real_test)
    synth_t = torch.from_numpy(synth)

    # ----------------------------
    # Run evaluation
    # ----------------------------
    out_dir = "_eval_test_outputs"

    report = compute_report(
        real_data=real_test_t,
        synth_data=synth_t,
        real_data_train=real_train_t,
        start_date="01/01/2024",
        features_extractor=ROCKET(),
        output_dir=out_dir,
        plot_set="none",
        log_metrics=False,
        log_plots=True,
        return_report=True,
    )

    # ----------------------------
    # Display results
    # ----------------------------
    print("\n=== Metrics (top-level) ===")
    for k, v in report.metrics.items():
        print(f"{k}: {v}")

    print("\n=== Saved plots ===")
    if report.plot_paths:
        for name, path in report.plot_paths.items():
            print(f"{name}: {path}")
    else:
        print("(no plots saved)")
