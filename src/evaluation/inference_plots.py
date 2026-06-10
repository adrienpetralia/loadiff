"""Plotting helpers for inference-time evaluation.

These functions were relocated verbatim from the (now removed)
``scripts/inference/inference_loadit_with_cond.py`` so the rich per-label
evaluation plots are preserved. They are used by the optional evaluation path of
the unified entrypoint (``inference.evaluate=true``). No plotting / metric logic
was changed during the move.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.evaluate import compute_report


def plot_consumption_examples_and_profiles(
    gen_np_w: np.ndarray,
    out_dir: str,
    start_date_str: str,
    indices: Sequence[int] = (0, 1),
    true_np_w: Optional[np.ndarray] = None,
    patch_per_day: int = 48,
    granularities: Sequence[str] = ("year", "month", "week", "day"),
    window_offsets: Optional[Dict[str, Sequence[int]]] = None,
    dpi: int = 150,
    profile_months: Optional[Sequence[int]] = None,
    profile_suffix: str = "",
) -> None:
    """Plot example consumption windows and daily/weekly/monthly profiles.

    profile_months: if provided, profiles are computed only on these months (1..12).
    profile_suffix: appended to saved profile filenames (useful for CLIM summer-only).
    """
    os.makedirs(out_dir, exist_ok=True)

    gen_np_w = np.asarray(gen_np_w)
    if gen_np_w.ndim != 2:
        raise ValueError(f"gen_np_w must be 2D (B,T). Got shape={gen_np_w.shape}.")

    B, T = gen_np_w.shape
    if T == 0:
        print("[WARN] Empty time series (T=0). Skipping plots.")
        return

    if true_np_w is not None:
        true_np_w = np.asarray(true_np_w)
        if true_np_w.shape != gen_np_w.shape:
            raise ValueError(
                f"true_np_w must have same shape as gen_np_w. "
                f"Got true={true_np_w.shape}, gen={gen_np_w.shape}."
            )

    if patch_per_day <= 0:
        raise ValueError("patch_per_day must be > 0.")
    if 1440 % patch_per_day != 0:
        raise ValueError(f"patch_per_day={patch_per_day} does not evenly divide 1440 minutes.")
    step_minutes = int(1440 // patch_per_day)

    def _parse_start_date(s: str) -> pd.Timestamp:
        s = (s or "").strip()
        if not s:
            raise ValueError("start_date_str is empty.")
        if "/" in s:
            dt_ = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
            if pd.isna(dt_):
                raise ValueError(
                    f"Could not parse start_date_str='{s}' as 'dd/mm/yyyy'. "
                    f"Expected like '01/01/2024'."
                )
            return pd.Timestamp(dt_)
        dt_ = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt_):
            raise ValueError(f"Could not parse start_date_str='{s}'.")
        return pd.Timestamp(dt_)

    start_dt = _parse_start_date(start_date_str)
    freq = f"{step_minutes}min"
    x_full = pd.date_range(start=start_dt, periods=T, freq=freq)

    window_offsets = dict(window_offsets or {})
    for g in granularities:
        window_offsets.setdefault(g, [0])

    valid_granularities = {"year", "month", "week", "day"}
    for g in granularities:
        if g not in valid_granularities:
            raise ValueError(f"Invalid granularity '{g}'. Must be one of {sorted(valid_granularities)}.")

    def _series_from_array(arr_1d: np.ndarray) -> pd.Series:
        return pd.Series(arr_1d, index=x_full)

    def _window_bounds(granularity: str, offset: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
        if granularity == "day":
            w_start = start_dt + pd.Timedelta(days=int(offset))
            w_end = w_start + pd.Timedelta(days=1)
        elif granularity == "week":
            w_start = start_dt + pd.Timedelta(weeks=int(offset))
            w_end = w_start + pd.Timedelta(days=7)
        elif granularity == "month":
            w_start = start_dt + pd.DateOffset(months=int(offset))
            w_end = w_start + pd.DateOffset(months=1)
        elif granularity == "year":
            if int(offset) == 0:
                w_start = x_full[0]
                w_end = x_full[-1] + pd.Timedelta(minutes=step_minutes)
            else:
                w_start = start_dt + pd.DateOffset(years=int(offset))
                w_end = w_start + pd.DateOffset(years=1)
        else:
            raise RuntimeError("Unreachable.")
        return w_start, w_end

    def _slice_to_available(s: pd.Series, w_start: pd.Timestamp, w_end: pd.Timestamp) -> pd.Series:
        last_inclusive = w_end - pd.Timedelta(minutes=step_minutes)
        return s.loc[w_start:last_inclusive]

    def _save_series_plot(idx: int, granularity: str, offset: int, s_gen: pd.Series, s_true: Optional[pd.Series]) -> None:
        w_start, w_end = _window_bounds(granularity, offset)
        g_slice = _slice_to_available(s_gen, w_start, w_end)
        if g_slice.empty:
            print(f"[WARN] idx={idx}: empty slice for {granularity} offset={offset}. Skipping.")
            return

        plt.figure(figsize=(14, 4))
        plt.plot(g_slice.index, g_slice.values, label="Generated", linewidth=1.0, alpha=0.9)
        if s_true is not None:
            t_slice = _slice_to_available(s_true, w_start, w_end)
            if not t_slice.empty:
                plt.plot(t_slice.index, t_slice.values, label="True", linewidth=1.0, alpha=0.9)

        plt.grid(True, linewidth=0.4, alpha=0.6)
        plt.legend()
        plt.title(f"{granularity.capitalize()} example — sample #{idx} (offset={offset}, start={w_start}, steps={len(g_slice)})")
        plt.xlabel("Time")
        plt.ylabel("Power (W)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"example_{idx}_{granularity}_offset{offset}.png"), dpi=dpi)
        plt.close()

    # ---------------------------
    # PROFILES (optionally month-filtered)
    # ---------------------------
    _months = None
    if profile_months is not None:
        _months = sorted({int(m) for m in profile_months})
        if any(m < 1 or m > 12 for m in _months):
            raise ValueError(f"profile_months must be in 1..12. Got {profile_months}.")

    def _filter_for_profiles(s: pd.Series) -> pd.Series:
        if _months is None:
            return s
        return s[s.index.month.isin(_months)]

    def _daily_profile(s: pd.Series) -> pd.Series:
        minutes = s.index.hour * 60 + s.index.minute
        slot = (minutes // step_minutes).astype(int)
        prof = s.groupby(slot).mean()
        return prof.reindex(range(patch_per_day))

    def _weekly_profile_matrix(s: pd.Series) -> pd.DataFrame:
        minutes = s.index.hour * 60 + s.index.minute
        slot = (minutes // step_minutes).astype(int)
        dow = s.index.dayofweek
        mat = s.groupby([dow, slot]).mean().unstack(level=-1)
        return mat.reindex(index=range(7), columns=range(patch_per_day))

    def _monthly_profile_matrix(s: pd.Series) -> pd.DataFrame:
        minutes = s.index.hour * 60 + s.index.minute
        slot = (minutes // step_minutes).astype(int)
        month = s.index.month
        mat = s.groupby([month, slot]).mean().unstack(level=-1)
        return mat.reindex(index=range(1, 13), columns=range(patch_per_day))

    def _save_profile_plots(idx: int, s_gen: pd.Series, s_true: Optional[pd.Series]) -> None:
        # Apply month filter ONLY here
        s_gen_p = _filter_for_profiles(s_gen)
        s_true_p = _filter_for_profiles(s_true) if s_true is not None else None

        if s_gen_p.empty:
            print(f"[WARN] idx={idx}: profile slice is empty (profile_months={_months}). Skipping profiles.")
            return

        suffix = profile_suffix or ""
        if _months is not None and not suffix:
            suffix = "_months" + "-".join(str(m) for m in _months)

        dgen = _daily_profile(s_gen_p)
        plt.figure(figsize=(10, 4))
        plt.plot(dgen.index, dgen.values, label="Generated", linewidth=1.2)
        if s_true_p is not None and not s_true_p.empty:
            dtru = _daily_profile(s_true_p)
            plt.plot(dtru.index, dtru.values, label="True", linewidth=1.2)
        plt.grid(True, linewidth=0.4, alpha=0.6)
        plt.legend()
        plt.title(f"Daily profile (mean by time-of-day) — sample #{idx}{suffix}")
        plt.xlabel(f"Time slot (0..{patch_per_day-1})")
        plt.ylabel("Power (W)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"profile_daily_{idx}{suffix}.png"), dpi=dpi)
        plt.close()

        wgen = _weekly_profile_matrix(s_gen_p)
        plt.figure(figsize=(12, 4))
        plt.imshow(wgen.values, aspect="auto", interpolation="nearest")
        plt.title(f"Weekly profile heatmap — sample #{idx} — Generated{suffix}")
        plt.xlabel(f"Time slot (0..{patch_per_day-1})")
        plt.ylabel("Day of week (0=Mon..6=Sun)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"profile_weekly_heatmap_gen_{idx}{suffix}.png"), dpi=dpi)
        plt.close()

        if s_true_p is not None and not s_true_p.empty:
            wtru = _weekly_profile_matrix(s_true_p)
            plt.figure(figsize=(12, 4))
            plt.imshow(wtru.values, aspect="auto", interpolation="nearest")
            plt.title(f"Weekly profile heatmap — sample #{idx} — True{suffix}")
            plt.xlabel(f"Time slot (0..{patch_per_day-1})")
            plt.ylabel("Day of week (0=Mon..6=Sun)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"profile_weekly_heatmap_true_{idx}{suffix}.png"), dpi=dpi)
            plt.close()

        mgen = _monthly_profile_matrix(s_gen_p)
        plt.figure(figsize=(12, 5))
        plt.imshow(mgen.values, aspect="auto", interpolation="nearest")
        plt.title(f"Monthly profile heatmap — sample #{idx} — Generated{suffix}")
        plt.xlabel(f"Time slot (0..{patch_per_day-1})")
        plt.ylabel("Month (1=Jan..12=Dec)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"profile_monthly_heatmap_gen_{idx}{suffix}.png"), dpi=dpi)
        plt.close()

        if s_true_p is not None and not s_true_p.empty:
            mtru = _monthly_profile_matrix(s_true_p)
            plt.figure(figsize=(12, 5))
            plt.imshow(mtru.values, aspect="auto", interpolation="nearest")
            plt.title(f"Monthly profile heatmap — sample #{idx} — True{suffix}")
            plt.xlabel(f"Time slot (0..{patch_per_day-1})")
            plt.ylabel("Month (1=Jan..12=Dec)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"profile_monthly_heatmap_true_{idx}{suffix}.png"), dpi=dpi)
            plt.close()

    for idx in indices:
        if idx < 0 or idx >= B:
            print(f"[WARN] idx={idx} out of bounds for B={B}. Skipping.")
            continue

        s_gen = _series_from_array(gen_np_w[idx, :])
        s_true = _series_from_array(true_np_w[idx, :]) if true_np_w is not None else None

        for g in granularities:
            for off in window_offsets.get(g, [0]):
                _save_series_plot(idx, g, int(off), s_gen, s_true)

        _save_profile_plots(idx, s_gen, s_true)

    print(f"[OK] Plots saved in: {out_dir}")


def plot_by_binary_per_label(
    real_data,
    synth_data,
    y_np,
    label_names,
    run_dir,
    features_extractor,
    start_date_str="01/01/2024",
    patch_per_day=48,
    min_group_size=32,
    max_examples=10,
    ac_labels: Sequence[str] = ("CLIM", "AC"),
    ac_profile_months: Optional[Sequence[int]] = (6, 7, 8),
    ac_profile_apply_to_classes: Sequence[int] = (0, 1),
):
    """For each label i, plot subsets where ``y[:, i] == 0`` and ``== 1``."""
    base_dir = os.path.join(run_dir, "by_label_binary")
    os.makedirs(base_dir, exist_ok=True)

    y_bin = (y_np > 0.5).astype(np.int64)

    ac_labels_u = {str(x).strip().upper() for x in ac_labels}
    clim_classes_u = {int(c) for c in ac_profile_apply_to_classes}

    summary = {}

    for i, lab in enumerate(label_names):
        lab_str = str(lab)
        lab_dir = os.path.join(base_dir, lab_str.replace(" ", "_"))
        os.makedirs(lab_dir, exist_ok=True)

        is_clim = lab_str.strip().upper() in ac_labels_u

        for cls in [0, 1]:
            mask = (y_bin[:, i] == cls)
            n = int(mask.sum())
            key = f"{lab}::{cls}"
            summary[key] = n

            if n < min_group_size:
                print(f"[INFO] Skipping label='{lab}' class={cls} (N={n} < {min_group_size}).")
                continue

            out_dir = os.path.join(lab_dir, f"class_{cls}")
            os.makedirs(out_dir, exist_ok=True)

            _ = compute_report(
                real_data=real_data[mask],
                synth_data=synth_data[mask],
                real_data_train=real_data[mask],
                start_date=start_date_str,
                features_extractor=features_extractor,
                output_dir=out_dir,
                plot_set="full",
                log_metrics=True,
                log_plots=True,
                return_report=True,
            )

            idxs = tuple(range(min(max_examples, n)))

            # apply summer-only month filter to PROFILES for CLIM/AC (typically class=1)
            profile_months = None
            profile_suffix = ""
            if is_clim and (ac_profile_months is not None) and (int(cls) in clim_classes_u):
                profile_months = ac_profile_months
                profile_suffix = "_summer"

            plot_consumption_examples_and_profiles(
                gen_np_w=synth_data[mask],
                out_dir=out_dir,
                start_date_str=start_date_str,
                indices=idxs,
                patch_per_day=patch_per_day,
                granularities=("year", "month", "week", "day"),
                profile_months=profile_months,
                profile_suffix=profile_suffix,
                window_offsets={"month": [7], "week": [28], "day": [182]},
            )

    with open(os.path.join(base_dir, "group_counts.json"), "w") as f:
        json.dump(summary, f, indent=2)