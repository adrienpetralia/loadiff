#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract every TSTR result (all approaches, all datasets) into consolidated tables.

This is the cross-approach companion to ``utils/summarize_results.py`` (which produces a
nested ``summary.json`` for a *single* results tree). It walks the result trees of every
dataset and collects every ``metrics.json`` produced by ``evaluate_tstr`` — for **loadiff**
(Train-Real reference, pure-synthetic Exp1, mixed Exp2) **and** for every generative
baseline (on-the-fly: timegan / timevae / diffusion_ts; pre-generated: timevqvae /
energydiff / gmm / timeweaver), in both their pure-synthetic and mixed Synthetic + N% Real
phases — and writes:

* ``tstr_results_long.csv`` — one tidy row per ``metrics.json`` (every scalar metric),
* ``tstr_<metric>.csv``     — a pivot of the chosen metric (default ``BALANCED_ACCURACY``)
  with rows ``(dataset, classifier, appliance)`` and one column per method, and
* a compact Markdown table per dataset printed to stdout.

Results layout (see the README): smach lives at ``results/tstr_experiments/`` while
``cer`` / ``cer_bis`` live at ``results/tstr_experiments/<dataset>/``. Paths recognised::

    baseline_train_real_test_real/<classifier>/<appliance>                     (loadiff, Train Real)
    exp1_pure_synthetic/<classifier>/<appliance>                               (loadiff, pure synthetic)
    exp2_mixed/<pct>pct/<classifier>/<appliance>                               (loadiff, mixed)
    baselines/<dataset>/<baseline>/<mode>/<classifier>/<appliance>             (baseline, pure synthetic)
    baselines/<dataset>/<baseline>/<mode>/exp2_mixed/<pct>pct/<classifier>/<appliance>  (baseline, mixed)

Example::

    python -m scripts.tstr_evaluation.extract_results
    python -m scripts.tstr_evaluation.extract_results \\
        --results_root results/tstr_experiments --datasets cer \\
        --metric BALANCED_ACCURACY --output_dir results/tstr_experiments
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Scalar metrics surfaced in the long table (BALANCED_ACCURACY first, the primary metric).
METRIC_KEYS = ("BALANCED_ACCURACY", "ACCURACY", "PRECISION", "RECALL", "F1")
# SUPPORT sub-fields flattened into their own columns.
SUPPORT_KEYS = ("n_total", "n_positive", "n_negative")
DEFAULT_DATASETS = ("smach", "cer", "cer_bis")


def _dataset_root(results_root: str, dataset: str, *, primary: str = "smach") -> str:
    """Resolve a dataset's results tree (smach at the root; others one level down)."""
    return results_root if dataset == primary else os.path.join(results_root, dataset)


def _pct_to_int(token: str) -> Optional[int]:
    """``'5pct'`` -> ``5``; returns ``None`` if it does not match the convention."""
    if token.endswith("pct") and token[:-3].isdigit():
        return int(token[:-3])
    return None


def classify_path(parts: List[str]) -> Optional[Dict[str, Any]]:
    """Map a results-relative path (split on os.sep) to a record, or ``None`` if unknown.

    The returned dict carries ``approach`` (``loadiff`` or the baseline name), ``phase``
    (``train_real`` / ``pure_synthetic`` / ``mixed``), ``mix_pct``, ``mode``,
    ``classifier`` and ``appliance`` — everything needed to label the row.
    """
    n = len(parts)
    # ---- loadiff phases (live directly under the dataset root) ----
    if parts[0] == "baseline_train_real_test_real" and n == 3:
        return dict(approach="loadiff", phase="train_real", mix_pct=None, mode=None,
                    classifier=parts[1], appliance=parts[2])
    if parts[0] == "exp1_pure_synthetic" and n == 3:
        return dict(approach="loadiff", phase="pure_synthetic", mix_pct=None, mode=None,
                    classifier=parts[1], appliance=parts[2])
    if parts[0] == "exp2_mixed" and n == 4 and _pct_to_int(parts[1]) is not None:
        return dict(approach="loadiff", phase="mixed", mix_pct=_pct_to_int(parts[1]),
                    mode=None, classifier=parts[2], appliance=parts[3])
    # ---- generative baselines ----
    if parts[0] == "baselines":
        if n == 6:  # baselines/<dataset>/<baseline>/<mode>/<classifier>/<appliance>
            return dict(approach=parts[2], phase="pure_synthetic", mix_pct=None,
                        mode=parts[3], classifier=parts[4], appliance=parts[5])
        if n == 8 and parts[4] == "exp2_mixed" and _pct_to_int(parts[5]) is not None:
            return dict(approach=parts[2], phase="mixed", mix_pct=_pct_to_int(parts[5]),
                        mode=parts[3], classifier=parts[6], appliance=parts[7])
    return None


def method_label(rec: Dict[str, Any]) -> str:
    """Short comparable method name, e.g. ``loadiff:pure`` / ``energydiff:mixed@50``."""
    approach = rec["approach"]
    phase = rec["phase"]
    if phase == "train_real":
        return f"{approach}:train_real"
    if phase == "pure_synthetic":
        return f"{approach}:pure"
    return f"{approach}:mixed@{int(rec['mix_pct'])}"


def _method_sort_key(label: str):
    """Order columns: loadiff first, then baselines A-Z; within a method train_real < pure < mixed@pct."""
    approach, _, phase = label.partition(":")
    approach_rank = (0, "") if approach == "loadiff" else (1, approach)
    if phase == "train_real":
        phase_rank = (0, 0)
    elif phase == "pure":
        phase_rank = (1, 0)
    else:  # mixed@<pct>
        phase_rank = (2, int(phase.split("@", 1)[1]))
    return (approach_rank, phase_rank)


def collect_records(
    results_root: str, datasets=DEFAULT_DATASETS, *, primary: str = "smach"
) -> List[Dict[str, Any]]:
    """Walk every dataset tree and return one record per ``metrics.json`` found."""
    other_roots = {d for d in datasets if d != primary}
    records: List[Dict[str, Any]] = []
    for dataset in datasets:
        root = _dataset_root(results_root, dataset, primary=primary)
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            if "metrics.json" not in filenames:
                continue
            rel = os.path.relpath(dirpath, root)
            if rel == ".":
                continue
            parts = rel.split(os.sep)
            # When scanning the primary root, skip the other datasets' nested trees.
            if dataset == primary and parts[0] in other_roots:
                continue
            rec = classify_path(parts)
            if rec is None:
                continue
            metrics_path = os.path.join(dirpath, "metrics.json")
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                # A job may still be writing this file (or it is empty/corrupt). Skip it
                # rather than crash, so the extractor stays safe to run mid-experiment.
                logger.warning("Skipping unreadable %s (%s).", metrics_path, exc)
                continue
            rec["dataset"] = dataset
            for k in METRIC_KEYS:
                rec[k] = data.get(k)
            support = data.get("SUPPORT") or {}
            for k in SUPPORT_KEYS:
                rec[k] = support.get(k)
            rec["path"] = os.path.relpath(os.path.join(dirpath, "metrics.json"), results_root)
            records.append(rec)
    return records


def write_long_csv(records: List[Dict[str, Any]], path: str) -> None:
    """Write the tidy long-form table (one row per result)."""
    cols = (["dataset", "approach", "phase", "mix_pct", "mode", "classifier", "appliance"]
            + list(METRIC_KEYS) + list(SUPPORT_KEYS) + ["path"])
    rows = sorted(
        records,
        key=lambda r: (r["dataset"], _method_sort_key(method_label(r)),
                       r["classifier"], str(r["appliance"])),
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_pivot(records: List[Dict[str, Any]], metric: str):
    """Return ``(row_keys, methods, table)`` pivoting ``metric`` by (dataset, clf, appliance)."""
    methods = sorted({method_label(r) for r in records}, key=_method_sort_key)
    table: Dict[tuple, Dict[str, Any]] = {}
    for r in records:
        key = (r["dataset"], r["classifier"], str(r["appliance"]))
        table.setdefault(key, {})[method_label(r)] = r.get(metric)
    row_keys = sorted(table.keys())
    return row_keys, methods, table


def write_pivot_csv(records: List[Dict[str, Any]], metric: str, path: str) -> None:
    """Write the pivot of ``metric`` (rows = dataset/classifier/appliance, cols = methods)."""
    row_keys, methods, table = build_pivot(records, metric)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "classifier", "appliance"] + methods)
        for key in row_keys:
            vals = table[key]
            w.writerow(list(key) + [_fmt(vals.get(m)) for m in methods])


def _fmt(v: Any) -> str:
    return "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def to_markdown(records: List[Dict[str, Any]], metric: str) -> str:
    """Render a compact per-dataset Markdown pivot of ``metric``."""
    row_keys, methods, table = build_pivot(records, metric)
    if not row_keys:
        return f"_No results found for metric {metric}._"
    lines: List[str] = []
    for dataset in sorted({k[0] for k in row_keys}):
        lines.append(f"### {dataset} — {metric}\n")
        header = ["classifier", "appliance"] + methods
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for key in [k for k in row_keys if k[0] == dataset]:
            _, classifier, appliance = key
            vals = table[key]
            cells = [classifier, appliance] + [_fmt(vals.get(m)) for m in methods]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract all TSTR results into consolidated tables.")
    p.add_argument("--results_root", default="results/tstr_experiments",
                   help="Root of the results tree (smach lives here; cer/cer_bis one level down).")
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p.add_argument("--primary", default="smach", help="Dataset stored at the results root.")
    p.add_argument("--metric", default="BALANCED_ACCURACY", choices=list(METRIC_KEYS))
    p.add_argument("--output_dir", default=None, help="Defaults to --results_root.")
    p.add_argument("--no_markdown", dest="markdown", action="store_false", default=True)
    return p


def main(argv: Optional[list] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    output_dir = args.output_dir or args.results_root

    records = collect_records(args.results_root, args.datasets, primary=args.primary)
    long_path = os.path.join(output_dir, "tstr_results_long.csv")
    pivot_path = os.path.join(output_dir, f"tstr_{args.metric.lower()}.csv")
    write_long_csv(records, long_path)
    write_pivot_csv(records, args.metric, pivot_path)

    n_methods = len({method_label(r) for r in records})
    n_datasets = len({r["dataset"] for r in records})
    print(f"Collected {len(records)} result(s): {n_methods} method(s) across {n_datasets} dataset(s).")
    print(f"  long  -> {long_path}")
    print(f"  pivot -> {pivot_path}")

    # Per-(dataset, method) result counts — a baseline still running shows up here with
    # fewer results than its peers (missing cells are simply blank in the pivot).
    counts: Dict[tuple, int] = {}
    for r in records:
        counts[(r["dataset"], method_label(r))] = counts.get((r["dataset"], method_label(r)), 0) + 1
    if counts:
        print("  results per (dataset, method):")
        for key in sorted(counts):
            print(f"    {key[0]:8s} {key[1]:24s} {counts[key]}")
    if args.markdown and records:
        md_path = os.path.join(output_dir, f"tstr_{args.metric.lower()}.md")
        md = to_markdown(records, args.metric)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(f"  table -> {md_path}\n")
        print(md)


if __name__ == "__main__":
    main()
