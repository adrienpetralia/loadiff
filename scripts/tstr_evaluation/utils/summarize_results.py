#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate every ``metrics.json`` under a results tree into a single summary.

Walks ``results/tstr_experiments/`` and collects the metrics produced by each run,
keyed by experiment / classifier / appliance (and mix percentage for Experiment 2),
producing the ``summary.json`` described in the task specification.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

# Metrics surfaced in the summary (BALANCED_ACCURACY first, as the primary metric).
SUMMARY_KEYS = ("BALANCED_ACCURACY", "ACCURACY", "PRECISION", "RECALL", "F1")


def _read_metrics(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: data[k] for k in SUMMARY_KEYS if k in data}


def _ensure(d: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    for k in keys:
        d = d.setdefault(k, {})
    return d


def summarize_results(results_dir: str, output_file: str) -> Dict[str, Any]:
    """Aggregate all ``metrics.json`` under ``results_dir`` into ``output_file``.

    The directory layout produced by ``run_experiments.sh`` is::

        <exp>/<classifier>/<appliance>/metrics.json                 (baseline, exp1)
        exp2_mixed/<pct>pct/<classifier>/<appliance>/metrics.json   (exp2)

    Returns the aggregated summary dict (also written as JSON to ``output_file``).
    """
    summary: Dict[str, Any] = {}

    for dirpath, _dirnames, filenames in os.walk(results_dir):
        if "metrics.json" not in filenames:
            continue
        rel = os.path.relpath(dirpath, results_dir)
        parts = rel.split(os.sep)
        metrics = _read_metrics(os.path.join(dirpath, "metrics.json"))

        # baselines/<dataset>/<baseline>/<mode>/exp2_mixed/<pct>pct/<classifier>/<appliance>
        if parts[0] == "baselines" and len(parts) >= 8 and parts[4] == "exp2_mixed":
            _, dataset, baseline, mode, _exp, pct, classifier, appliance = parts[:8]
            _ensure(
                summary, "baselines", dataset, baseline, mode, "exp2_mixed", pct, classifier
            )[appliance] = metrics
        # baselines/<dataset>/<baseline>/<mode>/<classifier>/<appliance>
        elif parts[0] == "baselines" and len(parts) >= 6:
            _, dataset, baseline, mode, classifier, appliance = parts[:6]
            _ensure(summary, "baselines", dataset, baseline, mode, classifier)[appliance] = metrics
        # exp2_mixed/<pct>pct/<classifier>/<appliance>
        elif parts[0] == "exp2_mixed" and len(parts) >= 4:
            _exp, pct, classifier, appliance = parts[0], parts[1], parts[2], parts[3]
            _ensure(summary, "exp2_mixed", pct, classifier)[appliance] = metrics
        # <experiment>/<classifier>/<appliance>
        elif len(parts) >= 3:
            experiment, classifier, appliance = parts[0], parts[1], parts[2]
            _ensure(summary, experiment, classifier)[appliance] = metrics

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarise TSTR experiment metrics.")
    p.add_argument("--results_dir", default="results/tstr_experiments")
    p.add_argument("--output_file", default=None, help="Defaults to <results_dir>/summary.json")
    return p


def main(argv: Optional[list] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    output_file = args.output_file or os.path.join(args.results_dir, "summary.json")
    summary = summarize_results(args.results_dir, output_file)
    print(f"Wrote summary with {len(summary)} experiment group(s) to {output_file}")


if __name__ == "__main__":
    main()
