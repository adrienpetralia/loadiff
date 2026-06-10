#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launch per-appliance baseline trainings (one dedicated model per appliance).

For each (baseline, appliance) pair, runs the baseline's Hydra training entrypoint
with ``training.filter_by_label.<appliance>=1`` so the model is trained ONLY on the
clients owning that appliance. Appliance/baseline lists are configurable; nothing is
hardcoded to a specific dataset schema.

Examples:
    # cer_bis appliances on all three baselines
    python scripts/training/train_all_appliances.py

    # custom appliances/dataset, dry-run (print commands only)
    python scripts/training/train_all_appliances.py \\
        --appliances ev heater water_heater --dataset cer_bis --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List

DEFAULT_BASELINES = ["timegan", "timevae", "diffusion_ts"]
DEFAULT_APPLIANCES = ["ev", "heater", "water_heater"]


def build_command(baseline: str, appliance: str, dataset: str, value: int) -> List[str]:
    """Build the Hydra training command for one (baseline, appliance) pair."""
    return [
        sys.executable, "-m", f"scripts.training.train_{baseline}",
        "--config-name", baseline,
        f"data.dataset={dataset}",
        f"model_name={baseline}_{appliance}",
        # `+` adds the key into the (initially empty) filter_by_label dict.
        f"+training.filter_by_label.{appliance}={value}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES,
                        help="Baselines to train (default: timegan timevae diffusion_ts).")
    parser.add_argument("--appliances", nargs="+", default=DEFAULT_APPLIANCES,
                        help="Appliance label names (must exist in the dataset metadata).")
    parser.add_argument("--dataset", default="cer_bis", help="Dataset name (data.dataset).")
    parser.add_argument("--value", type=int, default=1, choices=[-1, 0, 1],
                        help="Target label state to filter on (default: 1 = present).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands without running them.")
    args = parser.parse_args()

    failures = []
    for appliance in args.appliances:
        for baseline in args.baselines:
            cmd = build_command(baseline, appliance, args.dataset, args.value)
            print("🚀 " + " ".join(cmd))
            if args.dry_run:
                continue
            result = subprocess.run(cmd)
            if result.returncode != 0:
                failures.append((baseline, appliance, result.returncode))

    if failures:
        print(f"\n{len(failures)} run(s) failed: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()