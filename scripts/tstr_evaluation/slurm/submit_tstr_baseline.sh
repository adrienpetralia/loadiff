#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=tstr_baseline
#SBATCH -o ./jobs/tstr_baseline_%j.out
#SBATCH -e ./jobs/tstr_baseline_%j.err

# Baseline: Train Real, Test Real (every classifier x appliance).
set -euo pipefail
mkdir -p jobs
source .venv/bin/activate
source scripts/tstr_evaluation/slurm/_tstr_common.sh

for classifier in "${CLASSIFIERS[@]}"; do
  for appliance in "${APPLIANCES[@]}"; do
    echo "[baseline] ${classifier} / ${appliance}"
    train_and_eval "${classifier}" "${appliance}" "${REAL_DATA_DIR}" \
      "${RESULTS_DIR}/baseline_train_real_test_real/${classifier}/${appliance}" "train"
  done
done
