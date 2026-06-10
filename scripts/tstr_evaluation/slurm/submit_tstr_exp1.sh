#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=tstr_exp1
#SBATCH -o ./jobs/tstr_exp1_%j.out
#SBATCH -e ./jobs/tstr_exp1_%j.err

# Experiment 1: Train on 100% synthetic, Test Real (every classifier x appliance).
set -euo pipefail
mkdir -p jobs
source .venv/bin/activate
source scripts/tstr_evaluation/slurm/_tstr_common.sh

for classifier in "${CLASSIFIERS[@]}"; do
  for appliance in "${APPLIANCES[@]}"; do
    echo "[exp1] ${classifier} / ${appliance}"
    train_and_eval "${classifier}" "${appliance}" "${SYNTHETIC_DATA_DIR}" \
      "${RESULTS_DIR}/exp1_pure_synthetic/${classifier}/${appliance}"
  done
done
