#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=tstr_exp2
#SBATCH -o ./jobs/tstr_exp2_%j.out
#SBATCH -e ./jobs/tstr_exp2_%j.err

# Experiment 2: Train on Synthetic + N% Real, Test Real
# (every mix percentage x classifier x appliance).
set -euo pipefail
mkdir -p jobs
source .venv/bin/activate
source scripts/tstr_evaluation/slurm/_tstr_common.sh

for pct in "${MIX_PERCENTAGES[@]}"; do
  for appliance in "${APPLIANCES[@]}"; do
    mix_dir="${RESULTS_DIR}/exp2_mixed/${pct}pct/_datasets/${appliance}"
    echo "[exp2 ${pct}%] building mixed dataset for ${appliance}"
    make_mixed_dataset "${appliance}" "${pct}" "${mix_dir}"
    for classifier in "${CLASSIFIERS[@]}"; do
      echo "[exp2 ${pct}%] ${classifier} / ${appliance}"
      train_and_eval "${classifier}" "${appliance}" "${mix_dir}" \
        "${RESULTS_DIR}/exp2_mixed/${pct}pct/${classifier}/${appliance}"
    done
  done
done
