#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=tstr_summary
#SBATCH -o ./jobs/tstr_summary_%j.out
#SBATCH -e ./jobs/tstr_summary_%j.err

# Aggregate every metrics.json into results/tstr_experiments/summary.json.
# Pure CPU aggregation -> no GPU requested. Submitted with a dependency on the
# phase jobs by run_experiments.sh so it runs only after they finish.
set -euo pipefail
mkdir -p jobs
source .venv/bin/activate
source scripts/tstr_evaluation/slurm/_tstr_common.sh

${SRUN} python3 -m scripts.tstr_evaluation.utils.summarize_results \
  --results_dir "${RESULTS_DIR}" \
  --output_file "${RESULTS_DIR}/summary.json"
