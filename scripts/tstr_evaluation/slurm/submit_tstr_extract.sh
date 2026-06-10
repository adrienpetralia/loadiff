#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=tstr_extract
#SBATCH -o ./jobs/tstr_extract_%j.out
#SBATCH -e ./jobs/tstr_extract_%j.err

# Extract every TSTR result (all approaches, all datasets) into consolidated tables:
#   <RESULTS_ROOT>/tstr_results_long.csv          (one row per metrics.json)
#   <RESULTS_ROOT>/tstr_<metric>.csv / .md        (pivot of the chosen metric)
# Pure CPU aggregation -> no GPU requested. Spans datasets, so it uses RESULTS_ROOT
# (the base results tree) rather than the per-dataset RESULTS_DIR.
#
#   sbatch scripts/tstr_evaluation/slurm/submit_tstr_extract.sh
#   SRUN= bash scripts/tstr_evaluation/slurm/submit_tstr_extract.sh   # local, no SLURM
set -euo pipefail
mkdir -p jobs
source .venv/bin/activate
source scripts/tstr_evaluation/slurm/_tstr_common.sh   # for SRUN

RESULTS_ROOT="${RESULTS_ROOT:-results/tstr_experiments}"
DATASETS="${TSTR_DATASETS:-cer}"
METRIC="${TSTR_METRIC:-BALANCED_ACCURACY}"

${SRUN} python3 -m scripts.tstr_evaluation.extract_results \
  --results_root "${RESULTS_ROOT}" \
  --datasets ${DATASETS} \
  --metric "${METRIC}"
