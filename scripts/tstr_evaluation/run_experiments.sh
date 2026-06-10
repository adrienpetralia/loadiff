#!/bin/bash
# =============================================================================
# Master script: submit every TSTR phase as a separate SLURM job
# (à l'image de scripts/bash/selena/run_all_ablations.sh).
#
#   baseline -> Train Real, Test Real
#   exp1     -> Train 100% Synthetic, Test Real
#   exp2     -> Train Synthetic + {5,20,50,100}% Real, Test Real
#   summary  -> aggregate all metrics.json (runs after the phase jobs)
#
# Tests on the real test split of DATASET (data/<dataset>, default cer). Set DATASET=cer
# or cer_bis to run the same experiment on those datasets (see slurm/_tstr_common.sh).
# Submit from the repository root so the jobs' relative paths (./jobs, .venv,
# scripts.tstr_evaluation...) resolve correctly:
#
#   ./scripts/tstr_evaluation/run_experiments.sh [--dry-run]
#
# Config is taken from the environment (see slurm/_tstr_common.sh), e.g.:
#   SYNTHETIC_DATA_DIR=runs_inference/my_run ./scripts/tstr_evaluation/run_experiments.sh
#
# Options:
#   --dry-run   Print the sbatch commands without submitting them.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_DIR="${SCRIPT_DIR}/slurm"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# Phase jobs (submission script names, without extension). The summary job is
# submitted separately with a dependency on these.
PHASES=(
    "submit_tstr_baseline"
    "submit_tstr_exp1"
    "submit_tstr_exp2"
)
SUMMARY="submit_tstr_summary"

echo "Submitting ${#PHASES[@]} TSTR phase jobs (+ summary)..."
phase_ids=()
for PHASE in "${PHASES[@]}"; do
    SCRIPT="${SLURM_DIR}/${PHASE}.sh"
    if [[ ! -f "${SCRIPT}" ]]; then
        echo "WARNING: missing ${SCRIPT}, skipping." >&2
        continue
    fi
    if [[ "${DRY_RUN}" == true ]]; then
        echo "sbatch ${SCRIPT}"
    else
        JID="$(sbatch --parsable "${SCRIPT}")"
        echo "Submitted ${PHASE} as job ${JID}"
        phase_ids+=("${JID}")
    fi
done

# Summary: aggregate once the phase jobs finish (afterany = even if some fail,
# so partial results are still summarised).
SUMMARY_SCRIPT="${SLURM_DIR}/${SUMMARY}.sh"
if [[ "${DRY_RUN}" == true ]]; then
    echo "sbatch --dependency=afterany:<phase_jobs> ${SUMMARY_SCRIPT}"
elif [[ ${#phase_ids[@]} -gt 0 ]]; then
    DEP="$(IFS=:; echo "${phase_ids[*]}")"
    JID="$(sbatch --parsable --dependency="afterany:${DEP}" "${SUMMARY_SCRIPT}")"
    echo "Submitted ${SUMMARY} as job ${JID} (after ${DEP})"
else
    echo "No phase jobs submitted; submitting summary standalone."
    sbatch "${SUMMARY_SCRIPT}"
fi
