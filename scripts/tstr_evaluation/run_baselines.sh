#!/bin/bash
# =============================================================================
# Master: submit the baseline TSTR experiments, one SLURM job per baseline
# — à l'image de run_experiments.sh. Two families of baselines are supported:
#
#   * on-the-fly  (timegan / timevae / diffusion_ts): synthetic curves are
#     generated at submit time with the specialised checkpoints
#     (slurm/submit_tstr_baselines.sh);
#   * pre-generated (timevqvae / energydiff / gmm / timeweaver): synthetic curves
#     already exist on disk and are loaded directly — no inference is run
#     (slurm/submit_tstr_baselines_pregenerated.sh).
#
# Each job runs the full TSTR train+test (rocket + transapp) on the real test
# split. Submit from the repository root. The baseline run root defaults to
# <RUNS_BASE>/runs_<dataset>, so switching DATASET targets the right root automatically;
# prefer overriding the dataset-agnostic RUNS_BASE over RUNS_ROOT:
#
#   DATASET=cer RUNS_BASE=runs_evaluation/ \
#     ./scripts/tstr_evaluation/run_baselines.sh [--dry-run]
#
# Each baseline job runs the pure-synthetic phase (skip with TSTR_BASELINE_PURE=0) and
# the mixed Synthetic + N% Real sweep (skip with TSTR_BASELINE_MIX=0) over
# TSTR_MIX_PERCENTAGES (default {5,20,50,100}), à la loadiff's Exp2. Set
# TSTR_BASELINE_PURE=0 to backfill only the mixed results without recomputing the pure ones.
#
# Overridable env vars: DATASET, RUNS_BASE, RUNS_ROOT, N_SAMPLES, MAX_PER_FILE, GEN_ROOT,
# TSTR_BASELINES, TSTR_PREGENERATED_BASELINES, TSTR_BASELINE_PURE, TSTR_BASELINE_MIX,
# TSTR_MIX_PERCENTAGES, plus the usual TSTR ones
# (TSTR_APPLIANCES, TSTR_CLASSIFIERS, RESULTS_DIR, DEVICE).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB="${SCRIPT_DIR}/slurm/submit_tstr_baselines.sh"
JOB_PREGEN="${SCRIPT_DIR}/slurm/submit_tstr_baselines_pregenerated.sh"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# Baselines whose synthetic data is already generated on disk (loaded, not inferred).
PREGENERATED_BASELINES=" ${TSTR_PREGENERATED_BASELINES:-timevqvae energydiff timeweaver gmm} "

BASELINES=(${TSTR_BASELINES:-timegan timevae diffusion_ts timevqvae energydiff timeweaver gmm})

echo "Submitting ${#BASELINES[@]} baseline TSTR job(s)..."
for BASELINE in "${BASELINES[@]}"; do
    if [[ "${PREGENERATED_BASELINES}" == *" ${BASELINE} "* ]]; then
        job="${JOB_PREGEN}"
    else
        job="${JOB}"
    fi
    if [[ "${DRY_RUN}" == true ]]; then
        echo "sbatch --export=ALL,BASELINE=${BASELINE} ${job}"
    else
        JID="$(sbatch --parsable --export=ALL,BASELINE="${BASELINE}" "${job}")"
        echo "Submitted baseline=${BASELINE} as job ${JID}"
    fi
done
