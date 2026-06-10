#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=tstr_baseline_pregen
#SBATCH -o ./jobs/tstr_baseline_pregen_%j.out
#SBATCH -e ./jobs/tstr_baseline_pregen_%j.err

# TSTR with a *pre-generated* baseline generator (one job per baseline, BASELINE env).
# Unlike submit_tstr_baselines.sh, NO inference is run on the fly: the synthetic curves
# already exist on disk. Two layouts are handled transparently by the prepare step:
#   * per-label      (timevqvae / energydiff / gmm):
#       <RUNS_ROOT>/<baseline>/<baseline>_<appliance>/label<value>.npy
#   * single-dir     (timeweaver), multilabel population split by the appliance column:
#       <RUNS_ROOT>/timeweaver/{samples.npy, y.npy, logs_summary.json}
# For each appliance we materialise a balanced population (label0 + label1), capping
# each class to at most MAX_PER_FILE curves (reproducible, seeded), then train+test
# ROCKET and TransApp (Test on the real test split). No post-processing is applied.
#
#   BASELINE=timeweaver DATASET=cer_bis sbatch \
#     scripts/tstr_evaluation/slurm/submit_tstr_baselines_pregenerated.sh
set -euo pipefail
mkdir -p jobs
source .venv/bin/activate
source scripts/tstr_evaluation/slurm/_tstr_common.sh

BASELINE="${BASELINE:?Set BASELINE=timevqvae|energydiff|gmm|timeweaver}"
RUNS_ROOT="$(resolve_runs_root)" || exit 1   # <RUNS_BASE>/runs_<dataset>, guarded vs. stale export
MAX_PER_FILE="${MAX_PER_FILE:-1024}"   # max synthetic curves loaded per label file
GEN_ROOT="${GEN_ROOT:-runs_inference/tstr_baselines_pregenerated}"
SEED="${SEED:-0}"
# Inference-mode tag in the results nomenclature (kept identical to the on-the-fly
# baselines so timevqvae/energydiff line up with timegan/timevae/diffusion_ts).
MODE="user_conditioned"

echo "Pre-generated baseline: ${BASELINE} | runs_root=${RUNS_ROOT} | max_per_file=${MAX_PER_FILE}"

for appliance in "${APPLIANCES[@]}"; do
  gen_dir="${GEN_ROOT}/${BASELINE}_${DATASET}_${appliance}"
  echo "=== [prepare] ${BASELINE} / ${appliance} -> ${gen_dir}"
  ${SRUN} python3 -m scripts.tstr_evaluation.utils.prepare_pregenerated_baseline \
    --runs_root "${RUNS_ROOT}" --baseline "${BASELINE}" \
    --target_label "${appliance}" --dataset "${DATASET}" \
    --output_dir "${gen_dir}" --max_per_file "${MAX_PER_FILE}" --seed "${SEED}"

  # Pure-synthetic (Exp1 analog) + mixed Synthetic + N% Real sweep (Exp2 analog).
  baseline_tstr_all "${BASELINE}" "${MODE}" "${appliance}" "${gen_dir}"
done

echo "=== Done pre-generated baseline TSTR for ${BASELINE}."
