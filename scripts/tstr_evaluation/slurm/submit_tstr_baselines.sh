#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=tstr_baseline
#SBATCH -o ./jobs/tstr_baseline_%j.out
#SBATCH -e ./jobs/tstr_baseline_%j.err

# TSTR with a baseline generator (one job per baseline, set via BASELINE env).
# For each appliance: generate a balanced (label0/label1) synthetic population with the
# specialised baseline checkpoints, then train+test ROCKET and TransApp (Test on the real
# test split). No post-processing is applied to the baseline data.
#
#   BASELINE=timegan DATASET=smach sbatch scripts/tstr_evaluation/slurm/submit_tstr_baselines.sh
set -euo pipefail
mkdir -p jobs
source .venv/bin/activate
source scripts/tstr_evaluation/slurm/_tstr_common.sh

BASELINE="${BASELINE:?Set BASELINE=timegan|timevae|diffusion_ts}"
RUNS_ROOT="$(resolve_runs_root)" || exit 1   # <RUNS_BASE>/runs_<dataset>, guarded vs. stale export
N_SAMPLES="${N_SAMPLES:-1024}"           # per label (label0 + label1 => 2*N per appliance)
GEN_ROOT="${GEN_ROOT:-runs_inference/tstr_baselines}"
MODE="user_conditioned"

# Generation batch size. diffusion_ts is a heavy diffusion transformer -> small default
# to avoid CUDA OOM; override with GEN_BATCH_SIZE (e.g. GEN_BATCH_SIZE=4).
if [[ -n "${GEN_BATCH_SIZE:-}" ]]; then
  GEN_BS="${GEN_BATCH_SIZE}"
elif [[ "${BASELINE}" == "diffusion_ts" ]]; then
  GEN_BS=8
else
  GEN_BS=256
fi
echo "Generation batch size for ${BASELINE}: ${GEN_BS}"

for appliance in "${APPLIANCES[@]}"; do
  gen_dir="${GEN_ROOT}/${BASELINE}_${DATASET}_${appliance}"
  echo "=== [gen] ${BASELINE} / ${appliance} -> ${gen_dir}"
  ${SRUN} python3 -m scripts.inference.inference_baseline \
    inference.baseline="${BASELINE}" inference.dataset="${DATASET}" \
    inference.mode="${MODE}" inference.runs_root="${RUNS_ROOT}" \
    inference.batch_size="${GEN_BS}" inference.device="${DEVICE}" \
    "hydra.run.dir=${gen_dir}" \
    "inference.conditioning.combinations=[{values:{${appliance}:0},num_samples:${N_SAMPLES}},{values:{${appliance}:1},num_samples:${N_SAMPLES}}]"

  # Pure-synthetic (Exp1 analog) + mixed Synthetic + N% Real sweep (Exp2 analog).
  # No post-processing is ever applied to the baseline synthetic data.
  baseline_tstr_all "${BASELINE}" "${MODE}" "${appliance}" "${gen_dir}"
done

echo "=== Done baseline TSTR for ${BASELINE}."
