#!/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --job-name=inference_baseline
#SBATCH -o ./jobs/%j.out
#SBATCH -e ./jobs/%j.err

# Unified baseline inference (timegan / timevae / diffusion_ts), both modes.
#   BASELINE=timegan DATASET=cer MODE=unconditional N_SAMPLES=1024 \
#     sbatch scripts/bash/selena/submit_inference_baseline.sh
#   BASELINE=timegan DATASET=cer MODE=user_conditioned APPLIANCE=cooker N_SAMPLES=512 \
#     sbatch scripts/bash/selena/submit_inference_baseline.sh   # -> balanced label0/label1
set -euo pipefail
mkdir -p jobs
source .venv/bin/activate

BASELINE="${BASELINE:?Set BASELINE=timegan|timevae|diffusion_ts}"
DATASET="${DATASET:-cer}"
MODE="${MODE:-unconditional}"
N_SAMPLES="${N_SAMPLES:-1024}"
RUNS_ROOT="${RUNS_ROOT:-runs_${DATASET}}"
OUTPUT_DIR="${OUTPUT_DIR:-runs_inference}"
DEVICE="${DEVICE:-cuda}"
# Sampling batch size (lower it for diffusion_ts if you hit CUDA OOM, e.g. BATCH_SIZE=8).
BATCH_SIZE="${BATCH_SIZE:-256}"

COMMON=(
  inference.baseline="${BASELINE}"
  inference.dataset="${DATASET}"
  inference.runs_root="${RUNS_ROOT}"
  inference.output_dir="${OUTPUT_DIR}"
  inference.device="${DEVICE}"
  inference.batch_size="${BATCH_SIZE}"
)

if [[ "${MODE}" == "unconditional" ]]; then
  srun python3 -m scripts.inference.inference_baseline "${COMMON[@]}" \
    inference.mode=unconditional inference.n_samples="${N_SAMPLES}"
else
  APPLIANCE="${APPLIANCE:?Set APPLIANCE for user_conditioned mode}"
  COMBOS="[{values:{${APPLIANCE}:0},num_samples:${N_SAMPLES}},{values:{${APPLIANCE}:1},num_samples:${N_SAMPLES}}]"
  srun python3 -m scripts.inference.inference_baseline "${COMMON[@]}" \
    inference.mode=user_conditioned "inference.conditioning.combinations=${COMBOS}"
fi