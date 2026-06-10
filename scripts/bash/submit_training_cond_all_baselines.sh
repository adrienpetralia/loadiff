#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=train_array
#SBATCH --output=./jobs/%A_%a_%x.out
#SBATCH --error=./jobs/%A_%a_%x.err
#SBATCH --array=1-9

set -euo pipefail

# --- Configuration ---
BASELINES=("timegan" "timevae" "diffusion_ts")
APPLIANCES=("cooker" "dishwasher" "tumble_dryer" "water_heater")
DATASET="cer"
VALUE=1

TOTAL_JOBS=$(( ${#BASELINES[@]} * ${#APPLIANCES[@]} ))

# --- Vérification de l'index ---
JOB_INDEX=$(( SLURM_ARRAY_TASK_ID - 1 ))

if (( JOB_INDEX < 0 || JOB_INDEX >= TOTAL_JOBS )); then
    echo "Index Slurm invalide : ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

BASELINE_INDEX=$(( JOB_INDEX / ${#APPLIANCES[@]} ))
APPLIANCE_INDEX=$(( JOB_INDEX % ${#APPLIANCES[@]} ))

BASELINE="${BASELINES[$BASELINE_INDEX]}"
APPLIANCE="${APPLIANCES[$APPLIANCE_INDEX]}"

echo "Baseline : ${BASELINE}"
echo "Appareil : ${APPLIANCE}"
echo "Dataset  : ${DATASET}"
echo "Label    : ${VALUE}"

# --- Environnement ---
source .venv/bin/activate

# --- Exécution ---
srun python3 -m "scripts.training.train_${BASELINE}" \
    --config-name="${BASELINE}" \
    "data.dataset=${DATASET}" \
    "model_name=${BASELINE}_${APPLIANCE}_label${VALUE}" \
    "++training.filter_by_label.${APPLIANCE}=${VALUE}"