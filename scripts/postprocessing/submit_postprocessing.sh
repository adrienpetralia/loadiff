#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --time=0-08:00:00
#SBATCH --job-name=inference_loadiff
#SBATCH -o ./jobs/inference_loadiff_%j.out
#SBATCH -e ./jobs/inference_loadiff_%j.err

set -euo pipefail

# --- Configuration (MODIFIABLE) ---
CKPT_PATH="runs_cer/cond/loadiff_cooker/checkpoints/0300000.pt"
MODE="user_conditioned"  # unconditional | dataset_conditioned | user_conditioned
DATASET="cer"
N_SAMPLES=1000       # Nombre d'échantillons à générer
SPLIT="test"        # Pour dataset_conditioned : test | val | train

# --- Préparation ---
mkdir -p jobs
source .venv/bin/activate

# --- Commande d'inférence ---
case "$MODE" in
    "unconditional")
        CMD=(
            python -m scripts.inference.inference_loadiff
            --config-name inference_loadiff_unconditional
            "inference.ckpt_path=$CKPT_PATH"
            "inference.n_samples=$N_SAMPLES"
            "+data.dataset=$DATASET"
        )
        ;;

    "dataset_conditioned")
        CMD=(
            python -m scripts.inference.inference_loadiff
            --config-name inference_loadiff_dataset_conditioned
            "inference.ckpt_path=$CKPT_PATH"
            "inference.split=$SPLIT"
            "inference.n_samples=$N_SAMPLES"
            "+data.dataset=$DATASET"
        )
        ;;

    "user_conditioned")
        CMD=(
            python -m scripts.inference.inference_loadiff
            --config-name inference_loadiff_user_conditioned
            "inference.ckpt_path=$CKPT_PATH"
            "inference.n_samples=$N_SAMPLES"
            "+data.dataset=$DATASET"
            "inference.conditioning.combinations=[{values:{heater:0},num_samples:$N_SAMPLES},{values:{heater:1},num_samples:$N_SAMPLES}]"
        )
        ;;

    *)
        echo "❌ Mode invalide: $MODE (attendu: unconditional | dataset_conditioned | user_conditioned)"
        exit 1
        ;;
esac

echo "🚀 Lancement de l'inférence Loadiff en mode '$MODE'..."
echo "Checkpoint: $CKPT_PATH"
echo "Dataset: $DATASET"
echo "Nombre d'échantillons: $N_SAMPLES"

printf 'Commande exécutée :'
printf ' %q' "${CMD[@]}"
printf '\n'

srun "${CMD[@]}"