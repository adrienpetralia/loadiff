#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --time=0-08:00:00
#SBATCH --job-name=inference_loadiff
#SBATCH -o ./jobs/inference_loadiff_%j.out
#SBATCH -e ./jobs/inference_loadiff_%j.err

set -euo pipefail

# --- Configuration (MODIFIABLE) ---
MODE="dataset_conditioned"  # unconditional | dataset_conditioned | user_conditioned
DATASET="cer"      # cer
N_SAMPLES=1024      # Nombre d'échantillons à générer
SPLIT="test"        # Pour dataset_conditioned : test | val | train
MODEL_NAME="loadiff/loadiff_unconditional"
CKPT_ITER="0300000"
CKPT_PATH="runs_$DATASET/$MODEL_NAME/checkpoints/$CKPT_ITER.pt"

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
            "inference.conditioning.combinations=[{values:{cooker:0},num_samples:$N_SAMPLES},{values:{cooker:1},num_samples:$N_SAMPLES}]"
        )
        ;;

    *)
        echo "Mode invalide: $MODE (attendu: unconditional | dataset_conditioned | user_conditioned)"
        exit 1
        ;;
esac

echo "Lancement de l'inférence Loadiff en mode '$MODE'..."
echo "Checkpoint: $CKPT_PATH"
echo "Dataset: $DATASET"
echo "Nombre d'échantillons: $N_SAMPLES"

printf 'Commande exécutée :'
printf ' %q' "${CMD[@]}"
printf '\n'

srun "${CMD[@]}"