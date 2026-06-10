#!/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --job-name=training
#SBATCH -o ./jobs/%j.out
#SBATCH -e ./jobs/%j.err

# Lance l'entrainement d'un modele LoaDiff, conditionne ou non.
#
# Les arguments CLI surchargent dynamiquement les valeurs de
# configs/loadiff_with_conditioning.yaml. Le nom du modele est genere
# automatiquement a partir des appareils utilises pour le conditionnement
# et de la presence eventuelle de la temperature.
#
# Usage :
#   sbatch submit_training_loadiff.sh \
#       --dataset cer \
#       --appliances cooker,dishwasher \
#       --temperature \
#       [hydra.override=valeur ...]
#
# Options :
#   --dataset NAME        Dataset cible (defaut: cer).
#   --appliances LIST     Liste d'appareils separes par des virgules pour le
#                         conditionnement multi-label. Vide => inconditionnel.
#   --temperature         Active le conditionnement par la temperature.
#   --no-temperature      Desactive le conditionnement par la temperature (defaut).
#   Tout argument supplementaire est transmis tel quel a Hydra.

# --- Valeurs par defaut ---
DATASET="cer"
APPLIANCES=""
TEMPERATURE=false
EXTRA_OVERRIDES=()

# --- Parsing dynamique des arguments CLI ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --appliances)
            APPLIANCES="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE=true
            shift
            ;;
        --no-temperature)
            TEMPERATURE=false
            shift
            ;;
        *)
            # Surcharge Hydra transmise telle quelle (ex: training.lr=1e-4)
            EXTRA_OVERRIDES+=("$1")
            shift
            ;;
    esac
done

# --- Construction de la liste des appareils ---
# "CHAUFF_ELEC,ECS" => tableau ("CHAUFF_ELEC" "ECS")
APPLIANCE_ARRAY=()
if [[ -n "${APPLIANCES}" ]]; then
    IFS=',' read -r -a APPLIANCE_ARRAY <<< "${APPLIANCES}"
fi

# --- Generation automatique et explicite du nom du modele ---
# Forme : loadiff_<APP1>_<APP2>...[_temp], ou loadiff_unconditional si aucun
#         appareil ni temperature.
MODEL_NAME="loadiff"
if [[ ${#APPLIANCE_ARRAY[@]} -gt 0 ]]; then
    MODEL_NAME="${MODEL_NAME}_$(IFS=_; echo "${APPLIANCE_ARRAY[*]}")"
fi
if [[ "${TEMPERATURE}" == true ]]; then
    MODEL_NAME="${MODEL_NAME}_temp"
fi
if [[ ${#APPLIANCE_ARRAY[@]} -eq 0 && "${TEMPERATURE}" == false ]]; then
    MODEL_NAME="loadiff_unconditional"
fi

# --- Construction des surcharges Hydra ---
# Liste d'appareils au format Hydra : [CHAUFF_ELEC,ECS] (ou [] si inconditionnel)
BOOL_COL_NAMES="[$(IFS=,; echo "${APPLIANCE_ARRAY[*]}")]"

OVERRIDES=(
    "data.dataset=${DATASET}"
    "data.bool_col_names=${BOOL_COL_NAMES}"
    "model_name=${MODEL_NAME}"
    "ditmodelargs.temperature=${TEMPERATURE}"
)

# La temperature n'est chargee que si un chemin est fourni : on l'annule sinon.
if [[ "${TEMPERATURE}" == false ]]; then
    OVERRIDES+=("data.path_temperature=null")
fi

if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
    OVERRIDES+=("${EXTRA_OVERRIDES[@]}")
fi

echo "Dataset     : ${DATASET}"
echo "Appareils   : ${APPLIANCES:-<aucun>}"
echo "Temperature : ${TEMPERATURE}"
echo "Modele      : ${MODEL_NAME}"

# --- Environnement ---
mkdir -p jobs
source .venv/bin/activate

# --- Execution ---
srun python3 -m scripts.training.train_loadiff "${OVERRIDES[@]}"