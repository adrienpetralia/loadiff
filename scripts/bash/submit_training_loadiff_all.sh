#!/bin/bash
#
# Script maitre : boucle sur toutes les combinaisons de conditionnement
# possibles pour un dataset donne et soumet un entrainement LoaDiff par
# combinaison via submit_training_loadiff.sh.
#
# Combinaisons couvertes :
#   - aucun appareil ;
#   - chaque sous-ensemble des appareils ;
#   - avec et sans temperature.
#
# Pour N appareils, cela represente 2^(N+1) entrainements.
#
# Usage :
#   ./submit_training_loadiff_all.sh [--dry-run]
#
# Options :
#   --dry-run   Affiche les commandes generees sans soumettre les jobs.

# --- Configuration (prerempli pour cer) ---
DATASET="cer"
APPLIANCES=("cooker" "dishwasher" "water_heater")

# Repertoire du script, pour localiser submit_training_loadiff.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_training_loadiff.sh"

# --- Parsing des arguments ---
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Argument inconnu : $1" >&2
            exit 1
            ;;
    esac
done

N=${#APPLIANCES[@]}
TOTAL=$(( (1 << N) * 2 ))
echo "Dataset    : ${DATASET}"
echo "Appareils  : ${APPLIANCES[*]}"
echo "Combinaisons : ${TOTAL} (${N} appareils, avec/sans temperature)"
echo

# --- Boucle sur tous les sous-ensembles d'appareils (masque binaire) ---
for (( mask = 0; mask < (1 << N); mask++ )); do

    # Construit le sous-ensemble correspondant au masque courant
    subset=()
    for (( i = 0; i < N; i++ )); do
        if (( (mask >> i) & 1 )); then
            subset+=("${APPLIANCES[$i]}")
        fi
    done
    appliances_arg="$(IFS=,; echo "${subset[*]}")"

    # Avec et sans temperature
    for temp_flag in "--no-temperature" "--temperature"; do

        cmd=(sbatch "${SUBMIT_SCRIPT}" --dataset "${DATASET}")
        if [[ -n "${appliances_arg}" ]]; then
            cmd+=(--appliances "${appliances_arg}")
        fi
        cmd+=("${temp_flag}")

        if [[ "${DRY_RUN}" == true ]]; then
            echo "${cmd[*]}"
        else
            "${cmd[@]}"
        fi
    done
done