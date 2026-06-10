#!/bin/bash
# =============================================================================
# Shared configuration + helpers for the TSTR SLURM jobs.
#
# Sourced by each scripts/tstr_evaluation/slurm/submit_tstr_*.sh job *after* its
# SLURM header. Jobs run with the working directory set to the repository root
# (the directory from which `sbatch` was called), so all paths below are
# repo-root-relative — exactly like scripts/bash/selena/submit_ablation_*.sh.
#
# Every value can be overridden from the environment before submitting (sbatch
# exports the submitting environment by default), e.g.:
#   SYNTHETIC_DATA_DIR=runs_inference/my_run \
#   TSTR_CLASSIFIERS="rocket" \
#     scripts/tstr_evaluation/run_experiments.sh
# =============================================================================

DATASET="${DATASET:-smach}"
REAL_DATA_DIR="${REAL_DATA_DIR:-data/${DATASET}}"
SYNTHETIC_DATA_DIR="${SYNTHETIC_DATA_DIR:-runs_inference/inference_user_conditioned_latest}"
DEVICE="${DEVICE:-cuda}"

# Base directory holding the per-dataset run roots for the generative baselines, laid
# out as <RUNS_BASE>/runs_<dataset>/ (override RUNS_BASE for a different scratch path).
# RUNS_BASE is dataset-agnostic, so switching DATASET automatically targets the right
# run root — unlike exporting RUNS_ROOT directly, which is easy to leave stale.
RUNS_BASE="${RUNS_BASE:-runs_}"

# Default appliances per dataset (override with TSTR_APPLIANCES="a b c").
case "${DATASET}" in
  cer)     _DEFAULT_APPLIANCES="cooker dishwasher water_heater" ;;
  *) echo "Unknown DATASET=${DATASET} (expected cer)" >&2; exit 1 ;;
esac

# Results are separated per dataset; smach keeps its legacy path (unchanged behaviour).
if [[ "${DATASET}" == "smach" ]]; then
  RESULTS_DIR="${RESULTS_DIR:-results/tstr_experiments}"
else
  RESULTS_DIR="${RESULTS_DIR:-results/tstr_experiments/${DATASET}}"
fi

# Matrices (space-separated env overrides -> bash arrays).
APPLIANCES=(${TSTR_APPLIANCES:-${_DEFAULT_APPLIANCES}})
CLASSIFIERS=(${TSTR_CLASSIFIERS:-rocket transapp})
MIX_PERCENTAGES=(${TSTR_MIX_PERCENTAGES:-5 20 50 100})

SCRIPT_DIR="scripts/tstr_evaluation"
CONFIG_DIR="${SCRIPT_DIR}/configs"

# `srun` launches each Python invocation as a job step inside the allocation.
# Set SRUN="" to run a submit script directly (e.g. for local debugging).
SRUN="${SRUN-srun}"

config_for() {  # $1 = classifier -> path to its YAML config
  if [[ "$1" == "rocket" ]]; then
    echo "${CONFIG_DIR}/rocket_tstr.yaml"
  else
    echo "${CONFIG_DIR}/transapp_tstr.yaml"
  fi
}

resolve_runs_root() {
  # Echo the run root for the current DATASET (generative-baseline data).
  # Defaults to <RUNS_BASE>/runs_<dataset>. If RUNS_ROOT is set but follows the
  # runs_<name> convention for a *different* dataset (a stale export — the classic
  # "DATASET=cer_bis but path points to runs_smach" bug), fail fast with guidance.
  local default_root="${RUNS_BASE}/runs_${DATASET}"
  local root="${RUNS_ROOT:-${default_root}}"
  local base; base="$(basename "${root}")"
  if [[ "${base}" == runs_* && "${base}" != "runs_${DATASET}" ]]; then
    echo "ERROR: RUNS_ROOT=${root} targets '${base}' but DATASET=${DATASET}." >&2
    echo "       Unset RUNS_ROOT to use the default ${default_root}, or set" >&2
    echo "       RUNS_ROOT=${RUNS_BASE}/runs_${DATASET} (or RUNS_BASE=<scratch> with RUNS_ROOT unset)." >&2
    return 1
  fi
  echo "${root}"
}

train_and_eval() {  # $1=classifier $2=appliance $3=train_path $4=out_dir [$5=train_split]
  local classifier="$1" appliance="$2" train_path="$3" out_dir="$4" train_split="${5:-}"
  mkdir -p "${out_dir}"

  ${SRUN} python3 -m scripts.tstr_evaluation.train_classifiers \
    --classifier_type "${classifier}" \
    --target_label "${appliance}" \
    --dataset "${DATASET}" \
    --train_data_path "${train_path}" \
    ${train_split:+--train_split "${train_split}"} \
    --val_data_path "${REAL_DATA_DIR}" --val_split val \
    --output_dir "${out_dir}" \
    --config "$(config_for "${classifier}")" \
    --device "${DEVICE}"

  local model_path
  if [[ "${classifier}" == "rocket" ]]; then
    model_path="${out_dir}/model.pkl"
  else
    model_path="${out_dir}/checkpoint.pt"
  fi

  ${SRUN} python3 -m scripts.tstr_evaluation.evaluate_tstr \
    --classifier_type "${classifier}" \
    --target_label "${appliance}" \
    --dataset "${DATASET}" \
    --model_path "${model_path}" \
    --test_data_path "${REAL_DATA_DIR}" --test_split test \
    --output_dir "${out_dir}" \
    --device "${DEVICE}"
}

baseline_train_and_eval() {  # $1=classifier $2=appliance $3=train_path $4=out_dir
  # Like train_and_eval but for the generative baselines: the synthetic training
  # data is *never* post-processed (--no_postprocess). Test stays the real test split.
  local classifier="$1" appliance="$2" train_path="$3" out_dir="$4"
  mkdir -p "${out_dir}"

  ${SRUN} python3 -m scripts.tstr_evaluation.train_classifiers \
    --classifier_type "${classifier}" \
    --target_label "${appliance}" \
    --dataset "${DATASET}" \
    --train_data_path "${train_path}" --no_postprocess \
    --val_data_path "${REAL_DATA_DIR}" --val_split val \
    --output_dir "${out_dir}" \
    --config "$(config_for "${classifier}")" \
    --device "${DEVICE}"

  local model_path
  if [[ "${classifier}" == "rocket" ]]; then
    model_path="${out_dir}/model.pkl"
  else
    model_path="${out_dir}/checkpoint.pt"
  fi

  ${SRUN} python3 -m scripts.tstr_evaluation.evaluate_tstr \
    --classifier_type "${classifier}" \
    --target_label "${appliance}" \
    --dataset "${DATASET}" \
    --model_path "${model_path}" \
    --test_data_path "${REAL_DATA_DIR}" --test_split test \
    --output_dir "${out_dir}" \
    --device "${DEVICE}"
}

make_mixed_dataset() {  # $1=appliance $2=percentage $3=output_dir
  ${SRUN} python3 -m scripts.tstr_evaluation.utils.create_mixed_dataset \
    --synthetic_dir "${SYNTHETIC_DATA_DIR}" \
    --real_dir "${REAL_DATA_DIR}" \
    --output_dir "$3" \
    --percentage "$2" \
    --target_label "$1" \
    --dataset "${DATASET}"
}

make_mixed_dataset_baseline() {  # $1=appliance $2=percentage $3=synthetic_dir $4=output_dir
  # Like make_mixed_dataset but the synthetic source is the baseline's own population
  # ($3) and it is *never* post-processed (--no_postprocess), matching the rest of the
  # baseline flow. The real fraction comes from the dataset's real train split.
  ${SRUN} python3 -m scripts.tstr_evaluation.utils.create_mixed_dataset \
    --synthetic_dir "$3" \
    --real_dir "${REAL_DATA_DIR}" \
    --output_dir "$4" \
    --percentage "$2" \
    --target_label "$1" \
    --dataset "${DATASET}" \
    --no_postprocess
}

baseline_tstr_all() {  # $1=baseline $2=mode $3=appliance $4=synthetic_dir
  # Run the full baseline TSTR matrix for one appliance from an already-materialised
  # synthetic population ($4): pure-synthetic (Exp1 analog) for every classifier — unless
  # TSTR_BASELINE_PURE=0 — then — unless TSTR_BASELINE_MIX=0 — the mixed Synthetic + N% Real
  # sweep (Exp2 analog) over MIX_PERCENTAGES. Set TSTR_BASELINE_PURE=0 to run *only* the
  # mixed phase (e.g. to backfill missing mixes without recomputing the pure results).
  # Test always stays the real test split; no post-processing is applied.
  # Nomenclature:
  #   pure:  baselines/<dataset>/<baseline>/<mode>/<classifier>/<appliance>
  #   mixed: baselines/<dataset>/<baseline>/<mode>/exp2_mixed/<pct>pct/<classifier>/<appliance>
  local baseline="$1" mode="$2" appliance="$3" syn_dir="$4"
  local base_out="${RESULTS_DIR}/baselines/${DATASET}/${baseline}/${mode}"

  if [[ "${TSTR_BASELINE_PURE:-1}" == "1" ]]; then
    for classifier in "${CLASSIFIERS[@]}"; do
      echo "--- [tstr pure] ${baseline} / ${classifier} / ${appliance}"
      baseline_train_and_eval "${classifier}" "${appliance}" "${syn_dir}" \
        "${base_out}/${classifier}/${appliance}"
    done
  fi

  if [[ "${TSTR_BASELINE_MIX:-1}" != "1" ]]; then
    return 0
  fi
  for pct in "${MIX_PERCENTAGES[@]}"; do
    local mix_dir="${base_out}/exp2_mixed/${pct}pct/_datasets/${appliance}"
    echo "--- [tstr mix ${pct}%] building mixed dataset for ${baseline} / ${appliance}"
    make_mixed_dataset_baseline "${appliance}" "${pct}" "${syn_dir}" "${mix_dir}"
    for classifier in "${CLASSIFIERS[@]}"; do
      echo "--- [tstr mix ${pct}%] ${baseline} / ${classifier} / ${appliance}"
      baseline_train_and_eval "${classifier}" "${appliance}" "${mix_dir}" \
        "${base_out}/exp2_mixed/${pct}pct/${classifier}/${appliance}"
    done
  done
}
