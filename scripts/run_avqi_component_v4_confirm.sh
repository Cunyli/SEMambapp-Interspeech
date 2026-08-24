#!/bin/bash
# Submit or run the three locked AVQI v4 confirmation seeds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
DIAGNOSTIC_LAUNCHER="$SOURCE_ROOT/scripts/slurm_avqi_component_backprop_diagnostic.sh"
CONFIRM_KIND="${CONFIRM_KIND:-phase}"
case "$CONFIRM_KIND" in
  phase)
    DEFAULT_DATA_RUN_ROOT="$ROOT_DIR/runs/avqi_component_phaseaware_v4_data_20260816_02"
    DEFAULT_SCREEN_RUN_ROOT="$ROOT_DIR/runs/avqi_component_phaseaware_v4_screen_20260816_01"
    CONFIRM_RUN_STEM="avqi_component_phaseaware_v4_confirm"
    CONFIRM_JOB_PREFIX="avqi-v4-p"
    ;;
  direct)
    DEFAULT_DATA_RUN_ROOT="$ROOT_DIR/runs/avqi_component_direct_c_v5_data_20260816_02"
    DEFAULT_SCREEN_RUN_ROOT="$ROOT_DIR/runs/avqi_component_direct_c_v5_screen_20260817_01"
    CONFIRM_RUN_STEM="avqi_component_direct_c_v6_confirm"
    CONFIRM_JOB_PREFIX="avqi-v5-c"
    ;;
  full)
    DEFAULT_DATA_RUN_ROOT="$ROOT_DIR/runs/avqi_component_phaseaware_v4_data_20260816_02"
    DEFAULT_SCREEN_RUN_ROOT="$ROOT_DIR/runs/avqi_component_pretrained_full_tfgrid_v4_screen_20260816_01"
    CONFIRM_RUN_STEM="avqi_component_pretrained_full_tfgrid_v4_confirm"
    CONFIRM_JOB_PREFIX="avqi-v4-f"
    ;;
  *)
    echo "CONFIRM_KIND must be phase, direct, or full, got: $CONFIRM_KIND" >&2
    exit 2
    ;;
esac
DATA_RUN_ROOT="${DATA_RUN_ROOT:-$DEFAULT_DATA_RUN_ROOT}"
LABEL_RECEIPT="${LABEL_RECEIPT:-$DATA_RUN_ROOT/outputs/label_bank/receipt.json}"
SCREEN_RUN_ROOT="${SCREEN_RUN_ROOT:-$DEFAULT_SCREEN_RUN_ROOT}"
SCREEN_REPORT="${SCREEN_REPORT:-$SCREEN_RUN_ROOT/outputs/diagnostic_report.json}"
DEPENDENCY_JOB_ID="${DEPENDENCY_JOB_ID:-}"
CONFIRM_SEED="${CONFIRM_SEED:-}"
CONFIRMATION_SEEDS=(20260816 20260817 20260818)

PARTITION="${PARTITION:-gpu-v100-32g}"
GPU_TYPE="${GPU_TYPE:-v100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-48G}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
EXPECTED_TRAIN_SPEAKERS="${EXPECTED_TRAIN_SPEAKERS:-197}"
EXPECTED_CALIBRATION_SPEAKERS="${EXPECTED_CALIBRATION_SPEAKERS:-26}"
EXPECTED_HOLDOUT_SPEAKERS="${EXPECTED_HOLDOUT_SPEAKERS:-26}"
MAX_OPTIMIZER_STEPS="${MAX_OPTIMIZER_STEPS:-2000}"

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing to run from a dirty source tree: $SOURCE_ROOT" >&2
  exit 2
fi
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"

export ROOT_DIR SOURCE_ROOT DIAGNOSTIC_LAUNCHER DATA_RUN_ROOT LABEL_RECEIPT
export SCREEN_RUN_ROOT SCREEN_REPORT DEPENDENCY_JOB_ID PARTITION GPU_TYPE
export CPUS_PER_TASK MEMORY TIME_LIMIT EXPECTED_TRAIN_SPEAKERS
export EXPECTED_CALIBRATION_SPEAKERS EXPECTED_HOLDOUT_SPEAKERS
export MAX_OPTIMIZER_STEPS SOURCE_COMMIT
export CONFIRM_KIND CONFIRM_RUN_STEM CONFIRM_JOB_PREFIX

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  DEPENDENCY_ARGS=()
  if [[ -n "$DEPENDENCY_JOB_ID" ]]; then
    DEPENDENCY_ARGS=(--dependency="afterok:$DEPENDENCY_JOB_ID")
  elif [[ ! -f "$SCREEN_REPORT" ]]; then
    echo "Screen report is absent and no dependency job was supplied" >&2
    exit 2
  fi
  for seed in "${CONFIRMATION_SEEDS[@]}"; do
    run_root="$ROOT_DIR/runs/${CONFIRM_RUN_STEM}_seed${seed}_01"
    log_dir="$run_root/logs"
    output_dir="$run_root/outputs"
    checkpoint_dir="$ROOT_DIR/checkpoints/${CONFIRM_RUN_STEM}_seed${seed}_01"
    if [[ -e "$output_dir" || -e "$checkpoint_dir" ]]; then
      echo "Refusing to overwrite seed $seed outputs" >&2
      exit 2
    fi
    mkdir -p "$log_dir"
    sbatch \
      --parsable \
      --job-name="${CONFIRM_JOB_PREFIX}${seed: -2}" \
      --partition="$PARTITION" \
      --nodes=1 \
      --ntasks=1 \
      --gres="gpu:${GPU_TYPE}:1" \
      --cpus-per-task="$CPUS_PER_TASK" \
      --mem="$MEMORY" \
      --time="$TIME_LIMIT" \
      --output="$log_dir/slurm_%j.out" \
      --error="$log_dir/slurm_%j.err" \
      --export="ALL,CONFIRM_SEED=$seed,RUN_ROOT=$run_root,LOG_DIR=$log_dir,OUTPUT_DIR=$output_dir,CHECKPOINT_DIR=$checkpoint_dir" \
      "${DEPENDENCY_ARGS[@]}" \
      "$SELF_PATH"
  done
  exit 0
fi

if [[ ! " ${CONFIRMATION_SEEDS[*]} " =~ " ${CONFIRM_SEED} " ]]; then
  echo "Unexpected confirmation seed: $CONFIRM_SEED" >&2
  exit 2
fi
if [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" != "$SOURCE_COMMIT" ]]; then
  echo "Source HEAD drifted after submission" >&2
  exit 2
fi
for path in "$SCREEN_REPORT" "$LABEL_RECEIPT" "$DIAGNOSTIC_LAUNCHER"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required confirmation source: $path" >&2
    exit 2
  fi
done
SCREEN_DECISION="$(jq -er '.decision' "$SCREEN_REPORT")"
case "$CONFIRM_KIND:$SCREEN_DECISION" in
  direct:COMPLETED_ROUTE_C_SINGLE_SEED_SCREEN_NO_GENERATOR_UPDATE) ;;
  phase:COMPLETED_SINGLE_SEED_SCREEN_NO_GENERATOR_UPDATE) ;;
  full:COMPLETED_SINGLE_SEED_SCREEN_NO_GENERATOR_UPDATE) ;;
  *)
    echo "Screen report is not complete for $CONFIRM_KIND: $SCREEN_DECISION" >&2
    exit 2
    ;;
esac
if [[ "$(jq -er '.generator_optimizer_steps' "$SCREEN_REPORT")" != "0" ]]; then
  echo "Screen report contains generator updates" >&2
  exit 2
fi
if [[ "$(jq -er '.contract.source_commit' "$SCREEN_REPORT")" != "$SOURCE_COMMIT" ]]; then
  echo "Screen source commit differs from confirmation source" >&2
  exit 2
fi

ROUTE_SCOPE="$(jq -er '.contract.route_scope // "all"' "$SCREEN_REPORT")"
PYTHON_SCRIPT="$SOURCE_ROOT/scripts/evaluate_avqi_component_backprop.py"
if [[ "$CONFIRM_KIND" == "direct" ]]; then
  if [[ "$ROUTE_SCOPE" != "direct_only" ]]; then
    echo "Direct confirmation requires a direct_only screen" >&2
    exit 2
  fi
  SHARED_CANDIDATES=""
  WAVEFORM_ARCHITECTURES="$(jq -er '.routes.direct_differentiable_estimator.selected_architecture' "$SCREEN_REPORT")"
  PYTHON_SCRIPT="$SOURCE_ROOT/scripts/evaluate_avqi_component_direct_c.py"
  MAX_OPTIMIZER_STEPS=0
  case "$WAVEFORM_ARCHITECTURES" in
    direct_praat_hard_v2|direct_praat_hard_cpps_relative_log1p_v10|direct_praat_hard_cpps_view_input_v12|direct_praat_hard_hnr_pitch_path_v7|direct_praat_hard_cpps_view_input_v12_hnr_pitch_path_v7|direct_praat_hard_shimmer_rms_v3|direct_praat_hard_shimmer_raw_cc_surrogate_v4|direct_praat_hard_shimmer_pulse_chain_v5|direct_praat_hard_shimmer_pulse_path_v6) ;;
    *)
      echo "Unexpected locked Route C estimator: $WAVEFORM_ARCHITECTURES" >&2
      exit 2
      ;;
  esac
else
  if [[ "$ROUTE_SCOPE" != "all" ]]; then
    echo "$CONFIRM_KIND confirmation requires the two-route screen contract" >&2
    exit 2
  fi
  SHARED_CANDIDATES="$(jq -er '.routes.shared_dual_head.selected_candidate' "$SCREEN_REPORT")"
  WAVEFORM_ARCHITECTURES="$(jq -er '.routes.frozen_independent_predictor.selected_architecture' "$SCREEN_REPORT")"
  if [[ "$SHARED_CANDIDATES" != "output_phase_tfgrid" ]]; then
    echo "Unexpected locked shared candidate: $SHARED_CANDIDATES" >&2
    exit 2
  fi
  case "$CONFIRM_KIND:$WAVEFORM_ARCHITECTURES" in
    phase:frequency_aware|phase:phase_frequency_aware|phase:phase_compact_tfgrid) ;;
    full:pretrained_full_tfgrid) ;;
    *)
      echo "Unexpected locked $CONFIRM_KIND architecture: $WAVEFORM_ARCHITECTURES" >&2
      exit 2
      ;;
  esac
fi
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  echo "Missing scorer source: $PYTHON_SCRIPT" >&2
  exit 2
fi

LABEL_BANK="$(jq -er '.internal_label_bank' "$LABEL_RECEIPT")"
LABEL_BANK_SHA256="$(jq -er '.internal_label_bank_sha256' "$LABEL_RECEIPT")"
VCTK_EXTERNAL_LABEL_BANK="$(jq -er '.external_label_bank' "$LABEL_RECEIPT")"
VCTK_EXTERNAL_LABEL_BANK_SHA256="$(jq -er '.external_label_bank_sha256' "$LABEL_RECEIPT")"
CONFIG="$ROOT_DIR/runs/tau_s1_sv_threshold_ablation_20260719_01/configs/s_fidelity_m3_stage0500.yaml"
CHECKPOINT="$ROOT_DIR/checkpoints/S3_500/ln_g_00000500.pth"
EXTERNAL_EXACT_CSV="$ROOT_DIR/runs/tau_pathology_three_tracks_20260810_01/outputs/intensity_eval/exact_components_all.csv"
CONFIG_SHA256="$(sha256sum "$CONFIG" | awk '{print $1}')"
CHECKPOINT_SHA256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
EXTERNAL_EXACT_CSV_SHA256="$(sha256sum "$EXTERNAL_EXACT_CSV" | awk '{print $1}')"
SEED="$CONFIRM_SEED"
JOB_NAME="${CONFIRM_JOB_PREFIX}${CONFIRM_SEED: -2}"

export SHARED_CANDIDATES WAVEFORM_ARCHITECTURES LABEL_BANK LABEL_BANK_SHA256
export VCTK_EXTERNAL_LABEL_BANK VCTK_EXTERNAL_LABEL_BANK_SHA256 CONFIG
export CHECKPOINT EXTERNAL_EXACT_CSV CONFIG_SHA256 CHECKPOINT_SHA256
export EXTERNAL_EXACT_CSV_SHA256 SEED JOB_NAME RUN_ROOT LOG_DIR OUTPUT_DIR
export CHECKPOINT_DIR ROUTE_SCOPE PYTHON_SCRIPT MAX_OPTIMIZER_STEPS

exec bash "$DIAGNOSTIC_LAUNCHER"
