#!/bin/bash
# Run one frozen AVQI v4 scorer screen after the exact-label data job succeeds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
DIAGNOSTIC_LAUNCHER="$SOURCE_ROOT/scripts/slurm_avqi_component_backprop_diagnostic.sh"
DATA_RUN_ROOT="${DATA_RUN_ROOT:-$ROOT_DIR/runs/avqi_component_phaseaware_v4_data_20260816_02}"
LABEL_RECEIPT="${LABEL_RECEIPT:-$DATA_RUN_ROOT/outputs/label_bank/receipt.json}"
SCREEN_KIND="${SCREEN_KIND:-phase}"
DEPENDENCY_JOB_ID="${DEPENDENCY_JOB_ID:-}"
SEED="${SEED:-20260815}"

PARTITION="${PARTITION:-gpu-v100-32g}"
GPU_TYPE="${GPU_TYPE:-v100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-48G}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
EXPECTED_TRAIN_SPEAKERS="${EXPECTED_TRAIN_SPEAKERS:-197}"
EXPECTED_CALIBRATION_SPEAKERS="${EXPECTED_CALIBRATION_SPEAKERS:-26}"
EXPECTED_HOLDOUT_SPEAKERS="${EXPECTED_HOLDOUT_SPEAKERS:-26}"
MAX_OPTIMIZER_STEPS="${MAX_OPTIMIZER_STEPS:-2000}"
SHARED_CANDIDATES="output_phase_tfgrid"
FULL_TFGRID_CHECKPOINT="${FULL_TFGRID_CHECKPOINT:-}"

case "$SCREEN_KIND" in
  phase)
    JOB_NAME="avqi-v4-phase"
    WAVEFORM_ARCHITECTURES="frequency_aware,phase_frequency_aware,phase_compact_tfgrid"
    RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/avqi_component_phaseaware_v4_screen_20260816_01}"
    CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/checkpoints/avqi_component_phaseaware_v4_screen_20260816_01}"
    ;;
  direct)
    JOB_NAME="avqi-v4-direct"
    WAVEFORM_ARCHITECTURES="direct_praat_hard_v2"
    RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/avqi_component_direct_hard_v4_screen_20260816_01}"
    CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/checkpoints/avqi_component_direct_hard_v4_screen_20260816_01}"
    ;;
  full_tfgrid)
    JOB_NAME="avqi-v4-fullgrid"
    WAVEFORM_ARCHITECTURES="pretrained_full_tfgrid"
    RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/avqi_component_pretrained_full_tfgrid_v4_screen_20260816_01}"
    CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/checkpoints/avqi_component_pretrained_full_tfgrid_v4_screen_20260816_01}"
    FULL_TFGRID_CHECKPOINT="${FULL_TFGRID_CHECKPOINT:-/scratch/work/lil14/Hybrid_Unise/checkpoints/fusion_init_DISC-step122000_GEN-best-valnll2.358177-step092000/hybrid_unise_DISC_latest_step122000_epoch27.ckpt}"
    ;;
  *)
    echo "SCREEN_KIND must be phase, direct, or full_tfgrid, got: $SCREEN_KIND" >&2
    exit 2
    ;;
esac

LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
CONFIG="${CONFIG:-$ROOT_DIR/runs/tau_s1_sv_threshold_ablation_20260719_01/configs/s_fidelity_m3_stage0500.yaml}"
CHECKPOINT="${CHECKPOINT:-$ROOT_DIR/checkpoints/S3_500/ln_g_00000500.pth}"
EXTERNAL_EXACT_CSV="${EXTERNAL_EXACT_CSV:-$ROOT_DIR/runs/tau_pathology_three_tracks_20260810_01/outputs/intensity_eval/exact_components_all.csv}"

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing to run from a dirty source tree: $SOURCE_ROOT" >&2
  exit 2
fi
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"

REQUIRED_PATHS=("$DIAGNOSTIC_LAUNCHER" "$CONFIG" "$CHECKPOINT" "$EXTERNAL_EXACT_CSV")
if [[ "$SCREEN_KIND" == "full_tfgrid" ]]; then
  REQUIRED_PATHS+=("$FULL_TFGRID_CHECKPOINT")
fi
for path in "${REQUIRED_PATHS[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required source: $path" >&2
    exit 2
  fi
done

export ROOT_DIR SOURCE_ROOT DIAGNOSTIC_LAUNCHER DATA_RUN_ROOT LABEL_RECEIPT
export SCREEN_KIND DEPENDENCY_JOB_ID SEED PARTITION GPU_TYPE CPUS_PER_TASK MEMORY
export TIME_LIMIT EXPECTED_TRAIN_SPEAKERS EXPECTED_CALIBRATION_SPEAKERS
export EXPECTED_HOLDOUT_SPEAKERS MAX_OPTIMIZER_STEPS SHARED_CANDIDATES
export WAVEFORM_ARCHITECTURES JOB_NAME RUN_ROOT CHECKPOINT_DIR LOG_DIR OUTPUT_DIR
export CONFIG CHECKPOINT EXTERNAL_EXACT_CSV SOURCE_COMMIT
export FULL_TFGRID_CHECKPOINT

mkdir -p "$LOG_DIR"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  if [[ -e "$OUTPUT_DIR" || -e "$CHECKPOINT_DIR" ]]; then
    echo "Refusing to overwrite output or checkpoints: $OUTPUT_DIR $CHECKPOINT_DIR" >&2
    exit 2
  fi
  DEPENDENCY_ARGS=()
  if [[ -n "$DEPENDENCY_JOB_ID" ]]; then
    DEPENDENCY_ARGS=(--dependency="afterok:$DEPENDENCY_JOB_ID")
  elif [[ ! -f "$LABEL_RECEIPT" ]]; then
    echo "Label receipt is absent and no dependency job was supplied" >&2
    exit 2
  fi
  sbatch \
    --parsable \
    --job-name="$JOB_NAME" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --gres="gpu:${GPU_TYPE}:1" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEMORY" \
    --time="$TIME_LIMIT" \
    --output="$LOG_DIR/slurm_%j.out" \
    --error="$LOG_DIR/slurm_%j.err" \
    --export=ALL \
    "${DEPENDENCY_ARGS[@]}" \
    "$SELF_PATH"
  exit 0
fi

if [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" != "$SOURCE_COMMIT" ]]; then
  echo "Source HEAD drifted after submission" >&2
  exit 2
fi
if [[ ! -f "$LABEL_RECEIPT" ]]; then
  echo "Missing completed label receipt: $LABEL_RECEIPT" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" || -e "$CHECKPOINT_DIR" ]]; then
  echo "Refusing to overwrite output or checkpoints: $OUTPUT_DIR $CHECKPOINT_DIR" >&2
  exit 2
fi

LABEL_BANK="$(jq -er '.internal_label_bank' "$LABEL_RECEIPT")"
LABEL_BANK_SHA256="$(jq -er '.internal_label_bank_sha256' "$LABEL_RECEIPT")"
VCTK_EXTERNAL_LABEL_BANK="$(jq -er '.external_label_bank' "$LABEL_RECEIPT")"
VCTK_EXTERNAL_LABEL_BANK_SHA256="$(jq -er '.external_label_bank_sha256' "$LABEL_RECEIPT")"
if [[ "$(jq -er '.schema_version' "$LABEL_RECEIPT")" != "avqi-component-label-bank-v4" ]]; then
  echo "Unexpected label receipt schema" >&2
  exit 2
fi
if [[ "$(sha256sum "$LABEL_BANK" | awk '{print $1}')" != "$LABEL_BANK_SHA256" ]]; then
  echo "Internal label bank hash mismatch" >&2
  exit 2
fi
if [[ "$(sha256sum "$VCTK_EXTERNAL_LABEL_BANK" | awk '{print $1}')" != "$VCTK_EXTERNAL_LABEL_BANK_SHA256" ]]; then
  echo "VCTK external label bank hash mismatch" >&2
  exit 2
fi

CONFIG_SHA256="$(sha256sum "$CONFIG" | awk '{print $1}')"
CHECKPOINT_SHA256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
EXTERNAL_EXACT_CSV_SHA256="$(sha256sum "$EXTERNAL_EXACT_CSV" | awk '{print $1}')"
FULL_TFGRID_CHECKPOINT_SHA256=""
if [[ "$SCREEN_KIND" == "full_tfgrid" ]]; then
  FULL_TFGRID_CHECKPOINT_SHA256="$(sha256sum "$FULL_TFGRID_CHECKPOINT" | awk '{print $1}')"
fi
export LABEL_BANK LABEL_BANK_SHA256 VCTK_EXTERNAL_LABEL_BANK
export VCTK_EXTERNAL_LABEL_BANK_SHA256 CONFIG_SHA256 CHECKPOINT_SHA256
export EXTERNAL_EXACT_CSV_SHA256
export FULL_TFGRID_CHECKPOINT FULL_TFGRID_CHECKPOINT_SHA256

exec bash "$DIAGNOSTIC_LAUNCHER"
