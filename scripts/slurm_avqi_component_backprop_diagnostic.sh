#!/bin/bash
# AVQI scorer screen only: no generator optimizer step is implemented by this job.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$DEFAULT_ROOT}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-$SOURCE_ROOT/scripts/evaluate_avqi_component_backprop.py}"
ROUTE_SCOPE="${ROUTE_SCOPE:-all}"
JOB_NAME="${JOB_NAME:-avqi-predictor-screen}"
PARTITION="${PARTITION:-gpu-v100-32g}"
GPU_TYPE="${GPU_TYPE:-v100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-48G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/avqi_component_predictor_screen_20260813_01}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/checkpoints/avqi_component_predictor_screen_20260813_01}"
SEED="${SEED:-20260813}"
EXPECTED_TRAIN_SPEAKERS="${EXPECTED_TRAIN_SPEAKERS:-70}"
EXPECTED_CALIBRATION_SPEAKERS="${EXPECTED_CALIBRATION_SPEAKERS:-14}"
EXPECTED_HOLDOUT_SPEAKERS="${EXPECTED_HOLDOUT_SPEAKERS:-14}"
SHARED_CANDIDATES="${SHARED_CANDIDATES:-late_global,late_frequency,late_tfgrid}"
WAVEFORM_ARCHITECTURES="${WAVEFORM_ARCHITECTURES:-global_stats,frequency_aware,compact_tfgrid}"
MAX_OPTIMIZER_STEPS="${MAX_OPTIMIZER_STEPS:-0}"
LABEL_BANK="${LABEL_BANK:-$ROOT_DIR/runs/tau_pathology_preservation_eval_phase2_20260809_01/outputs/surrogate/exact_component_label_bank_v1.csv}"
CONFIG="${CONFIG:-$ROOT_DIR/runs/tau_s1_sv_threshold_ablation_20260719_01/configs/s_fidelity_m3_stage0500.yaml}"
CHECKPOINT="${CHECKPOINT:-$ROOT_DIR/checkpoints/S3_500/ln_g_00000500.pth}"
EXTERNAL_EXACT_CSV="${EXTERNAL_EXACT_CSV:-$ROOT_DIR/runs/tau_pathology_three_tracks_20260810_01/outputs/intensity_eval/exact_components_all.csv}"
VCTK_EXTERNAL_LABEL_BANK="${VCTK_EXTERNAL_LABEL_BANK:-}"
FULL_TFGRID_CHECKPOINT="${FULL_TFGRID_CHECKPOINT:-}"
LABEL_BANK_SHA256="${LABEL_BANK_SHA256:?LABEL_BANK_SHA256 is required}"
CONFIG_SHA256="${CONFIG_SHA256:?CONFIG_SHA256 is required}"
CHECKPOINT_SHA256="${CHECKPOINT_SHA256:?CHECKPOINT_SHA256 is required}"
EXTERNAL_EXACT_CSV_SHA256="${EXTERNAL_EXACT_CSV_SHA256:?EXTERNAL_EXACT_CSV_SHA256 is required}"
VCTK_EXTERNAL_LABEL_BANK_SHA256="${VCTK_EXTERNAL_LABEL_BANK_SHA256:-}"
FULL_TFGRID_CHECKPOINT_SHA256="${FULL_TFGRID_CHECKPOINT_SHA256:-}"

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing to run from a dirty source tree: $SOURCE_ROOT" >&2
  exit 2
fi
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"

# Slurm executes a copied spool script, so BASH_SOURCE no longer points into the
# repository inside the allocation. Preserve every resolved path and contract
# value explicitly across the submit boundary.
export ROOT_DIR SOURCE_ROOT PYTHON_SCRIPT
export ROUTE_SCOPE
export JOB_NAME PARTITION GPU_TYPE CPUS_PER_TASK MEMORY TIME_LIMIT
export SOFTWARE_STACK_MODULE COMPILER_MODULE
export RUN_ROOT LOG_DIR OUTPUT_DIR CHECKPOINT_DIR
export SEED EXPECTED_TRAIN_SPEAKERS EXPECTED_CALIBRATION_SPEAKERS
export EXPECTED_HOLDOUT_SPEAKERS SHARED_CANDIDATES WAVEFORM_ARCHITECTURES
export MAX_OPTIMIZER_STEPS
export LABEL_BANK CONFIG CHECKPOINT EXTERNAL_EXACT_CSV
export VCTK_EXTERNAL_LABEL_BANK VCTK_EXTERNAL_LABEL_BANK_SHA256
export LABEL_BANK_SHA256 CONFIG_SHA256 CHECKPOINT_SHA256
export EXTERNAL_EXACT_CSV_SHA256 FULL_TFGRID_CHECKPOINT
export FULL_TFGRID_CHECKPOINT_SHA256 SOURCE_COMMIT

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
  sbatch \
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
    "$SELF_PATH"
  exit 0
fi

if [[ -e "$OUTPUT_DIR" || -e "$CHECKPOINT_DIR" ]]; then
  echo "Refusing to overwrite output or checkpoints: $OUTPUT_DIR $CHECKPOINT_DIR" >&2
  exit 2
fi

cd "$ROOT_DIR"
module load "$SOFTWARE_STACK_MODULE"
module load "$COMPILER_MODULE"
eval "$(conda shell.bash hook)"
conda activate semambapp
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export PYTHONPATH="$SOURCE_ROOT:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LIVE_LOG="$LOG_DIR/avqi_component_diagnostic_${SLURM_JOB_ID}.log"
echo "event=start job=$SLURM_JOB_ID time=$(date -Is)" | tee -a "$LIVE_LOG"
python -c 'import os, torch; print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("device", torch.cuda.get_device_name(0)); print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))' | tee -a "$LIVE_LOG"

FULL_TFGRID_ARGS=()
if [[ -n "$FULL_TFGRID_CHECKPOINT" || -n "$FULL_TFGRID_CHECKPOINT_SHA256" ]]; then
  if [[ -z "$FULL_TFGRID_CHECKPOINT" || -z "$FULL_TFGRID_CHECKPOINT_SHA256" ]]; then
    echo "Both FULL_TFGRID_CHECKPOINT and its SHA256 are required" >&2
    exit 2
  fi
  FULL_TFGRID_ARGS=(
    --full-tfgrid-checkpoint "$FULL_TFGRID_CHECKPOINT"
    --full-tfgrid-checkpoint-sha256 "$FULL_TFGRID_CHECKPOINT_SHA256"
  )
fi

VCTK_EXTERNAL_ARGS=()
if [[ -n "$VCTK_EXTERNAL_LABEL_BANK" || -n "$VCTK_EXTERNAL_LABEL_BANK_SHA256" ]]; then
  if [[ -z "$VCTK_EXTERNAL_LABEL_BANK" || -z "$VCTK_EXTERNAL_LABEL_BANK_SHA256" ]]; then
    echo "Both VCTK external label bank and its SHA256 are required" >&2
    exit 2
  fi
  VCTK_EXTERNAL_ARGS=(
    --vctk-external-label-bank "$VCTK_EXTERNAL_LABEL_BANK"
    --vctk-external-label-bank-sha256 "$VCTK_EXTERNAL_LABEL_BANK_SHA256"
  )
fi

ROUTE_SCOPE_ARGS=()
SHARED_CANDIDATE_ARGS=(--shared-candidates "$SHARED_CANDIDATES")
case "$ROUTE_SCOPE" in
  all) ;;
  direct_only)
    if [[ "$MAX_OPTIMIZER_STEPS" != "0" ]]; then
      echo "Route C direct-only evaluation requires MAX_OPTIMIZER_STEPS=0" >&2
      exit 2
    fi
    if [[ -n "$FULL_TFGRID_CHECKPOINT" || -n "$FULL_TFGRID_CHECKPOINT_SHA256" ]]; then
      echo "Route C direct-only evaluation rejects full TF-GridNet inputs" >&2
      exit 2
    fi
    ROUTE_SCOPE_ARGS=(--route-scope direct_only)
    SHARED_CANDIDATE_ARGS=()
    ;;
  *)
    echo "ROUTE_SCOPE must be all or direct_only, got: $ROUTE_SCOPE" >&2
    exit 2
    ;;
esac

python "$PYTHON_SCRIPT" \
  --label-bank "$LABEL_BANK" \
  --label-bank-sha256 "$LABEL_BANK_SHA256" \
  --config "$CONFIG" \
  --config-sha256 "$CONFIG_SHA256" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --external-exact-csv "$EXTERNAL_EXACT_CSV" \
  --external-exact-csv-sha256 "$EXTERNAL_EXACT_CSV_SHA256" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --source-commit "$SOURCE_COMMIT" \
  --seed "$SEED" \
  --expected-train-speakers "$EXPECTED_TRAIN_SPEAKERS" \
  --expected-calibration-speakers "$EXPECTED_CALIBRATION_SPEAKERS" \
  --expected-holdout-speakers "$EXPECTED_HOLDOUT_SPEAKERS" \
  "${ROUTE_SCOPE_ARGS[@]}" \
  "${SHARED_CANDIDATE_ARGS[@]}" \
  --waveform-architectures "$WAVEFORM_ARCHITECTURES" \
  --max-optimizer-steps "$MAX_OPTIMIZER_STEPS" \
  "${VCTK_EXTERNAL_ARGS[@]}" \
  "${FULL_TFGRID_ARGS[@]}" \
  --device cuda \
  2>&1 | tee -a "$LIVE_LOG"

echo "event=complete job=$SLURM_JOB_ID time=$(date -Is)" | tee -a "$LIVE_LOG"
