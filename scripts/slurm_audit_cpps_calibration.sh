#!/bin/bash
# Frozen CPPS train/calibration audit only; no holdout or generator update.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
SOURCE_ROOT="${SOURCE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$SOURCE_ROOT/runs/avqi_route_c_cpps_calibration}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
LABEL_BANK="${LABEL_BANK:-/scratch/work/lil14/SEMambapp-Interspeech/runs/avqi_component_direct_c_v5_data_20260817_03/outputs/label_bank/exact_component_label_bank_v4.csv}"
LABEL_BANK_SHA256="${LABEL_BANK_SHA256:-03b8d5e3d0542dbfe60e54723bc89431e8dfd475dcc38284a6058465c5224760}"
MODES="${MODES:-praat_relative_log1p_v10,praat_pow2_highpass_v11}"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"
PARTITION="${PARTITION:-gpu-v100-32g}"
GPU_TYPE="${GPU_TYPE:-v100}"
TIME_LIMIT="${TIME_LIMIT:-00:20:00}"

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing to run from a dirty source tree: $SOURCE_ROOT" >&2
  exit 2
fi
if [[ ! -f "$LABEL_BANK" ]]; then
  echo "Missing label bank: $LABEL_BANK" >&2
  exit 2
fi

export SOURCE_ROOT RUN_ROOT OUTPUT_DIR LOG_DIR LABEL_BANK LABEL_BANK_SHA256
export MODES SOURCE_COMMIT PARTITION GPU_TYPE TIME_LIMIT

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite output: $OUTPUT_DIR" >&2
    exit 2
  fi
  mkdir -p "$LOG_DIR"
  sbatch \
    --parsable \
    --job-name="avqi-cpps-cal" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --gres="gpu:${GPU_TYPE}:1" \
    --cpus-per-task=4 \
    --mem=48G \
    --time="$TIME_LIMIT" \
    --output="$LOG_DIR/slurm_%j.out" \
    --error="$LOG_DIR/slurm_%j.err" \
    --export=ALL \
    "$SELF_PATH"
  exit 0
fi

if [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" != "$SOURCE_COMMIT" ]]; then
  echo "Source HEAD drifted after submission" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite output: $OUTPUT_DIR" >&2
  exit 2
fi

module load triton/2025.1-gcc
module load gcc/13.3.0
eval "$(conda shell.bash hook)"
conda activate semambapp
cd "$SOURCE_ROOT"
mkdir -p "$OUTPUT_DIR"
LIVE_LOG="$LOG_DIR/cpps_calibration_${SLURM_JOB_ID}.log"
REPORT="$OUTPUT_DIR/cpps_calibration_report.json"
RECEIPT="$OUTPUT_DIR/completion_receipt.json"
echo "event=start job=$SLURM_JOB_ID source_commit=$SOURCE_COMMIT time=$(date -Is)" | tee -a "$LIVE_LOG"
python -m pytest tests/test_avqi_components.py -k cpps -q 2>&1 | tee -a "$LIVE_LOG"
python scripts/audit_cpps_calibration.py \
  --label-bank "$LABEL_BANK" \
  --label-bank-sha256 "$LABEL_BANK_SHA256" \
  --output "$REPORT" \
  --source-commit "$SOURCE_COMMIT" \
  --modes "$MODES" \
  2>&1 | tee -a "$LIVE_LOG"
REPORT_SHA256="$(sha256sum "$REPORT" | awk '{print $1}')"
LOG_SHA256="$(sha256sum "$LIVE_LOG" | awk '{print $1}')"
jq -n \
  --arg source_commit "$SOURCE_COMMIT" \
  --arg slurm_job_id "$SLURM_JOB_ID" \
  --arg report "$REPORT" \
  --arg report_sha256 "$REPORT_SHA256" \
  --arg log_sha256 "$LOG_SHA256" \
  --arg selected_mode "$(jq -er '.selected_mode' "$REPORT")" \
  '{decision: "CPPS_CALIBRATION_AUDIT_COMPLETE", source_commit: $source_commit, slurm_job_id: $slurm_job_id, selected_mode: $selected_mode, report: $report, report_sha256: $report_sha256, log_sha256: $log_sha256, evaluated_splits: ["surrogate_train", "surrogate_calibration"], holdout_evaluated: false, external_evaluated: false, generator_optimizer_steps: 0, formal_generator_training_submitted: false}' \
  > "$RECEIPT"
echo "event=complete job=$SLURM_JOB_ID time=$(date -Is)" | tee -a "$LIVE_LOG"
