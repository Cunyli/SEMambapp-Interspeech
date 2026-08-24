#!/bin/bash
# Hash-locked LTAS-slope promotion audit. No generator training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
SOURCE_ROOT="${SOURCE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_SCRIPT="$SOURCE_ROOT/scripts/evaluate_avqi_ltas_slope_promotion.py"
GATE_HELPER="$SOURCE_ROOT/scripts/evaluate_avqi_ltas_slope_gate_alignment.py"
AUTHORITY_HELPER="$SOURCE_ROOT/scripts/evaluate_avqi_ltas_slope_lowpass_authority.py"

SCREEN_ROOT="${SCREEN_ROOT:-/scratch/work/lil14/SEMambapp-Interspeech/runs/avqi_route_c_shimmer_v6_screen_20260821_01/outputs}"
SCREEN_REPORT="${SCREEN_REPORT:-$SCREEN_ROOT/diagnostic_report.json}"
SCREEN_REPORT_SHA256="${SCREEN_REPORT_SHA256:-f242611ddf9b0245c93f326505a5298807dc204df99fa2157c8aa7bf8b934dc4}"
SCREEN_RECEIPT="${SCREEN_RECEIPT:-$SCREEN_ROOT/completion_receipt.json}"
SCREEN_RECEIPT_SHA256="${SCREEN_RECEIPT_SHA256:-afcc35d4b61e41980c6c5110198245b586ccd7b128d1c7fe184f66dfe599db45}"

LTAS_V8_ROOT="${LTAS_V8_ROOT:-/scratch/work/lil14/SEMambapp-Interspeech-avqi-shimmer-ltas-v8/runs}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-$LTAS_V8_ROOT/avqi_route_c_ltas_slope_gate_alignment_v8_calibration_20260823_01/outputs}"
CALIBRATION_REPORT="${CALIBRATION_REPORT:-$CALIBRATION_ROOT/diagnostic_report.json}"
CALIBRATION_REPORT_SHA256="${CALIBRATION_REPORT_SHA256:-0cedafc58e3edd0f27f3612a049bd6f33f934b583c8f410b8abb0fe3e59bd395}"
CALIBRATION_RECEIPT="${CALIBRATION_RECEIPT:-$CALIBRATION_ROOT/completion_receipt.json}"
CALIBRATION_RECEIPT_SHA256="${CALIBRATION_RECEIPT_SHA256:-3c19f31d50472472d86cb2ff3aa9bd727c22ed996d51978c80f387c189f2ac35}"
HOLDOUT_ROOT="${HOLDOUT_ROOT:-$LTAS_V8_ROOT/avqi_route_c_ltas_slope_gate_alignment_v8_holdout_20260823_01/outputs}"
HOLDOUT_REPORT="${HOLDOUT_REPORT:-$HOLDOUT_ROOT/diagnostic_report.json}"
HOLDOUT_REPORT_SHA256="${HOLDOUT_REPORT_SHA256:-2ccb5e43c4213c7754cca529f29d0d996e87e778e49b0bf3c1fca239632097fa}"
HOLDOUT_RECEIPT="${HOLDOUT_RECEIPT:-$HOLDOUT_ROOT/completion_receipt.json}"
HOLDOUT_RECEIPT_SHA256="${HOLDOUT_RECEIPT_SHA256:-9d550a602f7b063b6a04187b07a18ee19bbd9d514075c06c256cd3b56fa9cf9f}"

SVD_ROOT="${SVD_ROOT:-$LTAS_V8_ROOT/avqi_route_c_ltas_slope_svd_authority_v10_20260823_01/outputs}"
SVD_REPORT="${SVD_REPORT:-$SVD_ROOT/diagnostic_report.json}"
SVD_REPORT_SHA256="${SVD_REPORT_SHA256:-01e8ecfc9997ce3c02c8ad51034167a9a74f9f070ee0912c5ee85c73265519ee}"
SVD_RECEIPT="${SVD_RECEIPT:-$SVD_ROOT/completion_receipt.json}"
SVD_RECEIPT_SHA256="${SVD_RECEIPT_SHA256:-68a7a8e0b1eb1adeb84d856050317f192615827bff1aebf6e24cf5bec94ac7e9}"
SVD_PANEL_SEAL="${SVD_PANEL_SEAL:-$SVD_ROOT/panel_seal.json}"
SVD_PANEL_SEAL_SHA256="${SVD_PANEL_SEAL_SHA256:-02d27c3f05be3a3b1196e2178fa6fbb6a14dafefccd068094ded58e56f700e57}"
SVD_SEAL_RECEIPT="${SVD_SEAL_RECEIPT:-$SVD_ROOT/seal_receipt.json}"
SVD_SEAL_RECEIPT_SHA256="${SVD_SEAL_RECEIPT_SHA256:-13331ddcbd9c8621473a4bc01bbd351d39a0a150f8e0eaeac419569fc8f3bff9}"

LABEL_BANK="${LABEL_BANK:-/scratch/work/lil14/SEMambapp-Interspeech/runs/avqi_component_direct_c_v5_data_20260817_03/outputs/label_bank/exact_component_label_bank_v4.csv}"
LABEL_BANK_SHA256="${LABEL_BANK_SHA256:-03b8d5e3d0542dbfe60e54723bc89431e8dfd475dcc38284a6058465c5224760}"
PREDICTOR_CHECKPOINT="${PREDICTOR_CHECKPOINT:-/scratch/work/lil14/SEMambapp-Interspeech/checkpoints/avqi_route_c_shimmer_v6_screen_20260821_01/direct_direct_praat_hard_shimmer_pulse_path_v6_estimator.pt}"
PREDICTOR_CHECKPOINT_SHA256="${PREDICTOR_CHECKPOINT_SHA256:-40b819946abdcb8a4b643fe4238d1bb4d31168a3eb2a6d6c786a61752da629bc}"

RUNTIME_PYTHON="${RUNTIME_PYTHON:-/scratch/work/lil14/.conda_envs/semambapp/bin/python}"
EXACT_PYTHON="${EXACT_PYTHON:-/scratch/work/lil14/.conda_envs/avqi/bin/python}"
AVQI_CODE_ROOT="${AVQI_CODE_ROOT:-/scratch/work/lil14/avqi}"
PYTHON_VERSION_SHA256="${PYTHON_VERSION_SHA256:-6bbf41386a901f82127370bd23bd136b379b061ae283291853d94746985ac009}"
PRAAT_VERSION_SHA256="${PRAAT_VERSION_SHA256:-432b5157bc6ae03eb9d10d19aa0c0fc13aae711e172cd02fe21297bd581e85e0}"
HIGHPASS_PRAAT_SHA256="${HIGHPASS_PRAAT_SHA256:-e122cc43f347688a1349440ac0242f26256f35ae6ddce2fc50c0250bfd1e3a8d}"
SV_LENGTH_PRAAT_SHA256="${SV_LENGTH_PRAAT_SHA256:-fdbad298dcfb90f95358cbea737c4063a61785db7c32f5af8836e611928ce174}"
CS_VOICED_PRAAT_SHA256="${CS_VOICED_PRAAT_SHA256:-09e874ba3762d5529be3d3e83a737bd424295a831d57064bde5c4944305f578c}"
SLOPE_PRAAT_SHA256="${SLOPE_PRAAT_SHA256:-8ba59924ebfae16b8c55d1ea009d887182c31820d5c62a6b8d93ed174c2be8c2}"

RUN_ROOT="${RUN_ROOT:-$SOURCE_ROOT/runs/avqi_route_c_ltas_slope_promotion_v1_20260824_01}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
PARTITION="${PARTITION:-batch-milan}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-16G}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"
BASE_COMMIT="${BASE_COMMIT:-$SOURCE_COMMIT}"
PROMOTION_SCRIPT_SHA256="${PROMOTION_SCRIPT_SHA256:?PROMOTION_SCRIPT_SHA256 is required}"
GATE_HELPER_SHA256="${GATE_HELPER_SHA256:?GATE_HELPER_SHA256 is required}"
AUTHORITY_HELPER_SHA256="${AUTHORITY_HELPER_SHA256:?AUTHORITY_HELPER_SHA256 is required}"

verify_hash() {
  local path="$1"
  local expected="$2"
  local label="$3"
  if [[ "$(sha256sum "$path" | awk '{print $1}')" != "$expected" ]]; then
    echo "$label SHA-256 mismatch" >&2
    exit 2
  fi
}

for path in "$PYTHON_SCRIPT" "$GATE_HELPER" "$AUTHORITY_HELPER" \
  "$SCREEN_REPORT" "$SCREEN_RECEIPT" \
  "$CALIBRATION_REPORT" "$CALIBRATION_RECEIPT" \
  "$HOLDOUT_REPORT" "$HOLDOUT_RECEIPT" \
  "$SVD_REPORT" "$SVD_RECEIPT" "$SVD_PANEL_SEAL" "$SVD_SEAL_RECEIPT" \
  "$LABEL_BANK" "$PREDICTOR_CHECKPOINT" "$RUNTIME_PYTHON" "$EXACT_PYTHON" \
  "$AVQI_CODE_ROOT/avqi_code/python_version.py" \
  "$AVQI_CODE_ROOT/avqi_code/praat_version.py" \
  "$AVQI_CODE_ROOT/avqi_code/praat_scripts/highpass_filter.praat" \
  "$AVQI_CODE_ROOT/avqi_code/praat_scripts/length_normalize_sv.praat" \
  "$AVQI_CODE_ROOT/avqi_code/praat_scripts/voiced_segment_extraction.praat" \
  "$AVQI_CODE_ROOT/avqi_code/praat_scripts/slope.praat"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing LTAS promotion input: $path" >&2
    exit 2
  fi
done

verify_hash "$PYTHON_SCRIPT" "$PROMOTION_SCRIPT_SHA256" "promotion script"
verify_hash "$GATE_HELPER" "$GATE_HELPER_SHA256" "gate helper"
verify_hash "$AUTHORITY_HELPER" "$AUTHORITY_HELPER_SHA256" "authority helper"
verify_hash "$SCREEN_REPORT" "$SCREEN_REPORT_SHA256" "screen report"
verify_hash "$SCREEN_RECEIPT" "$SCREEN_RECEIPT_SHA256" "screen receipt"
verify_hash "$CALIBRATION_REPORT" "$CALIBRATION_REPORT_SHA256" "calibration report"
verify_hash "$CALIBRATION_RECEIPT" "$CALIBRATION_RECEIPT_SHA256" "calibration receipt"
verify_hash "$HOLDOUT_REPORT" "$HOLDOUT_REPORT_SHA256" "holdout report"
verify_hash "$HOLDOUT_RECEIPT" "$HOLDOUT_RECEIPT_SHA256" "holdout receipt"
verify_hash "$SVD_REPORT" "$SVD_REPORT_SHA256" "SVD report"
verify_hash "$SVD_RECEIPT" "$SVD_RECEIPT_SHA256" "SVD receipt"
verify_hash "$SVD_PANEL_SEAL" "$SVD_PANEL_SEAL_SHA256" "SVD panel seal"
verify_hash "$SVD_SEAL_RECEIPT" "$SVD_SEAL_RECEIPT_SHA256" "SVD seal receipt"
verify_hash "$LABEL_BANK" "$LABEL_BANK_SHA256" "label bank"
verify_hash "$PREDICTOR_CHECKPOINT" "$PREDICTOR_CHECKPOINT_SHA256" "checkpoint"
verify_hash "$AVQI_CODE_ROOT/avqi_code/python_version.py" "$PYTHON_VERSION_SHA256" "exact Python helper"
verify_hash "$AVQI_CODE_ROOT/avqi_code/praat_version.py" "$PRAAT_VERSION_SHA256" "exact Praat helper"
verify_hash "$AVQI_CODE_ROOT/avqi_code/praat_scripts/highpass_filter.praat" "$HIGHPASS_PRAAT_SHA256" "exact high-pass"
verify_hash "$AVQI_CODE_ROOT/avqi_code/praat_scripts/length_normalize_sv.praat" "$SV_LENGTH_PRAAT_SHA256" "exact SV length"
verify_hash "$AVQI_CODE_ROOT/avqi_code/praat_scripts/voiced_segment_extraction.praat" "$CS_VOICED_PRAAT_SHA256" "exact CS voiced"
verify_hash "$AVQI_CODE_ROOT/avqi_code/praat_scripts/slope.praat" "$SLOPE_PRAAT_SHA256" "exact slope"

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing dirty LTAS promotion source" >&2
  exit 2
fi
if [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" != "$BASE_COMMIT" ]]; then
  echo "LTAS promotion base commit drifted" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite LTAS promotion output: $OUTPUT_DIR" >&2
  exit 2
fi

export SOURCE_ROOT PYTHON_SCRIPT GATE_HELPER AUTHORITY_HELPER
export SCREEN_ROOT SCREEN_REPORT SCREEN_REPORT_SHA256 SCREEN_RECEIPT
export SCREEN_RECEIPT_SHA256 LTAS_V8_ROOT CALIBRATION_ROOT
export CALIBRATION_REPORT CALIBRATION_REPORT_SHA256 CALIBRATION_RECEIPT
export CALIBRATION_RECEIPT_SHA256 HOLDOUT_ROOT HOLDOUT_REPORT
export HOLDOUT_REPORT_SHA256 HOLDOUT_RECEIPT HOLDOUT_RECEIPT_SHA256
export SVD_ROOT SVD_REPORT SVD_REPORT_SHA256 SVD_RECEIPT SVD_RECEIPT_SHA256
export SVD_PANEL_SEAL SVD_PANEL_SEAL_SHA256 SVD_SEAL_RECEIPT
export SVD_SEAL_RECEIPT_SHA256 LABEL_BANK LABEL_BANK_SHA256
export PREDICTOR_CHECKPOINT PREDICTOR_CHECKPOINT_SHA256 RUNTIME_PYTHON
export EXACT_PYTHON AVQI_CODE_ROOT PYTHON_VERSION_SHA256 PRAAT_VERSION_SHA256
export HIGHPASS_PRAAT_SHA256 SV_LENGTH_PRAAT_SHA256 CS_VOICED_PRAAT_SHA256
export SLOPE_PRAAT_SHA256 RUN_ROOT OUTPUT_DIR LOG_DIR PARTITION CPUS_PER_TASK
export MEMORY TIME_LIMIT SOURCE_COMMIT BASE_COMMIT PROMOTION_SCRIPT_SHA256
export GATE_HELPER_SHA256 AUTHORITY_HELPER_SHA256

mkdir -p "$LOG_DIR"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  sbatch \
    --parsable \
    --job-name="avqi-ltas-promote" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEMORY" \
    --time="$TIME_LIMIT" \
    --output="$LOG_DIR/slurm_%j.out" \
    --error="$LOG_DIR/slurm_%j.err" \
    --export=ALL \
    "$SELF_PATH"
  exit 0
fi

module load triton/2025.1-gcc
module load gcc/13.3.0
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
LIVE_LOG="$LOG_DIR/ltas_slope_promotion_${SLURM_JOB_ID}.log"
echo "event=start job=$SLURM_JOB_ID source=$SOURCE_COMMIT time=$(date -Is)" | tee -a "$LIVE_LOG"
"$RUNTIME_PYTHON" "$PYTHON_SCRIPT" \
  --screen-report "$SCREEN_REPORT" \
  --screen-report-sha256 "$SCREEN_REPORT_SHA256" \
  --screen-receipt "$SCREEN_RECEIPT" \
  --screen-receipt-sha256 "$SCREEN_RECEIPT_SHA256" \
  --calibration-report "$CALIBRATION_REPORT" \
  --calibration-report-sha256 "$CALIBRATION_REPORT_SHA256" \
  --calibration-receipt "$CALIBRATION_RECEIPT" \
  --calibration-receipt-sha256 "$CALIBRATION_RECEIPT_SHA256" \
  --holdout-report "$HOLDOUT_REPORT" \
  --holdout-report-sha256 "$HOLDOUT_REPORT_SHA256" \
  --holdout-receipt "$HOLDOUT_RECEIPT" \
  --holdout-receipt-sha256 "$HOLDOUT_RECEIPT_SHA256" \
  --svd-report "$SVD_REPORT" \
  --svd-report-sha256 "$SVD_REPORT_SHA256" \
  --svd-receipt "$SVD_RECEIPT" \
  --svd-receipt-sha256 "$SVD_RECEIPT_SHA256" \
  --svd-panel-seal "$SVD_PANEL_SEAL" \
  --svd-panel-seal-sha256 "$SVD_PANEL_SEAL_SHA256" \
  --svd-seal-receipt "$SVD_SEAL_RECEIPT" \
  --svd-seal-receipt-sha256 "$SVD_SEAL_RECEIPT_SHA256" \
  --label-bank "$LABEL_BANK" \
  --label-bank-sha256 "$LABEL_BANK_SHA256" \
  --predictor-checkpoint "$PREDICTOR_CHECKPOINT" \
  --predictor-checkpoint-sha256 "$PREDICTOR_CHECKPOINT_SHA256" \
  --exact-python "$EXACT_PYTHON" \
  --avqi-code-root "$AVQI_CODE_ROOT" \
  --python-version-sha256 "$PYTHON_VERSION_SHA256" \
  --praat-version-sha256 "$PRAAT_VERSION_SHA256" \
  --highpass-praat-sha256 "$HIGHPASS_PRAAT_SHA256" \
  --sv-length-praat-sha256 "$SV_LENGTH_PRAAT_SHA256" \
  --cs-voiced-praat-sha256 "$CS_VOICED_PRAAT_SHA256" \
  --slope-praat-sha256 "$SLOPE_PRAAT_SHA256" \
  --output-dir "$OUTPUT_DIR" \
  --source-commit "$SOURCE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID" 2>&1 | tee -a "$LIVE_LOG"
echo "event=complete job=$SLURM_JOB_ID time=$(date -Is)" | tee -a "$LIVE_LOG"
