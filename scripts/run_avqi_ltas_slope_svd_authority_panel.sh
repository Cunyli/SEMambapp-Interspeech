#!/bin/bash
# Hash-locked, two-stage SVD external authority panel for LTAS slope. No training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
PYTHON_SCRIPT="$SOURCE_ROOT/scripts/evaluate_avqi_ltas_slope_svd_authority_panel.py"
GATE_SCRIPT="$SOURCE_ROOT/scripts/evaluate_avqi_ltas_slope_gate_alignment.py"
AUTHORITY_HELPER="$SOURCE_ROOT/scripts/evaluate_avqi_ltas_slope_lowpass_authority.py"
MODEL_SOURCE="$SOURCE_ROOT/model/avqi_components.py"

SVD_ROOT="${SVD_ROOT:-/scratch/work/lil14/data/SVD}"
SV_METADATA="${SV_METADATA:-$SVD_ROOT/sv_metadata.csv}"
CS_METADATA="${CS_METADATA:-$SVD_ROOT/cs_metadata.csv}"
SV_ROOT="${SV_ROOT:-$SVD_ROOT/sv}"
CS_ROOT="${CS_ROOT:-$SVD_ROOT/cs_raw}"
SV_METADATA_SHA256="${SV_METADATA_SHA256:-36d8a725a209578744a862e63b5990d348e3d17d066a0247cdcd2e657c7ffc03}"
CS_METADATA_SHA256="${CS_METADATA_SHA256:-465c15e46c9c9e325c14e5672abead050bbfd9a4bba75d0ace46bf5d58884966}"

LABEL_BANK="${LABEL_BANK:-/scratch/work/lil14/SEMambapp-Interspeech/runs/avqi_component_direct_c_v5_data_20260817_03/outputs/label_bank/exact_component_label_bank_v4.csv}"
LABEL_BANK_SHA256="${LABEL_BANK_SHA256:-03b8d5e3d0542dbfe60e54723bc89431e8dfd475dcc38284a6058465c5224760}"
PREDICTOR_CHECKPOINT="${PREDICTOR_CHECKPOINT:-/scratch/work/lil14/SEMambapp-Interspeech/checkpoints/avqi_route_c_shimmer_v6_screen_20260821_01/direct_direct_praat_hard_shimmer_pulse_path_v6_estimator.pt}"
PREDICTOR_CHECKPOINT_SHA256="${PREDICTOR_CHECKPOINT_SHA256:-40b819946abdcb8a4b643fe4238d1bb4d31168a3eb2a6d6c786a61752da629bc}"

EXACT_PYTHON="${EXACT_PYTHON:-/scratch/work/lil14/.conda_envs/avqi/bin/python}"
AVQI_CODE_ROOT="${AVQI_CODE_ROOT:-/scratch/work/lil14/avqi}"
PYTHON_VERSION_SHA256="${PYTHON_VERSION_SHA256:-6bbf41386a901f82127370bd23bd136b379b061ae283291853d94746985ac009}"
PRAAT_VERSION_SHA256="${PRAAT_VERSION_SHA256:-432b5157bc6ae03eb9d10d19aa0c0fc13aae711e172cd02fe21297bd581e85e0}"
HIGHPASS_PRAAT_SHA256="${HIGHPASS_PRAAT_SHA256:-e122cc43f347688a1349440ac0242f26256f35ae6ddce2fc50c0250bfd1e3a8d}"
SV_LENGTH_PRAAT_SHA256="${SV_LENGTH_PRAAT_SHA256:-fdbad298dcfb90f95358cbea737c4063a61785db7c32f5af8836e611928ce174}"
CS_VOICED_PRAAT_SHA256="${CS_VOICED_PRAAT_SHA256:-09e874ba3762d5529be3d3e83a737bd424295a831d57064bde5c4944305f578c}"
SLOPE_PRAAT_SHA256="${SLOPE_PRAAT_SHA256:-8ba59924ebfae16b8c55d1ea009d887182c31820d5c62a6b8d93ed174c2be8c2}"

STAGE="${STAGE:-seal}"
RUN_ROOT="${RUN_ROOT:-$SOURCE_ROOT/runs/avqi_route_c_ltas_slope_svd_authority_v9_20260823_01}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
PANEL_SEAL_SHA256="${PANEL_SEAL_SHA256:-}"
PARTITION="${PARTITION:-batch-milan}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-24G}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"
BASE_COMMIT="${BASE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"
MODEL_SHA256="${MODEL_SHA256:?MODEL_SHA256 is required}"
DIAGNOSTIC_SHA256="${DIAGNOSTIC_SHA256:?DIAGNOSTIC_SHA256 is required}"
GATE_SCRIPT_SHA256="${GATE_SCRIPT_SHA256:?GATE_SCRIPT_SHA256 is required}"
AUTHORITY_HELPER_SHA256="${AUTHORITY_HELPER_SHA256:?AUTHORITY_HELPER_SHA256 is required}"
ALLOW_HASH_LOCKED_SOURCE="${ALLOW_HASH_LOCKED_SOURCE:-0}"

if [[ "$STAGE" != "seal" && "$STAGE" != "score" ]]; then
  echo "STAGE must be seal or score" >&2
  exit 2
fi
for path in "$PYTHON_SCRIPT" "$GATE_SCRIPT" "$AUTHORITY_HELPER" \
  "$MODEL_SOURCE" "$SV_METADATA" "$CS_METADATA" "$LABEL_BANK" \
  "$PREDICTOR_CHECKPOINT" "$EXACT_PYTHON" \
  "$AVQI_CODE_ROOT/avqi_code/python_version.py" \
  "$AVQI_CODE_ROOT/avqi_code/praat_version.py" \
  "$AVQI_CODE_ROOT/avqi_code/praat_scripts/highpass_filter.praat" \
  "$AVQI_CODE_ROOT/avqi_code/praat_scripts/length_normalize_sv.praat" \
  "$AVQI_CODE_ROOT/avqi_code/praat_scripts/voiced_segment_extraction.praat" \
  "$AVQI_CODE_ROOT/avqi_code/praat_scripts/slope.praat"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing SVD LTAS authority input: $path" >&2
    exit 2
  fi
done
if [[ ! -d "$SV_ROOT" || ! -d "$CS_ROOT" ]]; then
  echo "Missing SVD CS/SV roots" >&2
  exit 2
fi

verify_hash() {
  local path="$1"
  local expected="$2"
  local label="$3"
  if [[ "$(sha256sum "$path" | awk '{print $1}')" != "$expected" ]]; then
    echo "$label SHA-256 mismatch" >&2
    exit 2
  fi
}
verify_hash "$PYTHON_SCRIPT" "$DIAGNOSTIC_SHA256" "SVD LTAS diagnostic"
verify_hash "$GATE_SCRIPT" "$GATE_SCRIPT_SHA256" "LTAS gate helper"
verify_hash "$AUTHORITY_HELPER" "$AUTHORITY_HELPER_SHA256" "LTAS authority helper"
verify_hash "$MODEL_SOURCE" "$MODEL_SHA256" "LTAS model source"
verify_hash "$SV_METADATA" "$SV_METADATA_SHA256" "SVD SV metadata"
verify_hash "$CS_METADATA" "$CS_METADATA_SHA256" "SVD CS metadata"
verify_hash "$LABEL_BANK" "$LABEL_BANK_SHA256" "LTAS label bank"
verify_hash "$PREDICTOR_CHECKPOINT" "$PREDICTOR_CHECKPOINT_SHA256" "LTAS checkpoint"
verify_hash "$AVQI_CODE_ROOT/avqi_code/python_version.py" "$PYTHON_VERSION_SHA256" "exact Python helper"
verify_hash "$AVQI_CODE_ROOT/avqi_code/praat_version.py" "$PRAAT_VERSION_SHA256" "exact Praat helper"
verify_hash "$AVQI_CODE_ROOT/avqi_code/praat_scripts/highpass_filter.praat" "$HIGHPASS_PRAAT_SHA256" "exact high-pass"
verify_hash "$AVQI_CODE_ROOT/avqi_code/praat_scripts/length_normalize_sv.praat" "$SV_LENGTH_PRAAT_SHA256" "exact SV length rule"
verify_hash "$AVQI_CODE_ROOT/avqi_code/praat_scripts/voiced_segment_extraction.praat" "$CS_VOICED_PRAAT_SHA256" "exact CS voiced rule"
verify_hash "$AVQI_CODE_ROOT/avqi_code/praat_scripts/slope.praat" "$SLOPE_PRAAT_SHA256" "exact slope"

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" && "$ALLOW_HASH_LOCKED_SOURCE" != "1" ]]; then
  echo "Refusing dirty source without ALLOW_HASH_LOCKED_SOURCE=1" >&2
  exit 2
fi
if [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" != "$BASE_COMMIT" ]]; then
  echo "SVD LTAS authority base commit drifted" >&2
  exit 2
fi
if [[ "$STAGE" == "seal" ]]; then
  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite SVD LTAS panel seal: $OUTPUT_DIR" >&2
    exit 2
  fi
elif [[ ! -f "$OUTPUT_DIR/panel_seal.json" ]]; then
  echo "Score stage requires an existing panel seal" >&2
  exit 2
elif [[ -z "$PANEL_SEAL_SHA256" ]]; then
  echo "Score stage requires PANEL_SEAL_SHA256" >&2
  exit 2
elif [[ -e "$OUTPUT_DIR/diagnostic_report.json" ]]; then
  echo "Refusing to reopen an already scored SVD LTAS panel" >&2
  exit 2
else
  verify_hash "$OUTPUT_DIR/panel_seal.json" "$PANEL_SEAL_SHA256" "SVD LTAS panel seal"
fi

export ROOT_DIR SOURCE_ROOT PYTHON_SCRIPT GATE_SCRIPT AUTHORITY_HELPER MODEL_SOURCE
export SVD_ROOT SV_METADATA CS_METADATA SV_ROOT CS_ROOT
export SV_METADATA_SHA256 CS_METADATA_SHA256 LABEL_BANK LABEL_BANK_SHA256
export PREDICTOR_CHECKPOINT PREDICTOR_CHECKPOINT_SHA256 EXACT_PYTHON
export AVQI_CODE_ROOT PYTHON_VERSION_SHA256 PRAAT_VERSION_SHA256
export HIGHPASS_PRAAT_SHA256 SV_LENGTH_PRAAT_SHA256 CS_VOICED_PRAAT_SHA256
export SLOPE_PRAAT_SHA256 STAGE RUN_ROOT OUTPUT_DIR LOG_DIR PANEL_SEAL_SHA256
export PARTITION CPUS_PER_TASK MEMORY TIME_LIMIT SOURCE_COMMIT BASE_COMMIT
export MODEL_SHA256 DIAGNOSTIC_SHA256 GATE_SCRIPT_SHA256
export AUTHORITY_HELPER_SHA256 ALLOW_HASH_LOCKED_SOURCE

mkdir -p "$LOG_DIR"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  sbatch \
    --parsable \
    --job-name="avqi-ltas-svd-${STAGE}" \
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
eval "$(conda shell.bash hook)"
conda activate semambapp
export PYTHONPATH="$SOURCE_ROOT:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

LIVE_LOG="$LOG_DIR/ltas_svd_authority_${STAGE}_${SLURM_JOB_ID}.log"
echo "event=start stage=$STAGE job=$SLURM_JOB_ID source=$SOURCE_COMMIT time=$(date -Is)" | tee -a "$LIVE_LOG"
arguments=(
  --stage "$STAGE"
  --sv-metadata "$SV_METADATA"
  --sv-metadata-sha256 "$SV_METADATA_SHA256"
  --cs-metadata "$CS_METADATA"
  --cs-metadata-sha256 "$CS_METADATA_SHA256"
  --sv-root "$SV_ROOT"
  --cs-root "$CS_ROOT"
  --label-bank "$LABEL_BANK"
  --label-bank-sha256 "$LABEL_BANK_SHA256"
  --predictor-checkpoint "$PREDICTOR_CHECKPOINT"
  --predictor-checkpoint-sha256 "$PREDICTOR_CHECKPOINT_SHA256"
  --exact-python "$EXACT_PYTHON"
  --avqi-code-root "$AVQI_CODE_ROOT"
  --python-version-sha256 "$PYTHON_VERSION_SHA256"
  --praat-version-sha256 "$PRAAT_VERSION_SHA256"
  --highpass-praat-sha256 "$HIGHPASS_PRAAT_SHA256"
  --sv-length-praat-sha256 "$SV_LENGTH_PRAAT_SHA256"
  --cs-voiced-praat-sha256 "$CS_VOICED_PRAAT_SHA256"
  --slope-praat-sha256 "$SLOPE_PRAAT_SHA256"
  --output-dir "$OUTPUT_DIR"
  --source-commit "$SOURCE_COMMIT"
  --slurm-job-id "$SLURM_JOB_ID"
  --device cpu
)
if [[ "$STAGE" == "score" ]]; then
  arguments+=(--panel-seal-sha256 "$PANEL_SEAL_SHA256")
fi
python "$PYTHON_SCRIPT" "${arguments[@]}" 2>&1 | tee -a "$LIVE_LOG"
echo "event=complete stage=$STAGE job=$SLURM_JOB_ID time=$(date -Is)" | tee -a "$LIVE_LOG"
