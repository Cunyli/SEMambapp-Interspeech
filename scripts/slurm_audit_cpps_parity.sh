#!/bin/bash
# CPPS parity Torch stage only: no optimizer step and no generator training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$SOURCE_ROOT/runs/avqi_route_c_cpps_v7_parity_20260823_02}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
LABEL_BANK="${LABEL_BANK:-/scratch/work/lil14/SEMambapp-Interspeech/runs/avqi_component_direct_c_v5_data_20260817_03/outputs/label_bank/exact_component_label_bank_v4.csv}"
LABEL_BANK_SHA256="${LABEL_BANK_SHA256:-03b8d5e3d0542dbfe60e54723bc89431e8dfd475dcc38284a6058465c5224760}"
AVQI_ROOT="${AVQI_ROOT:-/scratch/work/lil14/avqi}"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"
PARTITION="${PARTITION:-gpu-v100-32g}"
GPU_TYPE="${GPU_TYPE:-v100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-48G}"
TIME_LIMIT="${TIME_LIMIT:-00:20:00}"
AUDIT_STAGE="${AUDIT_STAGE:-torch}"
ROW_INDICES="${ROW_INDICES:-3,16,20}"
RUN_CPPS_TESTS="${RUN_CPPS_TESTS:-0}"

if [[ "$AUDIT_STAGE" != "torch" && "$AUDIT_STAGE" != "gradient" ]]; then
  echo "Unsupported AUDIT_STAGE: $AUDIT_STAGE" >&2
  exit 2
fi
if [[ "$RUN_CPPS_TESTS" != "0" && "$RUN_CPPS_TESTS" != "1" ]]; then
  echo "RUN_CPPS_TESTS must be 0 or 1" >&2
  exit 2
fi

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing to run from a dirty source tree: $SOURCE_ROOT" >&2
  exit 2
fi
if [[ ! -f "$OUTPUT_DIR/cpps_parity_prepare.json" || ! -f "$OUTPUT_DIR/cpps_parity_inputs.npz" ]]; then
  echo "Missing exact prepare artifacts under: $OUTPUT_DIR" >&2
  exit 2
fi
if [[ "$AUDIT_STAGE" == "torch" && -e "$OUTPUT_DIR/cpps_parity_report.json" ]]; then
  echo "Refusing to overwrite completed parity report: $OUTPUT_DIR/cpps_parity_report.json" >&2
  exit 2
fi
if [[ "$AUDIT_STAGE" == "gradient" && -e "$OUTPUT_DIR/cpps_gradient_decomposition.json" ]]; then
  echo "Refusing to overwrite completed gradient audit: $OUTPUT_DIR/cpps_gradient_decomposition.json" >&2
  exit 2
fi

export SOURCE_ROOT RUN_ROOT OUTPUT_DIR LOG_DIR LABEL_BANK LABEL_BANK_SHA256
export AVQI_ROOT SOURCE_COMMIT PARTITION GPU_TYPE CPUS_PER_TASK MEMORY TIME_LIMIT
export AUDIT_STAGE ROW_INDICES RUN_CPPS_TESTS

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  mkdir -p "$LOG_DIR"
  sbatch \
    --parsable \
    --job-name="avqi-cpps-$AUDIT_STAGE" \
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
    "$SCRIPT_DIR/$(basename "$BASH_SOURCE")"
  exit 0
fi

if [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" != "$SOURCE_COMMIT" ]]; then
  echo "Source HEAD drifted after submission" >&2
  exit 2
fi

module load triton/2025.1-gcc
module load gcc/13.3.0
eval "$(conda shell.bash hook)"
conda activate semambapp
cd "$SOURCE_ROOT"
mkdir -p "$LOG_DIR"

LIVE_LOG="$LOG_DIR/torch_stage_${SLURM_JOB_ID}.log"
echo "event=start job=$SLURM_JOB_ID stage=$AUDIT_STAGE source_commit=$SOURCE_COMMIT time=$(date -Is)" | tee -a "$LIVE_LOG"
python -c 'import os, torch; print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("device", torch.cuda.get_device_name(0)); print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))' | tee -a "$LIVE_LOG"
if [[ "$RUN_CPPS_TESTS" == "1" ]]; then
  python -m pytest tests/test_avqi_components.py -k cpps -q 2>&1 | tee -a "$LIVE_LOG"
fi
python "$SOURCE_ROOT/scripts/audit_cpps_parity.py" \
  --label-bank "$LABEL_BANK" \
  --label-bank-sha256 "$LABEL_BANK_SHA256" \
  --avqi-root "$AVQI_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --source-commit "$SOURCE_COMMIT" \
  --stage "$AUDIT_STAGE" \
  --max-rows 24 \
  --views cs,sv \
  --row-indices "$ROW_INDICES" \
  2>&1 | tee -a "$LIVE_LOG"
echo "event=complete job=$SLURM_JOB_ID time=$(date -Is)" | tee -a "$LIVE_LOG"
