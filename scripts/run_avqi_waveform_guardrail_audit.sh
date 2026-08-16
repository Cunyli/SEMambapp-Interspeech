#!/bin/bash
# Re-audit the preserved 12-case direct waveform pilot; no optimizer is run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
AUDIT_SCRIPT="$SOURCE_ROOT/scripts/audit_avqi_waveform_guardrails.py"
SOURCE_RUN="$ROOT_DIR/runs/avqi_direct_waveform_opt_balanced_hnr_tilt_final_20260814_01/outputs"
SOURCE_RESULTS_CSV="$SOURCE_RUN/results.csv"
SOURCE_RESULTS_CSV_SHA256="2d3766686f2c1aa99b58f24470ee973bb232d0c1fea29537a89359934c7caafb"
SOURCE_REPORT="$SOURCE_RUN/waveform_optimization_report.json"
SOURCE_REPORT_SHA256="0fa3cd09638c105fdcc31cd3a9460ffe9bcf85d9083159571750f7802c32678a"
EXTERNAL_EXACT_CSV="$ROOT_DIR/runs/tau_pathology_three_tracks_20260810_01/outputs/intensity_eval/exact_components_all.csv"
EXTERNAL_EXACT_CSV_SHA256="1e401d2d3343d5d5e8dc38245d14a2e4f9fbb568b11a26269e4ce0aca30c249a"
PREDICTOR_CHECKPOINT="$ROOT_DIR/checkpoints/avqi_component_direct_praat_v2_voicedmask_confirm_20260816_01/waveform_direct_praat_hard_v2_predictor.pt"
PREDICTOR_CHECKPOINT_SHA256="11779b5aee83ed8baa81d35a695f09477117ed2bc5830925085dc3ab5770807f"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/avqi_direct_waveform_guardrail_reaudit_20260816_01}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
PARTITION="${PARTITION:-batch-csl}"
CPUS_PER_TASK="${CPUS_PER_TASK:-2}"
MEMORY="${MEMORY:-8G}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing to run from a dirty source tree: $SOURCE_ROOT" >&2
  exit 2
fi
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"
for path in "$AUDIT_SCRIPT" "$SOURCE_RESULTS_CSV" "$SOURCE_REPORT" \
  "$EXTERNAL_EXACT_CSV" "$PREDICTOR_CHECKPOINT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required audit source: $path" >&2
    exit 2
  fi
done

export ROOT_DIR SOURCE_ROOT AUDIT_SCRIPT SOURCE_RUN SOURCE_RESULTS_CSV
export SOURCE_RESULTS_CSV_SHA256 SOURCE_REPORT SOURCE_REPORT_SHA256
export EXTERNAL_EXACT_CSV EXTERNAL_EXACT_CSV_SHA256 PREDICTOR_CHECKPOINT
export PREDICTOR_CHECKPOINT_SHA256 RUN_ROOT LOG_DIR OUTPUT_DIR PARTITION
export CPUS_PER_TASK MEMORY TIME_LIMIT SOURCE_COMMIT

mkdir -p "$LOG_DIR"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite output: $OUTPUT_DIR" >&2
    exit 2
  fi
  sbatch \
    --parsable \
    --job-name="avqi-v4-waudit" \
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
cd "$ROOT_DIR"
python "$AUDIT_SCRIPT" \
  --source-results-csv "$SOURCE_RESULTS_CSV" \
  --source-results-csv-sha256 "$SOURCE_RESULTS_CSV_SHA256" \
  --source-report "$SOURCE_REPORT" \
  --source-report-sha256 "$SOURCE_REPORT_SHA256" \
  --external-exact-csv "$EXTERNAL_EXACT_CSV" \
  --external-exact-csv-sha256 "$EXTERNAL_EXACT_CSV_SHA256" \
  --predictor-checkpoint "$PREDICTOR_CHECKPOINT" \
  --predictor-checkpoint-sha256 "$PREDICTOR_CHECKPOINT_SHA256" \
  --output-dir "$OUTPUT_DIR" \
  --source-commit "$SOURCE_COMMIT"
