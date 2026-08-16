#!/bin/bash
# Summarize three locked AVQI v4 scorer confirmations after all jobs succeed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
SUMMARY_SCRIPT="$SOURCE_ROOT/scripts/summarize_avqi_component_multiseed.py"
CONSENSUS_KIND="${CONSENSUS_KIND:-phase}"
DEPENDENCY_JOB_IDS="${DEPENDENCY_JOB_IDS:-}"
CONFIRMATION_SEEDS=(20260816 20260817 20260818)

case "$CONSENSUS_KIND" in
  phase)
    SCREEN_RUN_ROOT="$ROOT_DIR/runs/avqi_component_phaseaware_v4_screen_20260816_01"
    CONFIRM_RUN_STEM="avqi_component_phaseaware_v4_confirm"
    RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/avqi_component_phaseaware_v4_multiseed_20260816_01}"
    JOB_NAME="avqi-v4-pcons"
    ;;
  direct)
    SCREEN_RUN_ROOT="$ROOT_DIR/runs/avqi_component_direct_c_v5_screen_20260817_01"
    CONFIRM_RUN_STEM="avqi_component_direct_c_v5_confirm"
    RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/avqi_component_direct_c_v5_multiseed_20260817_01}"
    JOB_NAME="avqi-v5-ccons"
    ;;
  full)
    SCREEN_RUN_ROOT="$ROOT_DIR/runs/avqi_component_pretrained_full_tfgrid_v4_screen_20260816_01"
    CONFIRM_RUN_STEM="avqi_component_pretrained_full_tfgrid_v4_confirm"
    RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/avqi_component_pretrained_full_tfgrid_v4_multiseed_20260816_01}"
    JOB_NAME="avqi-v4-fcons"
    ;;
  *)
    echo "CONSENSUS_KIND must be phase, direct, or full, got: $CONSENSUS_KIND" >&2
    exit 2
    ;;
esac

SCREEN_REPORT="${SCREEN_REPORT:-$SCREEN_RUN_ROOT/outputs/diagnostic_report.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
PARTITION="${PARTITION:-batch-csl}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
MEMORY="${MEMORY:-4G}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
CONFIRMATION_REPORTS=()
for seed in "${CONFIRMATION_SEEDS[@]}"; do
  CONFIRMATION_REPORTS+=(
    "$ROOT_DIR/runs/${CONFIRM_RUN_STEM}_seed${seed}_01/outputs/diagnostic_report.json"
  )
done

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing to run from a dirty source tree: $SOURCE_ROOT" >&2
  exit 2
fi
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"
if [[ ! -f "$SUMMARY_SCRIPT" ]]; then
  echo "Missing multi-seed summary source: $SUMMARY_SCRIPT" >&2
  exit 2
fi

export ROOT_DIR SOURCE_ROOT SUMMARY_SCRIPT CONSENSUS_KIND DEPENDENCY_JOB_IDS
export SCREEN_RUN_ROOT SCREEN_REPORT CONFIRM_RUN_STEM RUN_ROOT OUTPUT_DIR LOG_DIR
export JOB_NAME PARTITION CPUS_PER_TASK MEMORY TIME_LIMIT SOURCE_COMMIT

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
  DEPENDENCY_ARGS=()
  if [[ -n "$DEPENDENCY_JOB_IDS" ]]; then
    NORMALIZED_JOB_IDS="${DEPENDENCY_JOB_IDS//,/:}"
    DEPENDENCY_ARGS=(--dependency="afterok:$NORMALIZED_JOB_IDS")
  else
    for path in "$SCREEN_REPORT" "${CONFIRMATION_REPORTS[@]}"; do
      if [[ ! -f "$path" ]]; then
        echo "Missing report and no dependency jobs supplied: $path" >&2
        exit 2
      fi
    done
  fi
  sbatch \
    --parsable \
    --job-name="$JOB_NAME" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
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
for path in "$SCREEN_REPORT" "${CONFIRMATION_REPORTS[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing completed scorer report: $path" >&2
    exit 2
  fi
done
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite output: $OUTPUT_DIR" >&2
  exit 2
fi

ARGS=(
  --screen-report "$SCREEN_REPORT"
  --output-dir "$OUTPUT_DIR"
)
for path in "${CONFIRMATION_REPORTS[@]}"; do
  ARGS+=(--confirmation-report "$path")
done
python3 "$SUMMARY_SCRIPT" "${ARGS[@]}"
