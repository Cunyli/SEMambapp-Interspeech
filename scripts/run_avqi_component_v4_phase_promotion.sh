#!/bin/bash
# Apply the frozen Compact-TFGrid promotion gate and conditionally submit full TFGrid.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
PROMOTION_SCRIPT="$SOURCE_ROOT/scripts/evaluate_avqi_component_v4_phase_promotion.py"
SCREEN_WRAPPER="$SOURCE_ROOT/scripts/run_avqi_component_v4_screen.sh"
SCREEN_RUN_ROOT="${SCREEN_RUN_ROOT:-$ROOT_DIR/runs/avqi_component_phaseaware_v4_screen_20260816_01}"
SCREEN_REPORT="${SCREEN_REPORT:-$SCREEN_RUN_ROOT/outputs/diagnostic_report.json}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/avqi_component_phaseaware_v4_promotion_20260816_01}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
DEPENDENCY_JOB_ID="${DEPENDENCY_JOB_ID:-}"
PARTITION="${PARTITION:-batch-csl}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
MEMORY="${MEMORY:-4G}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing to run from a dirty source tree: $SOURCE_ROOT" >&2
  exit 2
fi
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"

for path in "$PROMOTION_SCRIPT" "$SCREEN_WRAPPER"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required source: $path" >&2
    exit 2
  fi
done

export ROOT_DIR SOURCE_ROOT PROMOTION_SCRIPT SCREEN_WRAPPER SCREEN_RUN_ROOT
export SCREEN_REPORT RUN_ROOT LOG_DIR OUTPUT_DIR DEPENDENCY_JOB_ID PARTITION
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
  DEPENDENCY_ARGS=()
  if [[ -n "$DEPENDENCY_JOB_ID" ]]; then
    DEPENDENCY_ARGS=(--dependency="afterok:$DEPENDENCY_JOB_ID")
  elif [[ ! -f "$SCREEN_REPORT" ]]; then
    echo "Phase screen report is absent and no dependency job was supplied" >&2
    exit 2
  fi
  sbatch \
    --parsable \
    --job-name="avqi-v4-promote" \
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
if [[ ! -f "$SCREEN_REPORT" ]]; then
  echo "Missing completed phase screen report: $SCREEN_REPORT" >&2
  exit 2
fi
if [[ "$(jq -er '.contract.source_commit' "$SCREEN_REPORT")" != "$SOURCE_COMMIT" ]]; then
  echo "Phase screen source commit differs from promotion source" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite output: $OUTPUT_DIR" >&2
  exit 2
fi

python3 "$PROMOTION_SCRIPT" \
  --screen-report "$SCREEN_REPORT" \
  --output-dir "$OUTPUT_DIR"

PROMOTION_REPORT="$OUTPUT_DIR/promotion_report.json"
DECISION="$(jq -er '.decision' "$PROMOTION_REPORT")"
if [[ "$DECISION" == "PROMOTE_PRETRAINED_FULL_TFGRID_SCREEN" ]]; then
  SUBMISSION="$(
    env -u SLURM_JOB_ID \
    SCREEN_KIND=full_tfgrid \
    DEPENDENCY_JOB_ID="$SLURM_JOB_ID" \
    CONFIRM_SLURM_SUBMIT=1 \
    ROOT_DIR="$ROOT_DIR" \
    SOURCE_ROOT="$SOURCE_ROOT" \
    SOURCE_COMMIT="$SOURCE_COMMIT" \
    "$SCREEN_WRAPPER"
  )"
  FULL_TFGRID_JOB_ID="${SUBMISSION%%;*}"
  if [[ ! "$FULL_TFGRID_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "Unexpected full TFGrid submission receipt: $SUBMISSION" >&2
    exit 2
  fi
  jq -n \
    --arg decision "$DECISION" \
    --arg job_id "$FULL_TFGRID_JOB_ID" \
    --arg source_commit "$SOURCE_COMMIT" \
    --arg promotion_report_sha256 "$(sha256sum "$PROMOTION_REPORT" | awk '{print $1}')" \
    '{decision: $decision, full_tfgrid_job_id: $job_id, source_commit: $source_commit, promotion_report_sha256: $promotion_report_sha256, generator_optimizer_steps: 0, formal_pathology_training_submitted: false}' \
    > "$OUTPUT_DIR/full_tfgrid_submission.json"
elif [[ "$DECISION" != "KEEP_COMPACT_NO_FULL_TFGRID" ]]; then
  echo "Unexpected promotion decision: $DECISION" >&2
  exit 2
fi
