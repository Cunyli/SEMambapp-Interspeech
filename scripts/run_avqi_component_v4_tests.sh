#!/bin/bash
# Run the full repository test suite before any AVQI v4 GPU screen.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/avqi_component_phaseaware_v4_tests_20260816_02}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
PARTITION="${PARTITION:-batch-csl}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-16G}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
TEST_FILES=(
  tests/test_artifact_layout.py
  tests/test_avqi_components.py
  tests/test_avqi_shimmer_fresh_panel.py
  tests/test_direct_avqi_waveform_optimization.py
  tests/test_shifted_anechoic_target.py
)

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing to test a dirty source tree: $SOURCE_ROOT" >&2
  exit 2
fi
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"
for relative_path in "${TEST_FILES[@]}"; do
  if [[ ! -f "$SOURCE_ROOT/$relative_path" ]]; then
    echo "Missing test file: $relative_path" >&2
    exit 2
  fi
done

export ROOT_DIR SOURCE_ROOT RUN_ROOT LOG_DIR OUTPUT_DIR PARTITION CPUS_PER_TASK
export MEMORY TIME_LIMIT SOURCE_COMMIT

mkdir -p "$LOG_DIR"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite test output: $OUTPUT_DIR" >&2
    exit 2
  fi
  sbatch \
    --parsable \
    --job-name="avqi-v4-tests" \
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
  echo "Source HEAD drifted after test submission" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite test output: $OUTPUT_DIR" >&2
  exit 2
fi

module load triton/2025.1-gcc
module load gcc/13.3.0
eval "$(conda shell.bash hook)"
conda activate semambapp
cd "$SOURCE_ROOT"
mkdir -p "$OUTPUT_DIR"
PYTEST_LOG="$OUTPUT_DIR/pytest.log"
python -m pytest -q "${TEST_FILES[@]}" 2>&1 | tee "$PYTEST_LOG"
PYTEST_LOG_SHA256="$(sha256sum "$PYTEST_LOG" | awk '{print $1}')"
TEST_FILES_JSON='["tests/test_artifact_layout.py","tests/test_avqi_components.py","tests/test_avqi_shimmer_fresh_panel.py","tests/test_direct_avqi_waveform_optimization.py","tests/test_shifted_anechoic_target.py"]'
jq -n \
  --arg source_commit "$SOURCE_COMMIT" \
  --arg slurm_job_id "$SLURM_JOB_ID" \
  --arg pytest_log "$PYTEST_LOG" \
  --arg pytest_log_sha256 "$PYTEST_LOG_SHA256" \
  --argjson test_files "$TEST_FILES_JSON" \
  '{decision: "PASS_AVQI_V4_REPOSITORY_TESTS", source_commit: $source_commit, slurm_job_id: $slurm_job_id, test_files: $test_files, pytest_log: $pytest_log, pytest_log_sha256: $pytest_log_sha256, generator_optimizer_steps: 0, formal_pathology_training_submitted: false}' \
  > "$OUTPUT_DIR/completion_receipt.json"
