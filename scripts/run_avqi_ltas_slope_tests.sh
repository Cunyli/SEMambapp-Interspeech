#!/bin/bash
# Run LTAS promotion/fresh-panel regressions on a Slurm compute node.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
SOURCE_ROOT="${SOURCE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$SOURCE_ROOT/runs/avqi_route_c_ltas_slope_tests_v1_20260824_01}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
PARTITION="${PARTITION:-batch-milan}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-00:45:00}"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-/scratch/work/lil14/.conda_envs/semambapp/bin/python}"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"
BASE_COMMIT="${BASE_COMMIT:-$SOURCE_COMMIT}"
TEST_FILES=(
  tests/test_avqi_ltas_slope_promotion.py
  tests/test_avqi_ltas_slope_gate_alignment.py
  tests/test_avqi_ltas_slope_svd_authority_panel.py
  tests/test_avqi_ltas_slope_fresh_panel.py
  tests/test_avqi_shimmer_fresh_panel.py
  tests/test_avqi_hnr_fresh_panel.py
  tests/test_avqi_components.py
  tests/test_avqi_route_c_multicomponent.py
  tests/test_artifact_layout.py
  tests/test_direct_avqi_waveform_optimization.py
)

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing to test a dirty LTAS source tree" >&2
  exit 2
fi
if [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" != "$BASE_COMMIT" ]]; then
  echo "LTAS test source commit drifted" >&2
  exit 2
fi
if [[ ! -x "$RUNTIME_PYTHON" ]]; then
  echo "Missing absolute semambapp Python: $RUNTIME_PYTHON" >&2
  exit 2
fi
for relative_path in "${TEST_FILES[@]}"; do
  if [[ ! -f "$SOURCE_ROOT/$relative_path" ]]; then
    echo "Missing LTAS regression file: $relative_path" >&2
    exit 2
  fi
done
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite LTAS regression output: $OUTPUT_DIR" >&2
  exit 2
fi

export SOURCE_ROOT RUN_ROOT LOG_DIR OUTPUT_DIR PARTITION CPUS_PER_TASK MEMORY
export TIME_LIMIT RUNTIME_PYTHON SOURCE_COMMIT BASE_COMMIT

mkdir -p "$LOG_DIR"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  sbatch \
    --parsable \
    --job-name="avqi-ltas-tests" \
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
cd "$SOURCE_ROOT"
mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTEST_LOG="$OUTPUT_DIR/pytest.log"
"$RUNTIME_PYTHON" -m pytest -q "${TEST_FILES[@]}" 2>&1 | tee "$PYTEST_LOG"
PYTEST_LOG_SHA256="$(sha256sum "$PYTEST_LOG" | awk '{print $1}')"
TEST_FILES_JSON="$(printf '%s\n' "${TEST_FILES[@]}" | jq -R . | jq -s .)"
jq -n \
  --arg source_commit "$SOURCE_COMMIT" \
  --arg slurm_job_id "$SLURM_JOB_ID" \
  --arg pytest_log "$PYTEST_LOG" \
  --arg pytest_log_sha256 "$PYTEST_LOG_SHA256" \
  --argjson test_files "$TEST_FILES_JSON" \
  '{decision: "PASS_LTAS_SLOPE_RELEVANT_TESTS", source_commit: $source_commit, slurm_job_id: $slurm_job_id, test_files: $test_files, pytest_log: $pytest_log, pytest_log_sha256: $pytest_log_sha256, generator_optimizer_steps: 0, formal_pathology_training_submitted: false}' \
  > "$OUTPUT_DIR/completion_receipt.json"
