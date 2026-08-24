#!/bin/bash
# Run the topology-only FD23 pulse-time alignment diagnosis for v17.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
EVALUATOR="$SOURCE_ROOT/scripts/diagnose_avqi_shimmer_db_pulse_alignment_v17.py"
V15_ROOT="${V15_ROOT:-$ROOT_DIR/runs/avqi_route_c_shimmer_db_runtime_v15_fresh_panel_20260824_01/outputs}"
V16_ROOT="${V16_ROOT:-$ROOT_DIR/runs/avqi_route_c_shimmer_db_trust_region_v16_prototype_v15_20260824_02_instrumented/outputs}"
PANEL_CONTRACT="${PANEL_CONTRACT:-$V15_ROOT/panel_contract.json}"
PANEL_CONTRACT_SHA256="${PANEL_CONTRACT_SHA256:-b12fe527042cd4059f16856191338bb9c3e50744b6ecf1b71675f6273f619c62}"
ATTEMPTS_CSV="${ATTEMPTS_CSV:-$V16_ROOT/trust_region_attempts.csv}"
ATTEMPTS_CSV_SHA256="${ATTEMPTS_CSV_SHA256:-af017652b1416edd269732429d5febf609c302a34a77092b7816f3fc9ad09b2c}"
EXACT_PYTHON="${EXACT_PYTHON:-/scratch/work/lil14/.conda_envs/avqi/bin/python}"
AVQI_CODE_ROOT="${AVQI_CODE_ROOT:-/scratch/work/lil14/avqi}"
AVQI_CODE_TREE_SHA256="${AVQI_CODE_TREE_SHA256:-46987b3c447cb579aab4d34e87655938e4aa64e1b28c0e2348c4ea3e48f107f2}"
RUNTIME_WORKER_SCRIPT="${RUNTIME_WORKER_SCRIPT:-$SOURCE_ROOT/scripts/avqi_shimmer_exact_topology_worker.py}"
RUNTIME_WORKER_SCRIPT_SHA256="${RUNTIME_WORKER_SCRIPT_SHA256:-c78cdb277274a9f46153c80ca5ad8c47536e3c1009cf1b3c2b613aee744d276f}"
RUN_ROOT="${RUN_ROOT:-$SOURCE_ROOT/runs/avqi_route_c_shimmer_db_pulse_alignment_v17_20260824_01}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
PARTITION="${PARTITION:-gpu-v100-32g}"
GPU_TYPE="${GPU_TYPE:-v100}"
TIME_LIMIT="${TIME_LIMIT:-00:10:00}"

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing to run from dirty source: $SOURCE_ROOT" >&2
  exit 2
fi
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"

verify_sha256() {
  local file_path="$1"
  local expected="$2"
  local label="$3"
  local actual
  actual="$(sha256sum "$file_path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "$label hash drift: $actual != $expected" >&2
    exit 2
  fi
}

for path in "$EVALUATOR" "$PANEL_CONTRACT" "$ATTEMPTS_CSV" \
  "$EXACT_PYTHON" "$RUNTIME_WORKER_SCRIPT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing v17 diagnostic input: $path" >&2
    exit 2
  fi
done
verify_sha256 "$PANEL_CONTRACT" "$PANEL_CONTRACT_SHA256" "opened panel"
verify_sha256 "$ATTEMPTS_CSV" "$ATTEMPTS_CSV_SHA256" "v16 attempts"
verify_sha256 "$RUNTIME_WORKER_SCRIPT" "$RUNTIME_WORKER_SCRIPT_SHA256" "worker"

export ROOT_DIR SOURCE_ROOT EVALUATOR V15_ROOT V16_ROOT PANEL_CONTRACT
export PANEL_CONTRACT_SHA256 ATTEMPTS_CSV ATTEMPTS_CSV_SHA256 EXACT_PYTHON
export AVQI_CODE_ROOT AVQI_CODE_TREE_SHA256 RUNTIME_WORKER_SCRIPT
export RUNTIME_WORKER_SCRIPT_SHA256 RUN_ROOT OUTPUT_DIR LOG_DIR PARTITION
export GPU_TYPE TIME_LIMIT SOURCE_COMMIT

mkdir -p "$LOG_DIR"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite diagnostic output: $OUTPUT_DIR" >&2
    exit 2
  fi
  sbatch \
    --parsable \
    --job-name="avqi-shim-align-v17" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --gres="gpu:${GPU_TYPE}:1" \
    --cpus-per-task=4 \
    --mem=16G \
    --time="$TIME_LIMIT" \
    --output="$LOG_DIR/slurm_%j.out" \
    --error="$LOG_DIR/slurm_%j.err" \
    --export=ALL \
    "$SELF_PATH"
  exit 0
fi

if [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" != "$SOURCE_COMMIT" ]]; then
  echo "Diagnostic source HEAD drifted after submission" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite diagnostic output: $OUTPUT_DIR" >&2
  exit 2
fi

cd "$ROOT_DIR"
export PYTHONPATH="$SOURCE_ROOT:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
LIVE_LOG="$LOG_DIR/shimmer_db_pulse_alignment_v17_${SLURM_JOB_ID}.log"
echo "event=start job=$SLURM_JOB_ID commit=$SOURCE_COMMIT time=$(date -Is)" | tee -a "$LIVE_LOG"
"$EXACT_PYTHON" "$EVALUATOR" \
  --panel-contract "$PANEL_CONTRACT" \
  --panel-contract-sha256 "$PANEL_CONTRACT_SHA256" \
  --attempts-csv "$ATTEMPTS_CSV" \
  --attempts-csv-sha256 "$ATTEMPTS_CSV_SHA256" \
  --exact-python "$EXACT_PYTHON" \
  --avqi-code-root "$AVQI_CODE_ROOT" \
  --avqi-code-tree-sha256 "$AVQI_CODE_TREE_SHA256" \
  --runtime-worker-script "$RUNTIME_WORKER_SCRIPT" \
  --runtime-worker-script-sha256 "$RUNTIME_WORKER_SCRIPT_SHA256" \
  --output-dir "$OUTPUT_DIR" \
  --source-commit "$SOURCE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID" \
  2>&1 | tee -a "$LIVE_LOG"
echo "event=complete job=$SLURM_JOB_ID time=$(date -Is)" | tee -a "$LIVE_LOG"
