#!/bin/bash
# Run the four-case source-informed Candidate-D Shimmer-dB study.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
EVALUATOR="$SOURCE_ROOT/scripts/evaluate_avqi_shimmer_db_source_informed_v17.py"
INPUT_ROOT="${INPUT_ROOT:-$ROOT_DIR/runs/avqi_route_c_shimmer_db_runtime_v15_fresh_panel_20260824_01/outputs}"
DIAGNOSTIC_ROOT="${DIAGNOSTIC_ROOT:-$ROOT_DIR/runs/avqi_route_c_shimmer_db_pulse_alignment_v17_20260824_01/outputs}"
PANEL_CONTRACT="${PANEL_CONTRACT:-$INPUT_ROOT/panel_contract.json}"
PANEL_CONTRACT_SHA256="${PANEL_CONTRACT_SHA256:-b12fe527042cd4059f16856191338bb9c3e50744b6ecf1b71675f6273f619c62}"
FRESH_RESULTS="${FRESH_RESULTS:-$INPUT_ROOT/fresh_panel_results.csv}"
FRESH_RESULTS_SHA256="${FRESH_RESULTS_SHA256:-efee5a7f9a0d3e647a8167fe01b7e3cb114187328401c09dfb34a3f34ed5e8f6}"
PULSE_DIAGNOSTIC_REPORT="${PULSE_DIAGNOSTIC_REPORT:-$DIAGNOSTIC_ROOT/pulse_alignment_report.json}"
PULSE_DIAGNOSTIC_REPORT_SHA256="${PULSE_DIAGNOSTIC_REPORT_SHA256:-a7ae6f14e0f45b0bfe70c96dedfcf194d105bbec45c5aa004dd2a217a3b9b63b}"
PULSE_DIAGNOSTIC_RECEIPT="${PULSE_DIAGNOSTIC_RECEIPT:-$DIAGNOSTIC_ROOT/completion_receipt.json}"
PULSE_DIAGNOSTIC_RECEIPT_SHA256="${PULSE_DIAGNOSTIC_RECEIPT_SHA256:-02ddc14c602bb14fd1507c1b25da4afe9a42c81fc0b53e5162ccf302ec26b81d}"
PREDICTOR_CHECKPOINT="${PREDICTOR_CHECKPOINT:-/scratch/work/lil14/SEMambapp-Interspeech/checkpoints/avqi_route_c_shimmer_v6_screen_20260821_01/direct_direct_praat_hard_shimmer_pulse_path_v6_estimator.pt}"
PREDICTOR_CHECKPOINT_SHA256="${PREDICTOR_CHECKPOINT_SHA256:-40b819946abdcb8a4b643fe4238d1bb4d31168a3eb2a6d6c786a61752da629bc}"
EXACT_PYTHON="${EXACT_PYTHON:-/scratch/work/lil14/.conda_envs/avqi/bin/python}"
AVQI_CODE_ROOT="${AVQI_CODE_ROOT:-/scratch/work/lil14/avqi}"
AVQI_CODE_TREE_SHA256="${AVQI_CODE_TREE_SHA256:-46987b3c447cb579aab4d34e87655938e4aa64e1b28c0e2348c4ea3e48f107f2}"
RUNTIME_WORKER_SCRIPT="${RUNTIME_WORKER_SCRIPT:-$SOURCE_ROOT/scripts/avqi_shimmer_exact_topology_worker.py}"
RUNTIME_WORKER_SCRIPT_SHA256="${RUNTIME_WORKER_SCRIPT_SHA256:-c78cdb277274a9f46153c80ca5ad8c47536e3c1009cf1b3c2b613aee744d276f}"
RUN_ROOT="${RUN_ROOT:-$SOURCE_ROOT/runs/avqi_route_c_shimmer_db_source_informed_v17_4case_20260824_01}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
PARTITION="${PARTITION:-gpu-v100-32g}"
GPU_TYPE="${GPU_TYPE:-v100}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"

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

for path in "$EVALUATOR" "$PANEL_CONTRACT" "$FRESH_RESULTS" \
  "$PULSE_DIAGNOSTIC_REPORT" "$PULSE_DIAGNOSTIC_RECEIPT" \
  "$PREDICTOR_CHECKPOINT" "$EXACT_PYTHON" "$RUNTIME_WORKER_SCRIPT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing Candidate-D input: $path" >&2
    exit 2
  fi
done
verify_sha256 "$PANEL_CONTRACT" "$PANEL_CONTRACT_SHA256" "opened panel"
verify_sha256 "$FRESH_RESULTS" "$FRESH_RESULTS_SHA256" "opened results"
verify_sha256 "$PULSE_DIAGNOSTIC_REPORT" "$PULSE_DIAGNOSTIC_REPORT_SHA256" "pulse diagnostic report"
verify_sha256 "$PULSE_DIAGNOSTIC_RECEIPT" "$PULSE_DIAGNOSTIC_RECEIPT_SHA256" "pulse diagnostic receipt"
verify_sha256 "$PREDICTOR_CHECKPOINT" "$PREDICTOR_CHECKPOINT_SHA256" "predictor"
verify_sha256 "$RUNTIME_WORKER_SCRIPT" "$RUNTIME_WORKER_SCRIPT_SHA256" "worker"

export ROOT_DIR SOURCE_ROOT EVALUATOR INPUT_ROOT DIAGNOSTIC_ROOT PANEL_CONTRACT
export PANEL_CONTRACT_SHA256 FRESH_RESULTS FRESH_RESULTS_SHA256
export PULSE_DIAGNOSTIC_REPORT PULSE_DIAGNOSTIC_REPORT_SHA256
export PULSE_DIAGNOSTIC_RECEIPT PULSE_DIAGNOSTIC_RECEIPT_SHA256
export PREDICTOR_CHECKPOINT PREDICTOR_CHECKPOINT_SHA256 EXACT_PYTHON
export AVQI_CODE_ROOT AVQI_CODE_TREE_SHA256 RUNTIME_WORKER_SCRIPT
export RUNTIME_WORKER_SCRIPT_SHA256 RUN_ROOT OUTPUT_DIR LOG_DIR PARTITION
export GPU_TYPE SOFTWARE_STACK_MODULE COMPILER_MODULE SOURCE_COMMIT

mkdir -p "$LOG_DIR"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite Candidate-D output: $OUTPUT_DIR" >&2
    exit 2
  fi
  sbatch \
    --parsable \
    --job-name="avqi-shim-src-v17" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --gres="gpu:${GPU_TYPE}:1" \
    --cpus-per-task=4 \
    --mem=24G \
    --time=00:15:00 \
    --output="$LOG_DIR/slurm_%j.out" \
    --error="$LOG_DIR/slurm_%j.err" \
    --export=ALL \
    "$SELF_PATH"
  exit 0
fi

if [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" != "$SOURCE_COMMIT" ]]; then
  echo "Candidate-D source HEAD drifted after submission" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite Candidate-D output: $OUTPUT_DIR" >&2
  exit 2
fi

cd "$ROOT_DIR"
module load "$SOFTWARE_STACK_MODULE"
module load "$COMPILER_MODULE"
eval "$(conda shell.bash hook)"
conda activate semambapp
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export PYTHONPATH="$SOURCE_ROOT:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

LIVE_LOG="$LOG_DIR/shimmer_db_source_informed_v17_${SLURM_JOB_ID}.log"
echo "event=start job=$SLURM_JOB_ID commit=$SOURCE_COMMIT time=$(date -Is)" | tee -a "$LIVE_LOG"
python "$EVALUATOR" \
  --panel-contract "$PANEL_CONTRACT" \
  --panel-contract-sha256 "$PANEL_CONTRACT_SHA256" \
  --fresh-results "$FRESH_RESULTS" \
  --fresh-results-sha256 "$FRESH_RESULTS_SHA256" \
  --pulse-diagnostic-report "$PULSE_DIAGNOSTIC_REPORT" \
  --pulse-diagnostic-report-sha256 "$PULSE_DIAGNOSTIC_REPORT_SHA256" \
  --pulse-diagnostic-receipt "$PULSE_DIAGNOSTIC_RECEIPT" \
  --pulse-diagnostic-receipt-sha256 "$PULSE_DIAGNOSTIC_RECEIPT_SHA256" \
  --predictor-checkpoint "$PREDICTOR_CHECKPOINT" \
  --predictor-checkpoint-sha256 "$PREDICTOR_CHECKPOINT_SHA256" \
  --exact-python "$EXACT_PYTHON" \
  --avqi-code-root "$AVQI_CODE_ROOT" \
  --avqi-code-tree-sha256 "$AVQI_CODE_TREE_SHA256" \
  --runtime-worker-script "$RUNTIME_WORKER_SCRIPT" \
  --runtime-worker-script-sha256 "$RUNTIME_WORKER_SCRIPT_SHA256" \
  --output-dir "$OUTPUT_DIR" \
  --source-commit "$SOURCE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID" \
  --device cuda \
  2>&1 | tee -a "$LIVE_LOG"
echo "event=complete job=$SLURM_JOB_ID time=$(date -Is)" | tee -a "$LIVE_LOG"
