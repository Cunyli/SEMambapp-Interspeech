#!/bin/bash
# Run the hash-bound v18 family-equivalence audit or four-case selector.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
EVALUATOR="$SOURCE_ROOT/scripts/evaluate_avqi_shimmer_db_topology_family_selector_v18.py"
PHASE="${PHASE:-equivalence}"
if [[ "$PHASE" != "equivalence" && "$PHASE" != "selector4" ]]; then
  echo "PHASE must be equivalence or selector4" >&2
  exit 2
fi

INPUT_ROOT="${INPUT_ROOT:-$ROOT_DIR/runs/avqi_route_c_shimmer_db_runtime_v15_fresh_panel_20260824_01/outputs}"
PANEL_CONTRACT="${PANEL_CONTRACT:-$INPUT_ROOT/panel_contract.json}"
PANEL_CONTRACT_SHA256="${PANEL_CONTRACT_SHA256:-b12fe527042cd4059f16856191338bb9c3e50744b6ecf1b71675f6273f619c62}"
FRESH_RESULTS="${FRESH_RESULTS:-$INPUT_ROOT/fresh_panel_results.csv}"
FRESH_RESULTS_SHA256="${FRESH_RESULTS_SHA256:-efee5a7f9a0d3e647a8167fe01b7e3cb114187328401c09dfb34a3f34ed5e8f6}"
PREDICTOR_CHECKPOINT="${PREDICTOR_CHECKPOINT:-/scratch/work/lil14/SEMambapp-Interspeech/checkpoints/avqi_route_c_shimmer_v6_screen_20260821_01/direct_direct_praat_hard_shimmer_pulse_path_v6_estimator.pt}"
PREDICTOR_CHECKPOINT_SHA256="${PREDICTOR_CHECKPOINT_SHA256:-40b819946abdcb8a4b643fe4238d1bb4d31168a3eb2a6d6c786a61752da629bc}"
EXACT_PYTHON="${EXACT_PYTHON:-/scratch/work/lil14/.conda_envs/avqi/bin/python}"
AVQI_CODE_ROOT="${AVQI_CODE_ROOT:-/scratch/work/lil14/avqi}"
AVQI_CODE_TREE_SHA256="${AVQI_CODE_TREE_SHA256:-46987b3c447cb579aab4d34e87655938e4aa64e1b28c0e2348c4ea3e48f107f2}"
RUNTIME_WORKER_SCRIPT="${RUNTIME_WORKER_SCRIPT:-$SOURCE_ROOT/scripts/avqi_shimmer_exact_topology_worker.py}"
RUNTIME_WORKER_SCRIPT_SHA256="${RUNTIME_WORKER_SCRIPT_SHA256:-c78cdb277274a9f46153c80ca5ad8c47536e3c1009cf1b3c2b613aee744d276f}"
EQUIVALENCE_ROOT="${EQUIVALENCE_ROOT:-$SOURCE_ROOT/runs/avqi_route_c_shimmer_db_topology_family_selector_v18_equivalence_20260824_01/outputs}"
EQUIVALENCE_REPORT="${EQUIVALENCE_REPORT:-$EQUIVALENCE_ROOT/equivalence_report.json}"
EQUIVALENCE_REPORT_SHA256="${EQUIVALENCE_REPORT_SHA256:-}"
EQUIVALENCE_RECEIPT="${EQUIVALENCE_RECEIPT:-$EQUIVALENCE_ROOT/completion_receipt.json}"
EQUIVALENCE_RECEIPT_SHA256="${EQUIVALENCE_RECEIPT_SHA256:-}"
if [[ "$PHASE" == "equivalence" ]]; then
  DEFAULT_RUN_ROOT="$SOURCE_ROOT/runs/avqi_route_c_shimmer_db_topology_family_selector_v18_equivalence_20260824_01"
else
  DEFAULT_RUN_ROOT="$SOURCE_ROOT/runs/avqi_route_c_shimmer_db_topology_family_selector_v18_4case_20260824_01"
fi
RUN_ROOT="${RUN_ROOT:-$DEFAULT_RUN_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
PARTITION="${PARTITION:-gpu-v100-32g}"
GPU_TYPE="${GPU_TYPE:-v100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-00:25:00}"
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
  "$PREDICTOR_CHECKPOINT" "$EXACT_PYTHON" "$RUNTIME_WORKER_SCRIPT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing v18 input: $path" >&2
    exit 2
  fi
done
if [[ ! -d "$AVQI_CODE_ROOT" ]]; then
  echo "Missing exact AVQI root: $AVQI_CODE_ROOT" >&2
  exit 2
fi
verify_sha256 "$PANEL_CONTRACT" "$PANEL_CONTRACT_SHA256" "opened panel"
verify_sha256 "$FRESH_RESULTS" "$FRESH_RESULTS_SHA256" "opened results"
verify_sha256 "$PREDICTOR_CHECKPOINT" "$PREDICTOR_CHECKPOINT_SHA256" "predictor"
verify_sha256 "$RUNTIME_WORKER_SCRIPT" "$RUNTIME_WORKER_SCRIPT_SHA256" "worker"

if [[ "$PHASE" == "selector4" ]]; then
  if [[ -z "$EQUIVALENCE_REPORT_SHA256" || -z "$EQUIVALENCE_RECEIPT_SHA256" ]]; then
    echo "selector4 requires equivalence report and receipt hashes" >&2
    exit 2
  fi
  for path in "$EQUIVALENCE_REPORT" "$EQUIVALENCE_RECEIPT"; do
    if [[ ! -f "$path" ]]; then
      echo "Missing v18 equivalence evidence: $path" >&2
      exit 2
    fi
  done
  verify_sha256 "$EQUIVALENCE_REPORT" "$EQUIVALENCE_REPORT_SHA256" "equivalence report"
  verify_sha256 "$EQUIVALENCE_RECEIPT" "$EQUIVALENCE_RECEIPT_SHA256" "equivalence receipt"
fi

export ROOT_DIR SOURCE_ROOT EVALUATOR PHASE INPUT_ROOT PANEL_CONTRACT
export PANEL_CONTRACT_SHA256 FRESH_RESULTS FRESH_RESULTS_SHA256
export PREDICTOR_CHECKPOINT PREDICTOR_CHECKPOINT_SHA256 EXACT_PYTHON
export AVQI_CODE_ROOT AVQI_CODE_TREE_SHA256 RUNTIME_WORKER_SCRIPT
export RUNTIME_WORKER_SCRIPT_SHA256 EQUIVALENCE_ROOT EQUIVALENCE_REPORT
export EQUIVALENCE_REPORT_SHA256 EQUIVALENCE_RECEIPT
export EQUIVALENCE_RECEIPT_SHA256 RUN_ROOT OUTPUT_DIR LOG_DIR PARTITION
export GPU_TYPE CPUS_PER_TASK MEMORY TIME_LIMIT SOFTWARE_STACK_MODULE
export COMPILER_MODULE SOURCE_COMMIT

mkdir -p "$LOG_DIR"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite v18 output: $OUTPUT_DIR" >&2
    exit 2
  fi
  sbatch \
    --parsable \
    --job-name="avqi-shim-v18-$PHASE" \
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
    "$SELF_PATH"
  exit 0
fi

if [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" != "$SOURCE_COMMIT" ]]; then
  echo "v18 source HEAD drifted after submission" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite v18 output: $OUTPUT_DIR" >&2
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

LIVE_LOG="$LOG_DIR/shimmer_db_topology_family_selector_v18_${PHASE}_${SLURM_JOB_ID}.log"
echo "event=start job=$SLURM_JOB_ID phase=$PHASE commit=$SOURCE_COMMIT time=$(date -Is)" | tee -a "$LIVE_LOG"
EXTRA_ARGS=()
if [[ "$PHASE" == "selector4" ]]; then
  EXTRA_ARGS+=(
    --equivalence-report "$EQUIVALENCE_REPORT"
    --equivalence-report-sha256 "$EQUIVALENCE_REPORT_SHA256"
    --equivalence-receipt "$EQUIVALENCE_RECEIPT"
    --equivalence-receipt-sha256 "$EQUIVALENCE_RECEIPT_SHA256"
  )
fi
python "$EVALUATOR" \
  --phase "$PHASE" \
  --panel-contract "$PANEL_CONTRACT" \
  --panel-contract-sha256 "$PANEL_CONTRACT_SHA256" \
  --fresh-results "$FRESH_RESULTS" \
  --fresh-results-sha256 "$FRESH_RESULTS_SHA256" \
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
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "$LIVE_LOG"
echo "event=complete job=$SLURM_JOB_ID phase=$PHASE time=$(date -Is)" | tee -a "$LIVE_LOG"
