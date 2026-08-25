#!/bin/bash
# Run the hash-bound v19 paired Sinc70 certificate on immutable opened dev data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
EVALUATOR="$SOURCE_ROOT/scripts/evaluate_avqi_shimmer_db_runtime_v19_peak_certificate.py"
PEAK_CERTIFICATE_HELPER="$SOURCE_ROOT/scripts/avqi_shimmer_peak_certificate_v19.py"
FROZEN_WORKER="$SOURCE_ROOT/scripts/avqi_shimmer_exact_topology_worker.py"

EVALUATOR_SHA256="${EVALUATOR_SHA256:-18f2456b20861772488fa96e2f6bb54374b97c8082b48cfaa47b97c8f5004ad2}"
PEAK_CERTIFICATE_HELPER_SHA256="${PEAK_CERTIFICATE_HELPER_SHA256:-e77f832423153817917fc903177816c227814df3dd162881266ab5ba49653249}"
FROZEN_WORKER_SHA256="${FROZEN_WORKER_SHA256:-c78cdb277274a9f46153c80ca5ad8c47536e3c1009cf1b3c2b613aee744d276f}"

V14_ROOT="${V14_ROOT:-$ROOT_DIR/runs/avqi_route_c_shimmer_db_candidate_c_fresh_panel_v14_20260824_01/outputs}"
V14_PANEL_CONTRACT="${V14_PANEL_CONTRACT:-$V14_ROOT/panel_contract.json}"
V14_PANEL_CONTRACT_SHA256="${V14_PANEL_CONTRACT_SHA256:-28d9726b2ecfa97d40cc973d768cb852f6cf15138ad903e85c548e9398ee9af2}"

V15_ROOT="${V15_ROOT:-$ROOT_DIR/runs/avqi_route_c_shimmer_db_runtime_v15_fresh_panel_20260824_01/outputs}"
V15_PANEL_CONTRACT="${V15_PANEL_CONTRACT:-$V15_ROOT/panel_contract.json}"
V15_PANEL_CONTRACT_SHA256="${V15_PANEL_CONTRACT_SHA256:-b12fe527042cd4059f16856191338bb9c3e50744b6ecf1b71675f6273f619c62}"

V18_RUN_ROOT="${V18_RUN_ROOT:-$ROOT_DIR/runs/avqi_route_c_shimmer_db_topology_family_selector_v18_opened24_20260826_01}"
V18_OUTPUT_ROOT="$V18_RUN_ROOT/outputs"
V18_REPORT="${V18_REPORT:-$V18_OUTPUT_ROOT/diagnostic_report.json}"
V18_REPORT_SHA256="${V18_REPORT_SHA256:-6420f3301d5a07cc75c7ac516467da4448d6460ea0a71db73384a89debbb67ea}"
V18_PRESELECTION="${V18_PRESELECTION:-$V18_OUTPUT_ROOT/family_selector_preselection.csv}"
V18_PRESELECTION_SHA256="${V18_PRESELECTION_SHA256:-22c8226ede7ba5fe7da1d921e8274d4ad654beec0ad6653f59b66a4add3436d7}"
V18_RECEIPT="${V18_RECEIPT:-$V18_OUTPUT_ROOT/completion_receipt.json}"
V18_RECEIPT_SHA256="${V18_RECEIPT_SHA256:-6f392a1c9ce4b0ae45bd81da7673bc87f29be39652123963bfae5e77b5cf551e}"

EXACT_PYTHON="${EXACT_PYTHON:-/scratch/work/lil14/.conda_envs/avqi/bin/python}"
DEFAULT_RUN_ROOT="$SOURCE_ROOT/runs/avqi_route_c_shimmer_db_runtime_v19_peak_certificate_20260826_01"
RUN_ROOT="${RUN_ROOT:-$DEFAULT_RUN_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
REPEATS="${REPEATS:-3}"

PARTITION="${PARTITION:-gpu-v100-32g}"
GPU_TYPE="${GPU_TYPE:-v100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-00:20:00}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"

SOURCE_STATUS="$(git -C "$SOURCE_ROOT" status --porcelain=v1 --untracked-files=all)"
if [[ -n "$SOURCE_STATUS" ]]; then
  echo "Refusing to run v19 peak certificate from dirty source: $SOURCE_ROOT" >&2
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

REQUIRED_FILES=(
  "$EVALUATOR" "$PEAK_CERTIFICATE_HELPER" "$FROZEN_WORKER"
  "$V14_PANEL_CONTRACT" "$V15_PANEL_CONTRACT"
  "$V18_REPORT" "$V18_PRESELECTION" "$V18_RECEIPT" "$EXACT_PYTHON"
)
for path in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing v19 peak-certificate input: $path" >&2
    exit 2
  fi
done

verify_sha256 "$EVALUATOR" "$EVALUATOR_SHA256" "v19 evaluator"
verify_sha256 "$PEAK_CERTIFICATE_HELPER" "$PEAK_CERTIFICATE_HELPER_SHA256" "v19 peak helper"
verify_sha256 "$FROZEN_WORKER" "$FROZEN_WORKER_SHA256" "frozen exact worker"
verify_sha256 "$V14_PANEL_CONTRACT" "$V14_PANEL_CONTRACT_SHA256" "v14 panel"
verify_sha256 "$V15_PANEL_CONTRACT" "$V15_PANEL_CONTRACT_SHA256" "v15 panel"
verify_sha256 "$V18_REPORT" "$V18_REPORT_SHA256" "immutable v18 report"
verify_sha256 "$V18_PRESELECTION" "$V18_PRESELECTION_SHA256" "immutable v18 preselection"
verify_sha256 "$V18_RECEIPT" "$V18_RECEIPT_SHA256" "immutable v18 receipt"

export ROOT_DIR SOURCE_ROOT EVALUATOR PEAK_CERTIFICATE_HELPER FROZEN_WORKER
export EVALUATOR_SHA256 PEAK_CERTIFICATE_HELPER_SHA256 FROZEN_WORKER_SHA256
export V14_ROOT V14_PANEL_CONTRACT V14_PANEL_CONTRACT_SHA256
export V15_ROOT V15_PANEL_CONTRACT V15_PANEL_CONTRACT_SHA256
export V18_RUN_ROOT V18_OUTPUT_ROOT V18_REPORT V18_REPORT_SHA256
export V18_PRESELECTION V18_PRESELECTION_SHA256 V18_RECEIPT V18_RECEIPT_SHA256
export EXACT_PYTHON RUN_ROOT OUTPUT_DIR LOG_DIR REPEATS SOURCE_COMMIT
export PARTITION GPU_TYPE CPUS_PER_TASK MEMORY TIME_LIMIT
export SOFTWARE_STACK_MODULE COMPILER_MODULE

mkdir -p "$LOG_DIR"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit v19 peak certificate without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite v19 peak-certificate output: $OUTPUT_DIR" >&2
    exit 2
  fi
  sbatch \
    --parsable \
    --job-name="avqi-shim-v19-cert" \
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
  echo "v19 peak-certificate source HEAD drifted after submission" >&2
  exit 2
fi
if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "v19 peak-certificate source became dirty after submission" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite v19 peak-certificate output: $OUTPUT_DIR" >&2
  exit 2
fi

cd "$ROOT_DIR"
module load "$SOFTWARE_STACK_MODULE"
module load "$COMPILER_MODULE"
export PYTHONPATH="$SOURCE_ROOT:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

LIVE_LOG="$LOG_DIR/shimmer_db_runtime_v19_peak_certificate_${SLURM_JOB_ID}.log"
echo "event=start job=$SLURM_JOB_ID phase=topology_only_peak_certificate commit=$SOURCE_COMMIT time=$(date -Is)" | tee -a "$LIVE_LOG"
"$EXACT_PYTHON" "$EVALUATOR" \
  --v14-panel-contract "$V14_PANEL_CONTRACT" \
  --v14-panel-contract-sha256 "$V14_PANEL_CONTRACT_SHA256" \
  --v15-panel-contract "$V15_PANEL_CONTRACT" \
  --v15-panel-contract-sha256 "$V15_PANEL_CONTRACT_SHA256" \
  --v18-run-root "$V18_RUN_ROOT" \
  --v18-report "$V18_REPORT" \
  --v18-report-sha256 "$V18_REPORT_SHA256" \
  --v18-preselection "$V18_PRESELECTION" \
  --v18-preselection-sha256 "$V18_PRESELECTION_SHA256" \
  --v18-receipt "$V18_RECEIPT" \
  --v18-receipt-sha256 "$V18_RECEIPT_SHA256" \
  --repository-root "$SOURCE_ROOT" \
  --peak-certificate-helper-sha256 "$PEAK_CERTIFICATE_HELPER_SHA256" \
  --evaluator-sha256 "$EVALUATOR_SHA256" \
  --frozen-worker-sha256 "$FROZEN_WORKER_SHA256" \
  --output-dir "$OUTPUT_DIR" \
  --source-commit "$SOURCE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID" \
  --repeats "$REPEATS" \
  2>&1 | tee -a "$LIVE_LOG"
echo "event=complete job=$SLURM_JOB_ID phase=topology_only_peak_certificate time=$(date -Is)" | tee -a "$LIVE_LOG"
