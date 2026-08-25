#!/bin/bash
# Run the hash-bound opened v14+v15 24-case v18 selector audit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
EVALUATOR="$SOURCE_ROOT/scripts/evaluate_avqi_shimmer_db_topology_family_selector_opened24.py"

V14_ROOT="${V14_ROOT:-$ROOT_DIR/runs/avqi_route_c_shimmer_db_candidate_c_fresh_panel_v14_20260824_01/outputs}"
V14_PANEL_CONTRACT="${V14_PANEL_CONTRACT:-$V14_ROOT/panel_contract.json}"
V14_PANEL_CONTRACT_SHA256="${V14_PANEL_CONTRACT_SHA256:-28d9726b2ecfa97d40cc973d768cb852f6cf15138ad903e85c548e9398ee9af2}"
V14_TARGET_CONTRACT="${V14_TARGET_CONTRACT:-$V14_ROOT/target_label_contract.json}"
V14_TARGET_CONTRACT_SHA256="${V14_TARGET_CONTRACT_SHA256:-4d6a4f43d2a982e8d1862abc5bc722f44908d6221b1ff867064fbc44ab53fdd9}"
V14_FRESH_RESULTS="${V14_FRESH_RESULTS:-$V14_ROOT/fresh_panel_results.csv}"
V14_FRESH_RESULTS_SHA256="${V14_FRESH_RESULTS_SHA256:-809fadcfb48311d910b64fd001d2d2925dbe85bc0265a6229114e9ad01185795}"

V15_ROOT="${V15_ROOT:-$ROOT_DIR/runs/avqi_route_c_shimmer_db_runtime_v15_fresh_panel_20260824_01/outputs}"
V15_PANEL_CONTRACT="${V15_PANEL_CONTRACT:-$V15_ROOT/panel_contract.json}"
V15_PANEL_CONTRACT_SHA256="${V15_PANEL_CONTRACT_SHA256:-b12fe527042cd4059f16856191338bb9c3e50744b6ecf1b71675f6273f619c62}"
V15_TARGET_CONTRACT="${V15_TARGET_CONTRACT:-$V15_ROOT/target_label_contract.json}"
V15_TARGET_CONTRACT_SHA256="${V15_TARGET_CONTRACT_SHA256:-e003618c7be1ace9d01fd6c2fd0dc2346a4dec4d81ce675a3d89a87013a3d222}"
V15_FRESH_RESULTS="${V15_FRESH_RESULTS:-$V15_ROOT/fresh_panel_results.csv}"
V15_FRESH_RESULTS_SHA256="${V15_FRESH_RESULTS_SHA256:-efee5a7f9a0d3e647a8167fe01b7e3cb114187328401c09dfb34a3f34ed5e8f6}"

SELECTOR4_ROOT="${SELECTOR4_ROOT:-$ROOT_DIR/runs/avqi_route_c_shimmer_db_topology_family_selector_v18_4case_20260826_02_parallel_pcm/outputs}"
SELECTOR4_REPORT="${SELECTOR4_REPORT:-$SELECTOR4_ROOT/diagnostic_report.json}"
SELECTOR4_REPORT_SHA256="${SELECTOR4_REPORT_SHA256:-9c87982dfbf0bdc2cd685b3829f1e9bf6c2d97a953df6a4ca9f766ffe7c370e0}"
SELECTOR4_PRESELECTION="${SELECTOR4_PRESELECTION:-$SELECTOR4_ROOT/family_selector_preselection.csv}"
SELECTOR4_PRESELECTION_SHA256="${SELECTOR4_PRESELECTION_SHA256:-5d67e8960c965f6ced61223548620f6d7e2a37c46c6c995327c68874e3028be9}"
SELECTOR4_SEAL="${SELECTOR4_SEAL:-$SELECTOR4_ROOT/selector_seal.json}"
SELECTOR4_SEAL_SHA256="${SELECTOR4_SEAL_SHA256:-9ea07a264000d2821cd0b742a4d1de2bf5f503a6603d5f4357c3a1d969fb0d1e}"
SELECTOR4_RESULTS="${SELECTOR4_RESULTS:-$SELECTOR4_ROOT/family_selector_results.csv}"
SELECTOR4_RESULTS_SHA256="${SELECTOR4_RESULTS_SHA256:-72d408d53291d827ac0179c9e1cb7ef39f76aa99bf6328afa2041467fb20c43d}"
SELECTOR4_RECEIPT="${SELECTOR4_RECEIPT:-$SELECTOR4_ROOT/completion_receipt.json}"
SELECTOR4_RECEIPT_SHA256="${SELECTOR4_RECEIPT_SHA256:-a137b335074d1453515a1cb54e1b67a6683426fd430f11af625f3d600ff8e06f}"

SELECTOR_CORE_SCRIPT="${SELECTOR_CORE_SCRIPT:-$SOURCE_ROOT/scripts/evaluate_avqi_shimmer_db_topology_family_selector_v18.py}"
SELECTOR_CORE_SCRIPT_SHA256="${SELECTOR_CORE_SCRIPT_SHA256:-7401b4b80f6dbb546a4a88886c469bb4df6b4681bad9314f1244a046fbb2b69b}"
V16_FAMILY_SOURCE="${V16_FAMILY_SOURCE:-$SOURCE_ROOT/scripts/evaluate_avqi_shimmer_db_trust_region_v16.py}"
V16_FAMILY_SOURCE_SHA256="${V16_FAMILY_SOURCE_SHA256:-d8bfb0f31d9d98832d6c4409e5044b5d7cbe0b8b585e72f359fa3119d22aa662}"
V17_FAMILY_SOURCE="${V17_FAMILY_SOURCE:-$SOURCE_ROOT/scripts/evaluate_avqi_shimmer_db_source_informed_v17.py}"
V17_FAMILY_SOURCE_SHA256="${V17_FAMILY_SOURCE_SHA256:-324660709b2e6a4994d057c4d532cf89613f535ec96490f2cb038d7b33f55b22}"
PREDICTOR_CHECKPOINT="${PREDICTOR_CHECKPOINT:-/scratch/work/lil14/SEMambapp-Interspeech/checkpoints/avqi_route_c_shimmer_v6_screen_20260821_01/direct_direct_praat_hard_shimmer_pulse_path_v6_estimator.pt}"
PREDICTOR_CHECKPOINT_SHA256="${PREDICTOR_CHECKPOINT_SHA256:-40b819946abdcb8a4b643fe4238d1bb4d31168a3eb2a6d6c786a61752da629bc}"
EXACT_PYTHON="${EXACT_PYTHON:-/scratch/work/lil14/.conda_envs/avqi/bin/python}"
AVQI_CODE_ROOT="${AVQI_CODE_ROOT:-/scratch/work/lil14/avqi}"
AVQI_CODE_TREE_SHA256="${AVQI_CODE_TREE_SHA256:-46987b3c447cb579aab4d34e87655938e4aa64e1b28c0e2348c4ea3e48f107f2}"
RUNTIME_WORKER_SCRIPT="${RUNTIME_WORKER_SCRIPT:-$SOURCE_ROOT/scripts/avqi_shimmer_exact_topology_worker.py}"
RUNTIME_WORKER_SCRIPT_SHA256="${RUNTIME_WORKER_SCRIPT_SHA256:-c78cdb277274a9f46153c80ca5ad8c47536e3c1009cf1b3c2b613aee744d276f}"

DEFAULT_RUN_ROOT="$SOURCE_ROOT/runs/avqi_route_c_shimmer_db_topology_family_selector_v18_opened24_20260826_01"
RUN_ROOT="${RUN_ROOT:-$DEFAULT_RUN_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
PARTITION="${PARTITION:-gpu-v100-32g}"
GPU_TYPE="${GPU_TYPE:-v100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-00:35:00}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing to run opened24 from dirty source: $SOURCE_ROOT" >&2
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
  "$EVALUATOR"
  "$V14_PANEL_CONTRACT" "$V14_TARGET_CONTRACT" "$V14_FRESH_RESULTS"
  "$V15_PANEL_CONTRACT" "$V15_TARGET_CONTRACT" "$V15_FRESH_RESULTS"
  "$SELECTOR4_REPORT" "$SELECTOR4_PRESELECTION" "$SELECTOR4_SEAL"
  "$SELECTOR4_RESULTS" "$SELECTOR4_RECEIPT"
  "$SELECTOR_CORE_SCRIPT" "$V16_FAMILY_SOURCE" "$V17_FAMILY_SOURCE"
  "$PREDICTOR_CHECKPOINT" "$EXACT_PYTHON" "$RUNTIME_WORKER_SCRIPT"
)
for path in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing opened24 input: $path" >&2
    exit 2
  fi
done
if [[ ! -d "$AVQI_CODE_ROOT" ]]; then
  echo "Missing exact AVQI root: $AVQI_CODE_ROOT" >&2
  exit 2
fi

verify_sha256 "$V14_PANEL_CONTRACT" "$V14_PANEL_CONTRACT_SHA256" "v14 panel"
verify_sha256 "$V14_TARGET_CONTRACT" "$V14_TARGET_CONTRACT_SHA256" "v14 target"
verify_sha256 "$V14_FRESH_RESULTS" "$V14_FRESH_RESULTS_SHA256" "v14 results"
verify_sha256 "$V15_PANEL_CONTRACT" "$V15_PANEL_CONTRACT_SHA256" "v15 panel"
verify_sha256 "$V15_TARGET_CONTRACT" "$V15_TARGET_CONTRACT_SHA256" "v15 target"
verify_sha256 "$V15_FRESH_RESULTS" "$V15_FRESH_RESULTS_SHA256" "v15 results"
verify_sha256 "$SELECTOR4_REPORT" "$SELECTOR4_REPORT_SHA256" "selector4 report"
verify_sha256 "$SELECTOR4_PRESELECTION" "$SELECTOR4_PRESELECTION_SHA256" "selector4 preselection"
verify_sha256 "$SELECTOR4_SEAL" "$SELECTOR4_SEAL_SHA256" "selector4 seal"
verify_sha256 "$SELECTOR4_RESULTS" "$SELECTOR4_RESULTS_SHA256" "selector4 results"
verify_sha256 "$SELECTOR4_RECEIPT" "$SELECTOR4_RECEIPT_SHA256" "selector4 receipt"
verify_sha256 "$SELECTOR_CORE_SCRIPT" "$SELECTOR_CORE_SCRIPT_SHA256" "selector core"
verify_sha256 "$V16_FAMILY_SOURCE" "$V16_FAMILY_SOURCE_SHA256" "v16 family"
verify_sha256 "$V17_FAMILY_SOURCE" "$V17_FAMILY_SOURCE_SHA256" "v17 family"
verify_sha256 "$PREDICTOR_CHECKPOINT" "$PREDICTOR_CHECKPOINT_SHA256" "predictor"
verify_sha256 "$RUNTIME_WORKER_SCRIPT" "$RUNTIME_WORKER_SCRIPT_SHA256" "worker"

export ROOT_DIR SOURCE_ROOT EVALUATOR V14_ROOT V14_PANEL_CONTRACT
export V14_PANEL_CONTRACT_SHA256 V14_TARGET_CONTRACT V14_TARGET_CONTRACT_SHA256
export V14_FRESH_RESULTS V14_FRESH_RESULTS_SHA256 V15_ROOT V15_PANEL_CONTRACT
export V15_PANEL_CONTRACT_SHA256 V15_TARGET_CONTRACT V15_TARGET_CONTRACT_SHA256
export V15_FRESH_RESULTS V15_FRESH_RESULTS_SHA256 SELECTOR4_ROOT SELECTOR4_REPORT
export SELECTOR4_REPORT_SHA256 SELECTOR4_PRESELECTION
export SELECTOR4_PRESELECTION_SHA256 SELECTOR4_SEAL SELECTOR4_SEAL_SHA256
export SELECTOR4_RESULTS SELECTOR4_RESULTS_SHA256 SELECTOR4_RECEIPT
export SELECTOR4_RECEIPT_SHA256 SELECTOR_CORE_SCRIPT SELECTOR_CORE_SCRIPT_SHA256
export V16_FAMILY_SOURCE V16_FAMILY_SOURCE_SHA256 V17_FAMILY_SOURCE
export V17_FAMILY_SOURCE_SHA256 PREDICTOR_CHECKPOINT PREDICTOR_CHECKPOINT_SHA256
export EXACT_PYTHON AVQI_CODE_ROOT AVQI_CODE_TREE_SHA256 RUNTIME_WORKER_SCRIPT
export RUNTIME_WORKER_SCRIPT_SHA256 RUN_ROOT OUTPUT_DIR LOG_DIR PARTITION GPU_TYPE
export CPUS_PER_TASK MEMORY TIME_LIMIT SOFTWARE_STACK_MODULE COMPILER_MODULE
export SOURCE_COMMIT

mkdir -p "$LOG_DIR"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit opened24 without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite opened24 output: $OUTPUT_DIR" >&2
    exit 2
  fi
  sbatch \
    --parsable \
    --job-name="avqi-shim-v18-opened24" \
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
  echo "opened24 source HEAD drifted after submission" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite opened24 output: $OUTPUT_DIR" >&2
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

LIVE_LOG="$LOG_DIR/shimmer_db_topology_family_selector_opened24_${SLURM_JOB_ID}.log"
echo "event=start job=$SLURM_JOB_ID phase=opened24 commit=$SOURCE_COMMIT time=$(date -Is)" | tee -a "$LIVE_LOG"
python "$EVALUATOR" \
  --v14-panel-contract "$V14_PANEL_CONTRACT" \
  --v14-panel-contract-sha256 "$V14_PANEL_CONTRACT_SHA256" \
  --v14-target-contract "$V14_TARGET_CONTRACT" \
  --v14-target-contract-sha256 "$V14_TARGET_CONTRACT_SHA256" \
  --v14-fresh-results "$V14_FRESH_RESULTS" \
  --v14-fresh-results-sha256 "$V14_FRESH_RESULTS_SHA256" \
  --v15-panel-contract "$V15_PANEL_CONTRACT" \
  --v15-panel-contract-sha256 "$V15_PANEL_CONTRACT_SHA256" \
  --v15-target-contract "$V15_TARGET_CONTRACT" \
  --v15-target-contract-sha256 "$V15_TARGET_CONTRACT_SHA256" \
  --v15-fresh-results "$V15_FRESH_RESULTS" \
  --v15-fresh-results-sha256 "$V15_FRESH_RESULTS_SHA256" \
  --selector4-report "$SELECTOR4_REPORT" \
  --selector4-report-sha256 "$SELECTOR4_REPORT_SHA256" \
  --selector4-preselection "$SELECTOR4_PRESELECTION" \
  --selector4-preselection-sha256 "$SELECTOR4_PRESELECTION_SHA256" \
  --selector4-seal "$SELECTOR4_SEAL" \
  --selector4-seal-sha256 "$SELECTOR4_SEAL_SHA256" \
  --selector4-results "$SELECTOR4_RESULTS" \
  --selector4-results-sha256 "$SELECTOR4_RESULTS_SHA256" \
  --selector4-receipt "$SELECTOR4_RECEIPT" \
  --selector4-receipt-sha256 "$SELECTOR4_RECEIPT_SHA256" \
  --selector-core-script "$SELECTOR_CORE_SCRIPT" \
  --selector-core-script-sha256 "$SELECTOR_CORE_SCRIPT_SHA256" \
  --v16-family-source "$V16_FAMILY_SOURCE" \
  --v16-family-source-sha256 "$V16_FAMILY_SOURCE_SHA256" \
  --v17-family-source "$V17_FAMILY_SOURCE" \
  --v17-family-source-sha256 "$V17_FAMILY_SOURCE_SHA256" \
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
echo "event=complete job=$SLURM_JOB_ID phase=opened24 time=$(date -Is)" | tee -a "$LIVE_LOG"
