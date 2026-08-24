#!/bin/bash
# Audit exact-equivalent Candidate-C latency on the already-opened v14 dev panel.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
AUDIT_SCRIPT="$SOURCE_ROOT/scripts/evaluate_avqi_shimmer_db_runtime_v15_equivalence.py"
WORKER_SCRIPT="$SOURCE_ROOT/scripts/avqi_shimmer_exact_topology_worker.py"

RUN_ROOT="${RUN_ROOT:-$SOURCE_ROOT/runs/avqi_route_c_shimmer_db_runtime_v15_equivalence_20260824_07_sinc70_safe_bound}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
V14_ROOT="${V14_ROOT:-$SOURCE_ROOT/runs/avqi_route_c_shimmer_db_candidate_c_fresh_panel_v14_20260824_01/outputs}"
PANEL_CONTRACT="${PANEL_CONTRACT:-$V14_ROOT/panel_contract.json}"
TARGET_LABEL_CONTRACT="${TARGET_LABEL_CONTRACT:-$V14_ROOT/target_label_contract.json}"
CANDIDATE_SEAL="${CANDIDATE_SEAL:-$V14_ROOT/candidate_seal.json}"
PREDICTOR_CHECKPOINT="${PREDICTOR_CHECKPOINT:-/scratch/work/lil14/SEMambapp-Interspeech/checkpoints/avqi_route_c_shimmer_v6_screen_20260821_01/direct_direct_praat_hard_shimmer_pulse_path_v6_estimator.pt}"
EXACT_PYTHON="${EXACT_PYTHON:-/scratch/work/lil14/.conda_envs/avqi/bin/python}"
AVQI_CODE_ROOT="${AVQI_CODE_ROOT:-/scratch/work/lil14/avqi}"

PANEL_CONTRACT_SHA256="${PANEL_CONTRACT_SHA256:-28d9726b2ecfa97d40cc973d768cb852f6cf15138ad903e85c548e9398ee9af2}"
TARGET_LABEL_CONTRACT_SHA256="${TARGET_LABEL_CONTRACT_SHA256:-4d6a4f43d2a982e8d1862abc5bc722f44908d6221b1ff867064fbc44ab53fdd9}"
CANDIDATE_SEAL_SHA256="${CANDIDATE_SEAL_SHA256:-c7ed5dc5aa36ddcd8a807dc77400ba3c6524ff3dd6f8a8873e3f3d1c1fc8ecd6}"
PREDICTOR_CHECKPOINT_SHA256="${PREDICTOR_CHECKPOINT_SHA256:-40b819946abdcb8a4b643fe4238d1bb4d31168a3eb2a6d6c786a61752da629bc}"
AVQI_CODE_TREE_SHA256="${AVQI_CODE_TREE_SHA256:-46987b3c447cb579aab4d34e87655938e4aa64e1b28c0e2348c4ea3e48f107f2}"

PARTITION="${PARTITION:-gpu-v100-32g}"
GPU_TYPE="${GPU_TYPE:-v100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
MEMORY="${MEMORY:-24G}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"
WARM_REPEATS="${WARM_REPEATS:-3}"
HIGHPASS_MODE="${HIGHPASS_MODE:-numpy_official_praat_6_1_38_stop_hann_0_34_0p1}"

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing to run from dirty source: $SOURCE_ROOT" >&2
  exit 2
fi
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"

for path in "$AUDIT_SCRIPT" "$WORKER_SCRIPT" "$PANEL_CONTRACT" \
  "$TARGET_LABEL_CONTRACT" "$CANDIDATE_SEAL" "$PREDICTOR_CHECKPOINT" \
  "$EXACT_PYTHON"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing v15 equivalence input: $path" >&2
    exit 2
  fi
done
if [[ ! -d "$AVQI_CODE_ROOT" ]]; then
  echo "Missing exact AVQI code tree: $AVQI_CODE_ROOT" >&2
  exit 2
fi

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

verify_sha256 "$PANEL_CONTRACT" "$PANEL_CONTRACT_SHA256" "v14 panel contract"
verify_sha256 "$TARGET_LABEL_CONTRACT" "$TARGET_LABEL_CONTRACT_SHA256" "v14 target contract"
verify_sha256 "$CANDIDATE_SEAL" "$CANDIDATE_SEAL_SHA256" "v14 candidate seal"
verify_sha256 "$PREDICTOR_CHECKPOINT" "$PREDICTOR_CHECKPOINT_SHA256" "Shimmer v6 predictor"

export ROOT_DIR SOURCE_ROOT AUDIT_SCRIPT WORKER_SCRIPT RUN_ROOT LOG_DIR OUTPUT_DIR
export V14_ROOT PANEL_CONTRACT PANEL_CONTRACT_SHA256 TARGET_LABEL_CONTRACT
export TARGET_LABEL_CONTRACT_SHA256 CANDIDATE_SEAL CANDIDATE_SEAL_SHA256
export PREDICTOR_CHECKPOINT PREDICTOR_CHECKPOINT_SHA256 EXACT_PYTHON
export AVQI_CODE_ROOT AVQI_CODE_TREE_SHA256 PARTITION GPU_TYPE CPUS_PER_TASK
export MEMORY TIME_LIMIT SOFTWARE_STACK_MODULE COMPILER_MODULE WARM_REPEATS
export HIGHPASS_MODE SOURCE_COMMIT

mkdir -p "$LOG_DIR"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite v15 equivalence output: $OUTPUT_DIR" >&2
    exit 2
  fi
  sbatch \
    --parsable \
    --job-name=avqi-shim-db-v15-eq \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --gres="gpu:${GPU_TYPE}:1" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --hint=nomultithread \
    --mem="$MEMORY" \
    --time="$TIME_LIMIT" \
    --output="$LOG_DIR/slurm_%j.out" \
    --error="$LOG_DIR/slurm_%j.err" \
    --export=ALL \
    "$SELF_PATH"
  exit 0
fi

if [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" != "$SOURCE_COMMIT" ]]; then
  echo "v15 equivalence source HEAD drifted after submission" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite v15 equivalence output: $OUTPUT_DIR" >&2
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
export NUMEXPR_NUM_THREADS=1

LIVE_LOG="$LOG_DIR/shimmer_db_runtime_v15_equivalence_${SLURM_JOB_ID}.log"
echo "event=start job=$SLURM_JOB_ID commit=$SOURCE_COMMIT time=$(date -Is)" | tee -a "$LIVE_LOG"
python -c 'import torch; print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("device", torch.cuda.get_device_name(0))' | tee -a "$LIVE_LOG"

python "$AUDIT_SCRIPT" \
  --panel-contract "$PANEL_CONTRACT" \
  --panel-contract-sha256 "$PANEL_CONTRACT_SHA256" \
  --target-label-contract "$TARGET_LABEL_CONTRACT" \
  --target-label-contract-sha256 "$TARGET_LABEL_CONTRACT_SHA256" \
  --candidate-seal "$CANDIDATE_SEAL" \
  --candidate-seal-sha256 "$CANDIDATE_SEAL_SHA256" \
  --predictor-checkpoint "$PREDICTOR_CHECKPOINT" \
  --predictor-checkpoint-sha256 "$PREDICTOR_CHECKPOINT_SHA256" \
  --exact-python "$EXACT_PYTHON" \
  --avqi-code-root "$AVQI_CODE_ROOT" \
  --avqi-code-tree-sha256 "$AVQI_CODE_TREE_SHA256" \
  --worker-script "$WORKER_SCRIPT" \
  --output-dir "$OUTPUT_DIR" \
  --source-commit "$SOURCE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID" \
  --device cuda \
  --warm-repeats "$WARM_REPEATS" \
  --highpass-mode "$HIGHPASS_MODE" \
  2>&1 | tee -a "$LIVE_LOG"

echo "event=complete job=$SLURM_JOB_ID time=$(date -Is)" | tee -a "$LIVE_LOG"
