#!/bin/bash
# Run one authorization-bound Route C waveform pilot. No generator is trained.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
PILOT_SCRIPT="$SOURCE_ROOT/scripts/evaluate_direct_avqi_waveform_optimization.py"

RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/avqi_component_direct_c_v5_waveform_pilot_offset4_20260817_01}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
PARTITION="${PARTITION:-gpu-v100-32g}"
GPU_TYPE="${GPU_TYPE:-v100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-48G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"

CONSENSUS_REPORT="${CONSENSUS_REPORT:-$ROOT_DIR/runs/avqi_component_direct_c_v5_multiseed_20260817_01/outputs/multiseed_consensus.json}"
SCREEN_REPORT="${SCREEN_REPORT:-$ROOT_DIR/runs/avqi_component_direct_c_v5_screen_20260817_01/outputs/diagnostic_report.json}"
SCREEN_COMPLETION_RECEIPT="${SCREEN_COMPLETION_RECEIPT:-$ROOT_DIR/runs/avqi_component_direct_c_v5_screen_20260817_01/outputs/completion_receipt.json}"
PREDICTOR_CHECKPOINT="${PREDICTOR_CHECKPOINT:-$ROOT_DIR/checkpoints/avqi_component_direct_c_v5_screen_20260817_01/direct_direct_praat_hard_v2_estimator.pt}"
EXTERNAL_EXACT_CSV="${EXTERNAL_EXACT_CSV:-$ROOT_DIR/runs/tau_pathology_three_tracks_20260810_01/outputs/intensity_eval/exact_components_all.csv}"
AVQI_CODE_ROOT="${AVQI_CODE_ROOT:-/scratch/work/lil14/avqi}"
EXACT_PYTHON="${EXACT_PYTHON:-/scratch/work/lil14/.conda_envs/avqi/bin/python}"
CONSENSUS_REPORT_SHA256="${CONSENSUS_REPORT_SHA256:-0a2b29780a7db2b133e2e781dd78be047e0e92df309b89841fe2d00297678971}"
SCREEN_REPORT_SHA256="${SCREEN_REPORT_SHA256:-95851e82e041bc54e1c69bcd47a53122839fe67dfb5dbada0e83fce21e3c1055}"
SCREEN_COMPLETION_RECEIPT_SHA256="${SCREEN_COMPLETION_RECEIPT_SHA256:-c2a9f9d7fdb6df89b12731410473836d691e8d6918ec877f06ae6cf32d030e48}"
PREDICTOR_CHECKPOINT_SHA256="${PREDICTOR_CHECKPOINT_SHA256:-07b69e6722db46467626bd68ffaceb32844908c1a8378ee7e3bdc327fdc81aef}"
EXTERNAL_EXACT_CSV_SHA256="${EXTERNAL_EXACT_CSV_SHA256:-1e401d2d3343d5d5e8dc38245d14a2e4f9fbb568b11a26269e4ce0aca30c249a}"
AVQI_CODE_TREE_SHA256="${AVQI_CODE_TREE_SHA256:-46987b3c447cb579aab4d34e87655938e4aa64e1b28c0e2348c4ea3e48f107f2}"

SEED="${SEED:-20260817}"
SPEAKERS_PER_SEVERITY="${SPEAKERS_PER_SEVERITY:-3}"
SPEAKER_OFFSET="${SPEAKER_OFFSET:-4}"
EXPECTED_CASES="${EXPECTED_CASES:-12}"
STEPS="${STEPS:-20}"
LEARNING_RATE_SCALE="${LEARNING_RATE_SCALE:-0.0002}"
FIDELITY_WEIGHT="${FIDELITY_WEIGHT:-0.05}"
RESIDUAL_CEILING_DB="${RESIDUAL_CEILING_DB:--30.0}"

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing to run from a dirty source tree: $SOURCE_ROOT" >&2
  exit 2
fi
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"

for path in "$PILOT_SCRIPT" "$CONSENSUS_REPORT" "$SCREEN_REPORT" \
  "$SCREEN_COMPLETION_RECEIPT" "$PREDICTOR_CHECKPOINT" \
  "$EXTERNAL_EXACT_CSV" "$EXACT_PYTHON"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required Route C pilot source: $path" >&2
    exit 2
  fi
done
if [[ ! -d "$AVQI_CODE_ROOT" ]]; then
  echo "Missing exact AVQI code tree: $AVQI_CODE_ROOT" >&2
  exit 2
fi

verify_sha256() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "$label hash drift: $actual != $expected" >&2
    exit 2
  fi
}

verify_sha256 "$CONSENSUS_REPORT" "$CONSENSUS_REPORT_SHA256" "Route C consensus"
verify_sha256 "$SCREEN_REPORT" "$SCREEN_REPORT_SHA256" "Route C screen report"
verify_sha256 "$SCREEN_COMPLETION_RECEIPT" "$SCREEN_COMPLETION_RECEIPT_SHA256" "Route C screen receipt"
verify_sha256 "$PREDICTOR_CHECKPOINT" "$PREDICTOR_CHECKPOINT_SHA256" "Route C checkpoint"
verify_sha256 "$EXTERNAL_EXACT_CSV" "$EXTERNAL_EXACT_CSV_SHA256" "external exact CSV"

if [[ "$(jq -er '.promotion.decision' "$CONSENSUS_REPORT")" != "GO_BOUNDED_ROUTE_C_WAVEFORM_PILOT" ]]; then
  echo "Route C consensus does not authorize the bounded waveform pilot" >&2
  exit 2
fi
if [[ "$(jq -cer '.promotion.components' "$CONSENSUS_REPORT")" != '["hnr","tilt"]' ]]; then
  echo "Route C consensus components differ from the frozen pilot" >&2
  exit 2
fi

export ROOT_DIR SOURCE_ROOT PILOT_SCRIPT RUN_ROOT LOG_DIR OUTPUT_DIR
export PARTITION GPU_TYPE CPUS_PER_TASK MEMORY TIME_LIMIT
export SOFTWARE_STACK_MODULE COMPILER_MODULE SOURCE_COMMIT
export CONSENSUS_REPORT CONSENSUS_REPORT_SHA256 SCREEN_REPORT
export SCREEN_REPORT_SHA256 SCREEN_COMPLETION_RECEIPT
export SCREEN_COMPLETION_RECEIPT_SHA256 PREDICTOR_CHECKPOINT
export PREDICTOR_CHECKPOINT_SHA256 EXTERNAL_EXACT_CSV
export EXTERNAL_EXACT_CSV_SHA256 AVQI_CODE_ROOT AVQI_CODE_TREE_SHA256
export EXACT_PYTHON SEED SPEAKERS_PER_SEVERITY SPEAKER_OFFSET EXPECTED_CASES
export STEPS LEARNING_RATE_SCALE FIDELITY_WEIGHT RESIDUAL_CEILING_DB

mkdir -p "$LOG_DIR"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite pilot output: $OUTPUT_DIR" >&2
    exit 2
  fi
  sbatch \
    --parsable \
    --job-name=avqi-v5-cpilot \
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
  echo "Route C pilot source HEAD drifted after submission" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite pilot output: $OUTPUT_DIR" >&2
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LIVE_LOG="$LOG_DIR/avqi_direct_c_waveform_pilot_${SLURM_JOB_ID}.log"
echo "event=start job=$SLURM_JOB_ID commit=$SOURCE_COMMIT time=$(date -Is)" | tee -a "$LIVE_LOG"
python -c 'import os, torch; print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("device", torch.cuda.get_device_name(0)); print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))' | tee -a "$LIVE_LOG"

python "$PILOT_SCRIPT" \
  --external-exact-csv "$EXTERNAL_EXACT_CSV" \
  --external-exact-csv-sha256 "$EXTERNAL_EXACT_CSV_SHA256" \
  --predictor-checkpoint "$PREDICTOR_CHECKPOINT" \
  --predictor-checkpoint-sha256 "$PREDICTOR_CHECKPOINT_SHA256" \
  --authorization-consensus "$CONSENSUS_REPORT" \
  --authorization-consensus-sha256 "$CONSENSUS_REPORT_SHA256" \
  --screen-report "$SCREEN_REPORT" \
  --screen-report-sha256 "$SCREEN_REPORT_SHA256" \
  --screen-completion-receipt "$SCREEN_COMPLETION_RECEIPT" \
  --screen-completion-receipt-sha256 "$SCREEN_COMPLETION_RECEIPT_SHA256" \
  --avqi-code-root "$AVQI_CODE_ROOT" \
  --avqi-code-tree-sha256 "$AVQI_CODE_TREE_SHA256" \
  --exact-python "$EXACT_PYTHON" \
  --output-dir "$OUTPUT_DIR" \
  --source-commit "$SOURCE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID" \
  --device cuda \
  --seed "$SEED" \
  --speakers-per-severity "$SPEAKERS_PER_SEVERITY" \
  --speaker-offset "$SPEAKER_OFFSET" \
  --expected-cases "$EXPECTED_CASES" \
  --steps "$STEPS" \
  --learning-rate-scale "$LEARNING_RATE_SCALE" \
  --fidelity-weight "$FIDELITY_WEIGHT" \
  --residual-ceiling-db "$RESIDUAL_CEILING_DB" \
  2>&1 | tee -a "$LIVE_LOG"

echo "event=complete job=$SLURM_JOB_ID time=$(date -Is)" | tee -a "$LIVE_LOG"
