#!/bin/bash
# Run the authorization-bound CPPS-only fresh waveform pilot. No generator is trained.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
PILOT_SCRIPT="$SOURCE_ROOT/scripts/evaluate_direct_avqi_waveform_optimization.py"

RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/avqi_route_c_cpps_v12_waveform_pilot_20260823_28}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
PARTITION="${PARTITION:-gpu-v100-32g}"
GPU_TYPE="${GPU_TYPE:-v100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-48G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"

CONSENSUS_REPORT="${CONSENSUS_REPORT:-$ROOT_DIR/runs/avqi_route_c_cpps_v12_multiseed_20260823_25/outputs/multiseed_consensus.json}"
SCREEN_REPORT="${SCREEN_REPORT:-$ROOT_DIR/runs/avqi_route_c_cpps_v12_screen_20260823_20/outputs/diagnostic_report.json}"
SCREEN_COMPLETION_RECEIPT="${SCREEN_COMPLETION_RECEIPT:-$ROOT_DIR/runs/avqi_route_c_cpps_v12_screen_20260823_20/outputs/completion_receipt.json}"
PREDICTOR_CHECKPOINT="${PREDICTOR_CHECKPOINT:-$ROOT_DIR/checkpoints/avqi_route_c_cpps_v12_screen_20260823_20/direct_direct_praat_hard_cpps_view_input_v12_estimator.pt}"
EXTERNAL_EXACT_CSV="${EXTERNAL_EXACT_CSV:-/scratch/work/lil14/SEMambapp-Interspeech/runs/tau_pathology_three_tracks_20260810_01/outputs/intensity_eval/exact_components_all.csv}"
AVQI_CODE_ROOT="${AVQI_CODE_ROOT:-/scratch/work/lil14/avqi}"
EXACT_PYTHON="${EXACT_PYTHON:-/scratch/work/lil14/.conda_envs/avqi/bin/python}"

CONSENSUS_REPORT_SHA256="${CONSENSUS_REPORT_SHA256:-4b6712c0cef7a4f5e405d18abcd98664f0db9c93ec9c3d335f91f88bb2b9c1ea}"
SCREEN_REPORT_SHA256="${SCREEN_REPORT_SHA256:-8d2ee5e7cd094cfc8d65351484c773821756568641fc2759ac5e1ccb06466b0b}"
SCREEN_COMPLETION_RECEIPT_SHA256="${SCREEN_COMPLETION_RECEIPT_SHA256:-c0ade9c1662f371b8700be5e87e5175b26cb416aab26b1e671440e7b3a6b7762}"
PREDICTOR_CHECKPOINT_SHA256="${PREDICTOR_CHECKPOINT_SHA256:-f893783cccf09f3b4fe707e881afdfc889724861a6b86eb0600fb5400534c543}"
EXTERNAL_EXACT_CSV_SHA256="${EXTERNAL_EXACT_CSV_SHA256:-1e401d2d3343d5d5e8dc38245d14a2e4f9fbb568b11a26269e4ce0aca30c249a}"
AVQI_CODE_TREE_SHA256="${AVQI_CODE_TREE_SHA256:-46987b3c447cb579aab4d34e87655938e4aa64e1b28c0e2348c4ea3e48f107f2}"
AVQI_REPO_COMMIT="${AVQI_REPO_COMMIT:-861730e8e44aed190a9a2903d78596b0d480f4d9}"

PILOT_PROFILE="cpps_view_input_v12"
PANEL_SELECTION_SALT="${PANEL_SELECTION_SALT:-cpps-v12-fresh-speaker-panel-4b6712c0-20260823}"
SEED="${SEED:-20260823}"
SPEAKERS_PER_SEVERITY="${SPEAKERS_PER_SEVERITY:-3}"
SPEAKER_OFFSET=0
EXPECTED_CASES="${EXPECTED_CASES:-12}"
STEPS="${STEPS:-30}"
LEARNING_RATE_SCALE="${LEARNING_RATE_SCALE:-0.001}"
FIDELITY_WEIGHT="${FIDELITY_WEIGHT:-0.05}"
RESIDUAL_CEILING_DB="${RESIDUAL_CEILING_DB:--30.0}"

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing to run from a dirty source tree: $SOURCE_ROOT" >&2
  exit 2
fi
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"
if [[ "$(git -C "$AVQI_CODE_ROOT" rev-parse HEAD)" != "$AVQI_REPO_COMMIT" ]]; then
  echo "Exact AVQI repository commit drifted" >&2
  exit 2
fi

for path in "$PILOT_SCRIPT" "$CONSENSUS_REPORT" "$SCREEN_REPORT" \
  "$SCREEN_COMPLETION_RECEIPT" "$PREDICTOR_CHECKPOINT" \
  "$EXTERNAL_EXACT_CSV" "$EXACT_PYTHON"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required CPPS pilot source: $path" >&2
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

verify_sha256 "$CONSENSUS_REPORT" "$CONSENSUS_REPORT_SHA256" "CPPS consensus"
verify_sha256 "$SCREEN_REPORT" "$SCREEN_REPORT_SHA256" "CPPS screen report"
verify_sha256 "$SCREEN_COMPLETION_RECEIPT" "$SCREEN_COMPLETION_RECEIPT_SHA256" "CPPS screen receipt"
verify_sha256 "$PREDICTOR_CHECKPOINT" "$PREDICTOR_CHECKPOINT_SHA256" "CPPS checkpoint"
verify_sha256 "$EXTERNAL_EXACT_CSV" "$EXTERNAL_EXACT_CSV_SHA256" "external exact CSV"

if [[ "$(jq -er '.promotion.decision' "$CONSENSUS_REPORT")" != "GO_BOUNDED_ROUTE_C_WAVEFORM_PILOT" ]]; then
  echo "Route C consensus does not authorize a bounded waveform pilot" >&2
  exit 2
fi
if [[ "$(jq -er '.routes.direct_differentiable_estimator.selected_form' "$CONSENSUS_REPORT")" != "direct_praat_hard_cpps_view_input_v12" ]]; then
  echo "Route C consensus did not lock CPPS v12" >&2
  exit 2
fi
if [[ "$(jq -er '.routes.direct_differentiable_estimator.component_pass_counts.cpps' "$CONSENSUS_REPORT")" != "3" ]]; then
  echo "CPPS did not pass all three locked seeds" >&2
  exit 2
fi
if ! jq -e '.promotion.components | index("cpps") != null' "$CONSENSUS_REPORT" >/dev/null; then
  echo "CPPS is absent from the bounded-pilot authorization" >&2
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
export AVQI_REPO_COMMIT EXACT_PYTHON PILOT_PROFILE PANEL_SELECTION_SALT
export SEED SPEAKERS_PER_SEVERITY SPEAKER_OFFSET EXPECTED_CASES
export STEPS LEARNING_RATE_SCALE FIDELITY_WEIGHT RESIDUAL_CEILING_DB

mkdir -p "$LOG_DIR"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite CPPS pilot output: $OUTPUT_DIR" >&2
    exit 2
  fi
  sbatch \
    --parsable \
    --job-name=avqi-cpps-v12-pilot \
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
  echo "CPPS pilot source HEAD drifted after submission" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite CPPS pilot output: $OUTPUT_DIR" >&2
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

LIVE_LOG="$LOG_DIR/avqi_cpps_v12_waveform_pilot_${SLURM_JOB_ID}.log"
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
  --pilot-profile "$PILOT_PROFILE" \
  --panel-selection-salt "$PANEL_SELECTION_SALT" \
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
