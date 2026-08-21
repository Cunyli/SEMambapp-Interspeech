#!/bin/bash
# Run the authorization-bound Route C Shimmer fresh-panel pilot.
# S3_500 is frozen inference only; this script never trains the generator.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
PILOT_SCRIPT="$SOURCE_ROOT/scripts/evaluate_avqi_shimmer_fresh_panel.py"

RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/avqi_route_c_shimmer_v6_fresh_panel_20260821_01}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
PARTITION="${PARTITION:-gpu-v100-32g}"
GPU_TYPE="${GPU_TYPE:-v100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-48G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"

CONSENSUS_REPORT="${CONSENSUS_REPORT:-$ROOT_DIR/runs/avqi_route_c_shimmer_v6_multiseed_20260821_01/outputs/multiseed_consensus.json}"
CONSENSUS_RECEIPT="${CONSENSUS_RECEIPT:-$ROOT_DIR/runs/avqi_route_c_shimmer_v6_multiseed_20260821_01/outputs/completion_receipt.json}"
SCREEN_REPORT="${SCREEN_REPORT:-$ROOT_DIR/runs/avqi_route_c_shimmer_v6_screen_20260821_01/outputs/diagnostic_report.json}"
SCREEN_RECEIPT="${SCREEN_RECEIPT:-$ROOT_DIR/runs/avqi_route_c_shimmer_v6_screen_20260821_01/outputs/completion_receipt.json}"
PREDICTOR_CHECKPOINT="${PREDICTOR_CHECKPOINT:-$ROOT_DIR/checkpoints/avqi_route_c_shimmer_v6_screen_20260821_01/direct_direct_praat_hard_shimmer_pulse_path_v6_estimator.pt}"
GENERATOR_CONFIG="${GENERATOR_CONFIG:-$ROOT_DIR/runs/tau_s1_sv_threshold_ablation_20260719_01/configs/s_fidelity_m3_stage0500.yaml}"
GENERATOR_CHECKPOINT="${GENERATOR_CHECKPOINT:-$ROOT_DIR/checkpoints/S3_500/ln_g_00000500.pth}"
TAU_MANIFEST="${TAU_MANIFEST:-/scratch/work/lil14/use_simulation_pipeline/outputs/organized/csv/sampling/tau_clean_avqi_sampling_manifest.csv}"
DATA_POOL="${DATA_POOL:-/scratch/elec/t412-speechcom/Triton - Symptonic/lijie/gap_webdataset_active}"
FIXED_RECIPES="${FIXED_RECIPES:-$DATA_POOL/splits/hybrid_unise_v1_stream_80_10_10/test/fixed_recipes.jsonl}"
SIMULATION_ROOT="${SIMULATION_ROOT:-/scratch/work/lil14/USE_simulation}"
SIMULATION_CONFIG="${SIMULATION_CONFIG:-$SIMULATION_ROOT/configs/phone_room_22050.yaml}"
AVQI_CODE_ROOT="${AVQI_CODE_ROOT:-/scratch/work/lil14/avqi}"
EXACT_PYTHON="${EXACT_PYTHON:-/scratch/work/lil14/.conda_envs/avqi/bin/python}"

CONSENSUS_REPORT_SHA256="${CONSENSUS_REPORT_SHA256:-2a747f72a5af871ca5272dbda64fdf6fb5bdcef999145edfd83a47ac1a6a319a}"
CONSENSUS_RECEIPT_SHA256="${CONSENSUS_RECEIPT_SHA256:-54242fd0ba29ef77f8d85d7dd75bfd42a8f54a2f49c8a6ebc512c424e91230db}"
SCREEN_REPORT_SHA256="${SCREEN_REPORT_SHA256:-f242611ddf9b0245c93f326505a5298807dc204df99fa2157c8aa7bf8b934dc4}"
SCREEN_RECEIPT_SHA256="${SCREEN_RECEIPT_SHA256:-afcc35d4b61e41980c6c5110198245b586ccd7b128d1c7fe184f66dfe599db45}"
PREDICTOR_CHECKPOINT_SHA256="${PREDICTOR_CHECKPOINT_SHA256:-40b819946abdcb8a4b643fe4238d1bb4d31168a3eb2a6d6c786a61752da629bc}"
GENERATOR_CONFIG_SHA256="${GENERATOR_CONFIG_SHA256:-5c3f75ecd2b2a9fa9c938509e9ac8917bb19b22fdbdbf07b275c868685360407}"
GENERATOR_CHECKPOINT_SHA256="${GENERATOR_CHECKPOINT_SHA256:-d1ef31ec180b2378fce5a36b5a29ae5a601ed2fa1a26b1b18a81de7941bc6dd9}"
TAU_MANIFEST_SHA256="${TAU_MANIFEST_SHA256:-ea227a724ced6436b9aa7c75d4b1ca3d78bc28e157baa0bd73d662d28d2549bf}"
FIXED_RECIPES_SHA256="${FIXED_RECIPES_SHA256:-9f9654dd4e078cb111ee2fae0b039893b6ae61094f35fb328b305250555cc8c6}"
SIMULATION_CONFIG_SHA256="${SIMULATION_CONFIG_SHA256:-0e665b5f3d97ad617cd1dde22a84b1ec5a8089e31b7657c7cb9989363115e276}"
SIMULATION_SOURCE_SHA256="${SIMULATION_SOURCE_SHA256:-7f74a5727122bf3f8a6dbee297d9f3dd10165cba3bf2312bf2bd8704abc273bb}"
AVQI_CODE_TREE_SHA256="${AVQI_CODE_TREE_SHA256:-46987b3c447cb579aab4d34e87655938e4aa64e1b28c0e2348c4ea3e48f107f2}"
SEED="${SEED:-20260821}"

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing to run from a dirty source tree: $SOURCE_ROOT" >&2
  exit 2
fi
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"

for path in "$PILOT_SCRIPT" "$CONSENSUS_REPORT" "$CONSENSUS_RECEIPT" \
  "$SCREEN_REPORT" "$SCREEN_RECEIPT" "$PREDICTOR_CHECKPOINT" \
  "$GENERATOR_CONFIG" "$GENERATOR_CHECKPOINT" "$TAU_MANIFEST" \
  "$FIXED_RECIPES" "$SIMULATION_CONFIG" \
  "$SIMULATION_ROOT/simulate_degradation.py" "$EXACT_PYTHON"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required Shimmer pilot source: $path" >&2
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
verify_sha256 "$CONSENSUS_RECEIPT" "$CONSENSUS_RECEIPT_SHA256" "Route C consensus receipt"
verify_sha256 "$SCREEN_REPORT" "$SCREEN_REPORT_SHA256" "Route C screen report"
verify_sha256 "$SCREEN_RECEIPT" "$SCREEN_RECEIPT_SHA256" "Route C screen receipt"
verify_sha256 "$PREDICTOR_CHECKPOINT" "$PREDICTOR_CHECKPOINT_SHA256" "Route C checkpoint"
verify_sha256 "$GENERATOR_CONFIG" "$GENERATOR_CONFIG_SHA256" "S3_500 config"
verify_sha256 "$GENERATOR_CHECKPOINT" "$GENERATOR_CHECKPOINT_SHA256" "S3_500 checkpoint"
verify_sha256 "$TAU_MANIFEST" "$TAU_MANIFEST_SHA256" "TAU sampling manifest"
verify_sha256 "$FIXED_RECIPES" "$FIXED_RECIPES_SHA256" "fixed test recipes"
verify_sha256 "$SIMULATION_CONFIG" "$SIMULATION_CONFIG_SHA256" "simulation config"
verify_sha256 "$SIMULATION_ROOT/simulate_degradation.py" "$SIMULATION_SOURCE_SHA256" "simulation source"

if [[ "$(jq -er '.promotion.decision' "$CONSENSUS_REPORT")" != "GO_BOUNDED_ROUTE_C_WAVEFORM_PILOT" ]]; then
  echo "Route C consensus does not authorize a bounded waveform pilot" >&2
  exit 2
fi
if [[ "$(jq -cer '.promotion.components' "$CONSENSUS_REPORT")" != '["hnr","shimmer_percent","tilt"]' ]]; then
  echo "Route C consensus components differ from the frozen pilot" >&2
  exit 2
fi
if [[ "$(jq -er '.decision' "$CONSENSUS_RECEIPT")" != "GO_BOUNDED_ROUTE_C_WAVEFORM_PILOT" ]]; then
  echo "Route C consensus receipt differs from its report" >&2
  exit 2
fi

export ROOT_DIR SOURCE_ROOT PILOT_SCRIPT RUN_ROOT LOG_DIR OUTPUT_DIR
export PARTITION GPU_TYPE CPUS_PER_TASK MEMORY TIME_LIMIT
export SOFTWARE_STACK_MODULE COMPILER_MODULE SOURCE_COMMIT
export CONSENSUS_REPORT CONSENSUS_REPORT_SHA256 CONSENSUS_RECEIPT
export CONSENSUS_RECEIPT_SHA256 SCREEN_REPORT SCREEN_REPORT_SHA256
export SCREEN_RECEIPT SCREEN_RECEIPT_SHA256 PREDICTOR_CHECKPOINT
export PREDICTOR_CHECKPOINT_SHA256 GENERATOR_CONFIG GENERATOR_CONFIG_SHA256
export GENERATOR_CHECKPOINT GENERATOR_CHECKPOINT_SHA256 TAU_MANIFEST
export TAU_MANIFEST_SHA256 DATA_POOL FIXED_RECIPES FIXED_RECIPES_SHA256
export SIMULATION_ROOT SIMULATION_CONFIG SIMULATION_CONFIG_SHA256
export SIMULATION_SOURCE_SHA256 AVQI_CODE_ROOT AVQI_CODE_TREE_SHA256
export EXACT_PYTHON SEED

mkdir -p "$LOG_DIR"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite Shimmer pilot output: $OUTPUT_DIR" >&2
    exit 2
  fi
  sbatch \
    --parsable \
    --job-name=avqi-c-v6-shim-fp \
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
  echo "Shimmer pilot source HEAD drifted after submission" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite Shimmer pilot output: $OUTPUT_DIR" >&2
  exit 2
fi

cd "$ROOT_DIR"
module load "$SOFTWARE_STACK_MODULE"
module load "$COMPILER_MODULE"
eval "$(conda shell.bash hook)"
conda activate semambapp
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export PYTHONPATH="$SOURCE_ROOT:$SIMULATION_ROOT:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LIVE_LOG="$LOG_DIR/avqi_shimmer_fresh_panel_${SLURM_JOB_ID}.log"
echo "event=start job=$SLURM_JOB_ID commit=$SOURCE_COMMIT time=$(date -Is)" | tee -a "$LIVE_LOG"
python -c 'import os, torch; print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("device", torch.cuda.get_device_name(0)); print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))' | tee -a "$LIVE_LOG"

python "$PILOT_SCRIPT" \
  --authorization-consensus "$CONSENSUS_REPORT" \
  --authorization-consensus-sha256 "$CONSENSUS_REPORT_SHA256" \
  --authorization-consensus-receipt "$CONSENSUS_RECEIPT" \
  --authorization-consensus-receipt-sha256 "$CONSENSUS_RECEIPT_SHA256" \
  --screen-report "$SCREEN_REPORT" \
  --screen-report-sha256 "$SCREEN_REPORT_SHA256" \
  --screen-completion-receipt "$SCREEN_RECEIPT" \
  --screen-completion-receipt-sha256 "$SCREEN_RECEIPT_SHA256" \
  --predictor-checkpoint "$PREDICTOR_CHECKPOINT" \
  --predictor-checkpoint-sha256 "$PREDICTOR_CHECKPOINT_SHA256" \
  --generator-config "$GENERATOR_CONFIG" \
  --generator-config-sha256 "$GENERATOR_CONFIG_SHA256" \
  --generator-checkpoint "$GENERATOR_CHECKPOINT" \
  --generator-checkpoint-sha256 "$GENERATOR_CHECKPOINT_SHA256" \
  --tau-manifest "$TAU_MANIFEST" \
  --tau-manifest-sha256 "$TAU_MANIFEST_SHA256" \
  --fixed-recipes "$FIXED_RECIPES" \
  --fixed-recipes-sha256 "$FIXED_RECIPES_SHA256" \
  --simulation-root "$SIMULATION_ROOT" \
  --simulation-config "$SIMULATION_CONFIG" \
  --simulation-config-sha256 "$SIMULATION_CONFIG_SHA256" \
  --simulation-source-sha256 "$SIMULATION_SOURCE_SHA256" \
  --exact-python "$EXACT_PYTHON" \
  --avqi-code-root "$AVQI_CODE_ROOT" \
  --avqi-code-tree-sha256 "$AVQI_CODE_TREE_SHA256" \
  --output-dir "$OUTPUT_DIR" \
  --source-commit "$SOURCE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID" \
  --device cuda \
  --seed "$SEED" \
  2>&1 | tee -a "$LIVE_LOG"

echo "event=complete job=$SLURM_JOB_ID time=$(date -Is)" | tee -a "$LIVE_LOG"
