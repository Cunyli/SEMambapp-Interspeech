#!/bin/bash
# Run the authorization-bound HNR-only fresh waveform pilot. No generator is trained.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
PILOT_SCRIPT="$SOURCE_ROOT/scripts/evaluate_avqi_shimmer_fresh_panel.py"

RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/avqi_route_c_hnr_v7_fresh_panel_20260824_01}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
PARTITION="${PARTITION:-gpu-v100-32g}"
GPU_TYPE="${GPU_TYPE:-v100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-48G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"

CONSENSUS_REPORT="${CONSENSUS_REPORT:-$ROOT_DIR/runs/avqi_route_c_hnr_v7_multiseed_20260824_01/outputs/multiseed_consensus.json}"
CONSENSUS_RECEIPT="${CONSENSUS_RECEIPT:-$ROOT_DIR/runs/avqi_route_c_hnr_v7_multiseed_20260824_01/outputs/completion_receipt.json}"
SCREEN_REPORT="${SCREEN_REPORT:-$ROOT_DIR/runs/avqi_route_c_hnr_v7_screen_20260824_01/outputs/diagnostic_report.json}"
SCREEN_RECEIPT="${SCREEN_RECEIPT:-$ROOT_DIR/runs/avqi_route_c_hnr_v7_screen_20260824_01/outputs/completion_receipt.json}"
PREDICTOR_CHECKPOINT="${PREDICTOR_CHECKPOINT:-$ROOT_DIR/checkpoints/avqi_route_c_hnr_v7_screen_20260824_01/direct_direct_praat_hard_hnr_pitch_path_v7_estimator.pt}"
GENERATOR_CONFIG="${GENERATOR_CONFIG:-/scratch/work/lil14/SEMambapp-Interspeech/runs/tau_s1_sv_threshold_ablation_20260719_01/configs/s_fidelity_m3_stage0500.yaml}"
GENERATOR_CHECKPOINT="${GENERATOR_CHECKPOINT:-/scratch/work/lil14/SEMambapp-Interspeech/checkpoints/S3_500/ln_g_00000500.pth}"
TAU_MANIFEST="${TAU_MANIFEST:-/scratch/work/lil14/use_simulation_pipeline/outputs/organized/csv/sampling/tau_clean_avqi_sampling_manifest.csv}"
DATA_POOL="${DATA_POOL:-/scratch/elec/t412-speechcom/Triton - Symptonic/lijie/gap_webdataset_active}"
FIXED_RECIPES="${FIXED_RECIPES:-$DATA_POOL/splits/hybrid_unise_v1_stream_80_10_10/test/fixed_recipes.jsonl}"
SIMULATION_ROOT="${SIMULATION_ROOT:-/scratch/work/lil14/USE_simulation}"
SIMULATION_CONFIG="${SIMULATION_CONFIG:-$SIMULATION_ROOT/configs/phone_room_22050.yaml}"
AVQI_CODE_ROOT="${AVQI_CODE_ROOT:-/scratch/work/lil14/avqi}"
EXACT_PYTHON="${EXACT_PYTHON:-/scratch/work/lil14/.conda_envs/avqi/bin/python}"

CONSENSUS_REPORT_SHA256="${CONSENSUS_REPORT_SHA256:-e2be665715b1eff893d4db41db2860bbf6b95a049b551c2995255468ae9c4820}"
CONSENSUS_RECEIPT_SHA256="${CONSENSUS_RECEIPT_SHA256:-a1afefcddaa8a3539c88ab6a10c2375b49247e1ef24b5f827875453f49a72867}"
SCREEN_REPORT_SHA256="${SCREEN_REPORT_SHA256:-e4b22b5524438e0cb37e9779225ed77ab01660b218f38a945ccf11f8767d6cd5}"
SCREEN_RECEIPT_SHA256="${SCREEN_RECEIPT_SHA256:-9434330a9ebcb157323d5e717aece7aeddbae91cff8c7016c8b867377e0a5685}"
PREDICTOR_CHECKPOINT_SHA256="${PREDICTOR_CHECKPOINT_SHA256:-93f5b122486bcdc54215244fb894ffa3b34d1993fca32011dbf57650467c412c}"
GENERATOR_CONFIG_SHA256="${GENERATOR_CONFIG_SHA256:-5c3f75ecd2b2a9fa9c938509e9ac8917bb19b22fdbdbf07b275c868685360407}"
GENERATOR_CHECKPOINT_SHA256="${GENERATOR_CHECKPOINT_SHA256:-d1ef31ec180b2378fce5a36b5a29ae5a601ed2fa1a26b1b18a81de7941bc6dd9}"
TAU_MANIFEST_SHA256="${TAU_MANIFEST_SHA256:-ea227a724ced6436b9aa7c75d4b1ca3d78bc28e157baa0bd73d662d28d2549bf}"
FIXED_RECIPES_SHA256="${FIXED_RECIPES_SHA256:-9f9654dd4e078cb111ee2fae0b039893b6ae61094f35fb328b305250555cc8c6}"
SIMULATION_CONFIG_SHA256="${SIMULATION_CONFIG_SHA256:-0e665b5f3d97ad617cd1dde22a84b1ec5a8089e31b7657c7cb9989363115e276}"
SIMULATION_SOURCE_SHA256="${SIMULATION_SOURCE_SHA256:-7f74a5727122bf3f8a6dbee297d9f3dd10165cba3bf2312bf2bd8704abc273bb}"
AVQI_CODE_TREE_SHA256="${AVQI_CODE_TREE_SHA256:-46987b3c447cb579aab4d34e87655938e4aa64e1b28c0e2348c4ea3e48f107f2}"
AVQI_REPO_COMMIT="${AVQI_REPO_COMMIT:-861730e8e44aed190a9a2903d78596b0d480f4d9}"
SEED="${SEED:-20260824}"

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing to run from a dirty HNR source tree: $SOURCE_ROOT" >&2
  exit 2
fi
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"
if [[ "$(git -C "$AVQI_CODE_ROOT" rev-parse HEAD)" != "$AVQI_REPO_COMMIT" ]]; then
  echo "Exact AVQI repository commit drifted" >&2
  exit 2
fi

for path in "$PILOT_SCRIPT" "$CONSENSUS_REPORT" "$CONSENSUS_RECEIPT" \
  "$SCREEN_REPORT" "$SCREEN_RECEIPT" "$PREDICTOR_CHECKPOINT" \
  "$GENERATOR_CONFIG" "$GENERATOR_CHECKPOINT" "$TAU_MANIFEST" \
  "$FIXED_RECIPES" "$SIMULATION_CONFIG" \
  "$SIMULATION_ROOT/simulate_degradation.py" "$EXACT_PYTHON"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required HNR pilot source: $path" >&2
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

verify_sha256 "$CONSENSUS_REPORT" "$CONSENSUS_REPORT_SHA256" "HNR consensus"
verify_sha256 "$CONSENSUS_RECEIPT" "$CONSENSUS_RECEIPT_SHA256" "HNR consensus receipt"
verify_sha256 "$SCREEN_REPORT" "$SCREEN_REPORT_SHA256" "HNR screen report"
verify_sha256 "$SCREEN_RECEIPT" "$SCREEN_RECEIPT_SHA256" "HNR screen receipt"
verify_sha256 "$PREDICTOR_CHECKPOINT" "$PREDICTOR_CHECKPOINT_SHA256" "HNR checkpoint"
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
if [[ "$(jq -er '.routes.direct_differentiable_estimator.selected_form' "$CONSENSUS_REPORT")" != "direct_praat_hard_hnr_pitch_path_v7" ]]; then
  echo "Route C consensus did not lock HNR v7" >&2
  exit 2
fi
if [[ "$(jq -er '.routes.direct_differentiable_estimator.component_pass_counts.hnr' "$CONSENSUS_REPORT")" != "3" ]]; then
  echo "HNR did not pass all three locked seeds" >&2
  exit 2
fi
if ! jq -e '.promotion.components | index("hnr") != null' "$CONSENSUS_REPORT" >/dev/null; then
  echo "HNR is absent from the bounded-pilot authorization" >&2
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
export AVQI_REPO_COMMIT EXACT_PYTHON SEED

mkdir -p "$LOG_DIR"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite HNR pilot output: $OUTPUT_DIR" >&2
    exit 2
  fi
  sbatch \
    --parsable \
    --job-name=avqi-hnr-v7-fp \
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
  echo "HNR pilot source HEAD drifted after submission" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite HNR pilot output: $OUTPUT_DIR" >&2
  exit 2
fi

cd "$SOURCE_ROOT"
module load "$SOFTWARE_STACK_MODULE"
module load "$COMPILER_MODULE"
eval "$(conda shell.bash hook)"
conda activate semambapp
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export PYTHONPATH="$SOURCE_ROOT:$SIMULATION_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LIVE_LOG="$LOG_DIR/avqi_hnr_v7_fresh_panel_${SLURM_JOB_ID}.log"
echo "event=start job=$SLURM_JOB_ID commit=$SOURCE_COMMIT time=$(date -Is)" | tee -a "$LIVE_LOG"
python -c 'import os, torch; print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("device", torch.cuda.get_device_name(0)); print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))' | tee -a "$LIVE_LOG"

python "$PILOT_SCRIPT" \
  --pilot-profile hnr_pitch_path_v7 \
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
