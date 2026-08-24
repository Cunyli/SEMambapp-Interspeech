#!/bin/bash
# Run the promotion-bound LTAS-slope fresh waveform pilot. No generator training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
SOURCE_ROOT="${SOURCE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PILOT_SCRIPT="$SOURCE_ROOT/scripts/evaluate_avqi_shimmer_fresh_panel.py"

RUN_ROOT="${RUN_ROOT:-$SOURCE_ROOT/runs/avqi_route_c_ltas_slope_fresh_panel_v1_20260824_01}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
PARTITION="${PARTITION:-gpu-v100-32g}"
GPU_TYPE="${GPU_TYPE:-v100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-48G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"

PROMOTION_ROOT="${PROMOTION_ROOT:-$SOURCE_ROOT/runs/avqi_route_c_ltas_slope_promotion_v1_20260824_01/outputs}"
PROMOTION_REPORT="${PROMOTION_REPORT:-$PROMOTION_ROOT/diagnostic_report.json}"
PROMOTION_REPORT_SHA256="${PROMOTION_REPORT_SHA256:-077ed79a7d21ec31685a160e9ea16879c36c38d844c0a0e444ec589d8a636385}"
PROMOTION_RECEIPT="${PROMOTION_RECEIPT:-$PROMOTION_ROOT/completion_receipt.json}"
PROMOTION_RECEIPT_SHA256="${PROMOTION_RECEIPT_SHA256:-d812abd65498760f1c556e67747f081b3214625d4a0fc1059be96bb39e1bad98}"
SVD_PANEL_SEAL="${SVD_PANEL_SEAL:-/scratch/work/lil14/SEMambapp-Interspeech-avqi-shimmer-ltas-v8/runs/avqi_route_c_ltas_slope_svd_authority_v10_20260823_01/outputs/panel_seal.json}"
SVD_PANEL_SEAL_SHA256="${SVD_PANEL_SEAL_SHA256:-02d27c3f05be3a3b1196e2178fa6fbb6a14dafefccd068094ded58e56f700e57}"

PREDICTOR_CHECKPOINT="${PREDICTOR_CHECKPOINT:-/scratch/work/lil14/SEMambapp-Interspeech/checkpoints/avqi_route_c_shimmer_v6_screen_20260821_01/direct_direct_praat_hard_shimmer_pulse_path_v6_estimator.pt}"
PREDICTOR_CHECKPOINT_SHA256="${PREDICTOR_CHECKPOINT_SHA256:-40b819946abdcb8a4b643fe4238d1bb4d31168a3eb2a6d6c786a61752da629bc}"
GENERATOR_CONFIG="${GENERATOR_CONFIG:-/scratch/work/lil14/SEMambapp-Interspeech/runs/tau_s1_sv_threshold_ablation_20260719_01/configs/s_fidelity_m3_stage0500.yaml}"
GENERATOR_CONFIG_SHA256="${GENERATOR_CONFIG_SHA256:-5c3f75ecd2b2a9fa9c938509e9ac8917bb19b22fdbdbf07b275c868685360407}"
GENERATOR_CHECKPOINT="${GENERATOR_CHECKPOINT:-/scratch/work/lil14/SEMambapp-Interspeech/checkpoints/S3_500/ln_g_00000500.pth}"
GENERATOR_CHECKPOINT_SHA256="${GENERATOR_CHECKPOINT_SHA256:-d1ef31ec180b2378fce5a36b5a29ae5a601ed2fa1a26b1b18a81de7941bc6dd9}"

DATA_POOL="${DATA_POOL:-/scratch/elec/t412-speechcom/Triton - Symptonic/lijie/gap_webdataset_active}"
FIXED_RECIPES="${FIXED_RECIPES:-$DATA_POOL/splits/hybrid_unise_v1_stream_80_10_10/test/fixed_recipes.jsonl}"
FIXED_RECIPES_SHA256="${FIXED_RECIPES_SHA256:-9f9654dd4e078cb111ee2fae0b039893b6ae61094f35fb328b305250555cc8c6}"
SIMULATION_ROOT="${SIMULATION_ROOT:-/scratch/work/lil14/USE_simulation}"
SIMULATION_CONFIG="${SIMULATION_CONFIG:-$SIMULATION_ROOT/configs/phone_room_22050.yaml}"
SIMULATION_CONFIG_SHA256="${SIMULATION_CONFIG_SHA256:-0e665b5f3d97ad617cd1dde22a84b1ec5a8089e31b7657c7cb9989363115e276}"
SIMULATION_SOURCE_SHA256="${SIMULATION_SOURCE_SHA256:-7f74a5727122bf3f8a6dbee297d9f3dd10165cba3bf2312bf2bd8704abc273bb}"

RUNTIME_PYTHON="${RUNTIME_PYTHON:-/scratch/work/lil14/.conda_envs/semambapp/bin/python}"
EXACT_PYTHON="${EXACT_PYTHON:-/scratch/work/lil14/.conda_envs/avqi/bin/python}"
AVQI_CODE_ROOT="${AVQI_CODE_ROOT:-/scratch/work/lil14/avqi}"
AVQI_CODE_TREE_SHA256="${AVQI_CODE_TREE_SHA256:-46987b3c447cb579aab4d34e87655938e4aa64e1b28c0e2348c4ea3e48f107f2}"
AVQI_REPO_COMMIT="${AVQI_REPO_COMMIT:-861730e8e44aed190a9a2903d78596b0d480f4d9}"
SEED="${SEED:-20260824}"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"
BASE_COMMIT="${BASE_COMMIT:-$SOURCE_COMMIT}"
PILOT_SCRIPT_SHA256="${PILOT_SCRIPT_SHA256:?PILOT_SCRIPT_SHA256 is required}"

verify_hash() {
  local path="$1"
  local expected="$2"
  local label="$3"
  if [[ "$(sha256sum "$path" | awk '{print $1}')" != "$expected" ]]; then
    echo "$label SHA-256 mismatch" >&2
    exit 2
  fi
}

for path in "$PILOT_SCRIPT" "$PROMOTION_REPORT" "$PROMOTION_RECEIPT" \
  "$SVD_PANEL_SEAL" "$PREDICTOR_CHECKPOINT" "$GENERATOR_CONFIG" \
  "$GENERATOR_CHECKPOINT" "$FIXED_RECIPES" "$SIMULATION_CONFIG" \
  "$SIMULATION_ROOT/simulate_degradation.py" "$RUNTIME_PYTHON" \
  "$EXACT_PYTHON"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing LTAS fresh-panel input: $path" >&2
    exit 2
  fi
done
if [[ ! -d "$AVQI_CODE_ROOT" ]]; then
  echo "Missing exact AVQI code tree: $AVQI_CODE_ROOT" >&2
  exit 2
fi

verify_hash "$PILOT_SCRIPT" "$PILOT_SCRIPT_SHA256" "fresh-panel script"
verify_hash "$PROMOTION_REPORT" "$PROMOTION_REPORT_SHA256" "promotion report"
verify_hash "$PROMOTION_RECEIPT" "$PROMOTION_RECEIPT_SHA256" "promotion receipt"
verify_hash "$SVD_PANEL_SEAL" "$SVD_PANEL_SEAL_SHA256" "SVD panel seal"
verify_hash "$PREDICTOR_CHECKPOINT" "$PREDICTOR_CHECKPOINT_SHA256" "LTAS checkpoint"
verify_hash "$GENERATOR_CONFIG" "$GENERATOR_CONFIG_SHA256" "S3_500 config"
verify_hash "$GENERATOR_CHECKPOINT" "$GENERATOR_CHECKPOINT_SHA256" "S3_500 checkpoint"
verify_hash "$FIXED_RECIPES" "$FIXED_RECIPES_SHA256" "fixed test recipes"
verify_hash "$SIMULATION_CONFIG" "$SIMULATION_CONFIG_SHA256" "simulation config"
verify_hash "$SIMULATION_ROOT/simulate_degradation.py" "$SIMULATION_SOURCE_SHA256" "simulation source"

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing dirty LTAS fresh-panel source" >&2
  exit 2
fi
if [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" != "$BASE_COMMIT" ]]; then
  echo "LTAS fresh-panel source commit drifted" >&2
  exit 2
fi
if [[ "$(git -C "$AVQI_CODE_ROOT" rev-parse HEAD)" != "$AVQI_REPO_COMMIT" ]]; then
  echo "Exact AVQI repository commit drifted" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite LTAS fresh-panel output: $OUTPUT_DIR" >&2
  exit 2
fi

export SOURCE_ROOT PILOT_SCRIPT RUN_ROOT LOG_DIR OUTPUT_DIR PARTITION GPU_TYPE
export CPUS_PER_TASK MEMORY TIME_LIMIT SOFTWARE_STACK_MODULE COMPILER_MODULE
export PROMOTION_ROOT PROMOTION_REPORT PROMOTION_REPORT_SHA256
export PROMOTION_RECEIPT PROMOTION_RECEIPT_SHA256 SVD_PANEL_SEAL
export SVD_PANEL_SEAL_SHA256 PREDICTOR_CHECKPOINT PREDICTOR_CHECKPOINT_SHA256
export GENERATOR_CONFIG GENERATOR_CONFIG_SHA256 GENERATOR_CHECKPOINT
export GENERATOR_CHECKPOINT_SHA256 DATA_POOL FIXED_RECIPES FIXED_RECIPES_SHA256
export SIMULATION_ROOT SIMULATION_CONFIG SIMULATION_CONFIG_SHA256
export SIMULATION_SOURCE_SHA256 RUNTIME_PYTHON EXACT_PYTHON AVQI_CODE_ROOT
export AVQI_CODE_TREE_SHA256 AVQI_REPO_COMMIT SEED SOURCE_COMMIT BASE_COMMIT
export PILOT_SCRIPT_SHA256

mkdir -p "$LOG_DIR"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  sbatch \
    --parsable \
    --job-name="avqi-ltas-fp" \
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

cd "$SOURCE_ROOT"
module load "$SOFTWARE_STACK_MODULE"
module load "$COMPILER_MODULE"
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export PYTHONPATH="$SOURCE_ROOT:$SIMULATION_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LIVE_LOG="$LOG_DIR/avqi_ltas_slope_fresh_panel_${SLURM_JOB_ID}.log"
echo "event=start job=$SLURM_JOB_ID commit=$SOURCE_COMMIT time=$(date -Is)" | tee -a "$LIVE_LOG"
nvidia-smi -L | tee -a "$LIVE_LOG"
"$RUNTIME_PYTHON" "$PILOT_SCRIPT" \
  --pilot-profile ltas_slope_authority_v1 \
  --ltas-promotion-report "$PROMOTION_REPORT" \
  --ltas-promotion-report-sha256 "$PROMOTION_REPORT_SHA256" \
  --ltas-promotion-receipt "$PROMOTION_RECEIPT" \
  --ltas-promotion-receipt-sha256 "$PROMOTION_RECEIPT_SHA256" \
  --svd-panel-seal "$SVD_PANEL_SEAL" \
  --svd-panel-seal-sha256 "$SVD_PANEL_SEAL_SHA256" \
  --predictor-checkpoint "$PREDICTOR_CHECKPOINT" \
  --predictor-checkpoint-sha256 "$PREDICTOR_CHECKPOINT_SHA256" \
  --generator-config "$GENERATOR_CONFIG" \
  --generator-config-sha256 "$GENERATOR_CONFIG_SHA256" \
  --generator-checkpoint "$GENERATOR_CHECKPOINT" \
  --generator-checkpoint-sha256 "$GENERATOR_CHECKPOINT_SHA256" \
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
  --seed "$SEED" 2>&1 | tee -a "$LIVE_LOG"
echo "event=complete job=$SLURM_JOB_ID time=$(date -Is)" | tee -a "$LIVE_LOG"
