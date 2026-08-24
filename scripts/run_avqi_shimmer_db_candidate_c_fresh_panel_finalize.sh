#!/bin/bash
# Finalize the already-sealed Candidate-C fresh panel from job 19906678.
# No simulation, generator inference, candidate update, or training is run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
FINALIZER_SCRIPT="$SOURCE_ROOT/scripts/finalize_avqi_shimmer_db_candidate_c_fresh_panel.py"

RUN_ROOT="${RUN_ROOT:-$SOURCE_ROOT/runs/avqi_route_c_shimmer_db_candidate_c_fresh_panel_finalize_v14_20260824_01}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
PARTITION="${PARTITION:-gpu-v100-32g}"
GPU_TYPE="${GPU_TYPE:-v100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"

SEALED_RUN_ROOT="${SEALED_RUN_ROOT:-$SOURCE_ROOT/runs/avqi_route_c_shimmer_db_candidate_c_fresh_panel_v14_20260824_01}"
SEALED_OUTPUT_DIR="${SEALED_OUTPUT_DIR:-$SEALED_RUN_ROOT/outputs}"
MECHANISM_ROOT="${MECHANISM_ROOT:-$ROOT_DIR/runs/avqi_route_c_shimmer_current_output_refresh_v13_20260824_01/outputs}"
MECHANISM_REPORT="${MECHANISM_REPORT:-$MECHANISM_ROOT/diagnostic_report.json}"
MECHANISM_RECEIPT="${MECHANISM_RECEIPT:-$MECHANISM_ROOT/completion_receipt.json}"
PREDICTOR_CHECKPOINT="${PREDICTOR_CHECKPOINT:-/scratch/work/lil14/SEMambapp-Interspeech/checkpoints/avqi_route_c_shimmer_v6_screen_20260821_01/direct_direct_praat_hard_shimmer_pulse_path_v6_estimator.pt}"
GENERATOR_CONFIG="${GENERATOR_CONFIG:-/scratch/work/lil14/SEMambapp-Interspeech/runs/tau_s1_sv_threshold_ablation_20260719_01/configs/s_fidelity_m3_stage0500.yaml}"
GENERATOR_CHECKPOINT="${GENERATOR_CHECKPOINT:-/scratch/work/lil14/SEMambapp-Interspeech/checkpoints/S3_500/ln_g_00000500.pth}"
AVQI_CODE_ROOT="${AVQI_CODE_ROOT:-/scratch/work/lil14/avqi}"
EXACT_PYTHON="${EXACT_PYTHON:-/scratch/work/lil14/.conda_envs/avqi/bin/python}"

MECHANISM_REPORT_SHA256="${MECHANISM_REPORT_SHA256:-547e1a3dd106f5a24e218440644ef1e88a9497e6fd3d4f873eb889b7e1c86bb6}"
MECHANISM_RECEIPT_SHA256="${MECHANISM_RECEIPT_SHA256:-9caa69fa3cc967af6a8851c802cbf2c8d1baf52f8e50f131b81e65028b6c2d48}"
PREDICTOR_CHECKPOINT_SHA256="${PREDICTOR_CHECKPOINT_SHA256:-40b819946abdcb8a4b643fe4238d1bb4d31168a3eb2a6d6c786a61752da629bc}"
GENERATOR_CONFIG_SHA256="${GENERATOR_CONFIG_SHA256:-5c3f75ecd2b2a9fa9c938509e9ac8917bb19b22fdbdbf07b275c868685360407}"
GENERATOR_CHECKPOINT_SHA256="${GENERATOR_CHECKPOINT_SHA256:-d1ef31ec180b2378fce5a36b5a29ae5a601ed2fa1a26b1b18a81de7941bc6dd9}"
AVQI_CODE_TREE_SHA256="${AVQI_CODE_TREE_SHA256:-46987b3c447cb579aab4d34e87655938e4aa64e1b28c0e2348c4ea3e48f107f2}"
PANEL_CONTRACT_SHA256="${PANEL_CONTRACT_SHA256:-28d9726b2ecfa97d40cc973d768cb852f6cf15138ad903e85c548e9398ee9af2}"
TARGET_LABEL_CONTRACT_SHA256="${TARGET_LABEL_CONTRACT_SHA256:-4d6a4f43d2a982e8d1862abc5bc722f44908d6221b1ff867064fbc44ab53fdd9}"
CANDIDATE_SEAL_SHA256="${CANDIDATE_SEAL_SHA256:-c7ed5dc5aa36ddcd8a807dc77400ba3c6524ff3dd6f8a8873e3f3d1c1fc8ecd6}"
SEALED_SOURCE_COMMIT="60dd0fe9dc748ebb793937e67aa0e38a7909876f"
SEALED_JOB_ID="19906678"

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing to finalize from dirty source: $SOURCE_ROOT" >&2
  exit 2
fi
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"

for path in "$FINALIZER_SCRIPT" "$MECHANISM_REPORT" "$MECHANISM_RECEIPT" \
  "$PREDICTOR_CHECKPOINT" "$GENERATOR_CONFIG" "$GENERATOR_CHECKPOINT" \
  "$SEALED_OUTPUT_DIR/panel_contract.json" \
  "$SEALED_OUTPUT_DIR/target_label_contract.json" \
  "$SEALED_OUTPUT_DIR/candidate_seal.json" "$EXACT_PYTHON"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing sealed-panel finalizer input: $path" >&2
    exit 2
  fi
done
if [[ ! -d "$AVQI_CODE_ROOT" ]]; then
  echo "Missing exact AVQI code tree: $AVQI_CODE_ROOT" >&2
  exit 2
fi
for name in fresh_panel_results.csv fresh_panel_report.json completion_receipt.json; do
  if [[ -e "$SEALED_OUTPUT_DIR/$name" ]]; then
    echo "Refusing to overwrite finalized artifact: $name" >&2
    exit 2
  fi
done

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

verify_sha256 "$MECHANISM_REPORT" "$MECHANISM_REPORT_SHA256" "Candidate-C v13 report"
verify_sha256 "$MECHANISM_RECEIPT" "$MECHANISM_RECEIPT_SHA256" "Candidate-C v13 receipt"
verify_sha256 "$PREDICTOR_CHECKPOINT" "$PREDICTOR_CHECKPOINT_SHA256" "Shimmer v6 checkpoint"
verify_sha256 "$GENERATOR_CONFIG" "$GENERATOR_CONFIG_SHA256" "S3_500 config"
verify_sha256 "$GENERATOR_CHECKPOINT" "$GENERATOR_CHECKPOINT_SHA256" "S3_500 checkpoint"
verify_sha256 "$SEALED_OUTPUT_DIR/panel_contract.json" "$PANEL_CONTRACT_SHA256" "sealed panel contract"
verify_sha256 "$SEALED_OUTPUT_DIR/target_label_contract.json" "$TARGET_LABEL_CONTRACT_SHA256" "sealed target labels"
verify_sha256 "$SEALED_OUTPUT_DIR/candidate_seal.json" "$CANDIDATE_SEAL_SHA256" "sealed candidates"

export ROOT_DIR SOURCE_ROOT FINALIZER_SCRIPT RUN_ROOT LOG_DIR
export PARTITION GPU_TYPE CPUS_PER_TASK MEMORY TIME_LIMIT
export SOFTWARE_STACK_MODULE COMPILER_MODULE SOURCE_COMMIT
export SEALED_RUN_ROOT SEALED_OUTPUT_DIR SEALED_SOURCE_COMMIT SEALED_JOB_ID
export MECHANISM_REPORT MECHANISM_REPORT_SHA256 MECHANISM_RECEIPT
export MECHANISM_RECEIPT_SHA256 PREDICTOR_CHECKPOINT
export PREDICTOR_CHECKPOINT_SHA256 GENERATOR_CONFIG GENERATOR_CONFIG_SHA256
export GENERATOR_CHECKPOINT GENERATOR_CHECKPOINT_SHA256 AVQI_CODE_ROOT
export AVQI_CODE_TREE_SHA256 EXACT_PYTHON PANEL_CONTRACT_SHA256
export TARGET_LABEL_CONTRACT_SHA256 CANDIDATE_SEAL_SHA256

mkdir -p "$LOG_DIR"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  sbatch \
    --parsable \
    --job-name=avqi-shim-db-finalize \
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
  echo "sealed-panel finalizer source HEAD drifted after submission" >&2
  exit 2
fi

cd "$ROOT_DIR"
module load "$SOFTWARE_STACK_MODULE"
module load "$COMPILER_MODULE"
eval "$(conda shell.bash hook)"
conda activate semambapp
export PYTHONPATH="$SOURCE_ROOT:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

LIVE_LOG="$LOG_DIR/shimmer_db_candidate_c_finalize_${SLURM_JOB_ID}.log"
echo "event=start job=$SLURM_JOB_ID commit=$SOURCE_COMMIT sealed_job=$SEALED_JOB_ID time=$(date -Is)" | tee -a "$LIVE_LOG"
python -c 'import torch; print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("device", torch.cuda.get_device_name(0))' | tee -a "$LIVE_LOG"

python "$FINALIZER_SCRIPT" \
  --mechanism-report "$MECHANISM_REPORT" \
  --mechanism-report-sha256 "$MECHANISM_REPORT_SHA256" \
  --mechanism-receipt "$MECHANISM_RECEIPT" \
  --mechanism-receipt-sha256 "$MECHANISM_RECEIPT_SHA256" \
  --predictor-checkpoint "$PREDICTOR_CHECKPOINT" \
  --predictor-checkpoint-sha256 "$PREDICTOR_CHECKPOINT_SHA256" \
  --generator-config "$GENERATOR_CONFIG" \
  --generator-config-sha256 "$GENERATOR_CONFIG_SHA256" \
  --generator-checkpoint "$GENERATOR_CHECKPOINT" \
  --generator-checkpoint-sha256 "$GENERATOR_CHECKPOINT_SHA256" \
  --exact-python "$EXACT_PYTHON" \
  --avqi-code-root "$AVQI_CODE_ROOT" \
  --avqi-code-tree-sha256 "$AVQI_CODE_TREE_SHA256" \
  --sealed-output-dir "$SEALED_OUTPUT_DIR" \
  --panel-contract-sha256 "$PANEL_CONTRACT_SHA256" \
  --target-label-contract-sha256 "$TARGET_LABEL_CONTRACT_SHA256" \
  --candidate-seal-sha256 "$CANDIDATE_SEAL_SHA256" \
  --source-commit "$SOURCE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID" \
  --sealed-source-commit "$SEALED_SOURCE_COMMIT" \
  --sealed-job-id "$SEALED_JOB_ID" \
  --device cuda \
  2>&1 | tee -a "$LIVE_LOG"

echo "event=complete job=$SLURM_JOB_ID time=$(date -Is)" | tee -a "$LIVE_LOG"
