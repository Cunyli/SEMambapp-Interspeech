#!/bin/bash
# Run the opened-panel Shimmer hybrid/Candidate-C mechanism diagnostic on Slurm.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
PYTHON_SCRIPT="$SOURCE_ROOT/scripts/evaluate_avqi_shimmer_hybrid_topology.py"

RUN_ROOT="${RUN_ROOT:-$SOURCE_ROOT/runs/avqi_route_c_shimmer_current_output_refresh_v13_20260824_01}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
PANEL_ROOT="${PANEL_ROOT:-/scratch/work/lil14/SEMambapp-Interspeech/runs/avqi_route_c_shimmer_v6_fresh_panel_20260821_02/outputs}"
PANEL_CONTRACT="${PANEL_CONTRACT:-$PANEL_ROOT/panel_contract.json}"
CALIBRATION_RESULTS="${CALIBRATION_RESULTS:-$PANEL_ROOT/calibration_alpha_results.csv}"
FINAL_RESULTS="${FINAL_RESULTS:-$PANEL_ROOT/final_results.csv}"
PREDICTOR_CHECKPOINT="${PREDICTOR_CHECKPOINT:-/scratch/work/lil14/SEMambapp-Interspeech/checkpoints/avqi_route_c_shimmer_v6_screen_20260821_01/direct_direct_praat_hard_shimmer_pulse_path_v6_estimator.pt}"
EXACT_PYTHON="${EXACT_PYTHON:-/scratch/work/lil14/.conda_envs/avqi/bin/python}"
AVQI_CODE_ROOT="${AVQI_CODE_ROOT:-/scratch/work/lil14/avqi}"

PANEL_CONTRACT_SHA256="${PANEL_CONTRACT_SHA256:?set PANEL_CONTRACT_SHA256}"
CALIBRATION_RESULTS_SHA256="${CALIBRATION_RESULTS_SHA256:?set CALIBRATION_RESULTS_SHA256}"
FINAL_RESULTS_SHA256="${FINAL_RESULTS_SHA256:?set FINAL_RESULTS_SHA256}"
PREDICTOR_CHECKPOINT_SHA256="${PREDICTOR_CHECKPOINT_SHA256:-40b819946abdcb8a4b643fe4238d1bb4d31168a3eb2a6d6c786a61752da629bc}"
AVQI_CODE_TREE_SHA256="${AVQI_CODE_TREE_SHA256:-46987b3c447cb579aab4d34e87655938e4aa64e1b28c0e2348c4ea3e48f107f2}"

PARTITION="${PARTITION:-gpu-v100-32g}"
GPU_TYPE="${GPU_TYPE:-v100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-00:45:00}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing to run from dirty source: $SOURCE_ROOT" >&2
  exit 2
fi
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"

for path in "$PYTHON_SCRIPT" "$PANEL_CONTRACT" "$CALIBRATION_RESULTS" \
  "$FINAL_RESULTS" "$PREDICTOR_CHECKPOINT" "$EXACT_PYTHON"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing Shimmer hybrid input: $path" >&2
    exit 2
  fi
done
if [[ ! -d "$AVQI_CODE_ROOT" ]]; then
  echo "Missing exact AVQI code root: $AVQI_CODE_ROOT" >&2
  exit 2
fi

export ROOT_DIR SOURCE_ROOT PYTHON_SCRIPT RUN_ROOT LOG_DIR OUTPUT_DIR
export PANEL_CONTRACT PANEL_CONTRACT_SHA256 CALIBRATION_RESULTS
export CALIBRATION_RESULTS_SHA256 FINAL_RESULTS FINAL_RESULTS_SHA256
export PREDICTOR_CHECKPOINT PREDICTOR_CHECKPOINT_SHA256 EXACT_PYTHON
export AVQI_CODE_ROOT AVQI_CODE_TREE_SHA256 SOURCE_COMMIT
export SOFTWARE_STACK_MODULE COMPILER_MODULE

mkdir -p "$LOG_DIR"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite Shimmer hybrid output: $OUTPUT_DIR" >&2
    exit 2
  fi
  sbatch \
    --parsable \
    --job-name=avqi-shim-refresh-v13 \
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
  echo "Shimmer hybrid source HEAD drifted after submission" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite Shimmer hybrid output: $OUTPUT_DIR" >&2
  exit 2
fi

cd "$ROOT_DIR"
module load "$SOFTWARE_STACK_MODULE"
module load "$COMPILER_MODULE"
eval "$(conda shell.bash hook)"
conda activate semambapp
export PYTHONPATH="$SOURCE_ROOT:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

LIVE_LOG="$LOG_DIR/shimmer_refresh_${SLURM_JOB_ID}.log"
echo "event=start job=$SLURM_JOB_ID commit=$SOURCE_COMMIT time=$(date -Is)" | tee -a "$LIVE_LOG"
python -c 'import torch; print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("device", torch.cuda.get_device_name(0))' | tee -a "$LIVE_LOG"
python "$PYTHON_SCRIPT" \
  --panel-contract "$PANEL_CONTRACT" \
  --panel-contract-sha256 "$PANEL_CONTRACT_SHA256" \
  --calibration-results "$CALIBRATION_RESULTS" \
  --calibration-results-sha256 "$CALIBRATION_RESULTS_SHA256" \
  --final-results "$FINAL_RESULTS" \
  --final-results-sha256 "$FINAL_RESULTS_SHA256" \
  --predictor-checkpoint "$PREDICTOR_CHECKPOINT" \
  --predictor-checkpoint-sha256 "$PREDICTOR_CHECKPOINT_SHA256" \
  --exact-python "$EXACT_PYTHON" \
  --avqi-code-root "$AVQI_CODE_ROOT" \
  --avqi-code-tree-sha256 "$AVQI_CODE_TREE_SHA256" \
  --output-dir "$OUTPUT_DIR" \
  --source-commit "$SOURCE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID" \
  --device cuda \
  2>&1 | tee -a "$LIVE_LOG"
echo "event=complete job=$SLURM_JOB_ID time=$(date -Is)" | tee -a "$LIVE_LOG"
