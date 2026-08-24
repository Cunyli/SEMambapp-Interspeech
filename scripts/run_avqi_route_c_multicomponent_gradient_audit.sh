#!/bin/bash
# Run a dev-only five-active Route C gradient interference audit. No training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/avqi_route_c_five_component_gradient_audit_v1_20260824_01}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR:-$RUN_ROOT/test_outputs}"
PYTEST_LOG="${PYTEST_LOG:-$TEST_OUTPUT_DIR/pytest.log}"
PARTITION="${PARTITION:-gpu-v100-32g}"
GPU_TYPE="${GPU_TYPE:-v100}"
TIME_LIMIT="${TIME_LIMIT:-00:45:00}"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-/scratch/work/lil14/.conda_envs/semambapp/bin/python}"
ACCEPTED_BASE_COMMIT="${ACCEPTED_BASE_COMMIT:-1ebdbfee4d667e201b0f0ecac6a9d46123417ac7}"

LABEL_BANK="${LABEL_BANK:-/scratch/work/lil14/SEMambapp-Interspeech/runs/avqi_component_direct_c_v5_data_20260817_03/outputs/label_bank/exact_component_label_bank_v4.csv}"
LABEL_BANK_SHA256="${LABEL_BANK_SHA256:-03b8d5e3d0542dbfe60e54723bc89431e8dfd475dcc38284a6058465c5224760}"
CPPS_CHECKPOINT="${CPPS_CHECKPOINT:-/scratch/work/lil14/SEMambapp-Interspeech-avqi-cpps-v7/checkpoints/avqi_route_c_cpps_v12_screen_20260823_20/direct_direct_praat_hard_cpps_view_input_v12_estimator.pt}"
CPPS_CHECKPOINT_SHA256="${CPPS_CHECKPOINT_SHA256:-f893783cccf09f3b4fe707e881afdfc889724861a6b86eb0600fb5400534c543}"
HNR_CHECKPOINT="${HNR_CHECKPOINT:-/scratch/work/lil14/SEMambapp-Interspeech-avqi-hnr-v7/checkpoints/avqi_route_c_hnr_v7_screen_20260824_01/direct_direct_praat_hard_hnr_pitch_path_v7_estimator.pt}"
HNR_CHECKPOINT_SHA256="${HNR_CHECKPOINT_SHA256:-93f5b122486bcdc54215244fb894ffa3b34d1993fca32011dbf57650467c412c}"
SHIMMER_CHECKPOINT="${SHIMMER_CHECKPOINT:-/scratch/work/lil14/SEMambapp-Interspeech/checkpoints/avqi_route_c_shimmer_v6_screen_20260821_01/direct_direct_praat_hard_shimmer_pulse_path_v6_estimator.pt}"
SHIMMER_CHECKPOINT_SHA256="${SHIMMER_CHECKPOINT_SHA256:-40b819946abdcb8a4b643fe4238d1bb4d31168a3eb2a6d6c786a61752da629bc}"
SLOPE_CHECKPOINT="${SLOPE_CHECKPOINT:-$SHIMMER_CHECKPOINT}"
SLOPE_CHECKPOINT_SHA256="${SLOPE_CHECKPOINT_SHA256:-$SHIMMER_CHECKPOINT_SHA256}"
TILT_CHECKPOINT="${TILT_CHECKPOINT:-/scratch/work/lil14/SEMambapp-Interspeech/checkpoints/avqi_component_direct_c_v5_screen_20260817_01/direct_direct_praat_hard_v2_estimator.pt}"
TILT_CHECKPOINT_SHA256="${TILT_CHECKPOINT_SHA256:-07b69e6722db46467626bd68ffaceb32844908c1a8378ee7e3bdc327fdc81aef}"

CPPS_REPORT="/scratch/work/lil14/SEMambapp-Interspeech-avqi-cpps-v7/runs/avqi_route_c_cpps_v12_waveform_pilot_20260823_28/outputs/waveform_optimization_report.json"
CPPS_REPORT_SHA256="dc4ae219706fb5b23af1ed4e38dfe7a80a8e3ee31daebf0ef1c47ad914dc6dcc"
CPPS_RECEIPT="/scratch/work/lil14/SEMambapp-Interspeech-avqi-cpps-v7/runs/avqi_route_c_cpps_v12_waveform_pilot_20260823_28/outputs/completion_receipt.json"
CPPS_RECEIPT_SHA256="c7c3fd22b7b713c59a1d9c2708289ddf64a723c96714cb5e185e6a4b765e41da"
HNR_REPORT="/scratch/work/lil14/SEMambapp-Interspeech-avqi-hnr-v7/runs/avqi_route_c_hnr_v7_fresh_panel_20260824_01/outputs/fresh_panel_report.json"
HNR_REPORT_SHA256="67c5d4327ddd33fa45d03ead9028f52031a1b2d0645200af223bcc3053137d7d"
HNR_RECEIPT="/scratch/work/lil14/SEMambapp-Interspeech-avqi-hnr-v7/runs/avqi_route_c_hnr_v7_fresh_panel_20260824_01/outputs/completion_receipt.json"
HNR_RECEIPT_SHA256="603d4935708907ee6c403ee1b062a3d70f9af7c0b7e9e7b224ef7aa7e7fbf1b7"
SHIMMER_REPORT="/scratch/work/lil14/SEMambapp-Interspeech/runs/avqi_route_c_shimmer_v6_fresh_panel_20260821_02/outputs/fresh_panel_report.json"
SHIMMER_REPORT_SHA256="f40a8e16c0467c0d52654173a48c1b515eb68c223a2097557dc14eb1999350ac"
SHIMMER_RECEIPT="/scratch/work/lil14/SEMambapp-Interspeech/runs/avqi_route_c_shimmer_v6_fresh_panel_20260821_02/outputs/completion_receipt.json"
SHIMMER_RECEIPT_SHA256="0a3a09591099106ccf25226516896ff4f3c81b22d93247901f96c4bb042be963"
SLOPE_REPORT="/scratch/work/lil14/SEMambapp-Interspeech-avqi-ltas-slope-promotion-v1/runs/avqi_route_c_ltas_slope_fresh_panel_v1_20260824_01/outputs/fresh_panel_report.json"
SLOPE_REPORT_SHA256="7301291f8f4ab3fe0454bce293b346865d81261ce6f4e86a31615dde153e26b0"
SLOPE_RECEIPT="/scratch/work/lil14/SEMambapp-Interspeech-avqi-ltas-slope-promotion-v1/runs/avqi_route_c_ltas_slope_fresh_panel_v1_20260824_01/outputs/completion_receipt.json"
SLOPE_RECEIPT_SHA256="35b5f787df7b4b053a3d502878df5d92938438c1cf9bb6dc494b564b65e3b74e"
SLOPE_FINAL_PANEL_SEAL="/scratch/work/lil14/SEMambapp-Interspeech-avqi-ltas-slope-promotion-v1/runs/avqi_route_c_ltas_slope_fresh_panel_v1_20260824_01/outputs/final_panel_seal.json"
SLOPE_FINAL_PANEL_SEAL_SHA256="52a2afff063825e9c68228b5fafc0a9073853fd07deaeb7aee36aa09334b151c"
SLOPE_FINAL_RESULTS="/scratch/work/lil14/SEMambapp-Interspeech-avqi-ltas-slope-promotion-v1/runs/avqi_route_c_ltas_slope_fresh_panel_v1_20260824_01/outputs/final_results.csv"
SLOPE_FINAL_RESULTS_SHA256="c4bc55d2074d93d7425a018d92aa00f38a402a1ea2b2394bc2b87ee5ca48ea2a"
TILT_REPORT="/scratch/work/lil14/SEMambapp-Interspeech/runs/avqi_direct_waveform_opt_balanced_hnr_tilt_final_20260814_01/outputs/waveform_optimization_report.json"
TILT_REPORT_SHA256="0fa3cd09638c105fdcc31cd3a9460ffe9bcf85d9083159571750f7802c32678a"
TILT_RECEIPT="/scratch/work/lil14/SEMambapp-Interspeech/runs/avqi_direct_waveform_opt_balanced_hnr_tilt_final_20260814_01/outputs/completion_receipt.json"
TILT_RECEIPT_SHA256="d969e61bab4df7bea9d3f439b36774ad86e07a56bf294481d9224298366fb897"
SELECTION_SALT="${SELECTION_SALT:-route-c-five-active-dev-audit-1ebdbfe-20260824}"

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing to run from a dirty source tree: $SOURCE_ROOT" >&2
  exit 2
fi
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"

REQUIRED_PATHS=(
  "$LABEL_BANK"
  "$CPPS_CHECKPOINT"
  "$HNR_CHECKPOINT"
  "$SHIMMER_CHECKPOINT"
  "$SLOPE_CHECKPOINT"
  "$TILT_CHECKPOINT"
  "$CPPS_REPORT"
  "$CPPS_RECEIPT"
  "$HNR_REPORT"
  "$HNR_RECEIPT"
  "$SHIMMER_REPORT"
  "$SHIMMER_RECEIPT"
  "$SLOPE_REPORT"
  "$SLOPE_RECEIPT"
  "$SLOPE_FINAL_PANEL_SEAL"
  "$SLOPE_FINAL_RESULTS"
  "$TILT_REPORT"
  "$TILT_RECEIPT"
  "$RUNTIME_PYTHON"
)
for path in "${REQUIRED_PATHS[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing Route C audit source: $path" >&2
    exit 2
  fi
done

export ROOT_DIR SOURCE_ROOT RUN_ROOT LOG_DIR OUTPUT_DIR TEST_OUTPUT_DIR PYTEST_LOG
export PARTITION GPU_TYPE TIME_LIMIT RUNTIME_PYTHON ACCEPTED_BASE_COMMIT
export SOURCE_COMMIT LABEL_BANK LABEL_BANK_SHA256 CPPS_CHECKPOINT
export CPPS_CHECKPOINT_SHA256 HNR_CHECKPOINT HNR_CHECKPOINT_SHA256
export SHIMMER_CHECKPOINT SHIMMER_CHECKPOINT_SHA256 SLOPE_CHECKPOINT
export SLOPE_CHECKPOINT_SHA256 TILT_CHECKPOINT TILT_CHECKPOINT_SHA256
export SELECTION_SALT

mkdir -p "$LOG_DIR"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  if [[ -e "$OUTPUT_DIR" || -e "$TEST_OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite output under: $RUN_ROOT" >&2
    exit 2
  fi
  sbatch --parsable \
    --job-name=avqi-c5-grad-audit \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --gres="gpu:${GPU_TYPE}:1" \
    --cpus-per-task=4 \
    --mem=48G \
    --time="$TIME_LIMIT" \
    --output="$LOG_DIR/slurm_%j.out" \
    --error="$LOG_DIR/slurm_%j.err" \
    --export=ALL \
    "$SELF_PATH"
  exit 0
fi

if [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" != "$SOURCE_COMMIT" ]]; then
  echo "Source HEAD drifted after submission" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" || -e "$TEST_OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite output under: $RUN_ROOT" >&2
  exit 2
fi

module load triton/2025.1-gcc
module load gcc/13.3.0
cd "$SOURCE_ROOT"
mkdir -p "$TEST_OUTPUT_DIR"
LIVE_LOG="$LOG_DIR/multicomponent_gradient_${SLURM_JOB_ID}.log"
echo "event=start job=$SLURM_JOB_ID source_commit=$SOURCE_COMMIT time=$(date -Is)" | tee -a "$LIVE_LOG"
"$RUNTIME_PYTHON" -m pytest \
  tests/test_avqi_route_c_multicomponent.py \
  tests/test_avqi_components.py \
  tests/test_direct_avqi_waveform_optimization.py \
  tests/test_avqi_hnr_fresh_panel.py \
  tests/test_avqi_shimmer_fresh_panel.py \
  tests/test_avqi_ltas_slope_fresh_panel.py \
  tests/test_artifact_layout.py \
  -q 2>&1 | tee "$PYTEST_LOG" | tee -a "$LIVE_LOG"
"$RUNTIME_PYTHON" -m scripts.evaluate_avqi_route_c_multicomponent_gradients \
  --label-bank "$LABEL_BANK" \
  --label-bank-sha256 "$LABEL_BANK_SHA256" \
  --cpps-checkpoint "$CPPS_CHECKPOINT" \
  --cpps-checkpoint-sha256 "$CPPS_CHECKPOINT_SHA256" \
  --hnr-checkpoint "$HNR_CHECKPOINT" \
  --hnr-checkpoint-sha256 "$HNR_CHECKPOINT_SHA256" \
  --shimmer-checkpoint "$SHIMMER_CHECKPOINT" \
  --shimmer-checkpoint-sha256 "$SHIMMER_CHECKPOINT_SHA256" \
  --slope-checkpoint "$SLOPE_CHECKPOINT" \
  --slope-checkpoint-sha256 "$SLOPE_CHECKPOINT_SHA256" \
  --tilt-checkpoint "$TILT_CHECKPOINT" \
  --tilt-checkpoint-sha256 "$TILT_CHECKPOINT_SHA256" \
  --evidence cpps_report "$CPPS_REPORT" "$CPPS_REPORT_SHA256" \
  --evidence cpps_receipt "$CPPS_RECEIPT" "$CPPS_RECEIPT_SHA256" \
  --evidence hnr_report "$HNR_REPORT" "$HNR_REPORT_SHA256" \
  --evidence hnr_receipt "$HNR_RECEIPT" "$HNR_RECEIPT_SHA256" \
  --evidence shimmer_percent_report "$SHIMMER_REPORT" "$SHIMMER_REPORT_SHA256" \
  --evidence shimmer_percent_receipt "$SHIMMER_RECEIPT" "$SHIMMER_RECEIPT_SHA256" \
  --evidence slope_report "$SLOPE_REPORT" "$SLOPE_REPORT_SHA256" \
  --evidence slope_receipt "$SLOPE_RECEIPT" "$SLOPE_RECEIPT_SHA256" \
  --evidence slope_final_panel_seal "$SLOPE_FINAL_PANEL_SEAL" "$SLOPE_FINAL_PANEL_SEAL_SHA256" \
  --evidence slope_final_results "$SLOPE_FINAL_RESULTS" "$SLOPE_FINAL_RESULTS_SHA256" \
  --evidence tilt_report "$TILT_REPORT" "$TILT_REPORT_SHA256" \
  --evidence tilt_receipt "$TILT_RECEIPT" "$TILT_RECEIPT_SHA256" \
  --selection-salt "$SELECTION_SALT" \
  --source-root "$SOURCE_ROOT" \
  --source-commit "$SOURCE_COMMIT" \
  --accepted-base-commit "$ACCEPTED_BASE_COMMIT" \
  --pytest-log "$PYTEST_LOG" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda 2>&1 | tee -a "$LIVE_LOG"
echo "event=complete job=$SLURM_JOB_ID time=$(date -Is)" | tee -a "$LIVE_LOG"
