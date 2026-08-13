#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$DEFAULT_ROOT}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
PREPARE_SCRIPT="${PREPARE_SCRIPT:-$SOURCE_ROOT/scripts/prepare_avqi_component_expanded_data.py}"
SCORE_SCRIPT="${SCORE_SCRIPT:-$SOURCE_ROOT/scripts/build_avqi_component_expanded_label_bank.py}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/avqi_component_expanded_data_20260813_01}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/outputs}"
DATA_DIR="${DATA_DIR:-$OUTPUT_ROOT/expanded_train_audio}"
OUTPUT_LABEL_BANK="${OUTPUT_LABEL_BANK:-$OUTPUT_ROOT/exact_component_label_bank_v2.csv}"

FULL_MANIFEST="${FULL_MANIFEST:-/scratch/work/lil14/use_simulation_pipeline/outputs/organized/csv/sampling/tau_clean_avqi_sampling_manifest.csv}"
FULL_MANIFEST_SHA256="${FULL_MANIFEST_SHA256:-ea227a724ced6436b9aa7c75d4b1ca3d78bc28e157baa0bd73d662d28d2549bf}"
SELECTED_MANIFEST="${SELECTED_MANIFEST:-/scratch/work/lil14/use_simulation_pipeline/outputs/organized/csv/sampling/tau_clean_avqi_selected_samples.csv}"
SELECTED_MANIFEST_SHA256="${SELECTED_MANIFEST_SHA256:-12e7899e700b71807d971b38442d29297235da89b3231ed2d8bfa1a42dcdb049}"
SIMULATION_ROOT="${SIMULATION_ROOT:-/scratch/work/lil14/USE_simulation}"
SIMULATION_CONFIG="${SIMULATION_CONFIG:-$SIMULATION_ROOT/configs/phone_room_22050.yaml}"
SIMULATION_CONFIG_SHA256="${SIMULATION_CONFIG_SHA256:-0e665b5f3d97ad617cd1dde22a84b1ec5a8089e31b7657c7cb9989363115e276}"
NOISE_ROOT="${NOISE_ROOT:-/scratch/elec/t412-speechcom/Triton - Symptonic/lijie/gap_webdataset_active/v1_dns5_noise/noise}"
NOISE_MANIFEST="${NOISE_MANIFEST:-$NOISE_ROOT/manifest.jsonl}"
NOISE_MANIFEST_SHA256="${NOISE_MANIFEST_SHA256:-c6f9441cdd76f50b4eb7f4fa5b83b994a509d3a925d7ae9b887059af31794d65}"
RIR_ROOT="${RIR_ROOT:-/scratch/elec/t412-speechcom/Triton - Symptonic/lijie/gap_webdataset_active/v1_verified/rir}"
RIR_MANIFEST="${RIR_MANIFEST:-$RIR_ROOT/manifest.jsonl}"
RIR_MANIFEST_SHA256="${RIR_MANIFEST_SHA256:-6ea8993a88438232ff97ad524f70002623c9944844d7c6a27a7e582bd872cfab}"

BASE_LABEL_BANK="${BASE_LABEL_BANK:-$ROOT_DIR/runs/tau_pathology_preservation_eval_phase2_20260809_01/outputs/surrogate/exact_component_label_bank_v1.csv}"
BASE_LABEL_BANK_SHA256="${BASE_LABEL_BANK_SHA256:-d0e2b9453e8b1c3521b8ab8a30e246f00f5b25759baf263ddd0932e847779e1f}"
EXTERNAL_EXACT_CSV="${EXTERNAL_EXACT_CSV:-$ROOT_DIR/runs/tau_pathology_three_tracks_20260810_01/outputs/intensity_eval/exact_components_all.csv}"
EXTERNAL_EXACT_CSV_SHA256="${EXTERNAL_EXACT_CSV_SHA256:-1e401d2d3343d5d5e8dc38245d14a2e4f9fbb568b11a26269e4ce0aca30c249a}"
AVQI_ROOT="${AVQI_ROOT:-/scratch/work/lil14/avqi}"
EXACT_RUNNER="${EXACT_RUNNER:-/scratch/work/lil14/Hybrid_Unise/scripts/validation_selected_tau_free_run.py}"
EXACT_RUNNER_SHA256="${EXACT_RUNNER_SHA256:-71771de0892df6434f1a693b1ee13b294d785d43db1263d95cc698c994a4670b}"
AVQI_MAIN_SHA256="${AVQI_MAIN_SHA256:-b3018da3b41aa17d1382a390f1c2bbf31520a824b82db871d5d67b4417f6bcc6}"
AVQI_PRAAT_SHA256="${AVQI_PRAAT_SHA256:-432b5157bc6ae03eb9d10d19aa0c0fc13aae711e172cd02fe21297bd581e85e0}"
AVQI_SCRIPT_SHA256="${AVQI_SCRIPT_SHA256:-2c98811ab1ccf684fe45b7f4cfbcf1a4719a775ea337f55d27f1ea81351782ea}"

JOB_NAME="${JOB_NAME:-avqi-expand-data}"
PARTITION="${PARTITION:-batch-milan}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
SEED="${SEED:-20260813}"
EXPECTED_NEW_SPEAKERS="${EXPECTED_NEW_SPEAKERS:-55}"

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing to run from a dirty source tree: $SOURCE_ROOT" >&2
  exit 2
fi
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"

export ROOT_DIR SOURCE_ROOT PREPARE_SCRIPT SCORE_SCRIPT
export RUN_ROOT LOG_DIR OUTPUT_ROOT DATA_DIR OUTPUT_LABEL_BANK
export FULL_MANIFEST FULL_MANIFEST_SHA256 SELECTED_MANIFEST SELECTED_MANIFEST_SHA256
export SIMULATION_ROOT SIMULATION_CONFIG SIMULATION_CONFIG_SHA256
export NOISE_ROOT NOISE_MANIFEST NOISE_MANIFEST_SHA256
export RIR_ROOT RIR_MANIFEST RIR_MANIFEST_SHA256
export BASE_LABEL_BANK BASE_LABEL_BANK_SHA256 EXTERNAL_EXACT_CSV
export EXTERNAL_EXACT_CSV_SHA256 AVQI_ROOT EXACT_RUNNER EXACT_RUNNER_SHA256
export AVQI_MAIN_SHA256 AVQI_PRAAT_SHA256 AVQI_SCRIPT_SHA256
export JOB_NAME PARTITION CPUS_PER_TASK MEMORY TIME_LIMIT SEED
export EXPECTED_NEW_SPEAKERS SOURCE_COMMIT

mkdir -p "$LOG_DIR"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite output: $OUTPUT_ROOT" >&2
    exit 2
  fi
  sbatch \
    --job-name="$JOB_NAME" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEMORY" \
    --time="$TIME_LIMIT" \
    --output="$LOG_DIR/slurm_%j.out" \
    --error="$LOG_DIR/slurm_%j.err" \
    --export=ALL \
    "$SELF_PATH"
  exit 0
fi

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to overwrite output: $OUTPUT_ROOT" >&2
  exit 2
fi

module load triton/2025.1-gcc
module load gcc/13.3.0
eval "$(conda shell.bash hook)"
conda activate semambapp
export PYTHONPATH="$SIMULATION_ROOT:$SOURCE_ROOT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "event=prepare_start job=$SLURM_JOB_ID source_commit=$SOURCE_COMMIT time=$(date -Is)"
python "$PREPARE_SCRIPT" \
  --full-manifest "$FULL_MANIFEST" \
  --full-manifest-sha256 "$FULL_MANIFEST_SHA256" \
  --selected-manifest "$SELECTED_MANIFEST" \
  --selected-manifest-sha256 "$SELECTED_MANIFEST_SHA256" \
  --simulation-config "$SIMULATION_CONFIG" \
  --simulation-config-sha256 "$SIMULATION_CONFIG_SHA256" \
  --noise-manifest "$NOISE_MANIFEST" \
  --noise-manifest-sha256 "$NOISE_MANIFEST_SHA256" \
  --rir-manifest "$RIR_MANIFEST" \
  --rir-manifest-sha256 "$RIR_MANIFEST_SHA256" \
  --noise-root "$NOISE_ROOT" \
  --rir-root "$RIR_ROOT" \
  --output-dir "$DATA_DIR" \
  --expected-new-speakers "$EXPECTED_NEW_SPEAKERS" \
  --seed "$SEED"

EXPANSION_METADATA="$DATA_DIR/metadata.json"
EXPANSION_METADATA_SHA256="$(sha256sum "$EXPANSION_METADATA" | awk '{print $1}')"

conda activate /scratch/work/lil14/.conda_envs/avqi
export PYTHONPATH="$AVQI_ROOT:$SOURCE_ROOT"

echo "event=exact_score_start job=$SLURM_JOB_ID time=$(date -Is)"
python "$SCORE_SCRIPT" \
  --base-label-bank "$BASE_LABEL_BANK" \
  --base-label-bank-sha256 "$BASE_LABEL_BANK_SHA256" \
  --expansion-metadata "$EXPANSION_METADATA" \
  --expansion-metadata-sha256 "$EXPANSION_METADATA_SHA256" \
  --external-exact-csv "$EXTERNAL_EXACT_CSV" \
  --external-exact-csv-sha256 "$EXTERNAL_EXACT_CSV_SHA256" \
  --output-csv "$OUTPUT_LABEL_BANK" \
  --workers "$CPUS_PER_TASK" \
  --expected-expansion-speakers "$EXPECTED_NEW_SPEAKERS" \
  --expected-train-speakers 125 \
  --expected-calibration-speakers 14 \
  --expected-holdout-speakers 14 \
  --exact-runner "$EXACT_RUNNER" \
  --expected-exact-runner-sha256 "$EXACT_RUNNER_SHA256" \
  --avqi-main "$AVQI_ROOT/avqi_code/main.py" \
  --expected-avqi-main-sha256 "$AVQI_MAIN_SHA256" \
  --avqi-praat "$AVQI_ROOT/avqi_code/praat_version.py" \
  --expected-avqi-praat-sha256 "$AVQI_PRAAT_SHA256" \
  --avqi-praat-script "$AVQI_ROOT/avqi_code/praat_scripts/avqi_v03_01.praat" \
  --expected-avqi-praat-script-sha256 "$AVQI_SCRIPT_SHA256"

echo "event=complete job=$SLURM_JOB_ID label_bank=$OUTPUT_LABEL_BANK time=$(date -Is)"
