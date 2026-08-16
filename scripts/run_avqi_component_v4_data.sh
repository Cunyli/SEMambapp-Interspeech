#!/bin/bash
# Prepare and exact-score the frozen VCTK v4 expansion. No GPU is requested.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/avqi_component_phaseaware_v4_data_20260816_01}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
DATA_DIR="${DATA_DIR:-$RUN_ROOT/outputs/vctk_data}"
LABEL_DIR="${LABEL_DIR:-$RUN_ROOT/outputs/label_bank}"
PARTITION="${PARTITION:-batch-bdw}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
SEED="${SEED:-20260816}"
MAX_OPEN_SHARDS="${MAX_OPEN_SHARDS:-4}"

POOL_ROOT="${POOL_ROOT:-/scratch/elec/t412-speechcom/Triton - Symptonic/lijie/gap_webdataset_active}"
VCTK_ROOT="${VCTK_ROOT:-$POOL_ROOT/v1_vctk_clean/clean}"
VCTK_MANIFEST="${VCTK_MANIFEST:-$VCTK_ROOT/manifest.jsonl}"
NOISE_ROOT="${NOISE_ROOT:-$POOL_ROOT/v1_dns5_noise/noise}"
NOISE_MANIFEST="${NOISE_MANIFEST:-$NOISE_ROOT/manifest.jsonl}"
RIR_ROOT="${RIR_ROOT:-$POOL_ROOT/v1_verified/rir}"
RIR_MANIFEST="${RIR_MANIFEST:-$RIR_ROOT/manifest.jsonl}"
BASE_LABEL_BANK="${BASE_LABEL_BANK:-$ROOT_DIR/runs/avqi_component_expanded_data_20260813_02/outputs/exact_component_label_bank_v2.csv}"
EXACT_PYTHON="${EXACT_PYTHON:-/scratch/work/lil14/.conda_envs/avqi/bin/python}"
EXACT_RUNNER="${EXACT_RUNNER:-/scratch/work/lil14/Hybrid_Unise/scripts/validation_selected_tau_free_run.py}"
AVQI_MAIN="${AVQI_MAIN:-/scratch/work/lil14/avqi/avqi_code/main.py}"
AVQI_CODE_ROOT="${AVQI_CODE_ROOT:-/scratch/work/lil14/avqi}"

if [[ -n "$(git -C "$ROOT_DIR" status --porcelain)" ]]; then
  echo "Refusing to run from a dirty source tree: $ROOT_DIR" >&2
  exit 2
fi
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$ROOT_DIR" rev-parse HEAD)}"

for path in "$VCTK_MANIFEST" "$NOISE_MANIFEST" "$RIR_MANIFEST" \
  "$BASE_LABEL_BANK" "$EXACT_PYTHON" "$EXACT_RUNNER" "$AVQI_MAIN"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required source: $path" >&2
    exit 2
  fi
done

VCTK_MANIFEST_SHA256="${VCTK_MANIFEST_SHA256:-$(sha256sum "$VCTK_MANIFEST" | awk '{print $1}')}"
NOISE_MANIFEST_SHA256="${NOISE_MANIFEST_SHA256:-$(sha256sum "$NOISE_MANIFEST" | awk '{print $1}')}"
RIR_MANIFEST_SHA256="${RIR_MANIFEST_SHA256:-$(sha256sum "$RIR_MANIFEST" | awk '{print $1}')}"
BASE_LABEL_BANK_SHA256="${BASE_LABEL_BANK_SHA256:-$(sha256sum "$BASE_LABEL_BANK" | awk '{print $1}')}"
EXACT_RUNNER_SHA256="${EXACT_RUNNER_SHA256:-$(sha256sum "$EXACT_RUNNER" | awk '{print $1}')}"
AVQI_MAIN_SHA256="${AVQI_MAIN_SHA256:-$(sha256sum "$AVQI_MAIN" | awk '{print $1}')}"

export ROOT_DIR RUN_ROOT LOG_DIR DATA_DIR LABEL_DIR PARTITION CPUS_PER_TASK MEMORY
export TIME_LIMIT SEED POOL_ROOT VCTK_ROOT VCTK_MANIFEST NOISE_ROOT NOISE_MANIFEST
export MAX_OPEN_SHARDS
export RIR_ROOT RIR_MANIFEST BASE_LABEL_BANK EXACT_PYTHON EXACT_RUNNER AVQI_MAIN
export AVQI_CODE_ROOT SOURCE_COMMIT VCTK_MANIFEST_SHA256 NOISE_MANIFEST_SHA256
export RIR_MANIFEST_SHA256 BASE_LABEL_BANK_SHA256 EXACT_RUNNER_SHA256
export AVQI_MAIN_SHA256

mkdir -p "$LOG_DIR"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  if [[ -e "$DATA_DIR" || -e "$LABEL_DIR" ]]; then
    echo "Refusing to overwrite data or labels: $DATA_DIR $LABEL_DIR" >&2
    exit 2
  fi
  sbatch \
    --job-name=avqi-v4-data \
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

if [[ -e "$DATA_DIR" || -e "$LABEL_DIR" ]]; then
  echo "Refusing to overwrite data or labels: $DATA_DIR $LABEL_DIR" >&2
  exit 2
fi

cd "$ROOT_DIR"
module load triton/2025.1-gcc
module load gcc/13.3.0
LIVE_LOG="$LOG_DIR/avqi_v4_data_${SLURM_JOB_ID}.log"
echo "event=start job=$SLURM_JOB_ID commit=$SOURCE_COMMIT time=$(date -Is)" | tee -a "$LIVE_LOG"

conda run -n semambapp python scripts/prepare_avqi_component_v4_vctk.py \
  --vctk-manifest "$VCTK_MANIFEST" \
  --vctk-manifest-sha256 "$VCTK_MANIFEST_SHA256" \
  --vctk-root "$VCTK_ROOT" \
  --noise-manifest "$NOISE_MANIFEST" \
  --noise-manifest-sha256 "$NOISE_MANIFEST_SHA256" \
  --noise-root "$NOISE_ROOT" \
  --rir-manifest "$RIR_MANIFEST" \
  --rir-manifest-sha256 "$RIR_MANIFEST_SHA256" \
  --rir-root "$RIR_ROOT" \
  --output-dir "$DATA_DIR" \
  --seed "$SEED" \
  --max-open-shards "$MAX_OPEN_SHARDS" \
  2>&1 | tee -a "$LIVE_LOG"

VCTK_METADATA="$DATA_DIR/metadata.csv"
VCTK_METADATA_SHA256="$(sha256sum "$VCTK_METADATA" | awk '{print $1}')"
export PYTHONPATH="$AVQI_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$EXACT_PYTHON" scripts/build_avqi_component_v4_label_bank.py \
  --base-label-bank "$BASE_LABEL_BANK" \
  --base-label-bank-sha256 "$BASE_LABEL_BANK_SHA256" \
  --vctk-metadata "$VCTK_METADATA" \
  --vctk-metadata-sha256 "$VCTK_METADATA_SHA256" \
  --output-dir "$LABEL_DIR" \
  --workers "$CPUS_PER_TASK" \
  --exact-runner "$EXACT_RUNNER" \
  --exact-runner-sha256 "$EXACT_RUNNER_SHA256" \
  --avqi-main "$AVQI_MAIN" \
  --avqi-main-sha256 "$AVQI_MAIN_SHA256" \
  2>&1 | tee -a "$LIVE_LOG"

echo "event=complete job=$SLURM_JOB_ID time=$(date -Is)" | tee -a "$LIVE_LOG"
