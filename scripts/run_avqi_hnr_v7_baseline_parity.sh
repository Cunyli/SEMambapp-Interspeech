#!/bin/bash
# Freeze exact-label vs legacy-NumPy vs raw_cc_v3 HNR parity. No optimization.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
SOURCE_ROOT="${SOURCE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_SCRIPT="$SOURCE_ROOT/scripts/evaluate_hnr_v7_baseline_parity.py"
AVQI_ROOT="${AVQI_ROOT:-/scratch/work/lil14/avqi}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/work/lil14/.conda_envs/avqi/bin/python}"

LABEL_BANK="${LABEL_BANK:-/scratch/work/lil14/SEMambapp-Interspeech/runs/avqi_component_direct_c_v5_data_20260817_03/outputs/label_bank/exact_component_label_bank_v4.csv}"
LABEL_BANK_SHA256="${LABEL_BANK_SHA256:-03b8d5e3d0542dbfe60e54723bc89431e8dfd475dcc38284a6058465c5224760}"
TORCH_RUN_ROOT="${TORCH_RUN_ROOT:-/scratch/work/lil14/SEMambapp-Interspeech/runs/avqi_direct_hnr_raw_cc_v3_diagnostic_20260817_02}"
TORCH_PREDICTIONS="$TORCH_RUN_ROOT/outputs/hnr_formula_predictions.csv"
TORCH_PREDICTIONS_SHA256="${TORCH_PREDICTIONS_SHA256:-2100bc8b044d1dee236fe020cbc8aea8b9af29d56e7a1015d38b8d8907ea111a}"
TORCH_REPORT="$TORCH_RUN_ROOT/outputs/hnr_formula_report.json"
TORCH_REPORT_SHA256="${TORCH_REPORT_SHA256:-626b90bc85a83cdab97669ee7f22e503b81ed1fe2dc107c53cb62bd9d38eca94}"

RUN_ROOT="${RUN_ROOT:-$SOURCE_ROOT/runs/avqi_route_c_hnr_v7_baseline_parity_20260824_01}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
PARTITION="${PARTITION:-batch-milan}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing to run from a dirty HNR source tree: $SOURCE_ROOT" >&2
  exit 2
fi
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"

for path in \
  "$PYTHON_SCRIPT" \
  "$PYTHON_BIN" \
  "$LABEL_BANK" \
  "$TORCH_PREDICTIONS" \
  "$TORCH_REPORT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing HNR baseline input: $path" >&2
    exit 2
  fi
done
for specification in \
  "$LABEL_BANK|$LABEL_BANK_SHA256" \
  "$TORCH_PREDICTIONS|$TORCH_PREDICTIONS_SHA256" \
  "$TORCH_REPORT|$TORCH_REPORT_SHA256"; do
  path="${specification%%|*}"
  expected="${specification##*|}"
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "HNR baseline hash mismatch: $path" >&2
    exit 2
  fi
done

export SOURCE_ROOT PYTHON_SCRIPT AVQI_ROOT PYTHON_BIN LABEL_BANK LABEL_BANK_SHA256
export TORCH_RUN_ROOT TORCH_PREDICTIONS TORCH_PREDICTIONS_SHA256
export TORCH_REPORT TORCH_REPORT_SHA256 RUN_ROOT LOG_DIR OUTPUT_DIR
export PARTITION CPUS_PER_TASK MEMORY TIME_LIMIT SOURCE_COMMIT

mkdir -p "$LOG_DIR"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite HNR baseline output: $OUTPUT_DIR" >&2
    exit 2
  fi
  sbatch \
    --parsable \
    --job-name=avqi-hnr-v7-base \
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

if [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" != "$SOURCE_COMMIT" ]]; then
  echo "HNR source HEAD drifted after submission" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite HNR baseline output: $OUTPUT_DIR" >&2
  exit 2
fi

cd "$SOURCE_ROOT"
export PYTHONPATH="$AVQI_ROOT:$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

LIVE_LOG="$LOG_DIR/avqi_hnr_v7_baseline_${SLURM_JOB_ID}.log"
echo "event=start job=$SLURM_JOB_ID commit=$SOURCE_COMMIT time=$(date -Is)" | tee -a "$LIVE_LOG"
"$PYTHON_BIN" -c 'import parselmouth, librosa, scipy; print("parselmouth", parselmouth.__version__); print("librosa", librosa.__version__); print("scipy", scipy.__version__)' | tee -a "$LIVE_LOG"

"$PYTHON_BIN" "$PYTHON_SCRIPT" \
  --label-bank "$LABEL_BANK" \
  --label-bank-sha256 "$LABEL_BANK_SHA256" \
  --torch-predictions "$TORCH_PREDICTIONS" \
  --torch-predictions-sha256 "$TORCH_PREDICTIONS_SHA256" \
  --torch-report "$TORCH_REPORT" \
  --torch-report-sha256 "$TORCH_REPORT_SHA256" \
  --source-commit "$SOURCE_COMMIT" \
  --output-dir "$OUTPUT_DIR" \
  --workers "$CPUS_PER_TASK" \
  --slurm-job-id "$SLURM_JOB_ID" \
  2>&1 | tee -a "$LIVE_LOG"

echo "event=complete job=$SLURM_JOB_ID time=$(date -Is)" | tee -a "$LIVE_LOG"
