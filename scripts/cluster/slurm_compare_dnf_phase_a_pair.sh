#!/bin/bash
# Historical cluster helper. Submission requires CONFIRM_SLURM_SUBMIT=1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$DEFAULT_ROOT}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
JOB_NAME="${JOB_NAME:-dnf-phase-a-compare}"
PARTITION="${PARTITION:-batch-skl}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-16G}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
STANDARD_DIR="${STANDARD_DIR:?STANDARD_DIR is required}"
DNF_DIR="${DNF_DIR:?DNF_DIR is required}"
OUTPUT_JSON="${OUTPUT_JSON:?OUTPUT_JSON is required}"
LISTENING_PACK_DIR="${LISTENING_PACK_DIR:-${OUTPUT_JSON%.json}__blind_listening}"
CONTRACT_PATH="${CONTRACT_PATH:-$ROOT_DIR/configs/train/dnf_phase_ab_v2_contract.json}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
SEED="${SEED:-1234}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"

mkdir -p "$LOG_DIR"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
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
    "$ROOT_DIR/scripts/cluster/slurm_compare_dnf_phase_a_pair.sh"
  exit 0
fi

if [[ -e "$OUTPUT_JSON" ]]; then
  echo "Refusing to overwrite immutable pair comparison: $OUTPUT_JSON" >&2
  exit 2
fi
if [[ -e "$LISTENING_PACK_DIR" ]]; then
  echo "Refusing to overwrite blind listening pack: $LISTENING_PACK_DIR" >&2
  exit 2
fi

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
LIVE_LOG="$LOG_DIR/dnf_phase_a_compare_${SLURM_JOB_ID}.log"
echo "event=phase_a_compare_start job=${SLURM_JOB_ID} time=$(date -Is)" \
  | tee -a "$LIVE_LOG"
echo "standard_dir=$STANDARD_DIR dnf_dir=$DNF_DIR" | tee -a "$LIVE_LOG"
echo "output_json=$OUTPUT_JSON contract=$CONTRACT_PATH" | tee -a "$LIVE_LOG"

module load "$SOFTWARE_STACK_MODULE"
module load "$COMPILER_MODULE"
eval "$(conda shell.bash hook)"
conda activate semambapp

python scripts/compare_dnf_phase_a_pair.py \
  --standard-dir "$STANDARD_DIR" \
  --dnf-dir "$DNF_DIR" \
  --contract "$CONTRACT_PATH" \
  --output-json "$OUTPUT_JSON" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --seed "$SEED" \
  2>&1 | tee -a "$LIVE_LOG"

python scripts/build_dnf_phase_a_blind_listening_pack.py \
  --standard-dir "$STANDARD_DIR" \
  --dnf-dir "$DNF_DIR" \
  --output-dir "$LISTENING_PACK_DIR" \
  --seed "$SEED" \
  2>&1 | tee -a "$LIVE_LOG"

python -c 'import json, pathlib, sys; payload=json.loads(pathlib.Path(sys.argv[1]).read_text()); raise SystemExit(0 if payload["controlled_gate_pass"] else 3)' "$OUTPUT_JSON"

echo "event=phase_a_compare_pass job=${SLURM_JOB_ID} time=$(date -Is)" \
  | tee -a "$LIVE_LOG"
