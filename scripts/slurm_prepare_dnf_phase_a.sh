#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch/work/lil14/SEMambapp-Interspeech}"
SPLIT_ROOT="${SPLIT_ROOT:-/scratch/elec/t412-speechcom/Triton - Symptonic/lijie/gap_webdataset_active/splits/hybrid_unise_v1_stream_80_10_10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/runs/semambapp_dnf_phase_a}"
MANIFEST_ID="${MANIFEST_ID:-phase-a-paper-v3-seed1234}"
MANIFEST_DIR="${MANIFEST_DIR:-$OUTPUT_ROOT/manifests/$MANIFEST_ID}"
SMOKE_JSON="${SMOKE_JSON:-$MANIFEST_DIR/data_smoke.json}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
SEED="${SEED:-1234}"
TRAIN_ROWS="${TRAIN_ROWS:-40000}"
VALID_ROWS="${VALID_ROWS:-200}"
PARTITION="${PARTITION:-batch-skl}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"

mkdir -p "$LOG_DIR"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ -e "$MANIFEST_DIR" ]]; then
    echo "Refusing to overwrite immutable manifest directory: $MANIFEST_DIR" >&2
    exit 2
  fi
  sbatch \
    --job-name="dnf-phase-a-prepare" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEMORY" \
    --time="$TIME_LIMIT" \
    --output="$LOG_DIR/slurm_%j.out" \
    --error="$LOG_DIR/slurm_%j.err" \
    --export=ALL \
    "$ROOT_DIR/scripts/slurm_prepare_dnf_phase_a.sh"
  exit 0
fi

if [[ -e "$MANIFEST_DIR" ]]; then
  echo "Refusing to overwrite immutable manifest directory: $MANIFEST_DIR" >&2
  exit 2
fi

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
LIVE_LOG="$LOG_DIR/dnf_phase_a_prepare_${SLURM_JOB_ID}.log"
echo "event=phase_a_prepare_start job=${SLURM_JOB_ID} time=$(date -Is)" \
  | tee -a "$LIVE_LOG"

module load "$SOFTWARE_STACK_MODULE"
module load "$COMPILER_MODULE"
eval "$(conda shell.bash hook)"
conda activate semambapp

python -m pytest -q \
  tests/test_dnf_controlled_phase_a.py \
  tests/test_dnf_phase_a_loss.py \
  tests/test_dnf_phase_a_pair_compare.py \
  tests/test_dnf_phase_a_blind_listening.py \
  tests/test_dnf_phase_a_training_contract.py \
  tests/test_dnf_phase_a_tau_gate.py \
  tests/test_dnf_phase_a_tau_compare.py \
  2>&1 | tee -a "$LIVE_LOG"

python scripts/build_dnf_phase_a_manifests.py \
  --split-root "$SPLIT_ROOT" \
  --output-dir "$MANIFEST_DIR" \
  --seed "$SEED" \
  --train-rows "$TRAIN_ROWS" \
  --valid-rows "$VALID_ROWS" \
  --noise-pairing-policy same_family_iid \
  2>&1 | tee -a "$LIVE_LOG"

python scripts/smoke_dnf_phase_a_data.py \
  --train-manifest "$MANIFEST_DIR/train_manifest.jsonl" \
  --valid-manifest "$MANIFEST_DIR/valid_manifest.jsonl" \
  --output-json "$SMOKE_JSON" \
  --train-samples 40 \
  --valid-samples "$VALID_ROWS" \
  --seed "$SEED" \
  2>&1 | tee -a "$LIVE_LOG"

cp "$ROOT_DIR/configs/train/dnf_phase_ab_v2_contract.json" \
  "$MANIFEST_DIR/contract_snapshot.json"
sha256sum \
  "$MANIFEST_DIR/train_manifest.jsonl" \
  "$MANIFEST_DIR/valid_manifest.jsonl" \
  "$MANIFEST_DIR/data_smoke.json" \
  "$MANIFEST_DIR/contract_snapshot.json" \
  > "$MANIFEST_DIR/artifact_sha256.txt"

echo "event=phase_a_prepare_complete job=${SLURM_JOB_ID} time=$(date -Is)" \
  | tee -a "$LIVE_LOG"
