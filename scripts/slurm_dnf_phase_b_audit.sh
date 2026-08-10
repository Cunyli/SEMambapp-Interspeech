#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch/work/lil14/SEMambapp-Interspeech}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/runs/dnf_phase_b_audit}"
JOB_NAME="${JOB_NAME:-dnf-phase-b-audit}"
PARTITION="${PARTITION:-batch-skl}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
AUDIT_ID="${AUDIT_ID:-phase-b-v1-$(date -u +%Y%m%dT%H%M%SZ)}"
SEED="${SEED:-3407}"

SPLIT_ROOT="${SPLIT_ROOT:-/scratch/elec/t412-speechcom/Triton - Symptonic/lijie/gap_webdataset_active/splits/hybrid_unise_v1_stream_80_10_10}"
FSD_GROUND_TRUTH="${FSD_GROUND_TRUTH:-$ROOT_DIR/tmp/dnf_phase_ab_v1/FSD50K.ground_truth.zip}"
DNSMOS_MODEL="${DNSMOS_MODEL:-$ROOT_DIR/pretrained/dnsmos_p835_v1/sig_bak_ovr.onnx}"
PYTHON_DEPS="${PYTHON_DEPS:-$ROOT_DIR/tmp/dnf_phase_ab_v1/python_deps}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"

AUDIT_DIR="$OUTPUT_ROOT/$AUDIT_ID"
LIVE_LOG="$LOG_DIR/dnf_phase_b_${AUDIT_ID}_${SLURM_JOB_ID:-submit}.log"

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
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
    --export="ALL,AUDIT_ID=$AUDIT_ID" \
    "$ROOT_DIR/scripts/slurm_dnf_phase_b_audit.sh"
  exit 0
fi

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
LIVE_LOG="$LOG_DIR/dnf_phase_b_${AUDIT_ID}_${SLURM_JOB_ID}.log"
echo "event=phase_b_start job=${SLURM_JOB_ID} audit_id=${AUDIT_ID} time=$(date -Is)" \
  | tee -a "$LIVE_LOG"
echo "split_root=$SPLIT_ROOT audit_dir=$AUDIT_DIR" | tee -a "$LIVE_LOG"

module load "$SOFTWARE_STACK_MODULE"
module load "$COMPILER_MODULE"
eval "$(conda shell.bash hook)"
conda activate semambapp
export OMP_NUM_THREADS="$CPUS_PER_TASK"
export MKL_NUM_THREADS="$CPUS_PER_TASK"

python scripts/audit_dnf_phase_b_sources.py \
  --split-root "$SPLIT_ROOT" \
  --fsd-ground-truth "$FSD_GROUND_TRUTH" \
  --output-dir "$AUDIT_DIR" \
  --seed "$SEED" \
  2>&1 | tee -a "$LIVE_LOG"

mkdir -p "$AUDIT_DIR/scores"
python scripts/score_dnf_phase_b_probe.py \
  --input-jsonl "$AUDIT_DIR/mls_clean_candidate_probe.jsonl" \
  --output-jsonl "$AUDIT_DIR/scores/mls_probe_scored.jsonl" \
  --summary-json "$AUDIT_DIR/scores/mls_probe_score_summary.json" \
  --dnsmos-model "$DNSMOS_MODEL" \
  --python-deps "$PYTHON_DEPS" \
  2>&1 | tee -a "$LIVE_LOG"

python scripts/score_dnf_phase_b_probe.py \
  --input-jsonl "$AUDIT_DIR/libri_clean_candidate_probe.jsonl" \
  --output-jsonl "$AUDIT_DIR/scores/libri_probe_scored.jsonl" \
  --summary-json "$AUDIT_DIR/scores/libri_probe_score_summary.json" \
  --dnsmos-model "$DNSMOS_MODEL" \
  --python-deps "$PYTHON_DEPS" \
  2>&1 | tee -a "$LIVE_LOG"

echo "event=phase_b_complete job=${SLURM_JOB_ID} audit_dir=${AUDIT_DIR} time=$(date -Is)" \
  | tee -a "$LIVE_LOG"
