#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch/work/lil14/SEMambapp-Interspeech}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/runs/dnf_phase_b_audit}"
AUDIT_ID="${AUDIT_ID:-phase-b-v2-indoor-stable-20260719}"
AUDIT_DIR="${AUDIT_DIR:-$OUTPUT_ROOT/$AUDIT_ID}"
PARTITION="${PARTITION:-batch-skl}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
SEED="${SEED:-3407}"
SPLIT_ROOT="${SPLIT_ROOT:-/scratch/elec/t412-speechcom/Triton - Symptonic/lijie/gap_webdataset_active/splits/hybrid_unise_v1_stream_80_10_10}"
FSD_GROUND_TRUTH="${FSD_GROUND_TRUTH:-$ROOT_DIR/tmp/dnf_phase_ab_v1/FSD50K.ground_truth.zip}"
OLD_SCORE_DIR="${OLD_SCORE_DIR:-$OUTPUT_ROOT/phase-b-v1-20260719/scores}"
TRAIN_SHARD_MANIFEST="${TRAIN_SHARD_MANIFEST:-$SPLIT_ROOT/train/clean_shards.jsonl}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ -e "$AUDIT_DIR" ]]; then
    echo "Refusing to overwrite versioned Phase-B audit: $AUDIT_DIR" >&2
    exit 2
  fi
  sbatch \
    --job-name="dnf-phase-b-finalize" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEMORY" \
    --time="$TIME_LIMIT" \
    --output="$LOG_DIR/slurm_%j.out" \
    --error="$LOG_DIR/slurm_%j.err" \
    --export=ALL \
    "$ROOT_DIR/scripts/slurm_finalize_dnf_phase_b_audit.sh"
  exit 0
fi

if [[ -e "$AUDIT_DIR" ]]; then
  echo "Refusing to overwrite versioned Phase-B audit: $AUDIT_DIR" >&2
  exit 2
fi

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
LIVE_LOG="$LOG_DIR/dnf_phase_b_finalize_${SLURM_JOB_ID}.log"
echo "event=phase_b_finalize_start job=${SLURM_JOB_ID} time=$(date -Is)" \
  | tee -a "$LIVE_LOG"

module load "$SOFTWARE_STACK_MODULE"
module load "$COMPILER_MODULE"
eval "$(conda shell.bash hook)"
conda activate semambapp

python -m pytest -q \
  tests/test_dnf_phase_b_audit.py \
  tests/test_dnf_phase_b_scoring.py \
  tests/test_dnf_phase_b_recalibrate.py \
  tests/test_dnf_phase_b_candidate_proposals.py \
  tests/test_dnf_phase_b_indoor_assets.py \
  tests/test_dnf_phase_b_speech_review.py \
  tests/test_dnf_phase_b_routing_contract.py \
  2>&1 | tee -a "$LIVE_LOG"

python scripts/audit_dnf_phase_b_sources.py \
  --split-root "$SPLIT_ROOT" \
  --fsd-ground-truth "$FSD_GROUND_TRUTH" \
  --output-dir "$AUDIT_DIR" \
  --seed "$SEED" \
  2>&1 | tee -a "$LIVE_LOG"

mkdir -p "$AUDIT_DIR/scores"
python scripts/recalibrate_dnf_phase_b_probe.py \
  --input-jsonl "$OLD_SCORE_DIR/mls_probe_scored.jsonl" \
  --probe-jsonl "$AUDIT_DIR/mls_clean_candidate_probe.jsonl" \
  --output-jsonl "$AUDIT_DIR/scores/mls_probe_scored_v2.jsonl" \
  --summary-json "$AUDIT_DIR/scores/mls_probe_score_summary_v2.json" \
  2>&1 | tee -a "$LIVE_LOG"

python scripts/recalibrate_dnf_phase_b_probe.py \
  --input-jsonl "$OLD_SCORE_DIR/libri_probe_scored.jsonl" \
  --probe-jsonl "$AUDIT_DIR/libri_clean_candidate_probe.jsonl" \
  --output-jsonl "$AUDIT_DIR/scores/libri_probe_scored_v2.jsonl" \
  --summary-json "$AUDIT_DIR/scores/libri_probe_score_summary_v2.json" \
  2>&1 | tee -a "$LIVE_LOG"

python scripts/build_dnf_phase_b_candidate_proposals.py \
  --mls-scored "$AUDIT_DIR/scores/mls_probe_scored_v2.jsonl" \
  --libri-scored "$AUDIT_DIR/scores/libri_probe_scored_v2.jsonl" \
  --train-shard-manifest "$TRAIN_SHARD_MANIFEST" \
  --output-dir "$AUDIT_DIR/candidate_proposals" \
  2>&1 | tee -a "$LIVE_LOG"

python scripts/audit_dnf_phase_b_indoor_assets.py \
  --noise-jsonl "$AUDIT_DIR/indoor_noise_candidates_pending.jsonl" \
  --rir-jsonl "$AUDIT_DIR/indoor_rir_candidates_pending.jsonl" \
  --output-dir "$AUDIT_DIR/indoor_asset_review" \
  --seed "$SEED" \
  2>&1 | tee -a "$LIVE_LOG"

python scripts/build_dnf_phase_b_speech_review_pack.py \
  --scored-jsonl \
    "$AUDIT_DIR/scores/mls_probe_scored_v2.jsonl" \
    "$AUDIT_DIR/scores/libri_probe_scored_v2.jsonl" \
  --output-dir "$AUDIT_DIR/speech_review" \
  --per-stratum 8 \
  --seed "$SEED" \
  2>&1 | tee -a "$LIVE_LOG"

cp "$ROOT_DIR/configs/train/dnf_phase_b_v2.json" \
  "$AUDIT_DIR/phase_b_config_snapshot.json"
cp "$ROOT_DIR/configs/train/dnf_source_routing_webdataset_v2_audit.json" \
  "$AUDIT_DIR/source_routing_contract_snapshot.json"
find "$AUDIT_DIR" -type f ! -name artifact_sha256.txt -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$AUDIT_DIR/artifact_sha256.txt"

echo "event=phase_b_finalize_complete job=${SLURM_JOB_ID} time=$(date -Is)" \
  | tee -a "$LIVE_LOG"
