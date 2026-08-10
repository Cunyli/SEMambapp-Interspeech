#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch/work/lil14/SEMambapp-Interspeech}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
PARTITION="${PARTITION:-gpu-interactive}"
GPU_TYPE="${GPU_TYPE:-v100}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-}"
VALID_MANIFEST="${VALID_MANIFEST:-}"
PAPER_PAIR_ID="${PAPER_PAIR_ID:-phase-a-v3-paper-exact-seed1234}"
MATCHED_PAIR_ID="${MATCHED_PAIR_ID:-phase-a-v3-matched-scale-seed1234}"

if [[ -z "$TRAIN_MANIFEST" || -z "$VALID_MANIFEST" ]]; then
  echo "TRAIN_MANIFEST and VALID_MANIFEST are required." >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  sbatch \
    --job-name="dnf-phase-a-rem3" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=3 \
    --cpus-per-task=8 \
    --gres="gpu:${GPU_TYPE}:3" \
    --mem=192G \
    --time="$TIME_LIMIT" \
    --output="$LOG_DIR/slurm_%j.out" \
    --error="$LOG_DIR/slurm_%j.err" \
    --export=ALL \
    "$ROOT_DIR/scripts/slurm_semambapp_dnf_phase_a_remaining3.sh"
  exit 0
fi

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
LIVE_LOG="$LOG_DIR/dnf_phase_a_remaining3_${SLURM_JOB_ID}.log"
echo "event=phase_a_remaining3_start job=${SLURM_JOB_ID} time=$(date -Is)" \
  | tee -a "$LIVE_LOG"
echo "paper_pair=$PAPER_PAIR_ID matched_pair=$MATCHED_PAIR_ID" \
  | tee -a "$LIVE_LOG"
echo "scratch_only=true arms=paper_standard,paper_dnf,matched_dnf" \
  | tee -a "$LIVE_LOG"

pids=()
labels=()

launch_arm() {
  local pair_id="$1"
  local loss_variant="$2"
  local task_id="$3"
  local log_tag="$4"
  srun \
    --exclusive \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --gres="gpu:${GPU_TYPE}:1" \
    --mem=64G \
    --output="$LOG_DIR/slurm_${SLURM_JOB_ID}_${log_tag}.out" \
    --error="$LOG_DIR/slurm_${SLURM_JOB_ID}_${log_tag}.err" \
    env \
      PAIR_ID="$pair_id" \
      LOSS_VARIANT="$loss_variant" \
      TRAIN_MANIFEST="$TRAIN_MANIFEST" \
      VALID_MANIFEST="$VALID_MANIFEST" \
      SLURM_ARRAY_TASK_ID="$task_id" \
      SLURM_ARRAY_JOB_ID="${SLURM_JOB_ID}_${log_tag}" \
      bash "$ROOT_DIR/scripts/slurm_semambapp_dnf_phase_a_array.sh" &
  pids+=("$!")
  labels+=("$log_tag")
}

launch_arm "$PAPER_PAIR_ID" paper_exact 0 paper_standard
launch_arm "$PAPER_PAIR_ID" paper_exact 1 paper_dnf
launch_arm "$MATCHED_PAIR_ID" matched_scale 1 matched_dnf

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "event=phase_a_remaining3_arm_failed arm=${labels[$index]}" \
      | tee -a "$LIVE_LOG"
    failed=1
  fi
done

if [[ "$failed" != "0" ]]; then
  exit 1
fi

echo "event=phase_a_remaining3_complete job=${SLURM_JOB_ID} time=$(date -Is)" \
  | tee -a "$LIVE_LOG"
