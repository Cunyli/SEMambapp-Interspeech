#!/bin/bash
set -euo pipefail

CKPT=/scratch/work/lil14/SEMambapp-Interspeech/exp/train_semambapp/ln_g_00180000.pth
IN_ROOT=/scratch/work/lil14/data/TAU_SD_degraded
OUT_ROOT=/scratch/work/lil14/data/TAU_SD_enhanced/semamba

mkdir -p "$OUT_ROOT"

echo "Checkpoint: $CKPT"
echo "Input root: $IN_ROOT"
echo "Output root: $OUT_ROOT"

find "$IN_ROOT" -type f -name "*.wav" | sort | while read -r in_file; do
  rel="${in_file#$IN_ROOT/}"
  out_file="$OUT_ROOT/$rel"
  mkdir -p "$(dirname "$out_file")"

  if [[ -f "$out_file" ]]; then
    echo "Skip existing: $rel"
    continue
  fi

  echo "Enhancing: $rel"

  python3 infer.py \
    --config config.yaml \
    --checkpoint "$CKPT" \
    --input "$in_file" \
    --output "$out_file" \
    --device cuda
done

echo "Done."
echo "Input wav count:"
find "$IN_ROOT" -type f -name "*.wav" | wc -l
echo "Output wav count:"
find "$OUT_ROOT" -type f -name "*.wav" | wc -l
