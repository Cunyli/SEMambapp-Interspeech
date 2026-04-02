#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/Volumes/t412_symptosonic/databases}"
WHAM_DIR="$ROOT/WHAM"
ARNI_DIR="$ROOT/Arni"

mkdir -p "$WHAM_DIR/raw" "$WHAM_DIR/extracted" "$ARNI_DIR/raw" "$ARNI_DIR/extracted"

download() {
  local url="$1"
  local out="$2"
  echo "[download] $out"
  curl -L -C - --fail -o "$out" "$url"
}

echo "[1/4] Download WHAM!"
download \
  "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/high_res_wham.zip" \
  "$WHAM_DIR/raw/high_res_wham.zip"

echo "[2/4] Extract WHAM!"
unzip -n "$WHAM_DIR/raw/high_res_wham.zip" -d "$WHAM_DIR/extracted"

echo "[3/4] Download Arni"
arni_base="https://zenodo.org/records/6985104/files"
arni_files=(
  "Arni_layout.jpg"
  "Arni_panels_numbers.pdf"
  "combinations_setup.csv"
  "IR_Arni_upload_numClosed_0-5.zip"
  "IR_Arni_upload_numClosed_6-15.zip"
  "IR_Arni_upload_numClosed_16-25.zip"
  "IR_Arni_upload_numClosed_26-35.zip"
  "IR_Arni_upload_numClosed_36-45.zip"
  "IR_Arni_upload_numClosed_46-55.zip"
)

for name in "${arni_files[@]}"; do
  download "$arni_base/$name?download=1" "$ARNI_DIR/raw/$name"
done

echo "[4/4] Extract Arni"
for zip_file in "$ARNI_DIR/raw/"*.zip; do
  unzip -n "$zip_file" -d "$ARNI_DIR/extracted"
done

echo
echo "Done."
echo "WHAM extracted under: $WHAM_DIR/extracted"
echo "Arni extracted under: $ARNI_DIR/extracted"
