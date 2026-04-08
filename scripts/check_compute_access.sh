#!/bin/bash
set -uo pipefail

NOISE_FILE="${NOISE_FILE:-/m/teamwork/t412_symptosonic/databases/DNS/noise/datasets_fullband.noise_fullband.audioset_003/datasets_fullband/noise_fullband/O8KGwB1UNXk.wav}"
SCRATCH_DIR="${SCRATCH_DIR:-/scratch/elec/t412-speechcom/symptonic-r2b}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-semambapp}"

echo "=== Host ==="
hostname
echo

echo "=== Identity ==="
whoami
id
echo

echo "=== Kerberos ==="
if command -v klist >/dev/null 2>&1; then
  klist -f || true
else
  echo "klist not found"
fi
echo

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV_NAME" || true
fi

echo "=== Python ==="
command -v python || true
python --version || true
echo

echo "=== Check Noise File Metadata ==="
ls -l "$NOISE_FILE" || true
echo

echo "=== Check Noise File Raw Read ==="
python - <<PY
path = r"""$NOISE_FILE"""
try:
    with open(path, "rb") as f:
        chunk = f.read(32)
    print("open_ok bytes_read=", len(chunk))
except Exception as exc:
    print("open_failed:", repr(exc))
PY
echo

echo "=== Check Noise File With librosa ==="
python - <<PY
path = r"""$NOISE_FILE"""
try:
    import librosa
    audio, sr = librosa.load(path, sr=16000)
    print("librosa_ok shape=", audio.shape, "sr=", sr)
except Exception as exc:
    print("librosa_failed:", repr(exc))
PY
echo

echo "=== Check Scratch Dir Metadata ==="
ls -ld "$SCRATCH_DIR" || true
df -h "$SCRATCH_DIR" || true
echo

echo "=== Check Scratch Dir Write ==="
TEST_FILE="$SCRATCH_DIR/.codex_access_test_${SLURM_JOB_ID:-manual}_$$"
if touch "$TEST_FILE"; then
  echo "touch_ok $TEST_FILE"
  rm -f "$TEST_FILE"
  echo "rm_ok"
else
  echo "touch_failed $TEST_FILE"
fi
