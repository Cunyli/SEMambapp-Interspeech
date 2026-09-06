#!/bin/bash
#SBATCH --job-name=avqi-tau-audit
#SBATCH --partition=batch-bdw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -euo pipefail

: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
: "${SOURCE_HASH_MANIFEST:?SOURCE_HASH_MANIFEST is required}"
: "${SOURCE_HASH_MANIFEST_SHA256:?SOURCE_HASH_MANIFEST_SHA256 is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUN_MODE:?RUN_MODE is required}"
: "${SLURM_JOB_ID:?Use a Slurm compute allocation}"

OUTPUT_DIR="$RUN_ROOT/outputs"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "refusing to overwrite output: $OUTPUT_DIR" >&2
  exit 2
fi
if [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" != "$SOURCE_COMMIT" ]]; then
  echo "source commit differs" >&2
  exit 2
fi
if [[ "$(sha256sum "$SOURCE_HASH_MANIFEST" | cut -d ' ' -f1)" != "$SOURCE_HASH_MANIFEST_SHA256" ]]; then
  echo "source hash manifest differs" >&2
  exit 2
fi
cd "$SOURCE_ROOT"
sha256sum -c "$SOURCE_HASH_MANIFEST"

module load triton/2025.1-gcc
module load gcc/13.3.0
RUNTIME_PYTHON="/scratch/work/lil14/.conda_envs/semambapp/bin/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$SOURCE_ROOT"

echo "event=start job=$SLURM_JOB_ID mode=$RUN_MODE source_commit=$SOURCE_COMMIT"
if [[ "$RUN_MODE" == "audit" ]]; then
  : "${CONTRACT_SHA256:?CONTRACT_SHA256 is required}"
  "$RUNTIME_PYTHON" -m scripts.audit_avqi_route_c_tau_history_capacity_v1 \
    --contract "$SOURCE_ROOT/configs/avqi_route_c_tau_history_capacity_contract_v1.json" \
    --contract-sha256 "$CONTRACT_SHA256" \
    --source-root "$SOURCE_ROOT" \
    --source-commit "$SOURCE_COMMIT" \
    --output-dir "$OUTPUT_DIR"
elif [[ "$RUN_MODE" == "focused" || "$RUN_MODE" == "full" ]]; then
  mkdir -p "$OUTPUT_DIR"
  cp "$SOURCE_HASH_MANIFEST" "$OUTPUT_DIR/source_files.sha256"
  git status --porcelain=v1 --untracked-files=all > "$OUTPUT_DIR/source_status.txt"
  if [[ "$RUN_MODE" == "focused" ]]; then
    TEST_ARGS=(tests/test_avqi_route_c_tau_history_capacity_v1.py)
  else
    TEST_ARGS=(tests)
  fi
  set +e
  "$RUNTIME_PYTHON" -m pytest -q "${TEST_ARGS[@]}" \
    --junitxml="$OUTPUT_DIR/pytest.xml" 2>&1 | tee "$OUTPUT_DIR/pytest.log"
  TEST_STATUS="${PIPESTATUS[0]}"
  set -e
  "$RUNTIME_PYTHON" - "$OUTPUT_DIR" "$RUN_MODE" "$TEST_STATUS" "$SLURM_JOB_ID" "$SOURCE_COMMIT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

directory, mode, status, job_id, commit = sys.argv[1:]
output = Path(directory)
artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())}
xml_path = output / "pytest.xml"
counts = None
if xml_path.is_file():
    suites = ET.parse(xml_path).getroot().findall("testsuite")
    counts = {key: sum(int(s.attrib.get(key, "0")) for s in suites) for key in ("tests", "errors", "failures", "skipped")}
receipt = {
    "schema_version": "avqi-route-c-tau-history-capacity-tests-receipt-v1",
    "decision": "PASS_TAU_AUDIT_CODE_TESTS" if int(status) == 0 else "NO_GO_TAU_AUDIT_CODE_TESTS",
    "scope": mode, "pytest_exit_code": int(status), "test_counts": counts,
    "slurm_job_id": job_id, "base_commit": commit, "artifact_sha256": artifacts,
    "synthetic_test_fixtures_only": True, "real_svd_data_evaluated": False,
    "joint_panel_authorized": False, "generator_optimizer_steps": 0,
    "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
}
with (output / "completion_receipt.json").open("x", encoding="utf-8") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
  if [[ "$TEST_STATUS" != "0" ]]; then
    exit "$TEST_STATUS"
  fi
else
  echo "unsupported RUN_MODE: $RUN_MODE" >&2
  exit 2
fi

sha256sum -c "$SOURCE_HASH_MANIFEST"
echo "event=complete job=$SLURM_JOB_ID mode=$RUN_MODE"
