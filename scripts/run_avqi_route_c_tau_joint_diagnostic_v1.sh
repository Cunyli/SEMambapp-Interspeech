#!/bin/bash
#SBATCH --job-name=avqi-tau-joint-diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00

set -euo pipefail
: "${SOURCE_ROOT:?}"
: "${SOURCE_COMMIT:?}"
: "${SOURCE_HASH_MANIFEST:?}"
: "${SOURCE_HASH_MANIFEST_SHA256:?}"
: "${RUN_ROOT:?}"
: "${RUN_STAGE:?}"
: "${SLURM_JOB_ID:?Use a Slurm compute allocation}"

cd "$SOURCE_ROOT"
if [[ "$(git rev-parse HEAD)" != "$SOURCE_COMMIT" ]]; then
  echo "source commit differs" >&2
  exit 2
fi
if [[ "$(sha256sum "$SOURCE_HASH_MANIFEST" | cut -d ' ' -f1)" != "$SOURCE_HASH_MANIFEST_SHA256" ]]; then
  echo "source file manifest differs" >&2
  exit 2
fi
sha256sum -c "$SOURCE_HASH_MANIFEST"
OUTPUT_DIR="$RUN_ROOT/outputs"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "refusing to overwrite run output: $OUTPUT_DIR" >&2
  exit 2
fi

module load triton/2025.1-gcc
module load gcc/13.3.0
RUNTIME_PYTHON="/scratch/work/lil14/.conda_envs/semambapp/bin/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$SOURCE_ROOT"
export PYTHONUNBUFFERED=1
echo "event=start stage=$RUN_STAGE job=$SLURM_JOB_ID source=$SOURCE_COMMIT"

if [[ "$RUN_STAGE" == "focused" || "$RUN_STAGE" == "full" ]]; then
  mkdir "$OUTPUT_DIR"
  if [[ "$RUN_STAGE" == "focused" ]]; then
    TEST_ARGS=(tests/test_avqi_route_c_tau_joint_diagnostic_v1.py)
  else
    TEST_ARGS=(tests)
  fi
  COMMAND=("$RUNTIME_PYTHON" -m pytest -q "${TEST_ARGS[@]}" --junitxml="$OUTPUT_DIR/pytest.xml")
else
  : "${CONTRACT_SHA256:?}"
  EXTRA_ARGS=()
  if [[ "$RUN_STAGE" != "seal" ]]; then
    : "${DEPENDENCY_RECEIPT:?}"
    : "${DEPENDENCY_RECEIPT_SHA256:?}"
    EXTRA_ARGS=(--dependency-receipt "$DEPENDENCY_RECEIPT" --dependency-receipt-sha256 "$DEPENDENCY_RECEIPT_SHA256")
  fi
  COMMAND=("$RUNTIME_PYTHON" -m scripts.avqi_route_c_tau_joint_diagnostic_v1
    --stage "$RUN_STAGE" --contract "$SOURCE_ROOT/configs/avqi_route_c_tau_joint_diagnostic_contract_v1.json"
    --contract-sha256 "$CONTRACT_SHA256" --source-commit "$SOURCE_COMMIT"
    --output-dir "$OUTPUT_DIR" --device "${RUN_DEVICE:-cuda}" "${EXTRA_ARGS[@]}")
fi
set +e
"${COMMAND[@]}" 2>&1 | tee "$RUN_ROOT/logs/execution.log"
EXECUTION_STATUS="${PIPESTATUS[0]}"
set -e
sha256sum -c "$SOURCE_HASH_MANIFEST"
"$RUNTIME_PYTHON" - "$RUN_ROOT" "$RUN_STAGE" "$EXECUTION_STATUS" "$SLURM_JOB_ID" "$SOURCE_COMMIT" "$SOURCE_HASH_MANIFEST" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

run, stage, status, job_id, commit, manifest_path = sys.argv[1:]
root = Path(run)
output = root / "outputs"
output.mkdir(exist_ok=True)
artifacts = {}
for path in sorted(output.rglob("*")):
    if path.is_file():
        artifacts[str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
test_counts = None
if (output / "pytest.xml").is_file():
    suites = ET.parse(output / "pytest.xml").getroot().findall("testsuite")
    test_counts = {key: sum(int(s.attrib.get(key, "0")) for s in suites) for key in ("tests", "errors", "failures", "skipped")}
log = root / "logs/execution.log"
result = {
    "schema_version": "avqi-route-c-tau-joint-diagnostic-execution-receipt-v1",
    "stage": stage, "execution_exit_code": int(status), "slurm_job_id": job_id,
    "source_commit": commit, "source_files_manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
    "artifact_sha256": artifacts, "test_counts": test_counts,
    "execution_log": {"path": str(log), "sha256": hashlib.sha256(log.read_bytes()).hexdigest()},
    "synthetic_tests_only": stage in {"focused", "full"},
    "real_svd_data_evaluated": False, "generator_optimizer_steps": 0,
    "authoritative_training_decision": "NO_GO_AVQI_T2_TRAINING",
}
with (output / "execution_receipt.json").open("x", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(result, sort_keys=True))
PY
echo "event=complete stage=$RUN_STAGE job=$SLURM_JOB_ID exit=$EXECUTION_STATUS"
exit "$EXECUTION_STATUS"
