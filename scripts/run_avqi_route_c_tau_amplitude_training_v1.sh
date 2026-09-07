#!/bin/bash
#SBATCH --job-name=avqi-tau-training-v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
set -euo pipefail
: "${SOURCE_ROOT:?}" "${SOURCE_COMMIT:?}" "${RUN_ROOT:?}" "${RUN_STAGE:?}"
: "${SOURCE_HASH_MANIFEST:?}" "${PROTOCOL_SHA256:?}" "${SLURM_JOB_ID:?}"
cd "$SOURCE_ROOT"
[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
sha256sum -c "$SOURCE_HASH_MANIFEST" > "$RUN_ROOT/logs/source_check_start.log"
[[ ! -e "$RUN_ROOT/outputs" ]]
module load triton/2025.1-gcc
module load gcc/13.3.0
RUNTIME_PYTHON=/scratch/work/lil14/.conda_envs/semambapp/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$SOURCE_ROOT" PYTHONUNBUFFERED=1
if [[ "$RUN_STAGE" == focused || "$RUN_STAGE" == full ]]; then
  mkdir "$RUN_ROOT/outputs"
  TEST_ARGS=(tests)
  if [[ "$RUN_STAGE" == focused ]]; then
    TEST_ARGS=(tests/test_waveform_output_safety.py tests/test_avqi_route_c_tau_amplitude_training_v1.py)
  fi
  COMMAND=("$RUNTIME_PYTHON" -m pytest -q "${TEST_ARGS[@]}" --junitxml="$RUN_ROOT/outputs/pytest.xml")
else
  ARGS=()
  [[ -z "${SAFETY_RECEIPT:-}" ]] || ARGS+=(--safety-receipt "$SAFETY_RECEIPT")
  [[ -z "${PREFLIGHT_RECEIPT:-}" ]] || ARGS+=(--preflight-receipt "$PREFLIGHT_RECEIPT")
  [[ -z "${TRAIN_RECEIPT:-}" ]] || ARGS+=(--train-receipt "$TRAIN_RECEIPT")
  [[ -z "${RESUME_RECEIPT:-}" ]] || ARGS+=(--resume-receipt "$RESUME_RECEIPT" --resume-receipt-sha256 "$RESUME_RECEIPT_SHA256")
  COMMAND=("$RUNTIME_PYTHON" -m scripts.avqi_route_c_tau_amplitude_training_v1 --stage "$RUN_STAGE"
    --protocol "$RUN_ROOT/inputs/protocol.json" --run-root "$RUN_ROOT" "${ARGS[@]}")
fi
set +e
"${COMMAND[@]}" 2>&1 | tee "$RUN_ROOT/logs/execution.log"
EXECUTION_STATUS="${PIPESTATUS[0]}"
set -e
sha256sum -c "$SOURCE_HASH_MANIFEST" > "$RUN_ROOT/logs/source_check_end.log"
"$RUNTIME_PYTHON" - "$RUN_ROOT" "$RUN_STAGE" "$EXECUTION_STATUS" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

run, stage, code = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
out = run / "outputs"
out.mkdir(exist_ok=True)
artifacts = []
for folder in ("outputs", "checkpoints"):
    for p in sorted((run / folder).rglob("*")):
        if p.is_file():
            artifacts.append({"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
steps = []
if (out / "training_steps.jsonl").exists():
    steps = [json.loads(s)["step"] for s in (out / "training_steps.jsonl").read_text().splitlines()]
restored = 0
if (out / "resume_receipt.json").exists():
    restored = json.loads((out / "resume_receipt.json").read_text())["restored_optimizer_step"]
counts = None
if (out / "pytest.xml").exists():
    suites = ET.parse(out / "pytest.xml").getroot().findall("testsuite")
    counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ("tests", "errors", "failures", "skipped")}
receipt = dict(stage=stage, exit_code=code, job_id=os.environ["SLURM_JOB_ID"],
    source_commit=os.environ["SOURCE_COMMIT"], protocol_sha256=os.environ["PROTOCOL_SHA256"],
    source_manifest_sha256=hashlib.sha256(Path(os.environ["SOURCE_HASH_MANIFEST"]).read_bytes()).hexdigest(),
    artifacts=artifacts, test_counts=counts, observed_local_optimizer_steps=max(steps, default=restored) - restored,
    restored_optimizer_step=restored,
    scientific_promotion=False, svd_used_for_new_testing=False)
with (out / "execution_receipt.json").open("x") as h:
    json.dump(receipt, h, indent=2, sort_keys=True, allow_nan=False)
    h.write("\n")
print(json.dumps({k:v for k,v in receipt.items() if k != "artifacts"}, sort_keys=True))
PY
exit "$EXECUTION_STATUS"
