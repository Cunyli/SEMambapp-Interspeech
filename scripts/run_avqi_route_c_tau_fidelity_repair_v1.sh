#!/bin/bash
#SBATCH --job-name=avqi-tau-fidelity-repair
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
module load triton/2025.1-gcc
module load gcc/13.3.0
RUNTIME_PYTHON=/scratch/work/lil14/.conda_envs/semambapp/bin/python
export PYTHONPATH="$SOURCE_ROOT" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ "$RUN_STAGE" == train ]]; then
  "$RUNTIME_PYTHON" - "$RUN_ROOT" <<'PY'
import json
import os
from pathlib import Path
import sys

import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise ValueError("training requires exactly one allocated GPU")
properties = torch.cuda.get_device_properties(0)
minimum_gib = 30 if os.environ.get("RUN_ARM") == "aligned_joint" else 0
record = dict(name=properties.name, total_memory_bytes=properties.total_memory,
              required_minimum_gib=minimum_gib, cuda_version=torch.version.cuda,
              torch_version=torch.__version__,
              matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
              cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
(Path(sys.argv[1]) / "inputs/gpu_environment.json").write_text(json.dumps(record, indent=2) + "\n")
if properties.total_memory < minimum_gib * 1024 ** 3:
    raise ValueError("six-component CPPS training requires a 32 GB class GPU; no optimizer has started")
PY
fi
if [[ "$RUN_STAGE" == full ]]; then
  mkdir "$RUN_ROOT/outputs"
  COMMAND=("$RUNTIME_PYTHON" -m pytest -q tests --junitxml="$RUN_ROOT/outputs/pytest.xml")
else
  ARGS=()
  [[ -z "${RUN_ARM:-}" ]] || ARGS+=(--arm "$RUN_ARM")
  [[ -z "${ALIGNMENT_PATH:-}" ]] || ARGS+=(--alignment "$ALIGNMENT_PATH" --alignment-sha256 "$ALIGNMENT_SHA256")
  COMMAND=("$RUNTIME_PYTHON" -m scripts.avqi_route_c_tau_fidelity_repair_v1
    --protocol "$RUN_ROOT/inputs/protocol.json" --run-root "$RUN_ROOT" --stage "$RUN_STAGE" "${ARGS[@]}")
fi
set +e
"${COMMAND[@]}" 2>&1 | tee "$RUN_ROOT/logs/execution.log"
EXECUTION_STATUS="${PIPESTATUS[0]}"
set -e
sha256sum -c "$SOURCE_HASH_MANIFEST" > "$RUN_ROOT/logs/source_check_end.log"
"$RUNTIME_PYTHON" - "$RUN_ROOT" "$EXECUTION_STATUS" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

run = Path(sys.argv[1])
out = run / "outputs"
out.mkdir(exist_ok=True)
steps = []
if (out / "training_steps.jsonl").exists():
    steps = [json.loads(line)["step"] for line in (out / "training_steps.jsonl").read_text().splitlines()]
counts = None
if (out / "pytest.xml").exists():
    suites = ET.parse(out / "pytest.xml").getroot().findall("testsuite")
    counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ("tests", "errors", "failures", "skipped")}
artifacts = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for folder in ("outputs", "checkpoints") for p in sorted((run / folder).rglob("*")) if p.is_file()]
receipt = dict(job_id=os.environ["SLURM_JOB_ID"], stage=os.environ["RUN_STAGE"],
               arm=os.environ.get("RUN_ARM"), exit_code=int(sys.argv[2]),
               source_commit=os.environ["SOURCE_COMMIT"], protocol_sha256=os.environ["PROTOCOL_SHA256"],
               observed_optimizer_steps=max(steps, default=0), test_counts=counts, artifacts=artifacts,
               scientific_promotion=False, evaluation_reserve_used_for_training=False)
(out / "execution_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
print(json.dumps({k: v for k, v in receipt.items() if k != "artifacts"}))
PY
exit "$EXECUTION_STATUS"
