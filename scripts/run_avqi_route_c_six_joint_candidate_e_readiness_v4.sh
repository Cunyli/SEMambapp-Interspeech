#!/bin/bash
# Interpret one passing Candidate-E external receipt. No exact scoring or training.
set -euo pipefail

if [[ -z "${RUNTIME_PYTHON:-}" ]]; then
  echo "RUNTIME_PYTHON must name the reviewed project Python executable" >&2
  exit 2
fi
if [[ "$RUNTIME_PYTHON" != /* || ! -x "$RUNTIME_PYTHON" ]]; then
  echo "RUNTIME_PYTHON must be an executable absolute path" >&2
  exit 2
fi

required_flags=(
  --contract
  --contract-sha256
  --promotion-report
  --promotion-report-sha256
  --promotion-receipt
  --promotion-receipt-sha256
  --speaker-ledger
  --speaker-ledger-sha256
  --source-root
  --source-commit
  --output-dir
)
for required_flag in "${required_flags[@]}"; do
  found=false
  for argument in "$@"; do
    if [[ "$argument" == "$required_flag" ]]; then
      found=true
      break
    fi
  done
  if [[ "$found" != true ]]; then
    echo "Missing required fail-closed argument: $required_flag" >&2
    exit 2
  fi
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
cd "$project_root"
exec "$RUNTIME_PYTHON" \
  -m scripts.audit_avqi_route_c_six_joint_candidate_e_readiness_v4 \
  "$@"
