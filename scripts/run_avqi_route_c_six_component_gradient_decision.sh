#!/bin/bash
# Apply the reviewed JSON-only six-gradient decision contract. No submission.
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
  --raw-report
  --raw-report-sha256
  --raw-receipt
  --raw-receipt-sha256
  --five-precedent-report
  --five-precedent-report-sha256
  --five-precedent-receipt
  --five-precedent-receipt-sha256
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
  -m scripts.decide_avqi_route_c_six_component_gradients \
  "$@"
