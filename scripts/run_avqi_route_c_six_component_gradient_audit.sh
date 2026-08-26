#!/bin/bash
# Run one dev-only raw six-component gradient measurement. No submission/training.
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
  --source-root
  --source-commit
  --label-bank
  --label-bank-sha256
  --source-evidence
  --v19-evidence-manifest
  --v19-evidence-manifest-sha256
  --topology-manifest
  --topology-manifest-sha256
  --selection-salt
  --test-evidence
  --test-evidence-sha256
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
  -m scripts.evaluate_avqi_route_c_six_component_gradients \
  "$@"
