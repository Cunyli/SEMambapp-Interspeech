# Script guide

Scripts are grouped by operational risk. Read the matching configuration or
contract before running any command that writes outputs.

## Inspection and preparation

- `audit_dnf_*.py`, `score_dnf_*.py`, and `recalibrate_dnf_*.py` inspect or
  score prepared DNF evidence.
- `build_dnf_*.py` creates manifests, proposals, or bounded listening packs.
- `smoke_*.py` performs local data or tensor-flow checks.
- `summarize_tau_s1_sv_guardrail_ablation.py` summarizes prepared guardrail
  results; it does not establish promotion by itself.

These tools may write to an explicitly selected output directory, but they do
not submit cluster jobs.

Shareable examples are limited to 3--5 deterministic sample IDs as documented
in `outputs/README.md`. Larger stratified blind-review packs are research audit
artifacts and belong under a run directory; they must not be presented as the
small advisor-facing example set.

## Training and evaluation

- `train_semambapp_dnf_*.py` contains experimental DNF training entry points.
- `eval_semambapp_dnf_*.py` evaluates an explicitly supplied checkpoint.
- `compare_dnf_phase_a_*.py` compares prepared experiment artifacts.

Training scripts require external data and may require a separate `DNF_USE`
checkout. Set `DNF_USE_ROOT` instead of relying on a hard-coded repository
location.

## Cluster-specific helpers

All `slurm*.sh` files are isolated under `cluster/`. They preserve historical
resource requests and machine-specific defaults. Any helper that can call
`sbatch` requires `CONFIRM_SLURM_SUBMIT=1` outside an existing allocation.

The root README intentionally contains no cluster submission example. Review
every input path, output path, checkpoint, and resource request before adapting
a cluster helper.
