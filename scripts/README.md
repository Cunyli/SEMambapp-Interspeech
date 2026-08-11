# Script guide

Scripts are grouped by operational risk. Read the matching configuration or
contract before running any command that writes outputs.

## Inspection and preparation

- `summarize_tau_s1_sv_guardrail_ablation.py` summarizes prepared guardrail
  results; it does not establish promotion by itself.

These tools may write to an explicitly selected output directory, but they do
not submit cluster jobs.

DNF-specific training, comparison, audit, and listening scripts are maintained
in the separate `DNF-SeMambaPP-Reproduction` repository.

## Cluster-specific helpers

Cluster helpers are isolated under `cluster/`. They preserve historical
resource requests and machine-specific defaults. Any helper that can call
`sbatch` requires `CONFIRM_SLURM_SUBMIT=1` outside an existing allocation.

The root README intentionally contains no cluster submission example. Review
every input path, output path, checkpoint, and resource request before adapting
a cluster helper.
