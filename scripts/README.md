# Script guide

Scripts are grouped by operational risk. Read the matching configuration or
contract before running any command that writes outputs.

## Inspection and evaluation

- `summarize_tau_s1_sv_guardrail_ablation.py` summarizes prepared guardrail
  results; it does not establish promotion by itself.
- `evaluate_avqi_component_backprop.py` compares the shared dual head and
  frozen waveform predictor without taking a generator optimizer step.

These tools write only to explicitly selected output and checkpoint locations;
they do not submit cluster jobs. Shareable examples are limited to 3–5 fixed
sample IDs as documented in `outputs/README.md`. Larger evaluation and blind
listening sets remain run artifacts and are not published with the repository.

DNF-specific training, comparison, audit, and listening scripts are maintained
in the separate `DNF-SeMambaPP-Reproduction` repository.

## Cluster-specific helpers

Cluster helpers are isolated under `cluster/`. They preserve historical
resource requests and machine-specific defaults. Any helper that can call
`sbatch` requires `CONFIRM_SLURM_SUBMIT=1` outside an existing allocation.

The root README intentionally contains no cluster submission example. Review
every input path, output path, checkpoint, and resource request before adapting
a cluster helper.
