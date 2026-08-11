# Configuration guide

The existing `train/` path is retained because active scripts and experiment
contracts refer to it. Files inside it fall into three practical groups.

## Baselines and readable entry points

| File | Purpose |
|---|---|
| `train/semambapp_default.yaml` | Preserved baseline model and training defaults |
| `train/semambapp_tau_fixed.yaml` | Main fixed-pair USE simulation entry point |
| `train/semambapp_shifted_anechoic_online_v1.yaml` | Historical online shifted-anechoic baseline |

## Experiment configurations

Files beginning with `semambapp_tau_` record fixed-pair, batch-size, identity,
speaker-verification, and guardrail experiments. Several preserve absolute
Triton paths so the original intent remains auditable. Adapt those paths before
reuse. A configuration file does not prove that a run completed or passed its
evaluation gates.

The inherited configuration key `pretrained_generator_checkpoint` can also mean
"initialize this run from an earlier project checkpoint." Such files still
belong under `checkpoints/`; only weights obtained outside this project belong
under `pretrained/`.

Historical TAU configurations still point to `exp/` when that is the verified
location of the corresponding local run and initialization checkpoints. Keep
those paths intact until the artifacts and their manifests are migrated
together. New runs should use `checkpoints/`.

`semambapp_legacy_pretrain_main_v1.yaml` is retained for historical comparison
and is not the recommended starting point.

Generated configurations belong in `configs/generated/` and are ignored by
Git. Datasets, checkpoints, and pretrained weights are not bundled with any
configuration in this directory.

DNF reproduction contracts are maintained in the separate
`DNF-SeMambaPP-Reproduction` repository and are intentionally not duplicated
here.
