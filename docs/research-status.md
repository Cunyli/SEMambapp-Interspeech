# Research status

This page separates repository facts from research conclusions. The project is
still active, so this is a review boundary rather than a final result report.

## Verified repository facts

- The main SeMamba++ training and inference implementation is present.
- Fixed-pair USE simulation adapters and metric helpers are present.
- Identity and speaker-verification guardrail code and configurations are
  present.
- Unit and contract tests cover many deterministic data and objective rules.

## Claims not established by the tree

- A saved experiment configuration does not prove that the experiment ran.
- A checkpoint path does not prove that the checkpoint exists in a shared copy.
- A passing engineering test does not prove convergence or perceptual quality.
- A single metric or gate does not select a final model.
- The guardrail variants are not presented here as successful final methods.

## Current review boundary

The repository is organized to preserve active research and make its boundaries
visible. No normalization step launches training, submits Slurm work, performs
GPU validation, or reproduces paper results. Future research conclusions should
point to immutable run IDs, checkpoint hashes, data manifests, complete metric
panels, and a fixed listening protocol.

The closed DNF reproduction is maintained separately in
`DNF-SeMambaPP-Reproduction`; its conclusions must not be inferred from this
repository's test or checkpoint state.
