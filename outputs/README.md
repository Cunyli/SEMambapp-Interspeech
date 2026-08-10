# Outputs and listening samples

Generated evaluation and inference outputs belong here. A stable listening set
should use 3--5 deterministic sample IDs under `outputs/examples/` and include
a manifest with the degraded, reference, and enhanced paths plus the config,
checkpoint, and generation date.

Large experiment-specific output sets belong in `runs/<run_id>/outputs/`.
Generated audio and result files are ignored by Git unless a later review
explicitly selects a small evidence set.
