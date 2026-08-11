# Outputs and listening samples

Generated evaluation and inference outputs belong here. A shareable listening
set is an allowlist of 3--5 deterministic sample IDs under
`outputs/examples/`; it is not a copy of every generated waveform.

Large experiment-specific output sets belong in `runs/<run_id>/outputs/`.
Generated audio and result files are ignored by Git unless a later review
explicitly selects a small evidence set.

Each selected item must have a manifest entry with its task, pathology group,
degradation, model, checkpoint hash, audio hash, source terms, and selection
reason. See [`examples/README.md`](examples/README.md).
