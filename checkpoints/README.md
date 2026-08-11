# Checkpoints

This directory is reserved for checkpoints trained by this SeMamba++ project.
New training entry points write to `checkpoints/<experiment>/<run_id>/` and
record that path in the matching run metadata.

Downloaded or upstream model weights belong in `pretrained/`. Runtime
checkpoint files are ignored by Git.

Historical assets under `exp/` or `runs/.../checkpoints/` are legacy evidence.
Do not move them independently: their hashes, manifests, resume references and
checkpoint paths must be migrated together.
