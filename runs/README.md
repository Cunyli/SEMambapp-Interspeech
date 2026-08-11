# Run records

`runs/<run_id>/` stores the record of an experiment: configuration snapshots,
manifests, metrics, reports, receipts, and run-specific generated outputs.

New model weights do not belong here. They are written to the corresponding
`checkpoints/<experiment>/<run_id>/` directory and referenced by path and hash
from the run record. Existing `runs/.../checkpoints/` folders on Triton are
legacy evidence and remain untouched until a verified migration is performed.
