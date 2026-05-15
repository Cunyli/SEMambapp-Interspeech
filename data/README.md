# Data layout

SEMamba++ does not track dataset manifests or generated audio under `data/`.
Training and inference should point to fixed noisy/clean pair manifests exported by
the external `USE_simulation` bundle.

The active training configs expect these environment variables when using TAU
fixed pairs:

- `USE_SIMULATION_ROOT`
- `SEMAMBAPP_TAU_FIXED_TRAIN_CSV`
- `SEMAMBAPP_TAU_FIXED_VALID_CSV`

Old source-list JSON files from the repository-local simulation flow are no
longer part of the project layout. Build or export manifests in `USE_simulation`
or another external data workspace, and keep local generated files ignored under
`data/` only when needed for ad hoc experiments.
