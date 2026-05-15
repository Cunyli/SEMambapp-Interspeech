# SEMambapp
(Submitted to Interspeech 2026) An official code repository for SEMamba++.


This repository provides the official codebase and resources for SEMamba++ as described in our research. This repository is currently anonymous and will remain so until the publication process is complete, after which it will be de-anonymized with full author and project details.
## Prerequisites
1. Install the dependencies.
```
pip install -r requirements.txt
```
2. For Mamba, we recommend installing through [SEMamba](https://github.com/RoyChao19477/SEMamba)'s implementation.

## Layout

- `model/`: SEMamba++ model code.
- `configs/train/`: human-maintained training configs.
- `dataloaders/use_simulation.py`: adapter from USE_simulation fixed pairs to SEMamba++ batches.
- `metrics/evaluation.py`: validation metric helpers used during training.
- `scripts/slurm.sh`: generic Slurm entry point for `train` and `infer`.
- `runs/wandb/`: local W&B run cache.
- `outputs/examples/`: local demo/example audio, including migrated raw and normalized demo files.
- `checkpoints/`, `logs/`, `outputs/`, `runs/`, `tmp/`, and `pretrained/`: local runtime artifacts ignored by git.

Local agent notes such as `AGENTS.md`, `CLAUDE.md`, and `docs/` are intentionally ignored. Repository conventions live in this README, `.gitignore`, configs, and scripts.

## Datasets

The default training flow points to fixed noisy/clean pair manifests exported by the external `USE_simulation` bundle. Use environment variables or config fields to select manifests:

```yaml
data_cfg:
  dataset_type: use_simulation_fixed
  use_simulation_root: ${USE_SIMULATION_ROOT:-../USE_simulation}
  train_pair_manifest: ${SEMAMBAPP_TAU_FIXED_TRAIN_CSV:-/path/to/tau/simulated/phone_room/train/paired.csv}
  valid_pair_manifest: ${SEMAMBAPP_TAU_FIXED_VALID_CSV:-/path/to/tau/simulated/phone_room/valid/paired.csv}
```

SEMamba++ does not contain its own degradation or simulation implementation.
Generate or update synthetic datasets in `USE_simulation`, then point this
repository at the exported manifests.

The repository-local `data/` directory is reserved for a README only. Generated
manifests, degraded validation audio, and old local source lists should stay
outside git, preferably under external data roots or ignored runtime folders.

## Slurm

Use generic task names and choose data through config/environment variables:

```bash
bash scripts/slurm.sh train
bash scripts/slurm.sh infer
```

Common overrides:

```bash
CONFIG_PATH=configs/train/semambapp_tau_fixed.yaml bash scripts/slurm.sh train
OUTPUT_DIR=/path/to/enhanced CKPT=/path/to/ln_g_00000100.pth bash scripts/slurm.sh infer
```

Inference intended for ABQI/ABQY-style evaluation writes:

```text
<output_dir>/wav/
<output_dir>/inf.scp
<output_dir>/ref.scp
```


## Link to datasets

1. Download [VCTK](https://datashare.ed.ac.uk/handle/10283/2950) for speech.
2. Download [DNS Challenge 2020](https://github.com/microsoft/DNS-Challenge) and [WHAM!](http://wham.whisper.ai/) for noise.
3. Download [Arni](https://github.com/AaltoAcousticsLab/aalto-datasets) and [DNS5](https://github.com/microsoft/DNS-Challenge) for reverberation.

Set `USE_SIMULATION_ROOT` to the directory that contains the external `data/` bundle when you submit training jobs.


## Notices

Pretrained models will be made publicly available upon completion of the publication process.

## References

SEMamba: [SEMamba](https://github.com/RoyChao19477/SEMamba)
BigVGAN: [BigVGAN](https://github.com/NVIDIA/BigVGAN)
MP-SENet: [MPSENet](https://github.com/yxlu-0102/MP-SENet)
