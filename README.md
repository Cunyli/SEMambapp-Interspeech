# SeMamba++ Research Repository

> **Status:** active research code organized for advisor review. The repository
> records implemented systems and experiment intent; it does not claim that
> every saved configuration was run or that a final model has been selected.

This repository contains the SeMamba++ speech-enhancement implementation,
fixed-pair data paths, experiment configurations, evaluation tools, and
contract tests. The goal of this tree is to make the general SeMamba++ research
easy to inspect while preserving the full Git history.

## Project at a glance

| Area | Current boundary |
|---|---|
| SeMamba++ model and training path | Implemented |
| Fixed-pair USE simulation adapter | Implemented; data remains external |
| Identity and speaker-verification guardrails | Active research code and configs |
| DNF reproduction | Maintained in the separate `DNF-SeMambaPP-Reproduction` repository |
| Project-trained checkpoints | Stored locally under `checkpoints/`; not tracked |
| Downloaded model weights | Stored locally under `pretrained/`; not tracked |
| Final model or paper-level result | Not established by repository structure alone |

## Start here

For a structured review, read:

1. [Architecture](docs/architecture.md)
2. [Research status](docs/research-status.md)
3. [Provenance and asset boundaries](docs/provenance.md)
4. [Configuration guide](configs/README.md)
5. [Script guide](scripts/README.md)

## Repository layout

```text
configs/train/              # maintained baselines, experiment configs, and contracts
dataloaders/                # fixed-pair and controlled research data routes
docs/                       # architecture, status, and provenance
metrics/                    # validation metric helpers
model/                      # SeMamba++ and experimental model components
scripts/                    # data preparation, evaluation, and training entry points
scripts/cluster/            # cluster-specific helpers; may submit Slurm jobs
tests/                      # CPU/static engineering contract tests
checkpoints/                # project-trained checkpoints; ignored except README
pretrained/                 # externally obtained weights; ignored except README
logs/ outputs/ runs/ tmp/   # ignored runtime artifacts
train.py / infer.py         # baseline training and single-file inference entry points
```

## Local setup

Python 3.10 is the original project baseline. Install runtime dependencies with:

```bash
python -m pip install -r requirements.txt
```

Install the test dependency and run the CPU/static suite with:

```bash
python -m pip install -r requirements-dev.txt
CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

Mamba-specific installation may require the environment guidance from the
[SEMamba project](https://github.com/RoyChao19477/SEMamba). A passing unit test
suite establishes engineering contracts only; it does not establish training
convergence or speech quality.

## Data boundary

Datasets and generated manifests are not stored in Git. The main fixed-pair
training route expects manifests exported by the separate `USE_simulation`
project. Configure their locations in YAML or with environment variables:

```yaml
data_cfg:
  dataset_type: use_simulation_fixed
  use_simulation_root: ${USE_SIMULATION_ROOT:-../USE_simulation}
  train_pair_manifest: ${SEMAMBAPP_TAU_FIXED_TRAIN_CSV:-/path/to/train/paired.csv}
  valid_pair_manifest: ${SEMAMBAPP_TAU_FIXED_VALID_CSV:-/path/to/valid/paired.csv}
```

The original experiments used VCTK and EARS speech, DNS/WHAM-style noise, and
external room-response resources. See [Provenance](docs/provenance.md) before
reusing any external asset.

## Related repository boundary

The closed DNF-on-SeMamba++ reproduction is maintained separately in
`DNF-SeMambaPP-Reproduction`. Its DNF objectives, checkpoints, metrics, and
listening samples are not part of this repository. The earlier `DNF_USE`
experiment is also a separate project. Shared SeMamba++ backbone ancestry does
not make these repositories the same study.

## Checkpoint and output policy

- Put checkpoints trained by this project in `checkpoints/<run_id>/`.
- Put downloaded or upstream weights in `pretrained/<source>/`.
- Put run records in `runs/<run_id>/` and generated audio in
  `runs/<run_id>/outputs/` or `outputs/`.
- A shareable listening set should contain 3--5 deterministic sample IDs and a
  manifest that records the config and checkpoint used to create them.
- Legacy configs may retain absolute Triton paths as experiment history. Their
  presence is not evidence that a run completed successfully.

## Operational safety

Cluster helpers are isolated under `scripts/cluster/`. A helper that can call
`sbatch` refuses submission unless `CONFIRM_SLURM_SUBMIT=1` is set explicitly.
Review all paths, resources, and output locations before enabling that gate.
No root documentation command submits a job or starts training.

## Reference implementations

- [SEMamba](https://github.com/RoyChao19477/SEMamba)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [MP-SENet](https://github.com/yxlu-0102/MP-SENet)
