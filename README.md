# SeMamba++: pathology-preserving speech enhancement

This project studies whether room noise can be removed **without erasing the
pathological voice characteristics of the same speaker**. The clean CS
(continuous speech) or SV (sustained vowel) recording from that speaker is the
reference. Pathology preservation is the primary objective; denoising is
evaluated only after that constraint is satisfied.

A lower AVQI score is not automatically better: it may mean that a pathological
voice was made artificially healthier.

## What we tested

| Workstream | Controlled comparison | Current conclusion |
|---|---|---|
| AVQI-component backpropagation | Shared dual head vs. frozen independent waveform predictor, using the same speaker split and budget | Both gradient paths work, but neither predictor is accurate and calibrated enough to use as an enhancement loss |
| CS and SV preservation | CS and SV are evaluated separately against the same speaker's clean recording | The complete test path is implemented; `B0_250`, `S3_500`, and `S3_2000` are a Pareto shortlist, not a single winner |
| Noise severity | Source, RIR, noise, offset, and seed stay fixed while SNR changes | Evaluate the full clean / RIR-only / 30 / 20 / 15 / 10 dB ladder; 10 dB is a stress case, not a universal breakpoint |

## 1. AVQI-component backpropagation

The verified AVQI v03.01 formula contains six terms:

$$
\mathrm{AVQI}=2.8902\,(4.152-0.177\,\mathrm{CPPS}-0.006\,\mathrm{HNR}
-0.037\,\mathrm{Shimmer}_{\%}+0.941\,\mathrm{Shimmer}_{dB}
+0.01\,\mathrm{LTAS\ slope}+0.093\,\mathrm{LTAS\ tilt}).
$$

Jitter is diagnostic-only and is not part of this formula. We do not minimize
the AVQI scalar. Instead, the six predicted components are matched to the same
speaker's clean pathological target:

$$
\mathcal L_{\mathrm{comp}}=
\frac{1}{\sum_k w_k}\sum_k w_k\,
\operatorname{SmoothL1}\!\left(
\frac{\hat c_k-c_k^{\mathrm{clean}}}{\sigma_k}\right).
$$

| Route | Prediction path | What receives the gradient | Result |
|---|---|---|---|
| Shared dual head | Late shared SeMamba++ feature → six-component head | Component head and shared backbone | `NO_GO`: intended gradient passes, but HNR and LTAS-slope accuracy/calibration gates fail |
| Frozen independent predictor | Enhanced waveform → log-STFT → small CNN | Predictor stays frozen; gradient reaches the complete enhancement model through its waveform input | `NO_GO`: more accurate than the shared head, but it compresses component changes and fails calibration |

The diagnostic used all 390 valid CS/SV rows from 98 speakers with a
speaker-disjoint 70/14/14 train/calibration/holdout split. The independent
predictor reached HNR and LTAS-slope rank correlations of `0.910` and `0.821`,
but their calibration slopes were only `0.748` and `0.645`; LTAS-slope
calibration fell to `0.515` on external enhanced audio. Job `19684821`
completed with zero generator optimizer steps.

**Plain conclusion:** the independent predictor is the better candidate, but
neither route is safe for formal enhancement training yet. Final waveforms must
still be checked with exact Praat measurements. See the
[full AVQI diagnostic](docs/avqi-backprop-diagnostic.md).

The earlier direct soft-HNR + soft-slope formula was also a `NO_GO`: HNR failed
calibration/gradient checks and slope passed only 1/8 anti-shortcut cases. This
rejects that formula, not every learned predictor.

## 2. CS/SV preservation and the SNR ladder

Each candidate is evaluated in this order:

```text
same-speaker clean identity
    -> severe pathological SV survival
    -> CS and SV component fidelity
    -> denoising and residual distortion
    -> fixed listening set
```

Results are separated by task (CS/SV), health group
(healthy/mild/severe pathology), and degradation level. The controlled audit
fixes all mixture factors except SNR and contains:

- 24 speakers × CS/SV × 6 degradation levels × 5 models;
- 1,440 enhanced waveforms and 2,664 exact-component rows;
- 15 pathological speakers: 7 mild and 8 severe;
- completed jobs `19640413`, `19640420`, and `19640638`.

All six exact AVQI components, coherent/RMS gain, residual distortion, and
ordinary denoising metrics are reported. The matched-budget CS+SV comparison
remains `INCONCLUSIVE_NON_PARETO`.

## Repository map and artifact policy

```text
model/             SeMamba++ and AVQI-component models
dataloaders/       fixed-pair data routes
configs/train/     experiment configurations and frozen contracts
scripts/           training, evaluation, and report entry points
scripts/cluster/   guarded Triton/Slurm launchers
tests/             CPU/static contract tests
docs/              architecture, provenance, and research status
checkpoints/       project-trained weights (ignored by Git)
pretrained/        external weights (ignored by Git)
runs/              run metadata, manifests, metrics, reports, and outputs
outputs/examples/  small allowlisted listening sets only
```

New checkpoints belong under `checkpoints/<experiment>/<run_id>/`, not under
`runs/`. Historical paths remain unchanged so their hashes and resume records
stay auditable. Datasets, bulk audio, checkpoints, logs, and private listening
mappings are not stored in Git. A shareable audio set is limited to 3–5 fixed
sample IDs with source and checkpoint hashes; pathological recordings remain
private until their sharing terms are confirmed.

## Start here

1. [Research status](docs/research-status.md)
2. [AVQI backpropagation diagnostic](docs/avqi-backprop-diagnostic.md)
3. [Architecture](docs/architecture.md)
4. [Provenance and sharing boundary](docs/provenance.md)
5. [Script guide](scripts/README.md)

## Local checks

Python 3.10 is the project baseline:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

Mamba installation follows the upstream
[SEMamba repository](https://github.com/RoyChao19477/SEMamba). Passing tests
verify engineering contracts; they do not by themselves prove speech quality.
Cluster launchers refuse Slurm submission unless `CONFIRM_SLURM_SUBMIT=1` is
set explicitly.
