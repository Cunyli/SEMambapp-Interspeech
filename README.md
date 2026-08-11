# SeMamba++: pathological-voice-preserving speech enhancement

> **Advisor-review snapshot.** The project asks whether room noise can be
> removed without erasing the pathological voice characteristics of the same
> speaker. Pathology preservation is the primary objective; denoising is
> evaluated only after that constraint is checked.

## What this repository tests

| Workstream | How the experiment is controlled | Current conclusion |
|---|---|---|
| AVQI-component backpropagation | Same speaker-disjoint data and budget for a shared dual head and a frozen waveform predictor | The earlier hand-written surrogate failed. The two learned routes are implemented but are not declared successful before the held-out diagnostic finishes |
| CS and SV preservation | Continuous speech (CS) and sustained vowels (SV) are scored separately against the same speaker's clean recording | The full CS/SV test path is supported. Current checkpoints form a Pareto shortlist; there is no universal winner |
| Noise severity | The utterance, RIR, noise, offset and seed are fixed while SNR alone changes | The complete clean/RIR-only/30/20/15/10 dB ladder must be used; 10 dB is a stress slice, not a universal breakpoint |

The evaluation order is deliberately strict:

```text
same-speaker clean identity
    -> severe pathological SV survival
    -> CS and SV component fidelity
    -> denoising and residual distortion
    -> fixed listening panel
```

A lower AVQI score is **not automatically better**. It may mean that a
pathological voice was made artificially healthier.

## 1. AVQI-component backpropagation

The verified AVQI v03.01 implementation contains six terms:

$$
\mathrm{AVQI}=2.8902\,(4.152-0.177\,\mathrm{CPPS}-0.006\,\mathrm{HNR}
-0.037\,\mathrm{Shimmer}_{\%}+0.941\,\mathrm{Shimmer}_{dB}
+0.01\,\mathrm{Slope}+0.093\,\mathrm{Tilt}).
$$

Jitter is diagnostic-only and is not part of this formula. We do not minimize
the AVQI scalar. Instead, predicted components are matched to the same
speaker's clean pathological target:

$$
\mathcal L_{\mathrm{comp}}=
\frac{1}{\sum_k w_k}\sum_k w_k
\operatorname{SmoothL1}\!\left(
\frac{\hat c_k-c_k^{\mathrm{clean}}}{\sigma_k}\right).
$$

One six-output model keeps the comparison small. All six terms are trained and
reported; the two correlated shimmer terms and two correlated LTAS terms each
receive half weight. HNR and LTAS slope are the pre-registered primary gates.

| Route | Predictor input | Training label | Backpropagation path |
|---|---|---|---|
| Shared dual head | Encoder or late shared SeMamba++ feature | Same-speaker clean exact Praat components | Component loss updates the head and shared backbone |
| Frozen independent predictor | Enhanced waveform → log-STFT → small CNN | Exact Praat components of the predictor's input waveform | Predictor weights stay frozen; gradients pass through it to the enhanced waveform and generator |

Both routes use 390 valid CS/SV rows from 98 speakers with a 70/14/14
train/calibration/holdout speaker split. A route must pass held-out rank
correlation, normalized error, calibration, anti-shortcut and finite-gradient
checks before any small generator pilot is justified. Exact Praat measurements
of final output waveforms remain the deciding evidence.

The direct soft-HNR + soft-slope prototype (Slurm job `19643154`) was a
**NO-GO**: HNR failed calibration/gradient checks and slope passed only 1/8
anti-shortcut cases. This rejects that formula, not the two learned routes.

## 2. CS/SV preservation and SNR ladder

Every candidate is evaluated by task (CS/SV), health group
(healthy/mild/severe pathology), and degradation level. The panel reports all
six exact AVQI components, coherent and RMS gain, residual distortion, and
ordinary denoising metrics.

The controlled severity audit contains:

- 24 speakers × CS/SV × 6 degradation levels × 5 models;
- 1,440 enhanced waveforms and 2,664 exact-component rows;
- 15 pathological speakers: 7 mild and 8 severe;
- completed jobs `19640413`, `19640420`, and `19640638`.

The current shortlist is `B0_250`, `S3_500`, and `S3_2000`. It is a
Pareto shortlist, not a winner: longer training improves some SV-survival
measures but not every phone-room component. The matched-budget CS+SV result
remains `INCONCLUSIVE_NON_PARETO`.

## Artifact and sharing policy

```text
checkpoints/<run_id>/        project-trained weights
pretrained/<source>/         external weights
runs/<run_id>/               config snapshots, manifests, metrics, reports, outputs
outputs/examples/            3–5 allowlisted advisor-listening sample IDs
logs/                        runtime logs
tmp/                         disposable scratch files
```

New code writes checkpoints only under `checkpoints/`; `runs/` does not own
model weights. Historical `exp/` and `runs/.../checkpoints` paths remain
read-only until their hashes, manifests and resume references can be migrated
together.

The repository does not expose every generated waveform. A small listening set
must include a manifest recording sample ID, task, severity, condition,
checkpoint and file hash. Pathological recordings are not placed in a public
GitHub branch until their sharing terms are confirmed.

## Repository map

```text
model/             SeMamba++ and AVQI-component models
dataloaders/       fixed-pair and controlled data routes
configs/train/     frozen experiment contracts
scripts/           training, evaluation and report entry points
scripts/cluster/   guarded Triton/Slurm launchers
tests/             CPU/static contract tests
docs/              architecture, provenance and research status
```

Implementation entry points for the AVQI diagnostic are
`model/avqi_components.py`,
`scripts/evaluate_avqi_component_backprop.py`, and
`scripts/cluster/slurm_avqi_component_backprop_diagnostic.sh`. The launcher
contains no generator optimizer step and refuses Slurm submission unless
`CONFIRM_SLURM_SUBMIT=1` is explicitly set.

## Local checks

Python 3.10 is the project baseline:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

Mamba installation follows the upstream
[SEMamba repository](https://github.com/RoyChao19477/SEMamba). A passing code
test proves an engineering contract, not model quality. See
[research status](docs/research-status.md) and
[provenance](docs/provenance.md) for the evidence boundary.
