# SeMamba++: pathology-preserving speech enhancement

This project extends SeMamba++ for real-room speech enhancement while keeping
the pathological voice characteristics of the same speaker. The same-speaker
clean continuous speech (CS) and sustained vowel (SV) are the references.
Denoising matters only after clean identity and severe-SV survival are safe.

A lower AVQI is therefore not automatically better: an enhancer can obtain a
healthier-looking score by erasing the pathology that we need to preserve.

## Experiment pipeline

### 1. Original pretraining

The base SeMamba++ generator and discriminator were trained for 180k steps with
online noise, reverberation, bandwidth limitation, clipping, and packet-loss
degradations. The target is a delay-aligned anechoic waveform rather than the
unshifted source.

- config: `configs/train/semambapp_shifted_anechoic_online_v1.yaml`
- retained pair: `checkpoints/pretrain_180k/`

### 2. Fixed-pair fine-tuning

The base model was adapted to fixed TAU phone-room pairs at a lower learning
rate. The main route uses batch 4 with four-step accumulation (effective batch
16). A separate 500-step plain fine-tune, `A_plainFT_500`, is retained as the
matched control with no identity rows and no SV-survival loss.

- main config: `configs/train/semambapp_tau_fixed_effective_bs16_v1.yaml`
- retained endpoints: `checkpoints/finetune_bs16_1999/` and
  `checkpoints/A_plainFT_500/`

### 3. B0, S6, and S3

All three arms use the same pretrained G/D, fold, seed, degraded exposure, and
training budget. Their only controlled difference is the SV-preservation term.

| Arm | Definition | Purpose |
|---|---|---|
| `B0` | clean SV identity rows; no extra survival loss | practical baseline |
| `S6` | `B0` plus a lag-robust signed coherent-gain floor at `-6 dB` | prevent large SV disappearance |
| `S3` | the same floor tightened to `-3 dB` | intervene earlier for SV fidelity |

For non-identity SV, the added one-sided loss is

$$
\mathcal L_{SV}=\lambda\,[\tau-g_{coh}(\hat y,y)]_+,
$$

with calibrated weights `0.098316` for S6 and `0.138937` for S3. CS does
not enter this specific loss; CS and SV are both tested at evaluation time.

The result is a trade-off, not one universal winner:

| Retained checkpoint | What it shows |
|---|---|
| `B0_250` | practical listening baseline |
| `S6_500` | explanatory anchor; it did not stably beat B0 or S3 |
| `S3_500` | lowest phone-room mean absolute AVQI gap: `1.221` |
| `S3_2000` | best clean-identity endpoint; severe-SV clean gain `-0.72 / -2.29 dB` mean/worst |

`S3_2000` is a preservation endpoint, not the best denoising endpoint.

## CS/SV and SNR evaluation

The controlled panel fixes source speech, RIR, noise, offset, and random seed;
only degradation strength changes:

`clean -> RIR only -> 30 -> 20 -> 15 -> 10 dB`.

The completed panel contains 24 speakers, separate CS/SV rows, five model
candidates, 1,440 enhanced waveforms, and 2,664 exact-component rows. Results
are sliced by CS/SV, healthy/mild/severe pathology, and degradation level.
Evaluation includes exact Praat components, coherent/RMS survival, residual
distortion, severe-SV failures, and ordinary denoising metrics. The current
matched-budget CS+SV comparison is `INCONCLUSIVE_NON_PARETO`; 10 dB is a
stress condition, not a universal SNR boundary.

## AVQI-component backpropagation

AVQI v03.01 uses six terms:

```math
\mathrm{AVQI}=2.8902\left(4.152-0.177\,\mathrm{CPPS}-0.006\,\mathrm{HNR}-0.037\,\mathrm{Shimmer}_{\%}+0.941\,\mathrm{Shimmer}_{\mathrm{dB}}+0.01\,\mathrm{LTAS}_{\mathrm{slope}}+0.093\,\mathrm{LTAS}_{\mathrm{tilt}}\right).
```

Jitter is diagnostic only, and the AVQI scalar itself is never minimized. Both
routes predict all six terms, with the same speaker's clean pathological CS/SV
recording as the preservation target. This stage tests whether the predictors
are trustworthy enough to become a loss; it does not update the enhancer.

| Route | Three forms screened | Locked form | Three-seed result |
|---|---|---|---|
| Shared dual head | global, frequency-aware, compact TF-GridNet late-feature heads | compact TF-GridNet | gradients and anti-shortcut checks pass, but clean-target stability and external calibration fail; `0/6` full-gate components in every seed |
| Frozen predictor | global-stat, frequency-aware CNN, compact TF-GridNet waveform models | frequency-aware CNN | stronger internal CPPS/HNR/LTAS-tilt prediction, but CS, severe-SV, patient, and 10 dB slices remain unstable; `0/6` full-gate components in every seed |

The screen used 390 valid CS/SV rows from 98 speakers with a disjoint 70/14/14
train/calibration/holdout split. Confirmation locked the chosen forms for three
new seeds. The external test used 24 unseen speakers and clean, RIR-only, 30,
20, 15, and 10 dB conditions. A component had to pass accuracy, calibration,
paired-change or clean-target stability, coverage, anti-shortcut, three-second
segment transfer, input-gradient, and every required external slice.

**Plain conclusion:** the frozen frequency-aware predictor is the more promising
of the two current routes, but neither route is reliable enough for AVQI-T2
backpropagation. The multi-seed decision is `NO_GO_AVQI_BACKPROP`: no component
was admitted, generator optimizer steps stayed at zero, and no formal pathology
training was submitted. This rejects the current predictor forms, not the
dual-head or independent-predictor ideas in principle. The hashed consensus is
under `runs/avqi_component_predictor_multiseed_20260813_01/outputs/` on Triton.

## Repository and retained artifacts

```text
model/             SeMamba++ and AVQI predictor modules
dataloaders/       online and fixed-pair data routes
configs/train/     pretrain and fine-tuning configurations
scripts/           training, inference, evaluation, and Slurm entry points
tests/             CPU/static contract tests
checkpoints/       retained project-trained weights
pretrained/        third-party evaluation assets only
runs/              logs, manifests, metrics, and reports; no model weights
outputs/examples/  one fixed four-sample advisor listening set
```

The curated checkpoint set used by current entry points is listed in
[`checkpoints/manifest.csv`](checkpoints/manifest.csv). Bulk historical runs
are excluded from the shared Git tree and kept as read-only provenance on
Triton.

The public GitHub repository does not contain participant audio. The advisor
pack has four fixed IDs and 24 mono 16 kHz files: noisy input, clean target, `B0_250`,
`S6_500`, `S3_500`, and `S3_2000` for each ID. Its filenames and hashes
are recorded in [`outputs/examples/manifest.csv`](outputs/examples/manifest.csv);
the audio itself is shared privately after checking the data-sharing boundary.

## Entry points

- pretrain/fine-tune: `train.py` and `scripts/slurm.sh`
- inference: `infer.py` or `TASK=infer scripts/slurm.sh`
- AVQI diagnostic: `scripts/evaluate_avqi_component_backprop.py`
- AVQI multi-seed consensus: `scripts/summarize_avqi_component_multiseed.py`
- local verification: `CUDA_VISIBLE_DEVICES='' python -m pytest -q`

Python 3.10 is the reference environment. Slurm launchers require
`CONFIRM_SLURM_SUBMIT=1`; repository tests verify engineering contracts, not
speech quality.
