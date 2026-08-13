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

Jitter is diagnostic only, and the AVQI scalar itself is never minimized. The
loss candidate is always the two-sided gap to the same speaker's clean
pathological CS/SV target; “healthier-looking” output is not rewarded.

Three backpropagation mechanisms were tested on the same speaker-disjoint
train/calibration/holdout split:

| Route | Gradient path | Forms tested |
|---|---|---|
| Shared dual head | SeMamba++ shared features -> six-parameter head | global, frequency-aware, and compact TF-GridNet heads |
| Frozen predictor | enhanced waveform -> frozen predictor -> SeMamba++ | global CNN, frequency-aware CNN, compact TF-GridNet, and an 8-block pretrained full TF-GridNet |
| Direct frozen estimator | enhanced waveform -> differentiable PyTorch formulas -> SeMamba++ | soft/hard Praat-aligned six-term estimators; no trainable predictor weights |

The original bank had 390 usable CS/SV rows; only 278 belonged to training. We
added 55 speaker-disjoint pairs (30 healthy, 25 pathological) with independent
noise and RIR draws at 10/15/20 dB. This increased usable training rows to 498
(`+79%`) and the full bank to 610, while calibration, holdout, and the 24-speaker
external panel remained unchanged.

| Experiment | Main observation | Result |
|---|---|---|
| Shared dual head after expansion | calibration loss `0.141 -> 0.119`; CPPS/HNR improved, but clean-target stability and required external slices still fail | `0/6`, no-go |
| Frequency-aware CNN after expansion | best learned predictor, `0.083 -> 0.073`; HNR is strongest, but CS and severe-SV coverage is incomplete | `0/6`, no-go |
| Pretrained full TF-GridNet | internal CPPS/HNR/tilt pass, but calibration loss is worse (`0.095`) and all six outputs are too sensitive to a 100 ms circular shift | `0/6`, no-go |
| Shared late TF-Grid head, final locked test | gradients reach the shared backbone, but prediction accuracy and required external slices fail in all three seeds | `0/6` in `3/3` seeds |
| Direct frozen estimator v2 | the hard peak form beats the soft form on calibration (`0.0508` vs `0.1316`); HNR, Shimmer %, and LTAS tilt pass every component gate in all three seeds | `3/6`, bounded backprop candidate |

Every component must independently pass accuracy, calibration, paired change,
coverage, anti-shortcut tests, three-second transfer, finite input gradients,
and the required CS/SV, patient, severe-SV, and 10 dB external slices.

The promoted direct estimator was then tested by optimizing only a bounded
waveform residual, with HNR and LTAS tilt matched to the same speaker's clean
pathological target. Learning rate selection and final evaluation used disjoint
speakers. Exact Praat, rather than the surrogate, made the final decision:

| Exact result on 12 final CS/SV cases | Outcome |
|---|---|
| LTAS tilt | improved in `12/12`; median normalized gap reduction `0.123` — pass |
| HNR | improved in `10/12`; median normalized gap reduction `0.009 < 0.02` — fail |
| Waveform safety | worst residual about `-48.1 dB`, cosine `>0.99999`, no clipping, no median degradation of the other four terms — pass |

**Plain conclusion:** the independent direct estimator is currently more
reliable than the tested dual head or neural frozen predictors. Its LTAS-tilt
gradient also survives exact Praat verification, but HNR moves too weakly at a
safe perturbation level. The present decision is therefore
`NO_GO_AVQI_T2_TRAINING`: generator optimizer steps remain zero. This is a
bounded negative result for the current loss, not a claim that dual-head or
independent-predictor research can never work.

The locked receipts are on Triton under
`runs/avqi_component_direct_praat_v2_voicedmask_consensus_20260814_01/` and
`runs/avqi_direct_waveform_opt_balanced_hnr_tilt_final_20260814_01/`.

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
- exact-scored waveform backprop check: `scripts/evaluate_direct_avqi_waveform_optimization.py`
- local verification: `CUDA_VISIBLE_DEVICES='' python -m pytest -q`

Python 3.10 is the reference environment. Slurm launchers require
`CONFIRM_SLURM_SUBMIT=1`; repository tests verify engineering contracts, not
speech quality.
