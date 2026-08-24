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
| Direct frozen estimator v6 | the hard v2 baseline qualified HNR and LTAS tilt; the Praat pulse-path v6 repair also raises Shimmer % paired-delta Spearman from `0.5115` to `0.7950` and passes all three confirmation seeds | `3/6` scorer gate; bounded waveform evidence required per component |

Every component must independently pass accuracy, calibration, paired change,
coverage, anti-shortcut tests, three-second transfer, finite input gradients,
and the required CS/SV, patient, severe-SV, and 10 dB external slices.

The promoted direct estimator was then tested by optimizing only a bounded
waveform residual, with HNR and LTAS tilt matched to the same speaker's clean
pathological target. Learning rate selection and final evaluation used disjoint
speakers. Exact Praat, rather than the surrogate, made the final decision:

| Exact result on 12 final CS/SV cases | Outcome |
|---|---|
| LTAS tilt | improved in `12/12`; median normalized gap reduction `0.1244` — pass |
| HNR | improved in `11/12`; median normalized gap reduction `0.0127 < 0.02` — fail |
| Waveform safety | worst residual about `-48.1 dB`, cosine `>0.99999`, no clipping, no median degradation of the other four terms — pass |

The HNR formula was then checked once more without reopening that final panel.
A fixed Praat-timed raw cross-correlation candidate used a one-period window,
a quarter-period step, and the same 75--600 Hz, 0.03 silence, and 0.45 voicing
settings as the exact metric branch. Selection used only the 26-speaker
calibration split; evaluation used the 26-speaker holdout and a separate
12-speaker VCTK panel. Of 2,148 internal CS/SV task rows, 2,134 had exact
labels and 2,106 also had a valid same-view clean pair.

| Frozen HNR formula check | Linear-AC hard v2 | Raw-CC hard v3 | Gate result |
|---|---:|---:|---|
| Calibration normalized MSE | `0.04598` | `0.05485` | v3 is `19.3%` worse; v2 selected |
| Holdout Spearman / normalized MAE | `0.9779 / 0.1409` | `0.9150 / 0.2022` | both pass, v2 is stronger |
| Holdout paired-delta Spearman | `0.8406` | `0.7391` | both pass, v2 is stronger |
| VCTK condition calibration slopes | `0.855--1.019` | `0.640--0.707` | all four v3 condition slices fail |
| Input gradient | finite and nonzero | finite and nonzero | pass for both; not sufficient for promotion |

The raw-CC candidate is therefore not integrated into the full Route C screen,
and no second HNR waveform pilot is authorized. This negative comparison is
about the fixed v3 approximation, not about Praat's exact raw-CC
implementation.

Shimmer was then isolated on local historical SV waveforms. Exact Praat pulses
first showed that the asymmetric-Hann amplitude gradient was effective when
pulse topology was frozen. The deployable v6 formula subsequently reproduced
Praat's raw-AC candidate-strength/unvoiced path and sample-time correlation
geometry without receiving exact pulses. A bounded proxy-only line search was
rescored by exact Praat only after each waveform had been frozen:

| Historical SV-only v6 check | Calibration speakers | Holdout speakers |
|---|---:|---:|
| Material Shimmer % improvements | `10/11` | `5/5` |
| Median normalized Shimmer % gap reduction | `0.06337` | `0.05768` |
| Median normalized Shimmer dB gap reduction | `0.08365` | `0.05826` |
| Median internal pulses within 5 samples of exact | `0.9732` | `1.0000` |
| Identity / safety | unchanged; all gates pass | unchanged; all gates pass |

Two independent PCM24 runs produced identical reports, case metrics, and 28
waveform hashes. This result is
`PASS_DEPLOYABLE_PATH_HISTORICAL_PILOT_ONLY`: it authorizes a fresh frozen
speaker-disjoint CS/SV panel, not generator training. The historical panel has
only four speakers, is SV-only, and does not provide all-six-component or
external-speaker evidence.

The fresh Shimmer panel then froze six previously unused pathological speakers
before simulation: three for calibration and three for final evaluation, with
six CS/SV cases per split and balanced RIR-only, 20 dB, and 10 dB conditions.
Only calibration exact scores selected one global normalized step size; the six
final candidate waveforms were hash-sealed before their exact scores were
opened.

| Fresh Shimmer v6 result | Calibration | Final |
|---|---:|---:|
| Selected global step | `0.001` | frozen from calibration |
| Material cases | `6/6` | `5/6` |
| Exact Shimmer % improvement rate | `5/6` | `5/5` material cases |
| Median normalized Shimmer % gap reduction | `0.02508` | `0.03062` |
| Median normalized Shimmer dB gap reduction | `0.01275` | `0.02559` |
| Worst residual / minimum cosine / clipping | `-60.00 dB / 0.99999905 / 0` | `-60.00 dB / 0.99999917 / 0` |
| Full-band pathology and denoising guardrails | pass | pass |

The final CS, SV, mild, and severe slices all pass their frozen gates. One
non-material final-SV case started almost on target (exact Shimmer % gap
`0.0540`) and worsened to `0.2300`; it is excluded only by the predeclared
material-gap threshold, not hidden from the result table. All five material
cases improve. A second end-to-end run selects the same step, passes the same
gates, and obtains a final median reduction of `0.03062`; small GPU numerical
variation keeps it from being byte-identical (maximum decoded sample difference
about `2.27e-6`). An independent repeat on the originally sealed waveforms
recomputes all 108 exact values with maximum absolute difference `0.0`.

This result is `PASS_SHIMMER_FRESH_SPEAKER_PANEL`. **Plain conclusion:** LTAS
tilt and Shimmer % have now each passed a speaker-disjoint exact waveform
panel. HNR still fails its waveform-effect threshold even though its
predictor is accurate. The next scientific step is one combined, bounded
LTAS-tilt + Shimmer-% waveform pilot; the present decision remains
`NO_GO_AVQI_T2_TRAINING`: generator optimizer steps remain zero. This is a
component-level promotion, not evidence for putting a combined AVQI loss into
generator training, and not a claim that dual-head or independent-predictor
research can never work.

Shimmer dB was subsequently isolated from the Shimmer-% path. Pure-Torch pulse
locators did not retain exact noisy-output topology, while a disclosed
Candidate C refreshed each current output's exact Praat pulse topology once,
detached that discrete topology, and differentiated only the existing live
asymmetric-Hann amplitude and dB-ratio path. The fixed mechanism step was
`0.001`; it is a Praat-assisted straight-through metric branch, not a
pure-Torch estimator. Six new hash-ranked pathological speakers produced 12
speaker-disjoint CS/SV cases with balanced RIR-only, SNR20, and SNR10 recipes.
Candidates were sealed before exact base/candidate outcomes were opened.

| Candidate-C fresh Shimmer-dB result | Value | Gate |
|---|---:|---|
| Material exact improvements | `10/11` (`90.9%`) | pass (`>=80%`) |
| Median normalized exact gap reduction | `0.04378` | pass (`>=0.02`) |
| Required CS/SV, mild/severe, RIR/SNR20/SNR10 slices | `7/7` | pass |
| Gradient L2 range | `0.0583--26.8139` | pass |
| Exact output topology stability | `12/12` | pass |
| Full-band pathology / denoising / non-target / waveform safety | all pass | pass |
| Current-output topology refresh | median `239.19 ms`, max `823.31 ms` | **fail** (`<=500 ms`) |

Two CS refreshes took `823.31 ms` and `595.62 ms`; the other ten took
`138.57--364.92 ms`. Therefore all waveform-effect and non-runtime gates pass,
but the component decision is
`NO_GO_SHIMMER_DB_CANDIDATE_C_FRESH_PANEL` under the unchanged 500 ms runtime
contract. This is evidence that the exact-topology amplitude gradient is
directionally useful, not permission to hide its Praat dependency or promote
the current runtime implementation. Job `19906678` created the immutable seal;
after an unused clean-target topology assertion interrupted reporting, job
`19906781` finalized only those hash-bound waveforms without rerunning
simulation, generator inference, or candidate generation. The report and
results SHA256 values are
`32488e52070d5555172d241b79cde72d2f31cb8d8c9a1962f6e95d5c498e73ad`
and
`809fadcfb48311d910b64fd001d2d2925dbe85bc0265a6229114e9ad01185795`.

LTAS slope required no further formula development. On its frozen 24-speaker,
48-case external authority panel, all 192 selected exact rows were valid,
overall exact/candidate Spearman was `0.9658`, and the candidate/exact 3 kHz
low-pass standardized-distance ratio was `1.0501` (CS `1.0714`, SV `0.9740`).
The existing absolute `0.10` anti-shortcut threshold is not uniformly
consistent with exact Praat SV behavior; the formula is authority-aligned, but
changing production to the proposed exact-relative `[0.75,1.25]` contract
remains a separate gate-review decision and was not performed here. Its frozen
authority report SHA256 is
`01e8ecfc9997ce3c02c8ad51034167a9a74f9f070ee0912c5ee85c73265519ee`.

The authoritative training decision therefore remains
`NO_GO_AVQI_T2_TRAINING`, with generator optimizer steps equal to zero.

The current locked receipts are on Triton under
`runs/avqi_component_direct_c_v5_multiseed_20260817_01/`,
`runs/avqi_component_direct_c_v5_waveform_pilot_offset4_20260817_02/`, and
`runs/avqi_direct_hnr_raw_cc_v3_diagnostic_20260817_02/`. The HNR formula
report is SHA256
`626b90bc85a83cdab97669ee7f22e503b81ed1fe2dc107c53cb62bd9d38eca94`.
The deterministic local Shimmer receipts are
`runs/avqi_shimmer_internal_v6c_historical_20260821_03/` and
`runs/avqi_shimmer_internal_v6c_historical_20260821_04/`; both report files
have SHA256
`a3eee583c799753a52c4c8a298aecabec96c3dfcd9c11080574515539b8e962c`.
The fresh primary and numerical-repeat receipts are
`runs/avqi_route_c_shimmer_v6_fresh_panel_20260821_02/` and
`runs/avqi_route_c_shimmer_v6_fresh_panel_20260821_03_repeat/`. Their report
SHA256 values are respectively
`f40a8e16c0467c0d52654173a48c1b515eb68c223a2097557dc14eb1999350ac`
and
`b73c3bfbf52a7bb688fbf4a699a67fa9a29ea694ad30a4adac805c0e78da3258`.
The sealed exact-repeat audit is Slurm job `19858012`; its log SHA256 is
`1a826d3f5b4f3dee9c280a469c5365f6ee3d3050bf06248e8d96ffecb1f04c78`.

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
- fresh Shimmer exact panel: `scripts/evaluate_avqi_shimmer_fresh_panel.py`
- local verification: `CUDA_VISIBLE_DEVICES='' python -m pytest -q`

Python 3.10 is the reference environment. Slurm launchers require
`CONFIRM_SLURM_SUBMIT=1`; repository tests verify engineering contracts, not
speech quality.
