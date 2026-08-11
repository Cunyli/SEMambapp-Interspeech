# Research status

This page separates completed evidence from open hypotheses. Repository
structure and passing unit tests are not treated as model-quality results.

| Workstream | Verified evidence | Decision boundary |
|---|---|---|
| Direct differentiable HNR+slope formula | Slurm `19643154` completed; HNR failed calibration/gradient and slope passed 1/8 anti-shortcut cases | `NO_GO` for this direct formula |
| Shared dual head | Job `19684821`: late feature selected; anti-shortcut and gradient pass, but HNR and slope accuracy/calibration gates fail | `NO_GO_GENERATOR_TRAINING` |
| Frozen independent predictor | Job `19684821`: stronger held-out rank accuracy and valid frozen-input gradient, but HNR/slope calibration and external slope fail | `NO_GO_GENERATOR_TRAINING`; better candidate for future calibration work |
| CS/SV preservation | Exact six-component, gain and residual panels run separately for CS and SV | `B0_250 / S3_500 / S3_2000` remain a Pareto shortlist |
| Controlled SNR ladder | Jobs `19640413`, `19640420` and `19640638` completed | No universal SNR knee; use all six levels |

## AVQI backpropagation contract

- Six targets: CPPS, HNR, shimmer %, shimmer dB, LTAS slope and LTAS tilt.
- Jitter is excluded from the AVQI task.
- One six-output model uses a standardized, concept-balanced loss for all six
  terms; each member of the shimmer and LTAS pairs receives half weight.
- HNR and LTAS slope are the pre-registered primary promotion gates.
- Split: 70 train, 14 calibration and 14 holdout speakers.
- Promotion evidence: held-out accuracy, calibration, anti-shortcut behavior,
  finite intended gradients, and exact Praat scoring of generated waveforms.
- No generator optimizer step is part of the diagnostic.

The completed diagnostic used all 390 expected rows and exited `0:0` after
4 minutes 31 seconds on a V100. It performed zero generator optimizer steps.
See [the compact result](avqi-backprop-diagnostic.md).

## Pathology-preservation contract

The same speaker's clean CS or SV recording is the target. Healthy, mild and
severe pathology are reported separately. A candidate is rejected if it loses
clean identity or severe SV, even when its average AVQI or denoising score
improves.

The controlled SNR panel fixes the source, RIR, noise, offset and random seed,
then evaluates clean, RIR-only, 30, 20, 15 and 10 dB. It contains 1,440
enhanced waveforms and 2,664 exact-component rows.

## Evidence location

Large artifacts remain on Triton and are not GitHub links:

- `runs/tau_s1_sv_threshold_ablation_20260719_01/`
- `runs/tau_pathology_three_tracks_20260810_01/`
- `runs/avqi_component_backprop_20260811_01/`

A result is considered complete only when its run ID, source/checkpoint hashes,
metrics, completion receipt and fixed listening protocol agree.
