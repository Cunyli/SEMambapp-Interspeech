# Research status

This page separates completed evidence from open hypotheses. Repository
structure and passing unit tests are not treated as model-quality results.

| Workstream | Verified evidence | Decision boundary |
|---|---|---|
| Direct differentiable HNR+slope formula | Slurm `19643154` completed; HNR failed calibration/gradient and slope passed 1/8 anti-shortcut cases | `NO_GO` for this direct formula |
| Shared dual head | Encoder and late-feature variants, speaker-disjoint labels, accuracy and gradient gates are implemented | No claim until the held-out Triton diagnostic completes |
| Frozen independent predictor | Waveform-to-six-component CNN, freeze/input-gradient test, enhanced-waveform stress test are implemented | No claim until prediction and anti-shortcut gates pass |
| CS/SV preservation | Exact six-component, gain and residual panels run separately for CS and SV | `B0_250 / S3_500 / S3_2000` remain a Pareto shortlist |
| Controlled SNR ladder | Jobs `19640413`, `19640420` and `19640638` completed | No universal SNR knee; use all six levels |

## AVQI backpropagation contract

- Six targets: CPPS, HNR, shimmer %, shimmer dB, LTAS slope and LTAS tilt.
- Jitter is excluded from the AVQI task.
- Pilot loss: HNR + LTAS slope; all six terms are still evaluated.
- Split: 70 train, 14 calibration and 14 holdout speakers.
- Promotion evidence: held-out accuracy, calibration, anti-shortcut behavior,
  finite intended gradients, and exact Praat scoring of generated waveforms.
- No generator optimizer step is part of the diagnostic.

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
