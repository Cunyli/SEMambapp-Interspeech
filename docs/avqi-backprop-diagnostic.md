# AVQI-component backpropagation diagnostic

## Plain conclusion

The frozen independent predictor is more accurate than the shared dual head,
but **neither route is reliable enough to train SeMamba++ yet**. Both routes
produce the intended gradients and reject simple shortcuts; both fail the
pre-registered HNR + LTAS-slope accuracy/calibration gate. No generator update
was run.

## Fixed experiment

- Slurm job: `19684821` (`COMPLETED`, exit `0:0`, 4 min 31 s, V100 16 GB)
- Source commit: `998fdf69e6e4779374b4ed9ff24fe4d3ffbae9bd`
- Coverage: 390/390 usable CS/SV rows, 98 speakers
- Speaker split: 70 train / 14 calibration / 14 holdout
- Output: one six-component model per route; correlated shimmer and LTAS terms
  receive half weight
- Primary promotion components: HNR and LTAS slope
- Generator optimizer steps: 0

The shared route compared encoder and late shared features; the late feature
won by calibration loss. The independent route used one small log-STFT CNN,
trained on exact Praat labels and then frozen before the input-gradient test.

## Speaker-disjoint holdout

Each cell reports `Spearman rank / calibration slope / normalized MAE`.
The gates are at least `0.70 / 0.75--1.25 / at most 0.50`; the independent
route also passed its paired-delta rank checks.

| AVQI term | Shared late feature | Frozen independent predictor |
|---|---:|---:|
| CPPS | `0.871 / 0.776 / 0.427` PASS | `0.943 / 0.799 / 0.258` PASS |
| HNR | `0.769 / 0.630 / 0.489` FAIL | `0.910 / 0.748 / 0.307` FAIL |
| Shimmer % | `0.736 / 0.503 / 0.483` FAIL | `0.841 / 0.678 / 0.443` FAIL |
| Shimmer dB | `0.741 / 0.554 / 0.487` FAIL | `0.857 / 0.696 / 0.418` FAIL |
| LTAS slope | `0.654 / 0.403 / 0.552` FAIL | `0.821 / 0.645 / 0.405` FAIL |
| LTAS tilt | `0.298 / 0.077 / 0.770` FAIL | `0.807 / 0.592 / 0.413` FAIL |

All six component-level anti-shortcut checks passed for both routes. The shared
head sent finite gradients to the shared backbone and none to the decoders, as
designed. The frozen independent predictor sent finite gradients through the
enhanced waveform to both backbone and decoders while its own weights remained
gradient-free.

On 288 external enhanced-waveform rows, the independent predictor passed HNR
but failed LTAS slope because its calibration slope was `0.515`. This confirms
the main failure mode: it preserves ordering better than it preserves the size
of component changes.

## Decision

`NO_GO_GENERATOR_TRAINING` for both routes. The result does not prove that the
ideas are invalid; it proves that these compact first implementations are not
yet trustworthy losses. If this line is reopened, the smallest defensible next
experiment is calibration-focused work on the independent predictor, followed
by a fresh speaker-disjoint confirmation set. The existing holdout thresholds
must not be relaxed after seeing this result.

Evidence hashes:

- experiment contract: `730466e09ba750e98caf031cf3a1b78470076f829a8b40668157714542ca866f`
- full diagnostic report: `b91c7a1973a33b0df00be6b5fd2ea38e3138062c564211b25d5bb6e78350aa8d`
- compact summary: `ae0cdacbd7a38705c13aed5a806205cc38d4957b74e6627c85788b07136b750b`
- completion receipt: `fc1b08a160fe37ccf5d093833143f33f5109fd9d5fc9dc0c17d31bcb550b7fa7`
  (records three predictor/head hashes and
  `formal_pathology_training_submitted=false`)
