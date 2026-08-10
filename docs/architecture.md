# Architecture

## Main SeMamba++ path

The baseline entry points are `train.py` and `infer.py`. Model code is organized
under `model/`, with `model/semambapp.py` providing the main SeMamba++ network
and `model/stfts.py` providing waveform-to-spectrum transformations.

The primary flow is:

```text
degraded waveform
  -> compressed magnitude and phase features
  -> SeMamba++ generator
  -> estimated magnitude and phase
  -> inverse STFT
  -> enhanced waveform
```

Training can also use discriminator and loss components in
`model/discriminator.py` and `model/loss.py`.

## Data paths

| Path | Role | Boundary |
|---|---|---|
| `dataloaders/use_simulation.py` | Fixed noisy/clean pairs | Manifests and audio remain external |
| `dataloaders/legacy_online_degradation.py` | Preserved online degradation route | Historical compatibility path |
| `dataloaders/dnf_controlled_webdataset.py` | Controlled additive noisy-target stream | External `DNF_USE` support is loaded only for the streaming route |
| `dataloaders/dnf_controlled_phase_a.py` | Frozen-manifest Phase A route | Contract-focused research path |

## Experimental extensions

The repository includes DNF objectives, controlled routing, active-RMS
variants, identity losses, and speaker-verification guardrails. These modules
are retained as research engineering work. Their implementation does not by
itself establish that the mechanism improves speech quality or pathology
preservation.

## Evaluation boundary

`metrics/evaluation.py` and the scripts under `scripts/` provide metric and
gate tooling. A metric pass, a saved configuration, or a checkpoint filename is
not treated as a final research conclusion without the corresponding data
provenance, complete evaluation panel, and listening evidence.
