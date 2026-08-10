# Provenance and asset boundaries

## Code origin

The project builds on the public
[SEMamba](https://github.com/RoyChao19477/SEMamba) implementation and retains
ideas or components associated with BigVGAN and MP-SENet. The current tree also
contains independent SeMamba++ research extensions, controlled DNF experiments,
and advisor-review documentation.

Git history is the source of truth for the exact development lineage. This
repository should not be described as an official implementation of external
projects.

## External assets

The repository does not bundle datasets, downloaded pretrained weights, or
project-trained checkpoints. Keep the boundaries explicit:

| Location | Intended contents |
|---|---|
| `pretrained/` | Weights obtained from external projects or model providers |
| `checkpoints/` | Weights trained by this SeMamba++ project |
| external data roots | Speech, noise, room responses, pair manifests, and WebDataset shards |
| `outputs/` or `runs/<run_id>/outputs/` | Generated evaluation and listening artifacts |

Historical configs and cluster helpers may contain absolute Triton paths. Those
paths document the original environment; they do not make the referenced data
or weights part of this repository.

## Sharing boundary

Before sharing any dataset excerpt, pretrained asset, or generated listening
sample beyond internal review, verify its source terms and record its provenance
and generating checkpoint. Source code organization alone does not grant reuse
rights for external assets.
