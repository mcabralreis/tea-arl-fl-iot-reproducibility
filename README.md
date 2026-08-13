# TEA-FL and ARL-FL reproducibility package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21923444.svg)](https://doi.org/10.5281/zenodo.21923444)

This repository contains the frozen code, experimental protocols, manifests,
run-level evidence, revised statistical analyses, and publication figures for:

> *Lightweight Trust- and Energy-Aware Federated Learning for Robust and
> Sustainable IoT Edge Intelligence: Full-System PAMAP2 Evaluation and
> Cross-Domain Robustness Validation on CICIoT2023*

The package covers 1020 accepted experimental runs:

- 600-run PAMAP2 main campaign;
- 240-run PAMAP2 component-ablation campaign; and
- 180-run CICIoT2023 cross-domain robustness campaign.

## Important inferential interpretation

The files in `results/revised_analysis/` are the confirmatory statistical
results for the revised manuscript.

- For PAMAP2, the held-out outer-subject fold is the primary data-level
  inferential unit. The two federated seeds are repeated algorithmic
  realizations within each fold and are averaged before confirmatory testing.
- For CICIoT2023, the experimental seed is the stochastic replication unit.
  The Dirichlet concentration parameter alpha is a repeated experimental
  condition and is averaged within seed for the primary joint analysis.

These results supersede significance statements based on treating
`outer fold x federated seed` or `alpha x experimental seed` as independent
replicates. With five independent units, exact two-sided tests have limited
resolution; effect direction, magnitude, consistency, and uncertainty are
therefore reported alongside p-values.

## Repository map

| Path | Contents |
|---|---|
| `src/frozen_method_sources/` | Exact frozen source snapshots for FedAvg, FedProx, Random Trimmed Mean, FedLE-adapted, TEA-FL, and ARL-FL |
| `scripts/data_preparation/` | Dataset auditing, preprocessing, split, and partition-generation code |
| `scripts/campaigns/` | Frozen PAMAP2 and CICIoT2023 campaign runners |
| `scripts/analysis/` | Result extraction, revised inference, audits, and figure generation |
| `manifests/` | Frozen partitions, malicious identities, participation schedules, campaign matrices, and protocol bindings |
| `results/campaign_outputs/` | Run-level and descriptive evidence from the accepted campaigns |
| `results/revised_analysis/` | Fold-level/seed-level confirmatory reanalysis and unequal-horizon energy analysis |
| `results/figures/` | Manuscript and supplementary figures in PNG and PDF formats |
| `environment/` | Exact package lock and execution-environment report |
| `docs/` | Reproduction, provenance, codebook, and release guidance |

## Quick reproduction of the revised analyses

Create a Python environment, install the packages listed in
`environment/requirements-lock.txt`, and run:

```text
python scripts/analysis/revision_reanalysis_20260812.py
python scripts/analysis/generate_revised_figure2_20260813.py
```

The first command reconstructs `results/revised_analysis/` from the included
frozen campaign outputs and manifests. The second regenerates the revised
PAMAP2 effectiveness-energy Figure 2 files in `results/figures/`.

Full campaign reproduction requires locally obtained copies of the source
datasets and substantially more computation. See
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Data availability

Source datasets are not redistributed here:

- [PAMAP2 Physical Activity Monitoring](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)
- [CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html)

The repository contains derived, non-record-level result tables and frozen
protocol metadata needed to audit the reported experiments. Dataset provenance
and exclusions are described in [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md).

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The immutable
version 1.0.0 archive is available at
[https://doi.org/10.5281/zenodo.21923445](https://doi.org/10.5281/zenodo.21923445).
The concept DOI
[https://doi.org/10.5281/zenodo.21923444](https://doi.org/10.5281/zenodo.21923444)
always resolves to the latest archived version.

## Licence status

No reuse licence has been granted for this repository or its Zenodo archive.
Copyright © 2026 The Authors. All rights reserved. Reuse permission should not
be inferred.
