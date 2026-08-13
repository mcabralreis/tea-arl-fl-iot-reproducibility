# v1.0.0 — TEA-FL and ARL-FL reproducibility package

This is the first archival release of the reproducibility package accompanying
the manuscript:

> *Lightweight Trust- and Energy-Aware Federated Learning for Robust and
> Sustainable IoT Edge Intelligence: Full-System PAMAP2 Evaluation and
> Cross-Domain Robustness Validation on CICIoT2023*

## Included evidence

- Frozen method sources and scientific campaign runners.
- PAMAP2 evaluation, partition, energy, malicious-client, and campaign
  manifests.
- CICIoT2023 split, partition, participation, malicious-identity, and execution
  manifests.
- Accepted evidence from the 600-run PAMAP2 campaign, 240-run PAMAP2 ablation
  campaign, and 180-run CICIoT2023 campaign.
- Revised confirmatory analyses using the outer fold as the PAMAP2 inferential
  unit and the experimental seed as the CICIoT2023 replication unit.
- PAMAP2 protocol-endpoint and matched-common-horizon energy comparisons.
- CICIoT2023 source-file partition-overlap audit.
- Publication figures in PNG and PDF formats.

## Reproduction status

The revised analysis was regenerated from the tracked campaign outputs. All 18
revised tabular outputs matched the archived originals by SHA-256, and the
revised Figure 2 PNG files were regenerated identically.

## Data boundary

PAMAP2 and CICIoT2023 source datasets, record-level derived arrays, model
checkpoints, and virtual environments are not redistributed. Obtain the source
datasets from their original providers and consult `docs/DATA_PROVENANCE.md`
and `docs/REPRODUCIBILITY.md`.

## Statistical interpretation

For PAMAP2, federated seeds are repeated algorithmic realizations within each
held-out outer-subject fold. For CICIoT2023, alpha is a repeated experimental
condition within each stochastic seed. The five independent folds/seeds give
exact two-sided tests limited resolution; the package therefore reports effect
direction, magnitude, consistency, and uncertainty alongside p-values.

## Licence

No reuse licence is granted in this release unless a licence file is added
before publication. Copyright law applies by default.
