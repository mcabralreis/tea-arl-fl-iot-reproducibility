# Reproducibility guide

This package supports three progressively more demanding forms of
reproduction.

## 1. Audit the archived evidence

No source dataset is required. Inspect the final bindings and SHA-256 manifests
under `manifests/` and `results/campaign_outputs/`. The principal accepted-run
tables are:

- `results/campaign_outputs/pamap2/postcampaign_freeze_v1/PAMAP2_RUN_LEVEL_RESULTS_600.csv`;
- `results/campaign_outputs/pamap2/ablation_analysis_v1/PAMAP2_ABLATION_RUN_RESULTS_240.csv`; and
- `results/campaign_outputs/ciciot2023/scientific_campaign_final_audit_v1/MASTER_RUN_RESULTS.csv`.

The CICIoT2023 per-run completion records and round-level validation metrics
are retained under `results/campaign_outputs/ciciot2023/run_evidence/runs/`.

## 2. Reproduce the revised statistics and Figure 2

Use Python 3.14 and the exact environment lock when possible:

```text
python -m venv .venv
.venv/Scripts/python -m pip install -r environment/requirements-lock.txt
.venv/Scripts/python scripts/analysis/revision_reanalysis_20260812.py
.venv/Scripts/python scripts/analysis/generate_revised_figure2_20260813.py
```

On Linux or macOS, replace `.venv/Scripts/python` with `.venv/bin/python`.
The XPU-specific PyTorch wheel may require Intel's package index; the revised
tabular analyses themselves require only NumPy, pandas, and Matplotlib.

Expected outputs are written to `results/revised_analysis/` and
`results/figures/`. Compare them with the tracked files before committing any
regeneration.

The confirmatory design is deliberately hierarchical:

- PAMAP2: average the two federated seeds inside each outer fold, then use the
  five held-out-subject folds as inferential blocks;
- CICIoT2023: average the two alpha conditions inside each experimental seed,
  then use the five seeds as inferential blocks.

## 3. Reproduce data preparation and full campaigns

Obtain PAMAP2 and CICIoT2023 from their original providers. Place local data
under a non-versioned `data/` directory; this path is ignored by Git. The exact
original workflow used a project-root argument and frozen, absolute provenance
paths. When running from a new machine, pass or adapt the new local project
root while preserving all frozen protocol values and seeds.

PAMAP2 workflow sources are under `scripts/data_preparation/pamap2/`, followed
by `scripts/campaigns/pamap2/run_pamap2_scientific_campaign.py`. The source
snapshots loaded by the runner are under `src/frozen_method_sources/`.

CICIoT2023 split, count-plan, assignment, participation, malicious-identity,
and execution-freeze engines are under `scripts/data_preparation/ciciot2023/`.
The scientific runner and context bridge are under
`scripts/campaigns/ciciot2023/`.

The campaign scale is substantial: 600 main PAMAP2 runs, 240 PAMAP2 ablation
runs, and 180 CICIoT2023 runs. Full reproduction should be performed on a
separate branch or outside the repository, with generated arrays, checkpoints,
and logs kept under ignored paths.

## Frozen-versus-portable distinction

Files labelled `frozen`, `binding`, or `manifest` are preserved as scientific
evidence and may contain the original machine's historical project-root paths.
Those paths are provenance records, not current installation instructions.
Changing a frozen file invalidates its recorded SHA-256 binding. Portable
reanalysis scripts use paths relative to this repository.

