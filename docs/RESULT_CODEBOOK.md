# Result codebook

## Primary campaign evidence

| File | Unit | Purpose |
|---|---|---|
| `PAMAP2_RUN_LEVEL_RESULTS_600.csv` | accepted PAMAP2 run | Final effectiveness, lifetime, energy, fairness, and binding fields |
| `PAMAP2_EVALUATION_TRAJECTORIES_12600_ROWS.csv` | run x evaluation round | Roundwise PAMAP2 evaluation values at the frozen evaluation rounds |
| `PAMAP2_RUN_RESULT_BINDING_LEDGER_600.csv` | accepted PAMAP2 run | Compact result and optimizer-step ledger |
| `PAMAP2_ABLATION_RUN_RESULTS_240.csv` | accepted ablation run | Component-ablation endpoints |
| `MASTER_RUN_RESULTS.csv` | accepted CICIoT2023 run | Final and best-round CICIoT2023 metrics |

## Revised confirmatory outputs

| Prefix/file | Interpretation |
|---|---|
| `PAMAP2_OVERALL_FIVE_FOLD_BLOCKS` | Method values after averaging seeds within each outer fold |
| `PAMAP2_FIVE_FOLD_OMNIBUS` | Five-fold Friedman/randomization omnibus tests |
| `PAMAP2_FIVE_FOLD_PAIRWISE` | Exact paired tests, multiplicity adjustment, effect sizes, and uncertainty |
| `PAMAP2_TABLE5_REVISED_VALUES` | Protocol-endpoint and matched-common-horizon descriptive values |
| `PAMAP2_COMMON_HORIZON_PARETO` | Pareto membership at a horizon completed by all paired runs |
| `PAMAP2_ABLATION_FIVE_FOLD_*` | Ablation results after averaging repeated seeds within fold |
| `CICIOT2023_PRIMARY_FIVE_SEED_BLOCKS` | Method values after averaging alpha within experimental seed |
| `CICIOT2023_FIVE_SEED_OMNIBUS` | Five-seed confirmatory omnibus tests |
| `CICIOT2023_FIVE_SEED_ARL_PAIRWISE` | ARL-FL focused exact paired comparisons by five seeds |

`ANALYSIS_METADATA.json` records the inferential units, number of randomization
draws, bootstrap replicates, and confirmation that no training was repeated.

## Energy endpoints

`protocol endpoint` is the cumulative normalized energy at round 100 or at the
protocol-defined federation-viability stop. `common horizon` is the largest
round reached by every paired run in the corresponding block. The former is a
valid outcome of the stopping protocol but is not an equal-duration comparison;
the latter isolates energy/effectiveness at a matched operating horizon.

