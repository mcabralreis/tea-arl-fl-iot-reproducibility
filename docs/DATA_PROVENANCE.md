# Data provenance and redistribution boundary

## PAMAP2

PAMAP2 is obtained from the UCI Machine Learning Repository. Source recordings
are not redistributed. The package includes the preprocessing code, frozen
outer-fold manifests, virtual-client assignments, class counts, client energy
profiles, malicious-client manifests, and the 600-row final campaign table.

The virtual-client construction creates controlled class-distribution
heterogeneity. It is not a one-to-one mapping between federated clients and
physical PAMAP2 subjects. Subject independence is enforced at the outer
leave-one-complete-subject-out evaluation level.

## CICIoT2023

CICIoT2023 is obtained from the Canadian Institute for Cybersecurity. Raw
traffic records and derived record-level arrays are not redistributed. The
package includes preprocessing-policy metadata, split targets, client count
plans, participation schedules, malicious-client rankings, execution seeds,
method bindings, and non-record-level result evidence.

Exact-vector deduplication prevents exact duplicate vectors from being placed
in multiple effective observations, but it does not establish independence by
source file, device, or capture session. The retained provenance permits a
source-file audit. That audit found that all 309 source files contribute to all
three observation-level partitions. Device and capture-session identifiers were
not available in the retained provenance. See:

- `results/revised_analysis/CICIOT2023_SOURCE_OVERLAP_SUMMARY.json`;
- `results/revised_analysis/CICIOT2023_SOURCE_FILE_PARTITION_AUDIT.csv`.

This dependence must be considered when interpreting cross-domain
generalization. The CICIoT2023 campaign assesses trust/risk updating and robust
aggregation under a common frozen participation schedule; it does not evaluate
method-specific energy-aware client selection.

## Files intentionally excluded

- PAMAP2 and CICIoT2023 source datasets;
- record-level derived arrays;
- virtual environments and package caches;
- PyTorch checkpoints and tensor files;
- temporary diagnostics and superseded failed-attempt outputs;
- manuscript working documents.

