from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import product
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "ERROR: pandas is required in the project environment."
    ) from exc


ALL_CORE_SUBJECTS = (101, 102, 103, 104, 105, 106, 107, 108)
COMPLETE_SUBJECTS = (101, 102, 105, 106, 108)
PILOT_HELDOUT_EXPOSED_SUBJECTS = (101, 108)

ACTIVITY_IDS = (1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24)

REPRESENTATIONS = {
    "core27_axes": {
        "channels": 27,
        "definition": (
            "raw-axis 16g accelerometer + gyroscope + magnetometer "
            "from hand, chest and ankle"
        ),
    },
    "magnitude6_acc16_gyro": {
        "channels": 6,
        "definition": (
            "per-location vector magnitudes of 16g acceleration and "
            "gyroscope: 3 body locations x 2 modalities"
        ),
    },
    "magnitude9_all": {
        "channels": 9,
        "definition": (
            "per-location vector magnitudes of 16g acceleration, "
            "gyroscope and magnetometer: 3 locations x 3 modalities"
        ),
    },
}

NORMALIZATION_LAYERS = ("batchnorm", "groupnorm")

INNER_SEEDS = (123, 456)
OUTER_SEEDS = (123, 456, 789)

MODEL_CONFIG = {
    "family": "LightweightCNN1D",
    "channels": [32, 64, 96],
    "dropout": 0.20,
    "normalization_layer_candidates": list(NORMALIZATION_LAYERS),
    "groupnorm_groups": 8,
}

OPTIMIZATION = {
    "loss": "cross_entropy",
    "optimizer": "AdamW",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 128,
    "max_epochs": 40,
    "early_stopping_patience": 8,
    "early_stopping_metric": "validation_macro_f1",
    "min_delta": 1e-4,
}

LIGHTWEIGHT_TOLERANCE = 0.005


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def subject_label(subject_number: int) -> str:
    return f"subject{subject_number}"


def subjects_to_text(subjects: tuple[int, ...] | list[int]) -> str:
    return ",".join(subject_label(subject) for subject in subjects)


def coverage_for_subjects(
    counts: pd.DataFrame,
    subjects: tuple[int, ...] | list[int],
) -> dict[int, int]:
    subject_names = {subject_label(subject) for subject in subjects}

    subset = counts[counts["subject"].isin(subject_names)]

    grouped = (
        subset.groupby("activity_id")["windows"]
        .sum()
        .reindex(ACTIVITY_IDS, fill_value=0)
    )

    return {
        int(activity_id): int(grouped.loc[activity_id])
        for activity_id in ACTIVITY_IDS
    }


def verify_complete_coverage(
    coverage: dict[int, int],
    context: str,
) -> None:
    missing = [
        activity_id
        for activity_id, windows in coverage.items()
        if windows <= 0
    ]
    if missing:
        raise RuntimeError(
            f"{context} does not cover all 12 activities. "
            f"Missing: {missing}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the nested subject-wise PAMAP2 Evaluation Protocol v2 "
            "before any v2 model training."
        )
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    processed_root = args.processed_root.expanduser().resolve()

    try:
        project_root = processed_root.parents[3]
    except IndexError:
        project_root = Path.cwd()

    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else project_root
        / "outputs"
        / "protocols"
        / "pamap2_evaluation_v2"
    )

    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(
            "ERROR: output directory already exists and is not empty:\n"
            f"  {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    distribution_path = processed_root / "window_distribution.csv"
    preprocessing_config_path = processed_root / "preprocessing_configuration.json"
    inventory_path = processed_root / "output_inventory_sha256.json"

    for path in (
        distribution_path,
        preprocessing_config_path,
        inventory_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    counts = pd.read_csv(distribution_path)
    counts["activity_id"] = counts["activity_id"].astype(int)
    counts["windows"] = counts["windows"].astype(int)

    observed_subjects = tuple(
        sorted(
            int(subject.replace("subject", ""))
            for subject in counts["subject"].unique()
        )
    )

    if observed_subjects != ALL_CORE_SUBJECTS:
        raise RuntimeError(f"Unexpected subjects: {observed_subjects}")

    print("=== Freeze PAMAP2 Evaluation Protocol v2 ===")
    print(f"Processed dataset: {processed_root}")
    print(f"Output:            {output_root}")
    print()
    print(f"All core subjects:       {list(ALL_CORE_SUBJECTS)}")
    print(f"Complete outer subjects: {list(COMPLETE_SUBJECTS)}")
    print()

    candidate_rows: list[dict[str, object]] = []

    for candidate_index, (representation, normalization_layer) in enumerate(
        product(REPRESENTATIONS, NORMALIZATION_LAYERS),
        start=1,
    ):
        candidate_rows.append(
            {
                "candidate_index": candidate_index,
                "representation": representation,
                "input_channels": REPRESENTATIONS[representation]["channels"],
                "representation_definition": REPRESENTATIONS[representation]["definition"],
                "normalization_layer": normalization_layer,
                "loss": OPTIMIZATION["loss"],
                "model_family": MODEL_CONFIG["family"],
                "model_channels": "-".join(
                    str(value) for value in MODEL_CONFIG["channels"]
                ),
                "dropout": MODEL_CONFIG["dropout"],
            }
        )

    candidate_df = pd.DataFrame(candidate_rows)
    candidate_df.to_csv(output_root / "candidate_grid.csv", index=False)

    outer_rows: list[dict[str, object]] = []
    inner_rows: list[dict[str, object]] = []

    for outer_fold_index, outer_test_subject in enumerate(
        COMPLETE_SUBJECTS,
        start=1,
    ):
        outer_train_subjects = tuple(
            subject
            for subject in ALL_CORE_SUBJECTS
            if subject != outer_test_subject
        )

        outer_train_coverage = coverage_for_subjects(
            counts,
            outer_train_subjects,
        )
        outer_test_coverage = coverage_for_subjects(
            counts,
            (outer_test_subject,),
        )

        verify_complete_coverage(
            outer_train_coverage,
            f"Outer fold {outer_fold_index} training set",
        )
        verify_complete_coverage(
            outer_test_coverage,
            f"Outer fold {outer_fold_index} test subject",
        )

        outer_rows.append(
            {
                "outer_fold": outer_fold_index,
                "outer_test_subject": subject_label(outer_test_subject),
                "outer_training_subjects": subjects_to_text(outer_train_subjects),
                "outer_training_windows": sum(outer_train_coverage.values()),
                "outer_test_windows": sum(outer_test_coverage.values()),
                "pilot_heldout_performance_previously_observed": (
                    outer_test_subject in PILOT_HELDOUT_EXPOSED_SUBJECTS
                ),
                "inner_validation_subjects": subjects_to_text(
                    tuple(
                        subject
                        for subject in COMPLETE_SUBJECTS
                        if subject != outer_test_subject
                    )
                ),
                "outer_final_seeds": ",".join(str(seed) for seed in OUTER_SEEDS),
            }
        )

        inner_validation_subjects = tuple(
            subject
            for subject in COMPLETE_SUBJECTS
            if subject != outer_test_subject
        )

        for inner_fold_index, inner_validation_subject in enumerate(
            inner_validation_subjects,
            start=1,
        ):
            inner_train_subjects = tuple(
                subject
                for subject in ALL_CORE_SUBJECTS
                if subject
                not in {
                    outer_test_subject,
                    inner_validation_subject,
                }
            )

            inner_train_coverage = coverage_for_subjects(
                counts,
                inner_train_subjects,
            )
            inner_validation_coverage = coverage_for_subjects(
                counts,
                (inner_validation_subject,),
            )

            verify_complete_coverage(
                inner_train_coverage,
                (
                    f"Outer fold {outer_fold_index}, "
                    f"inner fold {inner_fold_index} training set"
                ),
            )
            verify_complete_coverage(
                inner_validation_coverage,
                (
                    f"Outer fold {outer_fold_index}, "
                    f"inner fold {inner_fold_index} validation subject"
                ),
            )

            inner_rows.append(
                {
                    "outer_fold": outer_fold_index,
                    "outer_test_subject": subject_label(outer_test_subject),
                    "inner_fold": inner_fold_index,
                    "inner_validation_subject": subject_label(
                        inner_validation_subject
                    ),
                    "inner_training_subjects": subjects_to_text(
                        inner_train_subjects
                    ),
                    "inner_training_windows": sum(inner_train_coverage.values()),
                    "inner_validation_windows": sum(
                        inner_validation_coverage.values()
                    ),
                    "inner_seeds": ",".join(str(seed) for seed in INNER_SEEDS),
                }
            )

    outer_df = pd.DataFrame(outer_rows)
    inner_df = pd.DataFrame(inner_rows)

    outer_df.to_csv(output_root / "outer_fold_manifest.csv", index=False)
    inner_df.to_csv(output_root / "inner_fold_manifest.csv", index=False)

    protocol = {
        "protocol_name": "PAMAP2 Evaluation Protocol v2",
        "status": "FROZEN_BEFORE_V2_MODEL_TRAINING",
        "scientific_purpose": (
            "Estimate cross-subject generalization with nested subject-wise "
            "selection after the single-validation-subject pilot proved unstable."
        ),
        "dataset": {
            "processed_root": str(processed_root),
            "preprocessing_configuration_sha256": sha256_file(
                preprocessing_config_path
            ),
            "output_inventory_sha256_file_sha256": sha256_file(inventory_path),
            "core_subjects": list(ALL_CORE_SUBJECTS),
            "excluded_subject109": True,
        },
        "pilot_status": {
            "pilot_validation_subject": 101,
            "pilot_test_subject": 108,
            "pilot_heldout_metrics_already_observed": True,
            "interpretation": (
                "The previous 101/108 split is development/diagnostic only "
                "and is not used as the final evaluation design."
            ),
        },
        "outer_evaluation": {
            "design": "five-fold leave-one-complete-subject-out",
            "outer_test_subjects": list(COMPLETE_SUBJECTS),
            "reason_for_complete_subject_subset": (
                "Each outer test subject must contain all 12 protocol activities "
                "for directly comparable 12-class Macro-F1."
            ),
            "outer_training_subjects": (
                "all remaining core subjects, including incomplete subjects"
            ),
            "outer_test_metrics": [
                "macro_f1",
                "balanced_accuracy",
                "accuracy",
                "per_class_f1",
            ],
            "outer_seeds": list(OUTER_SEEDS),
            "all_outer_folds_reported": True,
        },
        "inner_selection": {
            "design": (
                "for each outer fold, each remaining complete subject serves "
                "once as inner validation"
            ),
            "inner_folds_per_outer_fold": 4,
            "inner_training_subjects": (
                "all core subjects except outer test and current inner validation"
            ),
            "inner_seeds": list(INNER_SEEDS),
            "selection_metric": (
                "mean validation Macro-F1 across 4 inner validation subjects "
                "and 2 seeds"
            ),
            "lightweight_tolerance": LIGHTWEIGHT_TOLERANCE,
            "selection_rule": (
                "Select the highest mean inner Macro-F1. Any candidate within "
                "0.005 absolute Macro-F1 of the best is considered equivalent; "
                "among equivalent candidates choose fewer input channels, then "
                "lower across-run standard deviation."
            ),
            "epoch_rule_for_outer_training": (
                "median best epoch across the 8 inner runs "
                "(4 validation subjects x 2 seeds) of the selected candidate"
            ),
        },
        "candidate_grid": {
            "representations": REPRESENTATIONS,
            "normalization_layers": list(NORMALIZATION_LAYERS),
            "candidates": candidate_rows,
        },
        "model": MODEL_CONFIG,
        "optimization": OPTIMIZATION,
        "normalization_of_inputs": {
            "raw_scale_reconstruction": (
                "invert the frozen v1 z-score to recover interpolated Full36 "
                "window values up to float32 roundoff"
            ),
            "representation_construction": (
                "construct candidate representation on the recovered raw scale"
            ),
            "fold_specific_z_score": (
                "fit per-channel mean and standard deviation using only the "
                "current inner-training subjects during inner selection, or "
                "only the current outer-training subjects during final outer training"
            ),
            "test_or_validation_statistics_used": False,
        },
        "reporting": {
            "primary": (
                "mean and standard deviation across the five complete-subject outer folds; "
                "also report all fold-level values"
            ),
            "pilot_exposure_stratification": {
                "previously_observed_as_heldout": list(
                    PILOT_HELDOUT_EXPOSED_SUBJECTS
                ),
                "not_previously_observed_as_heldout": [
                    subject
                    for subject in COMPLETE_SUBJECTS
                    if subject not in PILOT_HELDOUT_EXPOSED_SUBJECTS
                ],
                "note": (
                    "This stratification is reported transparently; the whole v2 protocol "
                    "is considered a post-pilot nested cross-validation study."
                ),
            },
        },
    }

    protocol_path = output_root / "EVALUATION_PROTOCOL_V2.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )

    total_inner_runs = (
        len(COMPLETE_SUBJECTS)
        * 4
        * len(candidate_rows)
        * len(INNER_SEEDS)
    )
    total_outer_runs = (
        len(COMPLETE_SUBJECTS)
        * len(OUTER_SEEDS)
    )

    report_lines = [
        "PAMAP2 EVALUATION PROTOCOL V2",
        "=" * 78,
        f"Processed dataset: {processed_root}",
        "",
        "STATUS",
        "-" * 78,
        "FROZEN BEFORE ANY V2 MODEL TRAINING",
        "",
        "WHY V2 IS REQUIRED",
        "-" * 78,
        "The pilot single-subject validation/test design was unstable across subjects.",
        "The previous subject101 validation and subject108 test results are retained",
        "as pilot diagnostics and are not used as the final article evaluation design.",
        "",
        "OUTER EVALUATION",
        "-" * 78,
        "Design: five-fold leave-one-complete-subject-out",
        f"Outer subjects: {list(COMPLETE_SUBJECTS)}",
        "Each outer test subject contains all 12 activities.",
        "All remaining core subjects contribute to outer training.",
        f"Outer seeds: {list(OUTER_SEEDS)}",
        "",
        "NESTED INNER SELECTION",
        "-" * 78,
        "Four inner validation subjects per outer fold.",
        f"Inner seeds: {list(INNER_SEEDS)}",
        "Selection: mean Macro-F1 across 8 inner runs per candidate.",
        "Equivalent candidates: within 0.005 Macro-F1 of the best.",
        "Tie-break: fewer input channels, then lower variability.",
        "",
        "CANDIDATE GRID",
        "-" * 78,
    ]

    for row in candidate_df.itertuples(index=False):
        report_lines.append(
            f"{row.candidate_index}. {row.representation} "
            f"({row.input_channels} ch) + {row.normalization_layer}"
        )

    report_lines.extend(
        [
            "",
            "FIXED MODEL AND OPTIMIZATION",
            "-" * 78,
            "Model: small LightweightCNN1D",
            "Convolution channels: 32 -> 64 -> 96",
            "Dropout: 0.20",
            "Loss: standard cross-entropy",
            "Optimizer: AdamW",
            "Learning rate: 0.001",
            "Weight decay: 0.0001",
            "Batch size: 128",
            "Maximum epochs: 40",
            "Early stopping patience: 8",
            "",
            "FOLD-SPECIFIC INPUT NORMALIZATION",
            "-" * 78,
            "Candidate representations are constructed on recovered raw-scale windows.",
            "Z-score statistics are fitted only on the current training subjects.",
            "No inner-validation or outer-test subject contributes normalization statistics.",
            "",
            "PLANNED COMPUTATION",
            "-" * 78,
            f"Inner screening runs: {total_inner_runs}",
            f"Outer final runs: {total_outer_runs}",
            f"Total planned model trainings: {total_inner_runs + total_outer_runs}",
            "",
            "IMPORTANT TRANSPARENCY NOTE",
            "-" * 78,
            "subject101 and subject108 were already observed as held-out subjects in the pilot.",
            "The v2 study is therefore reported as a post-pilot nested cross-validation study,",
            "not as a retrospectively untouched hold-out experiment.",
            "",
            "FILES",
            "-" * 78,
            "EVALUATION_PROTOCOL_V2.json",
            "candidate_grid.csv",
            "outer_fold_manifest.csv",
            "inner_fold_manifest.csv",
        ]
    )

    report_path = output_root / "PROTOCOL_V2_REPORT.txt"
    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("[OK] All fold coverage checks passed.")
    print(f"[OK] Candidate configurations: {len(candidate_rows)}")
    print(f"[OK] Outer folds: {len(outer_df)}")
    print(f"[OK] Inner folds: {len(inner_df)}")
    print(f"[OK] Planned inner runs: {total_inner_runs}")
    print(f"[OK] Planned outer runs: {total_outer_runs}")
    print()
    print("=== PAMAP2 Evaluation Protocol v2 frozen successfully ===")
    print(f"Protocol: {protocol_path}")
    print(f"Report:   {report_path}")
    print()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nProtocol freeze interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
