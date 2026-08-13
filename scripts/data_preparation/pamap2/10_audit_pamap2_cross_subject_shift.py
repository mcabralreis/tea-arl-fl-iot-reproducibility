from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
except ImportError as exc:
    raise SystemExit(
        "ERROR: numpy, pandas and scipy are required in the project environment."
    ) from exc


NUM_CLASSES = 12

ACTIVITY_NAMES = [
    "lying",
    "sitting",
    "standing",
    "walking",
    "running",
    "cycling",
    "Nordic walking",
    "ascending stairs",
    "descending stairs",
    "vacuum cleaning",
    "ironing",
    "rope jumping",
]

COMPLETE_SUBJECTS = (101, 102, 105, 106, 108)
ALL_CORE_SUBJECTS = (101, 102, 103, 104, 105, 106, 107, 108)

# Full36 layout frozen by preprocessing:
# 0:9   = 16g accelerometer, 3 axes x 3 positions
# 9:18  = 6g accelerometer, 3 axes x 3 positions
# 18:27 = gyroscope, 3 axes x 3 positions
# 27:36 = magnetometer, 3 axes x 3 positions

FEATURE_SETS = {
    "acc16_axes9": np.arange(0, 9, dtype=np.int64),
    "acc6_axes9": np.arange(9, 18, dtype=np.int64),
    "gyro_axes9": np.arange(18, 27, dtype=np.int64),
    "mag_axes9": np.arange(27, 36, dtype=np.int64),
    "core18_axes": np.concatenate(
        (
            np.arange(0, 9, dtype=np.int64),
            np.arange(18, 27, dtype=np.int64),
        )
    ),
    "core27_axes": np.concatenate(
        (
            np.arange(0, 9, dtype=np.int64),
            np.arange(18, 36, dtype=np.int64),
        )
    ),
}


def load_all_windows(
    processed_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all 8 core subjects and invert the frozen global z-score.

    The inverse transform is diagnostic only. It recovers the interpolated
    Full36 window values up to float32 roundoff:
        raw ~= normalized * training_std + training_mean
    """
    split_dir = processed_root / "splits"
    stats_dir = processed_root / "statistics"

    mean = np.load(
        stats_dir / "training_mean_full36.npy"
    ).astype(np.float32)
    std = np.load(
        stats_dir / "training_std_full36.npy"
    ).astype(np.float32)

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    subject_parts: list[np.ndarray] = []

    for split in ("train", "validation", "test"):
        x_norm = np.load(
            split_dir / f"{split}_X_full36.npy",
            mmap_mode="r",
        )
        y = np.asarray(
            np.load(split_dir / f"{split}_y.npy", mmap_mode="r"),
            dtype=np.int64,
        ).copy()
        subject = np.asarray(
            np.load(
                split_dir / f"{split}_subject_id.npy",
                mmap_mode="r",
            ),
            dtype=np.int64,
        ).copy()

        # Restore to the pre-normalization scale.
        x_raw = (
            np.asarray(x_norm, dtype=np.float32) * std[None, None, :]
            + mean[None, None, :]
        ).astype(np.float32, copy=False)

        x_parts.append(x_raw)
        y_parts.append(y)
        subject_parts.append(subject)

    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    subject = np.concatenate(subject_parts, axis=0)

    if x.shape != (14972, 256, 36):
        raise RuntimeError(f"Unexpected combined X shape: {x.shape}")

    observed_subjects = tuple(sorted(int(v) for v in np.unique(subject)))
    if observed_subjects != ALL_CORE_SUBJECTS:
        raise RuntimeError(
            f"Unexpected subject IDs: {observed_subjects}"
        )

    return x, y, subject


def axis_magnitudes(
    x: np.ndarray,
    start_index: int,
) -> np.ndarray:
    """
    Convert a 9-axis block (3 positions x xyz) to 3 magnitudes.
    """
    blocks = []
    for position in range(3):
        start = start_index + 3 * position
        triple = x[:, :, start : start + 3]
        magnitude = np.sqrt(
            np.sum(
                np.square(triple, dtype=np.float32),
                axis=2,
            )
        )
        blocks.append(magnitude[:, :, None])

    return np.concatenate(blocks, axis=2)


def build_feature_arrays(
    x_raw: np.ndarray,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}

    for name, indices in FEATURE_SETS.items():
        arrays[name] = np.ascontiguousarray(
            x_raw[:, :, indices],
            dtype=np.float32,
        )

    acc16_mag = axis_magnitudes(x_raw, 0)
    gyro_mag = axis_magnitudes(x_raw, 18)
    mag_mag = axis_magnitudes(x_raw, 27)

    arrays["magnitude6_acc16_gyro"] = np.concatenate(
        (acc16_mag, gyro_mag),
        axis=2,
    )
    arrays["magnitude9_all"] = np.concatenate(
        (acc16_mag, gyro_mag, mag_mag),
        axis=2,
    )
    arrays["hybrid24_core18_plus_mag6"] = np.concatenate(
        (
            arrays["core18_axes"],
            arrays["magnitude6_acc16_gyro"],
        ),
        axis=2,
    )

    return arrays


def window_summary_features(
    x: np.ndarray,
) -> np.ndarray:
    """
    Summarize each window with per-channel temporal mean and std.
    """
    means = x.mean(axis=1, dtype=np.float64)
    stds = x.std(axis=1, dtype=np.float64)
    return np.concatenate((means, stds), axis=1)


def class_centroid_shift(
    features: np.ndarray,
    labels: np.ndarray,
    reference_mask: np.ndarray,
    target_mask: np.ndarray,
) -> list[dict[str, object]]:
    """
    Per-class target-vs-reference centroid shift.

    Each dimension is normalized by the global reference standard deviation.
    """
    reference_features = features[reference_mask]
    scale = reference_features.std(axis=0, ddof=0)
    scale = np.where(scale < 1e-8, 1.0, scale)

    rows: list[dict[str, object]] = []

    for class_index in range(NUM_CLASSES):
        ref = features[
            reference_mask & (labels == class_index)
        ]
        tgt = features[
            target_mask & (labels == class_index)
        ]

        if ref.shape[0] == 0 or tgt.shape[0] == 0:
            continue

        standardized_delta = (
            tgt.mean(axis=0) - ref.mean(axis=0)
        ) / scale

        rows.append(
            {
                "class_index": class_index,
                "activity_name": ACTIVITY_NAMES[class_index],
                "reference_windows": int(ref.shape[0]),
                "target_windows": int(tgt.shape[0]),
                "centroid_rms_shift": float(
                    np.sqrt(
                        np.mean(np.square(standardized_delta))
                    )
                ),
                "centroid_max_abs_shift": float(
                    np.max(np.abs(standardized_delta))
                ),
            }
        )

    return rows


def complete_subject_pair_shift(
    features: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
) -> list[dict[str, object]]:
    """
    Pairwise shift among the five subjects containing all 12 activities.
    """
    rows: list[dict[str, object]] = []

    # Common scale across all complete-subject data for comparability.
    complete_mask = np.isin(subjects, COMPLETE_SUBJECTS)
    scale = features[complete_mask].std(axis=0, ddof=0)
    scale = np.where(scale < 1e-8, 1.0, scale)

    for subject_a, subject_b in combinations(COMPLETE_SUBJECTS, 2):
        class_shifts: list[float] = []

        for class_index in range(NUM_CLASSES):
            a = features[
                (subjects == subject_a)
                & (labels == class_index)
            ]
            b = features[
                (subjects == subject_b)
                & (labels == class_index)
            ]

            if a.shape[0] == 0 or b.shape[0] == 0:
                continue

            delta = (
                b.mean(axis=0) - a.mean(axis=0)
            ) / scale
            class_shifts.append(
                float(
                    np.sqrt(
                        np.mean(np.square(delta))
                    )
                )
            )

        rows.append(
            {
                "subject_a": subject_a,
                "subject_b": subject_b,
                "activities_compared": len(class_shifts),
                "mean_class_centroid_rms_shift": float(
                    np.mean(class_shifts)
                ),
                "std_class_centroid_rms_shift": float(
                    np.std(class_shifts, ddof=1)
                ),
                "max_class_centroid_rms_shift": float(
                    np.max(class_shifts)
                ),
            }
        )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose PAMAP2 cross-subject shift by modality and "
            "orientation-invariant magnitude representation."
        )
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--final-root",
        type=Path,
        default=None,
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

    final_root = (
        args.final_root.expanduser().resolve()
        if args.final_root is not None
        else project_root
        / "outputs"
        / "centralized"
        / "pamap2"
        / "final_test_v1"
    )

    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else project_root
        / "outputs"
        / "diagnostics"
        / "pamap2_cross_subject_shift_v2"
    )

    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(
            "ERROR: output directory already exists and is not empty:\n"
            f"  {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    print("=== PAMAP2 cross-subject modality-shift audit ===")
    print(f"Processed dataset: {processed_root}")
    print(f"Final result source: {final_root}")
    print(f"Output: {output_root}")
    print()

    print("Loading and inverse-transforming all 8 core subjects...")
    x_raw, y, subjects = load_all_windows(processed_root)
    print(f"[OK] Combined raw-scale approximation: {x_raw.shape}")
    print(f"[OK] Subjects: {sorted(np.unique(subjects).tolist())}")
    print()

    print("Building raw-axis and magnitude feature sets...")
    feature_arrays = build_feature_arrays(x_raw)
    print(
        "[OK] Feature sets: "
        + ", ".join(feature_arrays.keys())
    )
    print()

    # Subject108 is already observed; this is diagnostic only.
    target108_mask = subjects == 108
    development108_mask = subjects != 108

    all_108_class_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for index, (feature_set_name, array) in enumerate(
        feature_arrays.items(),
        start=1,
    ):
        print(
            f"[{index}/{len(feature_arrays)}] "
            f"Profiling {feature_set_name} "
            f"({array.shape[2]} channels)..."
        )

        summary_features = window_summary_features(array)

        shift108 = class_centroid_shift(
            summary_features,
            y,
            development108_mask,
            target108_mask,
        )

        for row in shift108:
            row["feature_set"] = feature_set_name
            row["channels"] = int(array.shape[2])
            all_108_class_rows.append(row)

        pair_shift = complete_subject_pair_shift(
            summary_features,
            y,
            subjects,
        )

        for row in pair_shift:
            row["feature_set"] = feature_set_name
            row["channels"] = int(array.shape[2])
            pair_rows.append(row)

        shift108_values = np.array(
            [
                row["centroid_rms_shift"]
                for row in shift108
            ],
            dtype=float,
        )
        pair_values = np.array(
            [
                row["mean_class_centroid_rms_shift"]
                for row in pair_shift
            ],
            dtype=float,
        )

        summary_rows.append(
            {
                "feature_set": feature_set_name,
                "channels": int(array.shape[2]),
                "subject108_mean_class_shift": float(
                    shift108_values.mean()
                ),
                "subject108_median_class_shift": float(
                    np.median(shift108_values)
                ),
                "subject108_max_class_shift": float(
                    shift108_values.max()
                ),
                "complete_subject_pair_mean_shift": float(
                    pair_values.mean()
                ),
                "complete_subject_pair_std_shift": float(
                    pair_values.std(ddof=1)
                ),
                "complete_subject_pair_max_shift": float(
                    pair_values.max()
                ),
            }
        )

    class108_df = pd.DataFrame(all_108_class_rows)
    class108_df.to_csv(
        output_root / "subject108_class_shift_by_feature_set.csv",
        index=False,
    )

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(
        output_root / "complete_subject_pair_shift.csv",
        index=False,
    )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        [
            "complete_subject_pair_mean_shift",
            "subject108_mean_class_shift",
        ],
        ascending=True,
    )
    summary_df["rank_pair_shift"] = np.arange(
        1,
        len(summary_df) + 1,
    )
    summary_df.to_csv(
        output_root / "feature_set_shift_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Correlate subject108 class shift with already-observed test F1.
    # Diagnostic only; never use this as a clean final selection criterion.
    # ------------------------------------------------------------------
    f1_path = (
        final_root / "final_test_per_class_f1_summary.csv"
    )
    if not f1_path.is_file():
        raise FileNotFoundError(f1_path)

    f1_df = pd.read_csv(f1_path)[
        ["class_index", "activity_name", "mean_test_f1"]
    ]

    correlation_rows: list[dict[str, object]] = []

    for feature_set_name in feature_arrays:
        shift_subset = class108_df[
            class108_df["feature_set"] == feature_set_name
        ][
            [
                "class_index",
                "centroid_rms_shift",
            ]
        ]

        merged = f1_df.merge(
            shift_subset,
            on="class_index",
            how="inner",
        )

        rho, p_value = spearmanr(
            merged["centroid_rms_shift"].to_numpy(),
            merged["mean_test_f1"].to_numpy(),
        )

        correlation_rows.append(
            {
                "feature_set": feature_set_name,
                "spearman_rho_shift_vs_test_f1": float(rho),
                "p_value": float(p_value),
                "classes": int(len(merged)),
            }
        )

    correlation_df = pd.DataFrame(
        correlation_rows
    ).sort_values(
        "spearman_rho_shift_vs_test_f1"
    )
    correlation_df.to_csv(
        output_root / "shift_vs_observed_test_f1_correlation.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Human-readable report
    # ------------------------------------------------------------------
    best_pair = summary_df.iloc[0]
    best_108 = summary_df.sort_values(
        "subject108_mean_class_shift"
    ).iloc[0]
    raw_core27 = summary_df[
        summary_df["feature_set"] == "core27_axes"
    ].iloc[0]

    report_lines = [
        "PAMAP2 CROSS-SUBJECT MODALITY-SHIFT AUDIT",
        "=" * 78,
        f"Processed dataset: {processed_root}",
        "",
        "STATUS",
        "-" * 78,
        "This is a diagnostic analysis after subject108 has already been observed.",
        "No model is trained and no clean-test claim is made.",
        "",
        "FEATURE-SET RANKING BY MEAN PAIRWISE SHIFT",
        "-" * 78,
    ]

    for row in summary_df.itertuples(index=False):
        report_lines.append(
            f"{int(row.rank_pair_shift)}. {row.feature_set} "
            f"({int(row.channels)} ch): "
            f"pair mean={row.complete_subject_pair_mean_shift:.3f}; "
            f"subject108 mean={row.subject108_mean_class_shift:.3f}; "
            f"subject108 max={row.subject108_max_class_shift:.3f}"
        )

    report_lines.extend(
        [
            "",
            "KEY COMPARISONS",
            "-" * 78,
            f"Lowest average complete-subject pair shift: "
            f"{best_pair['feature_set']} "
            f"({best_pair['complete_subject_pair_mean_shift']:.3f})",
            f"Lowest average subject108 class shift: "
            f"{best_108['feature_set']} "
            f"({best_108['subject108_mean_class_shift']:.3f})",
            f"Current Core27 pair mean shift: "
            f"{raw_core27['complete_subject_pair_mean_shift']:.3f}",
            f"Current Core27 subject108 mean shift: "
            f"{raw_core27['subject108_mean_class_shift']:.3f}",
            "",
            "SHIFT-VS-OBSERVED-TEST-F1 CORRELATION",
            "-" * 78,
        ]
    )

    for row in correlation_df.itertuples(index=False):
        report_lines.append(
            f"{row.feature_set}: "
            f"Spearman rho={row.spearman_rho_shift_vs_test_f1:.3f}; "
            f"p={row.p_value:.4f}"
        )

    report_lines.extend(
        [
            "",
            "INTERPRETATION RULE FOR THE NEXT STEP",
            "-" * 78,
            "1. If magnetometer axes are substantially more shifted than acceleration",
            "   and gyroscope, Core18 becomes the main raw-axis candidate.",
            "2. If magnitude6 has clearly lower pairwise shift, include an",
            "   orientation-invariant candidate in the redesign.",
            "3. The final paper evaluation must use a subject-wise multi-fold protocol;",
            "   the previous 101/108 split is retained only as a pilot diagnostic.",
        ]
    )

    report_path = output_root / "CROSS_SUBJECT_SHIFT_REPORT.txt"
    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=== Cross-subject shift audit completed successfully ===")
    print(
        "Best pairwise-shift feature set: "
        f"{best_pair['feature_set']}"
    )
    print(
        "Best subject108-shift feature set: "
        f"{best_108['feature_set']}"
    )
    print(f"Report: {report_path}")
    print()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nAudit interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
