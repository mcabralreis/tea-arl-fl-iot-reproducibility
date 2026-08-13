from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "ERROR: numpy and pandas are required. Install them with:\n"
        "  py -m pip install numpy pandas"
    ) from exc


IMU_POSITIONS = ("hand", "chest", "ankle")
IMU_FIELDS = (
    "temperature",
    "acceleration_16g_x",
    "acceleration_16g_y",
    "acceleration_16g_z",
    "acceleration_6g_x",
    "acceleration_6g_y",
    "acceleration_6g_z",
    "gyroscope_x",
    "gyroscope_y",
    "gyroscope_z",
    "magnetometer_x",
    "magnetometer_y",
    "magnetometer_z",
    "orientation_1",
    "orientation_2",
    "orientation_3",
    "orientation_4",
)

COLUMNS = ["timestamp", "activity_id", "heart_rate"]
for position in IMU_POSITIONS:
    COLUMNS.extend(f"{position}_{field}" for field in IMU_FIELDS)

ACTIVITY_NAMES = {
    0: "other/transient",
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "Nordic walking",
    12: "ascending stairs",
    13: "descending stairs",
    16: "vacuum cleaning",
    17: "ironing",
    24: "rope jumping",
}

PROTOCOL_ACTIVITY_IDS = (1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24)

ORIENTATION_COLUMNS = [
    f"{position}_orientation_{axis}"
    for position in IMU_POSITIONS
    for axis in range(1, 5)
]

TEMPERATURE_COLUMNS = [
    f"{position}_temperature" for position in IMU_POSITIONS
]

ACC16_COLUMNS = [
    f"{position}_acceleration_16g_{axis}"
    for position in IMU_POSITIONS
    for axis in ("x", "y", "z")
]

ACC6_COLUMNS = [
    f"{position}_acceleration_6g_{axis}"
    for position in IMU_POSITIONS
    for axis in ("x", "y", "z")
]

GYRO_COLUMNS = [
    f"{position}_gyroscope_{axis}"
    for position in IMU_POSITIONS
    for axis in ("x", "y", "z")
]

MAG_COLUMNS = [
    f"{position}_magnetometer_{axis}"
    for position in IMU_POSITIONS
    for axis in ("x", "y", "z")
]

FULL36_COLUMNS = ACC16_COLUMNS + ACC6_COLUMNS + GYRO_COLUMNS + MAG_COLUMNS
CORE27_COLUMNS = ACC16_COLUMNS + GYRO_COLUMNS + MAG_COLUMNS
GAP_AUDIT_COLUMNS = ["heart_rate"] + FULL36_COLUMNS

WINDOWS = {
    "w256_s128": {"window": 256, "stride": 128},
    "w512_s256": {"window": 512, "stride": 256},
}

SAMPLE_RATE_HZ = 100.0


def infer_project_root(dataset_root: Path) -> Path:
    resolved = dataset_root.resolve()
    lower_parts = [part.lower() for part in resolved.parts]
    try:
        idx = lower_parts.index("data")
        return Path(*resolved.parts[:idx])
    except ValueError:
        return Path.cwd()


def append_true_runs(
    mask: np.ndarray,
    current_open_run: int,
    completed_runs: list[int],
) -> int:
    """
    Update missing-value run lengths for one sequential boolean chunk.

    True means missing. The returned integer is the still-open run length
    at the end of the chunk (0 if no run remains open).
    """
    if mask.size == 0:
        return current_open_run

    mask = np.asarray(mask, dtype=bool)

    # Locate constant-value blocks without iterating over every sample.
    change_points = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    boundaries = np.concatenate(([0], change_points, [mask.size]))

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        is_missing = bool(mask[start])
        block_len = int(end - start)

        if is_missing:
            current_open_run += block_len
        elif current_open_run > 0:
            completed_runs.append(current_open_run)
            current_open_run = 0

    return current_open_run


def finalize_activity_segment(
    subject: str,
    activity_id: int | None,
    length: int,
    segment_lengths: dict[int, list[int]],
    window_counts: dict[str, Counter[tuple[str, int]]],
) -> None:
    if activity_id is None or activity_id == 0 or length <= 0:
        return

    segment_lengths[activity_id].append(length)

    for name, cfg in WINDOWS.items():
        window = cfg["window"]
        stride = cfg["stride"]
        n_windows = 0 if length < window else 1 + (length - window) // stride
        if n_windows > 0:
            window_counts[name][(subject, activity_id)] += int(n_windows)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2((a[mask] + eps) / (b[mask] + eps))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def count_vector(
    subjects: list[str],
    table: dict[str, dict[int, int]],
) -> np.ndarray:
    return np.array(
        [
            sum(table.get(subject, {}).get(activity_id, 0) for subject in subjects)
            for activity_id in PROTOCOL_ACTIVITY_IDS
        ],
        dtype=float,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Profile PAMAP2 missing-value gaps, contiguous activity segments, "
            "candidate window yields, and deterministic subject-wise splits."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd() / "data" / "raw" / "pamap2",
        help="Path to the extracted PAMAP2 directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: <project>/outputs/design/pamap2",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Rows per chunk (default: 100000).",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    protocol_dir = root / "Protocol"
    if not protocol_dir.is_dir():
        raise SystemExit(f"ERROR: Protocol directory not found: {protocol_dir}")

    project_root = infer_project_root(root)
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else project_root / "outputs" / "design" / "pamap2"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    files = sorted(protocol_dir.glob("*.dat"))
    if len(files) != 9:
        raise SystemExit(f"ERROR: expected 9 Protocol files, found {len(files)}.")

    # Missing-run states.
    gap_runs: dict[str, list[int]] = {column: [] for column in GAP_AUDIT_COLUMNS}
    open_gap: dict[str, int] = {column: 0 for column in GAP_AUDIT_COLUMNS}

    # Activity-segment states.
    segment_lengths: dict[int, list[int]] = defaultdict(list)
    window_counts: dict[str, Counter[tuple[str, int]]] = {
        name: Counter() for name in WINDOWS
    }

    print("=== PAMAP2 preprocessing-design profiler ===")
    print(f"Dataset root: {root}")
    print(f"Output root: {output_root}")
    print(f"Files: {len(files)}")
    print(f"Chunk size: {args.chunk_size:,}")
    print()

    for file_index, path in enumerate(files, start=1):
        subject = path.stem.lower()
        current_activity: int | None = None
        current_segment_len = 0

        # Missing runs are independent per recording; close any previous state.
        for column in GAP_AUDIT_COLUMNS:
            if open_gap[column] > 0:
                gap_runs[column].append(open_gap[column])
                open_gap[column] = 0

        print(f"[{file_index}/9] Profiling {path.name} ...")

        reader = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=COLUMNS,
            na_values=["NaN"],
            chunksize=args.chunk_size,
            low_memory=False,
        )

        file_rows = 0
        for chunk in reader:
            file_rows += len(chunk)

            # Missing-value run profiling.
            missing_matrix = chunk[GAP_AUDIT_COLUMNS].isna()
            for column in GAP_AUDIT_COLUMNS:
                open_gap[column] = append_true_runs(
                    missing_matrix[column].to_numpy(dtype=bool, copy=False),
                    open_gap[column],
                    gap_runs[column],
                )

            # Sequential activity-segment profiling.
            activity = pd.to_numeric(
                chunk["activity_id"], errors="raise"
            ).to_numpy(dtype=np.int64, copy=False)

            if activity.size == 0:
                continue

            change_points = np.flatnonzero(activity[1:] != activity[:-1]) + 1
            boundaries = np.concatenate(([0], change_points, [activity.size]))

            for start, end in zip(boundaries[:-1], boundaries[1:]):
                label = int(activity[start])
                block_len = int(end - start)

                if current_activity is None:
                    current_activity = label
                    current_segment_len = block_len
                elif label == current_activity:
                    current_segment_len += block_len
                else:
                    finalize_activity_segment(
                        subject,
                        current_activity,
                        current_segment_len,
                        segment_lengths,
                        window_counts,
                    )
                    current_activity = label
                    current_segment_len = block_len

        finalize_activity_segment(
            subject,
            current_activity,
            current_segment_len,
            segment_lengths,
            window_counts,
        )

        # Close missing runs at recording boundary.
        for column in GAP_AUDIT_COLUMNS:
            if open_gap[column] > 0:
                gap_runs[column].append(open_gap[column])
                open_gap[column] = 0

        print(f"      rows={file_rows:,}")

    # ------------------------------------------------------------------
    # Missing-gap summary
    # ------------------------------------------------------------------
    gap_rows = []
    for column, runs in gap_runs.items():
        arr = np.asarray(runs, dtype=np.int64)
        if arr.size == 0:
            stats = {
                "missing_runs": 0,
                "missing_samples": 0,
                "min_run_samples": 0,
                "median_run_samples": 0.0,
                "p95_run_samples": 0.0,
                "p99_run_samples": 0.0,
                "max_run_samples": 0,
                "max_run_seconds": 0.0,
            }
        else:
            stats = {
                "missing_runs": int(arr.size),
                "missing_samples": int(arr.sum()),
                "min_run_samples": int(arr.min()),
                "median_run_samples": float(np.median(arr)),
                "p95_run_samples": float(np.percentile(arr, 95)),
                "p99_run_samples": float(np.percentile(arr, 99)),
                "max_run_samples": int(arr.max()),
                "max_run_seconds": float(arr.max() / SAMPLE_RATE_HZ),
            }

        feature_set = (
            "heart_rate"
            if column == "heart_rate"
            else "core27"
            if column in CORE27_COLUMNS
            else "full36_only"
        )

        gap_rows.append(
            {
                "column": column,
                "feature_set": feature_set,
                **stats,
            }
        )

    gap_summary = pd.DataFrame(gap_rows).sort_values(
        ["feature_set", "max_run_samples", "column"],
        ascending=[True, False, True],
    )
    gap_summary.to_csv(output_root / "missing_gap_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Segment summary
    # ------------------------------------------------------------------
    segment_rows = []
    for activity_id in PROTOCOL_ACTIVITY_IDS:
        lengths = np.asarray(segment_lengths.get(activity_id, []), dtype=np.int64)
        if lengths.size == 0:
            continue
        segment_rows.append(
            {
                "activity_id": activity_id,
                "activity_name": ACTIVITY_NAMES[activity_id],
                "segments": int(lengths.size),
                "median_segment_samples": float(np.median(lengths)),
                "median_segment_seconds": float(np.median(lengths) / SAMPLE_RATE_HZ),
                "p10_segment_seconds": float(np.percentile(lengths, 10) / SAMPLE_RATE_HZ),
                "p90_segment_seconds": float(np.percentile(lengths, 90) / SAMPLE_RATE_HZ),
                "max_segment_seconds": float(lengths.max() / SAMPLE_RATE_HZ),
            }
        )

    segment_summary = pd.DataFrame(segment_rows)
    segment_summary.to_csv(output_root / "activity_segment_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Window-count tables
    # ------------------------------------------------------------------
    per_window_tables: dict[str, pd.DataFrame] = {}
    per_window_nested: dict[str, dict[str, dict[int, int]]] = {}

    all_subjects = [path.stem.lower() for path in files]

    for window_name, counts in window_counts.items():
        rows = []
        nested: dict[str, dict[int, int]] = defaultdict(dict)

        for subject in all_subjects:
            for activity_id in PROTOCOL_ACTIVITY_IDS:
                n = int(counts[(subject, activity_id)])
                nested[subject][activity_id] = n
                rows.append(
                    {
                        "subject": subject,
                        "activity_id": activity_id,
                        "activity_name": ACTIVITY_NAMES[activity_id],
                        "windows": n,
                    }
                )

        table = pd.DataFrame(rows)
        table.to_csv(
            output_root / f"window_counts_{window_name}.csv",
            index=False,
        )
        per_window_tables[window_name] = table
        per_window_nested[window_name] = nested

    # ------------------------------------------------------------------
    # Deterministic split search using the 256-sample candidate.
    # subject109 is excluded from the core experiment by design because
    # it contains only rope jumping and is not a representative full
    # protocol participant.
    # ------------------------------------------------------------------
    split_window_name = "w256_s128"
    split_table = per_window_nested[split_window_name]
    eligible_subjects = [s for s in all_subjects if s != "subject109"]

    complete_subjects = []
    for subject in eligible_subjects:
        counts = split_table[subject]
        if all(counts.get(activity_id, 0) > 0 for activity_id in PROTOCOL_ACTIVITY_IDS):
            complete_subjects.append(subject)

    overall = count_vector(eligible_subjects, split_table)
    total_all = float(overall.sum())

    split_candidates = []
    for val_subject, test_subject in permutations(complete_subjects, 2):
        train_subjects = [
            s
            for s in eligible_subjects
            if s not in {val_subject, test_subject}
        ]

        train_vec = count_vector(train_subjects, split_table)
        val_vec = count_vector([val_subject], split_table)
        test_vec = count_vector([test_subject], split_table)

        if np.any(train_vec == 0) or np.any(val_vec == 0) or np.any(test_vec == 0):
            continue

        js_train = js_divergence(train_vec, overall)
        js_val = js_divergence(val_vec, overall)
        js_test = js_divergence(test_vec, overall)

        fractions = np.array(
            [train_vec.sum(), val_vec.sum(), test_vec.sum()],
            dtype=float,
        ) / max(total_all, 1.0)
        target = np.array([0.75, 0.125, 0.125], dtype=float)
        fraction_penalty = float(np.abs(fractions - target).sum())

        # Distribution fidelity is primary; size balance is secondary.
        objective = js_train + js_val + js_test + 0.50 * fraction_penalty

        split_candidates.append(
            {
                "rank": 0,
                "objective": objective,
                "js_train": js_train,
                "js_validation": js_val,
                "js_test": js_test,
                "fraction_penalty": fraction_penalty,
                "train_fraction": fractions[0],
                "validation_fraction": fractions[1],
                "test_fraction": fractions[2],
                "train_subjects": ",".join(train_subjects),
                "validation_subject": val_subject,
                "test_subject": test_subject,
            }
        )

    if not split_candidates:
        raise RuntimeError("No valid complete-class subject-wise split was found.")

    split_candidates.sort(
        key=lambda row: (
            row["objective"],
            row["validation_subject"],
            row["test_subject"],
        )
    )
    for idx, row in enumerate(split_candidates, start=1):
        row["rank"] = idx

    split_df = pd.DataFrame(split_candidates)
    split_df.to_csv(output_root / "subject_split_candidates.csv", index=False)

    best_split = split_candidates[0]

    # ------------------------------------------------------------------
    # Window candidate summary
    # ------------------------------------------------------------------
    window_summary_rows = []
    for name, cfg in WINDOWS.items():
        table = per_window_tables[name]
        total_windows = int(table["windows"].sum())
        min_class_windows = int(
            table.groupby("activity_id")["windows"].sum().min()
        )
        window_summary_rows.append(
            {
                "candidate": name,
                "window_samples": cfg["window"],
                "stride_samples": cfg["stride"],
                "window_seconds": cfg["window"] / SAMPLE_RATE_HZ,
                "overlap_pct": 100.0 * (1.0 - cfg["stride"] / cfg["window"]),
                "total_windows": total_windows,
                "minimum_class_windows": min_class_windows,
            }
        )

    window_summary = pd.DataFrame(window_summary_rows)
    window_summary.to_csv(output_root / "window_candidate_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Machine-readable design recommendation
    # ------------------------------------------------------------------
    recommendation = {
        "dataset_mode": "Protocol only",
        "exclude_activity_id_0": True,
        "exclude_subject109_from_core_experiment": True,
        "subject109_reason": (
            "Only rope jumping is present; retaining it as a core subject would "
            "create a highly atypical one-class participant."
        ),
        "main_feature_storage": "full36 valid inertial channels",
        "main_model_feature_mask": "core27",
        "full36_definition": FULL36_COLUMNS,
        "core27_definition": CORE27_COLUMNS,
        "excluded_from_main_features": {
            "heart_rate": (
                "90.87% missing in the audit and asynchronously sampled relative "
                "to the 100 Hz IMU streams."
            ),
            "temperature": "Slow-varying auxiliary signal; excluded from lightweight core.",
            "orientation": "Dataset documentation marks these fields invalid.",
        },
        "candidate_windows": WINDOWS,
        "split_selection_window": split_window_name,
        "recommended_subject_split": {
            "train_subjects": best_split["train_subjects"].split(","),
            "validation_subject": best_split["validation_subject"],
            "test_subject": best_split["test_subject"],
            "selection_objective": best_split["objective"],
        },
    }

    (output_root / "preprocessing_design_recommendation.json").write_text(
        json.dumps(recommendation, indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Human-readable report
    # ------------------------------------------------------------------
    hr_row = gap_summary[gap_summary["column"] == "heart_rate"].iloc[0]
    inertial_gaps = gap_summary[gap_summary["column"].isin(FULL36_COLUMNS)]

    report_lines = [
        "PAMAP2 PREPROCESSING DESIGN PROFILE",
        "=" * 78,
        f"Dataset root: {root}",
        f"Output root: {output_root}",
        "",
        "FROZEN DECISIONS BEFORE FINAL PREPROCESSING",
        "-" * 78,
        "1. Use Protocol recordings only.",
        "2. Remove activity_id = 0 and never bridge windows across those transitions.",
        "3. Exclude subject109 from the core experiment.",
        "4. Drop heart_rate from the main model; reserve it for a possible ablation.",
        "5. Drop temperature and invalid orientation channels from the lightweight core.",
        "6. Preserve 36 valid inertial channels in processed storage.",
        "7. Use a 27-channel core mask: 16g accelerometer + gyroscope + magnetometer.",
        "8. Keep the 6g accelerometer channels available for a later 27-vs-36 feature ablation.",
        "",
        "MISSING-RUN PROFILE",
        "-" * 78,
        f"Heart-rate missing runs: {int(hr_row['missing_runs']):,}",
        f"Heart-rate median missing run: {hr_row['median_run_samples']:.1f} samples",
        f"Heart-rate p99 missing run: {hr_row['p99_run_samples']:.1f} samples",
        f"Heart-rate maximum missing run: {hr_row['max_run_samples']:,} samples "
        f"({hr_row['max_run_seconds']:.2f} s at 100 Hz grid)",
        f"Maximum missing run among 36 inertial channels: "
        f"{int(inertial_gaps['max_run_samples'].max()):,} samples",
        "",
        "WINDOW CANDIDATES",
        "-" * 78,
    ]

    for row in window_summary.itertuples(index=False):
        report_lines.append(
            f"{row.candidate}: {row.window_seconds:.2f} s, "
            f"{row.overlap_pct:.0f}% overlap, "
            f"{row.total_windows:,} total windows, "
            f"minimum class count={row.minimum_class_windows:,}"
        )

    report_lines.extend(
        [
            "",
            "DETERMINISTIC SUBJECT-INDEPENDENT SPLIT RECOMMENDATION",
            "-" * 78,
            f"Train: {best_split['train_subjects']}",
            f"Validation: {best_split['validation_subject']}",
            f"Test: {best_split['test_subject']}",
            f"Objective score: {best_split['objective']:.6f}",
            f"Window fractions: train={best_split['train_fraction']:.3f}, "
            f"validation={best_split['validation_fraction']:.3f}, "
            f"test={best_split['test_fraction']:.3f}",
            "",
            "IMPORTANT",
            "-" * 78,
            "The split was selected only from label coverage and window-count distributions.",
            "No model was trained and no validation/test predictive result was used.",
            "",
            "FILES GENERATED",
            "-" * 78,
            "missing_gap_summary.csv",
            "activity_segment_summary.csv",
            "window_candidate_summary.csv",
            "window_counts_w256_s128.csv",
            "window_counts_w512_s256.csv",
            "subject_split_candidates.csv",
            "preprocessing_design_recommendation.json",
            "DESIGN_PROFILE_REPORT.txt",
        ]
    )

    report_path = output_root / "DESIGN_PROFILE_REPORT.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print()
    print("=== Profiling completed successfully ===")
    print(f"Complete subjects eligible for validation/test: {complete_subjects}")
    print(f"Recommended validation subject: {best_split['validation_subject']}")
    print(f"Recommended test subject: {best_split['test_subject']}")
    print(f"Report: {report_path}")
    print()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nProfiling interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
