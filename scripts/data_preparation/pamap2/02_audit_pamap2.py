from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "ERROR: pandas is required. Install it with:\n"
        "  py -m pip install pandas"
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

if len(COLUMNS) != 54:
    raise RuntimeError(f"Internal column definition error: expected 54, got {len(COLUMNS)}")

ACTIVITY_NAMES = {
    0: "other/transient",
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "Nordic walking",
    9: "watching TV",
    10: "computer work",
    11: "car driving",
    12: "ascending stairs",
    13: "descending stairs",
    16: "vacuum cleaning",
    17: "ironing",
    18: "folding laundry",
    19: "house cleaning",
    20: "playing soccer",
    24: "rope jumping",
}

PROTOCOL_ACTIVITY_IDS = {1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24}
ORIENTATION_COLUMNS = {
    f"{position}_orientation_{axis}"
    for position in IMU_POSITIONS
    for axis in range(1, 5)
}

SAMPLE_RATE_HZ = 100.0


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def infer_project_root(dataset_root: Path) -> Path:
    # Expected layout: <project>/data/raw/pamap2
    resolved = dataset_root.resolve()
    parts = [part.lower() for part in resolved.parts]
    try:
        idx = parts.index("data")
        return Path(*resolved.parts[:idx])
    except ValueError:
        return Path.cwd()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Perform a reproducible, chunked audit of the PAMAP2 raw files "
            "before preprocessing."
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
        help=(
            "Audit output directory. Default: "
            "<project>/outputs/audit/pamap2"
        ),
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Audit both Protocol and Optional recordings. Default: Protocol only.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Rows per pandas chunk (default: 100000).",
    )
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="Skip SHA-256 calculation for faster reruns.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    protocol_dir = root / "Protocol"
    optional_dir = root / "Optional"

    if not protocol_dir.is_dir():
        raise SystemExit(f"ERROR: Protocol directory not found: {protocol_dir}")

    project_root = infer_project_root(root)
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else project_root / "outputs" / "audit" / "pamap2"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    recordings: list[tuple[str, Path]] = [
        ("Protocol", path) for path in sorted(protocol_dir.glob("*.dat"))
    ]
    if args.include_optional:
        if not optional_dir.is_dir():
            raise SystemExit(f"ERROR: Optional directory not found: {optional_dir}")
        recordings.extend(
            ("Optional", path) for path in sorted(optional_dir.glob("*.dat"))
        )

    if not recordings:
        raise SystemExit("ERROR: no .dat files found.")

    total_missing = Counter()
    total_labels = Counter()
    subject_labels: dict[str, Counter[int]] = defaultdict(Counter)
    file_rows: Counter[str] = Counter()
    file_nonzero_rows: Counter[str] = Counter()
    file_min_timestamp: dict[str, float] = {}
    file_max_timestamp: dict[str, float] = {}
    file_timestamp_monotonic: dict[str, bool] = {}
    file_metadata: list[dict[str, object]] = []
    total_rows = 0

    print("=== PAMAP2 reproducible audit ===")
    print(f"Dataset root: {root}")
    print(f"Project root: {project_root}")
    print(f"Output root: {output_root}")
    print(f"Recordings: {len(recordings)}")
    print(f"Mode: {'Protocol + Optional' if args.include_optional else 'Protocol only'}")
    print(f"Chunk size: {args.chunk_size:,}")
    print()

    for index, (source, path) in enumerate(recordings, start=1):
        subject = path.stem.lower()
        key = f"{source}/{path.name}"
        rows = 0
        nonzero_rows = 0
        labels = Counter()
        missing = Counter()
        min_timestamp: float | None = None
        max_timestamp: float | None = None
        monotonic = True
        previous_last_timestamp: float | None = None

        print(f"[{index:>2}/{len(recordings)}] Auditing {key} ...")

        reader = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=COLUMNS,
            na_values=["NaN"],
            chunksize=args.chunk_size,
            low_memory=False,
        )

        for chunk in reader:
            chunk_rows = len(chunk)
            rows += chunk_rows
            total_rows += chunk_rows

            activity = pd.to_numeric(chunk["activity_id"], errors="coerce")
            if activity.isna().any():
                raise RuntimeError(f"Invalid activity_id found in {path}")

            activity_int = activity.astype("int64")
            counts = Counter(activity_int.value_counts().to_dict())
            labels.update(counts)
            total_labels.update(counts)
            subject_labels[subject].update(counts)

            nonzero_count = int((activity_int != 0).sum())
            nonzero_rows += nonzero_count

            missing_counts = chunk.isna().sum()
            for column, count in missing_counts.items():
                count_int = int(count)
                missing[column] += count_int
                total_missing[column] += count_int

            timestamps = pd.to_numeric(chunk["timestamp"], errors="coerce")
            if timestamps.isna().any():
                raise RuntimeError(f"Invalid timestamp found in {path}")

            chunk_min = float(timestamps.iloc[0])
            chunk_max = float(timestamps.iloc[-1])

            if min_timestamp is None:
                min_timestamp = chunk_min
            max_timestamp = chunk_max

            if not timestamps.is_monotonic_increasing:
                monotonic = False
            if previous_last_timestamp is not None and chunk_min < previous_last_timestamp:
                monotonic = False
            previous_last_timestamp = chunk_max

        file_rows[key] = rows
        file_nonzero_rows[key] = nonzero_rows
        file_min_timestamp[key] = float(min_timestamp if min_timestamp is not None else 0.0)
        file_max_timestamp[key] = float(max_timestamp if max_timestamp is not None else 0.0)
        file_timestamp_monotonic[key] = monotonic

        file_hash = None if args.skip_hash else sha256_file(path)

        file_metadata.append(
            {
                "source": source,
                "file": path.name,
                "subject": subject,
                "rows": rows,
                "usable_rows_activity_nonzero": nonzero_rows,
                "transient_rows_activity_zero": rows - nonzero_rows,
                "timestamp_start_s": file_min_timestamp[key],
                "timestamp_end_s": file_max_timestamp[key],
                "recording_span_s": file_max_timestamp[key] - file_min_timestamp[key],
                "timestamp_monotonic": monotonic,
                "activity_ids": ",".join(str(x) for x in sorted(labels)),
                "sha256": file_hash or "SKIPPED",
            }
        )

        print(
            f"       rows={rows:,}; usable={nonzero_rows:,}; "
            f"labels={sorted(labels)}; monotonic={monotonic}"
        )

    if total_rows == 0:
        raise SystemExit("ERROR: zero rows were audited.")

    # ------------------------------------------------------------------
    # Build audit tables
    # ------------------------------------------------------------------
    file_summary = pd.DataFrame(file_metadata)

    activity_rows = []
    for activity_id, count in sorted(total_labels.items()):
        activity_rows.append(
            {
                "activity_id": activity_id,
                "activity_name": ACTIVITY_NAMES.get(activity_id, "UNKNOWN"),
                "rows": count,
                "fraction_all_rows": count / total_rows,
                "estimated_minutes_at_100hz": count / SAMPLE_RATE_HZ / 60.0,
            }
        )
    activity_summary = pd.DataFrame(activity_rows)

    subject_activity_rows = []
    for subject, counts in sorted(subject_labels.items()):
        for activity_id, count in sorted(counts.items()):
            subject_activity_rows.append(
                {
                    "subject": subject,
                    "activity_id": activity_id,
                    "activity_name": ACTIVITY_NAMES.get(activity_id, "UNKNOWN"),
                    "rows": count,
                    "estimated_minutes_at_100hz": count / SAMPLE_RATE_HZ / 60.0,
                }
            )
    subject_activity_long = pd.DataFrame(subject_activity_rows)

    subject_activity_counts = (
        subject_activity_long.pivot_table(
            index="subject",
            columns="activity_id",
            values="rows",
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index()
        .reset_index()
    )
    subject_activity_counts.columns = [
        "subject" if col == "subject" else f"activity_{int(col)}_rows"
        for col in subject_activity_counts.columns
    ]

    missingness_rows = []
    for column in COLUMNS:
        missing_count = int(total_missing[column])
        missing_pct = 100.0 * missing_count / total_rows

        if column in {"timestamp", "activity_id"}:
            planned_action = "metadata/label"
            reason = "Required for temporal ordering or ground truth."
        elif column in ORIENTATION_COLUMNS:
            planned_action = "remove"
            reason = "Orientation fields are documented as invalid for this data collection."
        elif column == "heart_rate":
            planned_action = "retain_pending_imputation_audit"
            reason = "Low-rate heart-rate stream; missingness expected and must be handled temporally."
        else:
            planned_action = "retain"
            reason = "Raw sensor feature; preprocessing decision remains data-driven."

        missingness_rows.append(
            {
                "column": column,
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "planned_action_v1": planned_action,
                "reason": reason,
            }
        )
    missingness = pd.DataFrame(missingness_rows).sort_values(
        ["missing_pct", "column"], ascending=[False, True]
    )

    # Completeness matrix for protocol activities by subject.
    protocol_only = subject_activity_long[
        subject_activity_long["activity_id"].isin(PROTOCOL_ACTIVITY_IDS)
    ]
    coverage = (
        protocol_only.assign(present=1)
        .pivot_table(
            index="subject",
            columns="activity_id",
            values="present",
            aggfunc="max",
            fill_value=0,
        )
        .reindex(columns=sorted(PROTOCOL_ACTIVITY_IDS), fill_value=0)
        .reset_index()
    )
    coverage.columns = [
        "subject" if col == "subject" else f"activity_{int(col)}_present"
        for col in coverage.columns
    ]

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    file_summary.to_csv(output_root / "file_summary.csv", index=False)
    activity_summary.to_csv(output_root / "activity_summary.csv", index=False)
    subject_activity_long.to_csv(output_root / "subject_activity_long.csv", index=False)
    subject_activity_counts.to_csv(output_root / "subject_activity_counts.csv", index=False)
    coverage.to_csv(output_root / "subject_activity_coverage.csv", index=False)
    missingness.to_csv(output_root / "feature_missingness.csv", index=False)

    configuration = {
        "dataset_root": str(root),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "include_optional": bool(args.include_optional),
        "chunk_size": int(args.chunk_size),
        "hashes_computed": not args.skip_hash,
        "sample_rate_hz_for_duration_estimates": SAMPLE_RATE_HZ,
        "columns": COLUMNS,
        "protocol_activity_ids": sorted(PROTOCOL_ACTIVITY_IDS),
        "orientation_columns_planned_for_removal": sorted(ORIENTATION_COLUMNS),
    }
    (output_root / "audit_configuration.json").write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )

    nonzero_rows_total = total_rows - int(total_labels.get(0, 0))
    unexpected_protocol_labels = sorted(
        set(total_labels) - PROTOCOL_ACTIVITY_IDS - {0}
    )
    subjects_missing_protocol_activities: list[tuple[str, list[int]]] = []
    for subject, counts in sorted(subject_labels.items()):
        observed = set(counts) & PROTOCOL_ACTIVITY_IDS
        missing_ids = sorted(PROTOCOL_ACTIVITY_IDS - observed)
        if missing_ids:
            subjects_missing_protocol_activities.append((subject, missing_ids))

    highest_missing = missingness.head(15)

    report_lines = [
        "PAMAP2 AUDIT REPORT",
        "=" * 72,
        f"Dataset root: {root}",
        f"Mode: {'Protocol + Optional' if args.include_optional else 'Protocol only'}",
        f"Files audited: {len(recordings)}",
        f"Total rows: {total_rows:,}",
        f"Activity-0 transient rows: {int(total_labels.get(0, 0)):,} "
        f"({100.0 * int(total_labels.get(0, 0)) / total_rows:.2f}%)",
        f"Rows after planned removal of activity 0: {nonzero_rows_total:,}",
        f"Observed activity IDs: {sorted(total_labels)}",
        f"Unexpected labels for current mode: {unexpected_protocol_labels}",
        "",
        "SUBJECTS MISSING ONE OR MORE PROTOCOL ACTIVITIES",
        "-" * 72,
    ]

    if subjects_missing_protocol_activities:
        for subject, missing_ids in subjects_missing_protocol_activities:
            names = [ACTIVITY_NAMES.get(x, "UNKNOWN") for x in missing_ids]
            report_lines.append(
                f"{subject}: missing IDs {missing_ids} ({', '.join(names)})"
            )
    else:
        report_lines.append("None")

    report_lines.extend(
        [
            "",
            "TOP 15 FEATURES BY MISSINGNESS",
            "-" * 72,
        ]
    )
    for row in highest_missing.itertuples(index=False):
        report_lines.append(
            f"{row.column}: {row.missing_pct:.2f}% "
            f"[planned action: {row.planned_action_v1}]"
        )

    report_lines.extend(
        [
            "",
            "TIMESTAMP MONOTONICITY",
            "-" * 72,
        ]
    )
    for row in file_summary.itertuples(index=False):
        report_lines.append(
            f"{row.source}/{row.file}: {row.timestamp_monotonic}"
        )

    report_lines.extend(
        [
            "",
            "PREPROCESSING DECISIONS NOT YET FROZEN",
            "-" * 72,
            "1. Heart-rate imputation strategy.",
            "2. Exact temporal window length and overlap.",
            "3. Subject-independent train/validation/test split.",
            "4. Treatment of subject109 in the main protocol experiment.",
            "5. Whether both accelerometer ranges are retained in the final lightweight model.",
            "",
            "Files generated:",
            "  file_summary.csv",
            "  activity_summary.csv",
            "  subject_activity_long.csv",
            "  subject_activity_counts.csv",
            "  subject_activity_coverage.csv",
            "  feature_missingness.csv",
            "  audit_configuration.json",
        ]
    )

    report_path = output_root / "AUDIT_REPORT.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print()
    print("=== Audit completed successfully ===")
    print(f"Total rows: {total_rows:,}")
    print(
        f"Activity-0 rows: {int(total_labels.get(0, 0)):,} "
        f"({100.0 * int(total_labels.get(0, 0)) / total_rows:.2f}%)"
    )
    print(f"Usable nonzero rows: {nonzero_rows_total:,}")
    print(f"Observed labels: {sorted(total_labels)}")
    print(f"Output directory: {output_root}")
    print(f"Main report: {report_path}")
    print()

    if unexpected_protocol_labels and not args.include_optional:
        print(
            "WARNING: unexpected activity IDs were observed in Protocol mode: "
            f"{unexpected_protocol_labels}"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nAudit interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
