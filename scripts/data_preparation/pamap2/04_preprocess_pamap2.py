from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "ERROR: numpy and pandas are required. Install them with:\n"
        "  py -m pip install numpy pandas"
    ) from exc


# =============================================================================
# Frozen PAMAP2 protocol
# =============================================================================

SAMPLE_RATE_HZ = 100.0
WINDOW_SAMPLES = 256
STRIDE_SAMPLES = 128
MAX_INTERPOLATION_GAP = 100

TRAIN_SUBJECTS = (
    "subject102",
    "subject103",
    "subject104",
    "subject105",
    "subject106",
    "subject107",
)
VALIDATION_SUBJECTS = ("subject101",)
TEST_SUBJECTS = ("subject108",)
EXCLUDED_SUBJECTS = ("subject109",)

PROTOCOL_ACTIVITY_IDS = (1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24)

ACTIVITY_NAMES = {
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

CLASS_INDEX = {
    activity_id: class_index
    for class_index, activity_id in enumerate(PROTOCOL_ACTIVITY_IDS)
}

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

ALL_COLUMNS = ["timestamp", "activity_id", "heart_rate"]
for position in IMU_POSITIONS:
    ALL_COLUMNS.extend(f"{position}_{field}" for field in IMU_FIELDS)

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
CORE27_INDICES = [FULL36_COLUMNS.index(column) for column in CORE27_COLUMNS]

READ_COLUMNS = ["timestamp", "activity_id"] + FULL36_COLUMNS


# =============================================================================
# Utilities
# =============================================================================

def infer_project_root(dataset_root: Path) -> Path:
    resolved = dataset_root.resolve()
    lower_parts = [part.lower() for part in resolved.parts]
    try:
        idx = lower_parts.index("data")
        return Path(*resolved.parts[:idx])
    except ValueError:
        return Path.cwd()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def subject_to_number(subject: str) -> int:
    digits = "".join(character for character in subject if character.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse numeric subject ID from {subject!r}")
    return int(digits)


def split_for_subject(subject: str) -> str | None:
    if subject in TRAIN_SUBJECTS:
        return "train"
    if subject in VALIDATION_SUBJECTS:
        return "validation"
    if subject in TEST_SUBJECTS:
        return "test"
    if subject in EXCLUDED_SUBJECTS:
        return None
    raise ValueError(f"Subject {subject!r} is not covered by the frozen split.")


def read_subject(path: Path) -> pd.DataFrame:
    """
    Read only the columns required by the frozen preprocessing protocol.
    """
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=ALL_COLUMNS,
        usecols=READ_COLUMNS,
        na_values=["NaN"],
        low_memory=False,
    )

    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="raise")
    frame["activity_id"] = pd.to_numeric(
        frame["activity_id"], errors="raise"
    ).astype(np.int16)

    for column in FULL36_COLUMNS:
        frame[column] = pd.to_numeric(
            frame[column], errors="coerce"
        ).astype(np.float32)

    return frame


def contiguous_activity_runs(activity: np.ndarray) -> Iterator[tuple[int, int, int]]:
    """
    Yield (start, end, activity_id) for every contiguous constant-label run.
    """
    if activity.size == 0:
        return

    change_points = np.flatnonzero(activity[1:] != activity[:-1]) + 1
    boundaries = np.concatenate(([0], change_points, [activity.size]))

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        yield int(start), int(end), int(activity[start])


def interpolate_segment(features: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate only internal gaps, never extrapolating at segment edges.

    The segment is already guaranteed to contain a single activity label and
    never crosses activity_id = 0.
    """
    frame = pd.DataFrame(features, columns=FULL36_COLUMNS)

    frame = frame.interpolate(
        method="linear",
        axis=0,
        limit=MAX_INTERPOLATION_GAP,
        limit_direction="both",
        limit_area="inside",
    )

    return frame.to_numpy(dtype=np.float32, copy=False)


def valid_window_starts(features: np.ndarray) -> np.ndarray:
    """
    Return starts of 256-sample windows with no remaining non-finite values.
    """
    length = int(features.shape[0])
    if length < WINDOW_SAMPLES:
        return np.empty(0, dtype=np.int64)

    starts = np.arange(
        0,
        length - WINDOW_SAMPLES + 1,
        STRIDE_SAMPLES,
        dtype=np.int64,
    )

    invalid_row = ~np.isfinite(features).all(axis=1)
    prefix = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.cumsum(invalid_row, dtype=np.int64),
        )
    )
    invalid_counts = prefix[starts + WINDOW_SAMPLES] - prefix[starts]
    return starts[invalid_counts == 0]


def iter_prepared_segments(path: Path) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """
    Yield (activity_id, timestamps, interpolated_full36_features) for every
    nonzero, supported, contiguous activity segment.
    """
    frame = read_subject(path)
    activity = frame["activity_id"].to_numpy(dtype=np.int16, copy=False)
    timestamps_all = frame["timestamp"].to_numpy(dtype=np.float64, copy=False)
    features_all = frame[FULL36_COLUMNS].to_numpy(dtype=np.float32, copy=False)

    for start, end, activity_id in contiguous_activity_runs(activity):
        if activity_id == 0:
            continue
        if activity_id not in CLASS_INDEX:
            raise RuntimeError(
                f"Unexpected nonzero activity ID {activity_id} in {path.name}"
            )

        timestamps = timestamps_all[start:end]
        features = interpolate_segment(features_all[start:end])

        yield activity_id, timestamps, features


# =============================================================================
# Pass 1: training-only normalization statistics and exact window counts
# =============================================================================

class RunningChannelStats:
    def __init__(self, channels: int) -> None:
        self.count = np.zeros(channels, dtype=np.int64)
        self.sum = np.zeros(channels, dtype=np.float64)
        self.sum_sq = np.zeros(channels, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        finite = np.isfinite(values)
        safe = np.where(finite, values, 0.0).astype(np.float64, copy=False)

        self.count += finite.sum(axis=0, dtype=np.int64)
        self.sum += safe.sum(axis=0, dtype=np.float64)
        self.sum_sq += np.square(safe).sum(axis=0, dtype=np.float64)

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if np.any(self.count == 0):
            empty = [
                FULL36_COLUMNS[index]
                for index, value in enumerate(self.count)
                if value == 0
            ]
            raise RuntimeError(
                f"No finite training values were observed for channels: {empty}"
            )

        mean = self.sum / self.count
        variance = self.sum_sq / self.count - np.square(mean)
        variance = np.maximum(variance, 0.0)
        std = np.sqrt(variance)

        if np.any(std < 1e-12):
            near_constant = [
                FULL36_COLUMNS[index]
                for index, value in enumerate(std)
                if value < 1e-12
            ]
            raise RuntimeError(
                f"Near-zero training standard deviation for channels: {near_constant}"
            )

        return mean.astype(np.float32), std.astype(np.float32)


def pass_one(
    protocol_dir: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, int],
    Counter[tuple[str, str, int]],
    dict[str, int],
]:
    print("=== PASS 1/2: exact counts and training-only normalization statistics ===")

    stats = RunningChannelStats(len(FULL36_COLUMNS))
    split_counts = {"train": 0, "validation": 0, "test": 0}
    detail_counts: Counter[tuple[str, str, int]] = Counter()
    discarded_windows = {"train": 0, "validation": 0, "test": 0}

    subjects = (
        list(VALIDATION_SUBJECTS)
        + list(TRAIN_SUBJECTS)
        + list(TEST_SUBJECTS)
    )

    for subject_index, subject in enumerate(subjects, start=1):
        split = split_for_subject(subject)
        if split is None:
            continue

        path = protocol_dir / f"{subject}.dat"
        if not path.is_file():
            raise FileNotFoundError(path)

        print(f"[{subject_index}/{len(subjects)}] {subject} -> {split}")

        subject_kept = 0
        subject_discarded = 0

        for activity_id, _timestamps, features in iter_prepared_segments(path):
            # Training normalization uses unique interpolated samples only,
            # avoiding repeated weighting from overlapping windows.
            if split == "train":
                stats.update(features)

            length = int(features.shape[0])
            possible = (
                0
                if length < WINDOW_SAMPLES
                else 1 + (length - WINDOW_SAMPLES) // STRIDE_SAMPLES
            )
            starts = valid_window_starts(features)
            kept = int(starts.size)
            discarded = int(possible - kept)

            split_counts[split] += kept
            discarded_windows[split] += discarded
            detail_counts[(split, subject, activity_id)] += kept

            subject_kept += kept
            subject_discarded += discarded

        print(
            f"    kept_windows={subject_kept:,}; "
            f"discarded_due_to_remaining_NaN={subject_discarded:,}"
        )

    mean, std = stats.finalize()

    print()
    print("Exact window counts:")
    for split in ("train", "validation", "test"):
        print(
            f"  {split:>10}: {split_counts[split]:,} "
            f"(discarded={discarded_windows[split]:,})"
        )
    print(f"  {'total':>10}: {sum(split_counts.values()):,}")
    print()

    return mean, std, split_counts, detail_counts, discarded_windows


# =============================================================================
# Pass 2: materialize normalized windows as memory-mappable .npy files
# =============================================================================

def allocate_split_files(
    output_root: Path,
    split_counts: dict[str, int],
) -> dict[str, dict[str, np.memmap]]:
    split_dir = output_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, dict[str, np.memmap]] = {}

    for split, count in split_counts.items():
        prefix = split_dir / split

        arrays[split] = {
            "X_full36": np.lib.format.open_memmap(
                f"{prefix}_X_full36.npy",
                mode="w+",
                dtype=np.float32,
                shape=(count, WINDOW_SAMPLES, len(FULL36_COLUMNS)),
            ),
            "y": np.lib.format.open_memmap(
                f"{prefix}_y.npy",
                mode="w+",
                dtype=np.int16,
                shape=(count,),
            ),
            "activity_id": np.lib.format.open_memmap(
                f"{prefix}_activity_id.npy",
                mode="w+",
                dtype=np.int16,
                shape=(count,),
            ),
            "subject_id": np.lib.format.open_memmap(
                f"{prefix}_subject_id.npy",
                mode="w+",
                dtype=np.int16,
                shape=(count,),
            ),
            "start_timestamp_s": np.lib.format.open_memmap(
                f"{prefix}_start_timestamp_s.npy",
                mode="w+",
                dtype=np.float64,
                shape=(count,),
            ),
        }

    return arrays


def pass_two(
    protocol_dir: Path,
    output_root: Path,
    mean: np.ndarray,
    std: np.ndarray,
    split_counts: dict[str, int],
) -> Counter[tuple[str, str, int]]:
    print("=== PASS 2/2: materializing normalized windows ===")

    arrays = allocate_split_files(output_root, split_counts)
    offsets = {"train": 0, "validation": 0, "test": 0}
    written_detail: Counter[tuple[str, str, int]] = Counter()

    subjects = (
        list(VALIDATION_SUBJECTS)
        + list(TRAIN_SUBJECTS)
        + list(TEST_SUBJECTS)
    )

    for subject_index, subject in enumerate(subjects, start=1):
        split = split_for_subject(subject)
        if split is None:
            continue

        path = protocol_dir / f"{subject}.dat"
        subject_number = subject_to_number(subject)

        print(f"[{subject_index}/{len(subjects)}] {subject} -> {split}")

        for activity_id, timestamps, features in iter_prepared_segments(path):
            starts = valid_window_starts(features)
            if starts.size == 0:
                continue

            normalized = (features - mean) / std
            class_index = CLASS_INDEX[activity_id]

            for start in starts:
                destination = offsets[split]
                end = int(start) + WINDOW_SAMPLES

                arrays[split]["X_full36"][destination] = normalized[int(start):end]
                arrays[split]["y"][destination] = class_index
                arrays[split]["activity_id"][destination] = activity_id
                arrays[split]["subject_id"][destination] = subject_number
                arrays[split]["start_timestamp_s"][destination] = timestamps[int(start)]

                offsets[split] += 1
                written_detail[(split, subject, activity_id)] += 1

    for split, expected in split_counts.items():
        actual = offsets[split]
        if actual != expected:
            raise RuntimeError(
                f"Window-count mismatch for {split}: expected {expected}, wrote {actual}"
            )

    # Flush memory maps.
    for split_arrays in arrays.values():
        for array in split_arrays.values():
            array.flush()

    # Release references before later hashing/reading.
    arrays.clear()

    print()
    print("Materialization completed:")
    for split in ("train", "validation", "test"):
        print(f"  {split:>10}: {offsets[split]:,}")
    print()

    return written_detail


# =============================================================================
# Reports and manifests
# =============================================================================

def save_statistics(
    output_root: Path,
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    stats_dir = output_root / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    np.save(stats_dir / "training_mean_full36.npy", mean)
    np.save(stats_dir / "training_std_full36.npy", std)
    np.save(
        stats_dir / "core27_indices.npy",
        np.asarray(CORE27_INDICES, dtype=np.int16),
    )

    rows = []
    for index, column in enumerate(FULL36_COLUMNS):
        rows.append(
            {
                "full36_index": index,
                "feature": column,
                "training_mean": float(mean[index]),
                "training_std": float(std[index]),
                "in_core27": column in CORE27_COLUMNS,
                "core27_index": (
                    CORE27_COLUMNS.index(column)
                    if column in CORE27_COLUMNS
                    else ""
                ),
            }
        )

    pd.DataFrame(rows).to_csv(
        stats_dir / "normalization_statistics.csv",
        index=False,
    )


def save_window_summary(
    output_root: Path,
    detail_counts: Counter[tuple[str, str, int]],
) -> pd.DataFrame:
    rows = []

    for split in ("train", "validation", "test"):
        subjects = {
            "train": TRAIN_SUBJECTS,
            "validation": VALIDATION_SUBJECTS,
            "test": TEST_SUBJECTS,
        }[split]

        for subject in subjects:
            for activity_id in PROTOCOL_ACTIVITY_IDS:
                rows.append(
                    {
                        "split": split,
                        "subject": subject,
                        "activity_id": activity_id,
                        "class_index": CLASS_INDEX[activity_id],
                        "activity_name": ACTIVITY_NAMES[activity_id],
                        "windows": int(
                            detail_counts[(split, subject, activity_id)]
                        ),
                    }
                )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / "window_distribution.csv", index=False)
    return summary


def file_inventory_with_hashes(output_root: Path) -> list[dict[str, object]]:
    inventory = []

    for path in sorted(output_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "output_inventory_sha256.json":
            continue

        inventory.append(
            {
                "path": str(path.relative_to(output_root)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    return inventory


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create the frozen PAMAP2 subject-independent, leakage-safe, "
            "windowed dataset for ARL-FL experiments."
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
            "Default: <project>/data/processed/pamap2/"
            "protocol_v1_w256_s128"
        ),
    )
    parser.add_argument(
        "--skip-output-hashes",
        action="store_true",
        help="Skip SHA-256 hashes of generated files.",
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
        else project_root
        / "data"
        / "processed"
        / "pamap2"
        / "protocol_v1_w256_s128"
    )

    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(
            "ERROR: output directory already exists and is not empty:\n"
            f"  {output_root}\n"
            "Remove it explicitly before rerunning to prevent accidental overwrite."
        )

    output_root.mkdir(parents=True, exist_ok=True)

    print("=== PAMAP2 frozen preprocessing pipeline ===")
    print(f"Raw dataset: {root}")
    print(f"Output:      {output_root}")
    print()
    print("Frozen protocol:")
    print(f"  Window: {WINDOW_SAMPLES} samples ({WINDOW_SAMPLES / SAMPLE_RATE_HZ:.2f} s)")
    print(f"  Stride: {STRIDE_SAMPLES} samples (50% overlap)")
    print(f"  Max internal interpolation gap: {MAX_INTERPOLATION_GAP} samples")
    print(f"  Training subjects: {', '.join(TRAIN_SUBJECTS)}")
    print(f"  Validation subject: {', '.join(VALIDATION_SUBJECTS)}")
    print(f"  Test subject: {', '.join(TEST_SUBJECTS)}")
    print(f"  Excluded subject: {', '.join(EXCLUDED_SUBJECTS)}")
    print(f"  Stored channels: {len(FULL36_COLUMNS)}")
    print(f"  Main-model channels: {len(CORE27_COLUMNS)}")
    print()

    # Save configuration before processing.
    configuration = {
        "dataset": "PAMAP2 Physical Activity Monitoring",
        "dataset_mode": "Protocol only",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "window_samples": WINDOW_SAMPLES,
        "window_seconds": WINDOW_SAMPLES / SAMPLE_RATE_HZ,
        "stride_samples": STRIDE_SAMPLES,
        "overlap_fraction": 1.0 - STRIDE_SAMPLES / WINDOW_SAMPLES,
        "max_internal_interpolation_gap_samples": MAX_INTERPOLATION_GAP,
        "interpolation": (
            "linear, internal gaps only, within contiguous same-activity "
            "segments; no extrapolation across segment edges"
        ),
        "activity_id_zero": "removed; windows never cross it",
        "train_subjects": list(TRAIN_SUBJECTS),
        "validation_subjects": list(VALIDATION_SUBJECTS),
        "test_subjects": list(TEST_SUBJECTS),
        "excluded_subjects": list(EXCLUDED_SUBJECTS),
        "activity_ids": list(PROTOCOL_ACTIVITY_IDS),
        "activity_names": {
            str(key): value for key, value in ACTIVITY_NAMES.items()
        },
        "class_index": {
            str(key): value for key, value in CLASS_INDEX.items()
        },
        "full36_columns": FULL36_COLUMNS,
        "core27_columns": CORE27_COLUMNS,
        "core27_indices_in_full36": CORE27_INDICES,
        "excluded_main_features": {
            "heart_rate": "90.87% missing on the 100 Hz grid",
            "temperature": "excluded from lightweight core",
            "orientation": "documented as invalid in this collection",
            "acceleration_6g": (
                "preserved in Full36 storage but excluded from Core27 main model"
            ),
        },
        "normalization": (
            "per-channel z-score using only unique interpolated samples "
            "from training subjects"
        ),
        "output_dtype": "float32",
    }

    (output_root / "preprocessing_configuration.json").write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )

    mean, std, split_counts, detail_counts, discarded_windows = pass_one(
        protocol_dir
    )

    save_statistics(output_root, mean, std)

    written_detail = pass_two(
        protocol_dir,
        output_root,
        mean,
        std,
        split_counts,
    )

    if written_detail != detail_counts:
        raise RuntimeError(
            "Detailed window counts differ between pass 1 and pass 2."
        )

    summary = save_window_summary(output_root, written_detail)

    split_class = (
        summary.groupby(
            ["split", "activity_id", "class_index", "activity_name"],
            as_index=False,
        )["windows"]
        .sum()
        .sort_values(["split", "class_index"])
    )
    split_class.to_csv(output_root / "class_distribution_by_split.csv", index=False)

    total_windows = int(sum(split_counts.values()))
    expected_from_design_profile = 15023
    difference_from_profile = total_windows - expected_from_design_profile

    report_lines = [
        "PAMAP2 FROZEN PREPROCESSING REPORT",
        "=" * 78,
        f"Raw dataset: {root}",
        f"Processed dataset: {output_root}",
        "",
        "FROZEN PROTOCOL",
        "-" * 78,
        f"Window: {WINDOW_SAMPLES} samples ({WINDOW_SAMPLES / SAMPLE_RATE_HZ:.2f} s)",
        f"Stride: {STRIDE_SAMPLES} samples (50% overlap)",
        f"Internal interpolation limit: {MAX_INTERPOLATION_GAP} samples",
        "Windows never cross activity boundaries or activity_id = 0.",
        "No extrapolation is performed at segment edges.",
        "",
        "SUBJECT-INDEPENDENT SPLIT",
        "-" * 78,
        f"Train: {','.join(TRAIN_SUBJECTS)}",
        f"Validation: {','.join(VALIDATION_SUBJECTS)}",
        f"Test: {','.join(TEST_SUBJECTS)}",
        f"Excluded from core experiment: {','.join(EXCLUDED_SUBJECTS)}",
        "",
        "FEATURES",
        "-" * 78,
        f"Stored representation: Full36 ({len(FULL36_COLUMNS)} channels)",
        f"Main-model mask: Core27 ({len(CORE27_COLUMNS)} channels)",
        "Core27 = 16g accelerometer + gyroscope + magnetometer from all 3 IMUs.",
        "The 6g accelerometer channels remain available for a future feature ablation.",
        "",
        "NORMALIZATION",
        "-" * 78,
        "Per-channel z-score statistics were computed exclusively from training subjects.",
        "Statistics use unique interpolated training samples, not duplicated overlapping windows.",
        "",
        "WINDOW COUNTS",
        "-" * 78,
        f"Train: {split_counts['train']:,}",
        f"Validation: {split_counts['validation']:,}",
        f"Test: {split_counts['test']:,}",
        f"Total: {total_windows:,}",
        f"Difference from design-profile theoretical count (15,023): {difference_from_profile:+,}",
        "",
        "WINDOWS DISCARDED DUE TO REMAINING NON-FINITE VALUES",
        "-" * 78,
        f"Train: {discarded_windows['train']:,}",
        f"Validation: {discarded_windows['validation']:,}",
        f"Test: {discarded_windows['test']:,}",
        "",
        "OUTPUT ACCESS",
        "-" * 78,
        "Use numpy memory mapping to avoid loading all windows into RAM:",
        "  X = np.load('..._X_full36.npy', mmap_mode='r')",
        "  X_core27 = X[:, :, core27_indices]",
        "",
        "FILES",
        "-" * 78,
        "splits/*_X_full36.npy",
        "splits/*_y.npy",
        "splits/*_activity_id.npy",
        "splits/*_subject_id.npy",
        "splits/*_start_timestamp_s.npy",
        "statistics/training_mean_full36.npy",
        "statistics/training_std_full36.npy",
        "statistics/core27_indices.npy",
        "statistics/normalization_statistics.csv",
        "window_distribution.csv",
        "class_distribution_by_split.csv",
        "preprocessing_configuration.json",
    ]

    report_path = output_root / "PREPROCESSING_REPORT.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    if not args.skip_output_hashes:
        print("Computing SHA-256 hashes of generated outputs...")
        inventory = file_inventory_with_hashes(output_root)
        (output_root / "output_inventory_sha256.json").write_text(
            json.dumps(inventory, indent=2),
            encoding="utf-8",
        )
        print(f"[OK] Hashed {len(inventory)} generated files.")
    else:
        print("Output hashing skipped by request.")

    print()
    print("=== PAMAP2 preprocessing completed successfully ===")
    print(f"Processed dataset: {output_root}")
    print(f"Total windows: {total_windows:,}")
    print(f"Report: {report_path}")
    print()
    print("Core27 indices within Full36:")
    print(CORE27_INDICES)
    print()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nPreprocessing interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
