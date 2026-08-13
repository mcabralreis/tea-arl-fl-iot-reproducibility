from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "ERROR: numpy and pandas are required. Install them with:\n"
        "  py -m pip install numpy pandas"
    ) from exc


EXPECTED_SPLITS = {
    "train": {
        "count": 11014,
        "subjects": {102, 103, 104, 105, 106, 107},
    },
    "validation": {
        "count": 1932,
        "subjects": {101},
    },
    "test": {
        "count": 2026,
        "subjects": {108},
    },
}

EXPECTED_ACTIVITY_IDS = (1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24)
EXPECTED_CLASS_INDEX = {
    activity_id: class_index
    for class_index, activity_id in enumerate(EXPECTED_ACTIVITY_IDS)
}
EXPECTED_CORE27_INDICES = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8,
     18, 19, 20, 21, 22, 23, 24, 25, 26,
     27, 28, 29, 30, 31, 32, 33, 34, 35],
    dtype=np.int16,
)

WINDOW_SAMPLES = 256
FULL36_CHANNELS = 36


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise RuntimeError(message)


def check(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def load_memmap(path: Path) -> np.ndarray:
    if not path.is_file():
        fail(f"Missing required file: {path}")
    return np.load(path, mmap_mode="r")


def verify_inventory(dataset_root: Path) -> tuple[int, list[str]]:
    inventory_path = dataset_root / "output_inventory_sha256.json"
    if not inventory_path.is_file():
        fail(f"Hash inventory not found: {inventory_path}")

    entries = json.loads(inventory_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []

    print("=== CHECK 1/7: SHA-256 output inventory ===")
    for index, entry in enumerate(entries, start=1):
        relative = Path(entry["path"])
        path = dataset_root / relative
        if not path.is_file():
            mismatches.append(f"MISSING: {relative}")
            continue

        actual_size = path.stat().st_size
        if actual_size != int(entry["bytes"]):
            mismatches.append(
                f"SIZE: {relative} expected={entry['bytes']} actual={actual_size}"
            )
            continue

        actual_hash = sha256_file(path)
        if actual_hash != entry["sha256"]:
            mismatches.append(
                f"HASH: {relative} expected={entry['sha256']} actual={actual_hash}"
            )

        print(f"  [{index:>2}/{len(entries)}] {relative}")

    check(not mismatches, "Inventory verification failed:\n" + "\n".join(mismatches))
    print(f"[OK] {len(entries)} files match recorded sizes and SHA-256 hashes.\n")
    return len(entries), mismatches


def verify_configuration(dataset_root: Path) -> dict:
    print("=== CHECK 2/7: frozen preprocessing configuration ===")

    path = dataset_root / "preprocessing_configuration.json"
    config = json.loads(path.read_text(encoding="utf-8"))

    check(config["dataset_mode"] == "Protocol only", "Unexpected dataset_mode.")
    check(config["window_samples"] == 256, "Unexpected window size.")
    check(config["stride_samples"] == 128, "Unexpected stride.")
    check(
        config["max_internal_interpolation_gap_samples"] == 100,
        "Unexpected interpolation limit.",
    )
    check(
        config["train_subjects"]
        == ["subject102", "subject103", "subject104",
            "subject105", "subject106", "subject107"],
        "Unexpected training subjects.",
    )
    check(config["validation_subjects"] == ["subject101"], "Unexpected validation split.")
    check(config["test_subjects"] == ["subject108"], "Unexpected test split.")
    check(config["excluded_subjects"] == ["subject109"], "Unexpected excluded subjects.")
    check(len(config["full36_columns"]) == 36, "Full36 does not contain 36 channels.")
    check(len(config["core27_columns"]) == 27, "Core27 does not contain 27 channels.")

    print("[OK] Frozen protocol matches the approved design.\n")
    return config


def verify_statistics(dataset_root: Path, config: dict) -> dict[str, object]:
    print("=== CHECK 3/7: normalization statistics and Core27 mask ===")

    stats_dir = dataset_root / "statistics"
    mean = np.load(stats_dir / "training_mean_full36.npy")
    std = np.load(stats_dir / "training_std_full36.npy")
    core27 = np.load(stats_dir / "core27_indices.npy")

    check(mean.shape == (36,), f"Unexpected mean shape: {mean.shape}")
    check(std.shape == (36,), f"Unexpected std shape: {std.shape}")
    check(np.isfinite(mean).all(), "Training means contain non-finite values.")
    check(np.isfinite(std).all(), "Training standard deviations contain non-finite values.")
    check((std > 0).all(), "Training standard deviations must all be positive.")

    check(core27.shape == (27,), f"Unexpected Core27 shape: {core27.shape}")
    check(np.array_equal(core27, EXPECTED_CORE27_INDICES), "Core27 indices differ from frozen design.")
    check(len(np.unique(core27)) == 27, "Core27 indices are not unique.")
    check(int(core27.min()) >= 0 and int(core27.max()) < 36, "Core27 indices out of bounds.")

    config_indices = np.asarray(config["core27_indices_in_full36"], dtype=np.int16)
    check(np.array_equal(core27, config_indices), "Saved Core27 mask differs from configuration.")

    print("[OK] Training statistics are finite and Core27 mask is exact.\n")

    return {
        "training_mean_min": float(mean.min()),
        "training_mean_max": float(mean.max()),
        "training_std_min": float(std.min()),
        "training_std_max": float(std.max()),
    }


def verify_split_arrays(dataset_root: Path) -> tuple[dict[str, dict], list[tuple[int, int, float]]]:
    print("=== CHECK 4/7: split arrays, dtypes, labels, subjects and finiteness ===")

    split_dir = dataset_root / "splits"
    summaries: dict[str, dict] = {}
    metadata_keys_all: list[tuple[int, int, float]] = []

    for split, expected in EXPECTED_SPLITS.items():
        x = load_memmap(split_dir / f"{split}_X_full36.npy")
        y = load_memmap(split_dir / f"{split}_y.npy")
        activity = load_memmap(split_dir / f"{split}_activity_id.npy")
        subject = load_memmap(split_dir / f"{split}_subject_id.npy")
        timestamp = load_memmap(split_dir / f"{split}_start_timestamp_s.npy")

        n = expected["count"]

        check(x.shape == (n, WINDOW_SAMPLES, FULL36_CHANNELS), f"{split}: unexpected X shape {x.shape}")
        check(y.shape == (n,), f"{split}: unexpected y shape {y.shape}")
        check(activity.shape == (n,), f"{split}: unexpected activity shape {activity.shape}")
        check(subject.shape == (n,), f"{split}: unexpected subject shape {subject.shape}")
        check(timestamp.shape == (n,), f"{split}: unexpected timestamp shape {timestamp.shape}")

        check(x.dtype == np.float32, f"{split}: X dtype must be float32, got {x.dtype}")
        check(y.dtype == np.int16, f"{split}: y dtype must be int16, got {y.dtype}")
        check(activity.dtype == np.int16, f"{split}: activity dtype must be int16, got {activity.dtype}")
        check(subject.dtype == np.int16, f"{split}: subject dtype must be int16, got {subject.dtype}")
        check(timestamp.dtype == np.float64, f"{split}: timestamp dtype must be float64, got {timestamp.dtype}")

        # Chunked finiteness scan to avoid unnecessary RAM use.
        chunk_size = 256
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            check(
                np.isfinite(x[start:end]).all(),
                f"{split}: non-finite values found in X[{start}:{end}]",
            )

        check(np.isfinite(timestamp).all(), f"{split}: timestamps contain non-finite values.")

        unique_y = set(int(v) for v in np.unique(y))
        check(unique_y == set(range(12)), f"{split}: unexpected class indices {sorted(unique_y)}")

        unique_activity = set(int(v) for v in np.unique(activity))
        check(
            unique_activity == set(EXPECTED_ACTIVITY_IDS),
            f"{split}: unexpected activity IDs {sorted(unique_activity)}",
        )

        unique_subjects = set(int(v) for v in np.unique(subject))
        check(
            unique_subjects == expected["subjects"],
            f"{split}: subjects {sorted(unique_subjects)} != expected {sorted(expected['subjects'])}",
        )

        # y must agree exactly with the stored activity_id.
        mapped = np.array(
            [EXPECTED_CLASS_INDEX[int(value)] for value in activity],
            dtype=np.int16,
        )
        check(np.array_equal(mapped, y), f"{split}: y/activity_id mapping mismatch.")

        # Metadata tuple uniqueness within the split.
        keys = [
            (int(subject[i]), int(activity[i]), float(timestamp[i]))
            for i in range(n)
        ]
        check(
            len(keys) == len(set(keys)),
            f"{split}: duplicate (subject, activity, start_timestamp) keys detected.",
        )
        metadata_keys_all.extend(keys)

        # Report normalized-window statistics only as a sanity diagnostic.
        channel_sum = np.zeros(36, dtype=np.float64)
        channel_sq = np.zeros(36, dtype=np.float64)
        sample_count = 0

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            block = np.asarray(x[start:end], dtype=np.float64)
            channel_sum += block.sum(axis=(0, 1))
            channel_sq += np.square(block).sum(axis=(0, 1))
            sample_count += block.shape[0] * block.shape[1]

        channel_mean = channel_sum / sample_count
        channel_var = channel_sq / sample_count - np.square(channel_mean)
        channel_std = np.sqrt(np.maximum(channel_var, 0.0))

        summaries[split] = {
            "windows": n,
            "subjects": sorted(unique_subjects),
            "class_counts": {
                str(class_index): int(count)
                for class_index, count in sorted(Counter(int(v) for v in y).items())
            },
            "window_weighted_channel_mean_abs_max": float(np.max(np.abs(channel_mean))),
            "window_weighted_channel_std_min": float(channel_std.min()),
            "window_weighted_channel_std_max": float(channel_std.max()),
        }

        print(
            f"[OK] {split:>10}: X={x.shape}; subjects={sorted(unique_subjects)}; "
            f"all finite"
        )

    print()
    return summaries, metadata_keys_all


def verify_cross_split_disjointness(metadata_keys_all: list[tuple[int, int, float]]) -> None:
    print("=== CHECK 5/7: cross-split metadata disjointness ===")
    check(
        len(metadata_keys_all) == len(set(metadata_keys_all)),
        "Duplicate metadata keys exist across splits.",
    )

    train_subjects = EXPECTED_SPLITS["train"]["subjects"]
    validation_subjects = EXPECTED_SPLITS["validation"]["subjects"]
    test_subjects = EXPECTED_SPLITS["test"]["subjects"]

    check(train_subjects.isdisjoint(validation_subjects), "Train/validation subjects overlap.")
    check(train_subjects.isdisjoint(test_subjects), "Train/test subjects overlap.")
    check(validation_subjects.isdisjoint(test_subjects), "Validation/test subjects overlap.")

    print("[OK] Subject sets and window metadata are disjoint across splits.\n")


def verify_class_distribution_csv(dataset_root: Path, summaries: dict[str, dict]) -> None:
    print("=== CHECK 6/7: class-distribution table against array labels ===")

    path = dataset_root / "class_distribution_by_split.csv"
    table = pd.read_csv(path)

    for split in EXPECTED_SPLITS:
        rows = table[table["split"] == split].sort_values("class_index")
        check(len(rows) == 12, f"{split}: expected 12 rows in class-distribution CSV.")

        csv_counts = {
            str(int(row.class_index)): int(row.windows)
            for row in rows.itertuples(index=False)
        }
        check(
            csv_counts == summaries[split]["class_counts"],
            f"{split}: CSV class counts do not match y array.",
        )

    print("[OK] CSV class distributions exactly match the stored labels.\n")


def save_report(
    output_root: Path,
    dataset_root: Path,
    hashed_files: int,
    stats_summary: dict[str, object],
    split_summaries: dict[str, dict],
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "PROCESSED_DATA_VERIFICATION_REPORT.txt"

    total_windows = sum(summary["windows"] for summary in split_summaries.values())

    lines = [
        "PAMAP2 PROCESSED-DATA INDEPENDENT VERIFICATION REPORT",
        "=" * 78,
        f"Processed dataset: {dataset_root}",
        "",
        "OVERALL RESULT",
        "-" * 78,
        "PASS",
        "",
        "VERIFIED CONDITIONS",
        "-" * 78,
        f"Generated files matching SHA-256 inventory: {hashed_files}",
        "Frozen preprocessing configuration: exact match",
        "Training normalization statistics: finite and positive",
        "Core27 mask: exact match",
        "All split arrays: expected shape and dtype",
        "All feature values: finite",
        "All timestamps: finite",
        "All 12 classes: present in train, validation and test",
        "Activity ID to class-index mapping: exact",
        "Subject-independent split: exact",
        "Cross-split metadata overlap: none",
        "Class-distribution CSV versus arrays: exact match",
        "",
        "WINDOW COUNTS",
        "-" * 78,
        f"Train: {split_summaries['train']['windows']:,}",
        f"Validation: {split_summaries['validation']['windows']:,}",
        f"Test: {split_summaries['test']['windows']:,}",
        f"Total: {total_windows:,}",
        "",
        "NORMALIZATION STATISTICS FILES",
        "-" * 78,
        f"Training mean range: "
        f"{stats_summary['training_mean_min']:.6g} to "
        f"{stats_summary['training_mean_max']:.6g}",
        f"Training standard-deviation range: "
        f"{stats_summary['training_std_min']:.6g} to "
        f"{stats_summary['training_std_max']:.6g}",
        "",
        "WINDOW-WEIGHTED NORMALIZED ARRAY DIAGNOSTICS",
        "-" * 78,
    ]

    for split in ("train", "validation", "test"):
        summary = split_summaries[split]
        lines.extend(
            [
                f"{split}:",
                f"  maximum absolute channel mean = "
                f"{summary['window_weighted_channel_mean_abs_max']:.6f}",
                f"  channel std range = "
                f"{summary['window_weighted_channel_std_min']:.6f} to "
                f"{summary['window_weighted_channel_std_max']:.6f}",
            ]
        )

    lines.extend(
        [
            "",
            "NOTE ON THE 51-WINDOW DIFFERENCE FROM THE DESIGN PROFILE",
            "-" * 78,
            "The design profile counted 15,023 theoretical windows across all 9 subjects.",
            "The frozen core protocol excludes subject109, which contributes 48 theoretical",
            "rope-jumping windows. Three further windows were rejected because non-finite",
            "values remained after leakage-safe within-segment interpolation.",
            "Thus: 15,023 - 48 - 3 = 14,972 verified final windows.",
            "",
            "CONCLUSION",
            "-" * 78,
            "The processed PAMAP2 dataset is internally consistent and ready for model",
            "development and federated-learning experiments.",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Independently verify the frozen processed PAMAP2 dataset."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd()
        / "data"
        / "processed"
        / "pamap2"
        / "protocol_v1_w256_s128",
        help="Path to the processed PAMAP2 dataset directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: <project>/outputs/verification/pamap2",
    )
    args = parser.parse_args()

    dataset_root = args.root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"ERROR: processed dataset directory not found: {dataset_root}")

    # Expected project layout:
    # <project>/data/processed/pamap2/protocol_v1_w256_s128
    try:
        project_root = dataset_root.parents[3]
    except IndexError:
        project_root = Path.cwd()

    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else project_root / "outputs" / "verification" / "pamap2"
    )

    print("=== PAMAP2 independent processed-data verification ===")
    print(f"Processed dataset: {dataset_root}")
    print(f"Verification output: {output_root}")
    print()

    hashed_files, _ = verify_inventory(dataset_root)
    config = verify_configuration(dataset_root)
    stats_summary = verify_statistics(dataset_root, config)
    split_summaries, metadata_keys = verify_split_arrays(dataset_root)
    verify_cross_split_disjointness(metadata_keys)
    verify_class_distribution_csv(dataset_root, split_summaries)

    print("=== CHECK 7/7: final verification report ===")
    report_path = save_report(
        output_root,
        dataset_root,
        hashed_files,
        stats_summary,
        split_summaries,
    )
    print(f"[OK] Report written to: {report_path}\n")

    print("=== PAMAP2 verification completed successfully ===")
    print("OVERALL RESULT: PASS")
    print(f"Report: {report_path}")
    print()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nVerification interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
