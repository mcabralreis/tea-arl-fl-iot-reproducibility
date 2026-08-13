from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "derived" / "ciciot2023"
RAW = DATA / "task7_unique_exact_v1"
EFFECTIVE = DATA / "task7_transformed_unique_f32_index_v1"
SPLIT = DATA / "task7_transformed_unique_f32_index_v1_split_80_10_10_v1"
OUT = ROOT / "results" / "revised_analysis"
BUCKETS = 256
SPLIT_NAMES = ["train", "validation", "test"]


def bucket(root: Path, index: int) -> Path:
    return root / f"bucket_{index:03d}"


def build_global_raw_file_csr() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_lengths = []
    counts_parts = []
    file_parts = []
    for bucket_index in range(BUCKETS):
        root = bucket(RAW, bucket_index)
        metadata = np.load(root / "observation_metadata.npy", mmap_mode="r")
        indptr = np.load(root / "provenance_indptr_i64.npy", mmap_mode="r")
        files = np.load(root / "file_provenance_edges.npy", mmap_mode="r")
        raw_lengths.append(len(metadata))
        counts_parts.append(np.diff(indptr[:, 1]).astype(np.uint16, copy=False))
        file_parts.append(np.asarray(files["FileID"], dtype=np.uint16))
    raw_offsets = np.zeros(BUCKETS + 1, dtype=np.int64)
    raw_offsets[1:] = np.cumsum(np.asarray(raw_lengths, dtype=np.int64))
    counts = np.concatenate(counts_parts)
    global_indptr = np.empty(len(counts) + 1, dtype=np.int64)
    global_indptr[0] = 0
    np.cumsum(counts, dtype=np.int64, out=global_indptr[1:])
    global_file_ids = np.concatenate(file_parts)
    if global_indptr[-1] != len(global_file_ids):
        raise RuntimeError("Global raw-file provenance CSR is inconsistent")
    return raw_offsets, global_indptr, global_file_ids


def effective_file_pairs(
    bucket_index: int,
    raw_offsets: np.ndarray,
    raw_file_indptr: np.ndarray,
    raw_file_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    effective_root = bucket(EFFECTIVE, bucket_index)
    split_root = bucket(SPLIT, bucket_index)
    index = np.load(effective_root / "effective_observation_index.npy", mmap_mode="r")
    source_indptr = np.load(
        effective_root / "effective_source_indptr_i64.npy", mmap_mode="r"
    )
    source_edges = np.load(
        effective_root / "effective_source_edges.npy", mmap_mode="r"
    )
    split_ids = np.load(split_root / "split_id_u8.npy", mmap_mode="r")
    source_counts = np.diff(source_indptr)
    observation_for_source = np.repeat(
        np.arange(len(index), dtype=np.int64), source_counts
    )
    global_source_row = (
        raw_offsets[np.asarray(source_edges["SourceBucket"], dtype=np.int64)]
        + np.asarray(source_edges["SourceRowIndex"], dtype=np.int64)
    )
    starts = raw_file_indptr[global_source_row]
    file_counts = raw_file_indptr[global_source_row + 1] - starts
    prefix = np.cumsum(file_counts, dtype=np.int64) - file_counts
    flat_file_edge_index = np.arange(
        int(file_counts.sum()), dtype=np.int64
    ) + np.repeat(starts - prefix, file_counts)
    file_ids = raw_file_ids[flat_file_edge_index]
    observation_for_file = np.repeat(observation_for_source, file_counts)
    number_files = int(pd.read_csv(RAW / "RAW_FILE_MAP.csv")["FileID"].max()) + 1
    pair_key = observation_for_file * number_files + file_ids.astype(np.int64)
    unique_pair_key = np.unique(pair_key)
    unique_observation = unique_pair_key // number_files
    unique_file = (unique_pair_key % number_files).astype(np.int64)
    unique_split = np.asarray(split_ids[unique_observation], dtype=np.int64)
    unique_source_file_count = np.bincount(
        unique_observation, minlength=len(index)
    ).astype(np.int32)
    return (
        unique_observation,
        unique_file,
        unique_split,
        unique_source_file_count,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    file_map = pd.read_csv(RAW / "RAW_FILE_MAP.csv").sort_values("FileID")
    number_files = int(file_map["FileID"].max()) + 1
    raw_offsets, raw_file_indptr, raw_file_ids = build_global_raw_file_csr()

    file_split_counts = np.zeros((number_files, 3), dtype=np.int64)
    source_file_count_histogram: dict[int, int] = {}
    source_exact_count_histogram: dict[int, int] = {}
    split_observation_counts = np.zeros(3, dtype=np.int64)
    total_effective_observations = 0

    for bucket_index in range(BUCKETS):
        effective_root = bucket(EFFECTIVE, bucket_index)
        split_root = bucket(SPLIT, bucket_index)
        source_indptr = np.load(
            effective_root / "effective_source_indptr_i64.npy", mmap_mode="r"
        )
        split_ids = np.load(split_root / "split_id_u8.npy", mmap_mode="r")
        total_effective_observations += len(split_ids)
        split_observation_counts += np.bincount(split_ids, minlength=3)
        source_exact_counts = np.diff(source_indptr)
        values, counts = np.unique(source_exact_counts, return_counts=True)
        for value, count in zip(values, counts):
            source_exact_count_histogram[int(value)] = (
                source_exact_count_histogram.get(int(value), 0) + int(count)
            )
        _, files, splits, file_count_per_observation = effective_file_pairs(
            bucket_index, raw_offsets, raw_file_indptr, raw_file_ids
        )
        file_split_counts += np.bincount(
            files * 3 + splits, minlength=number_files * 3
        ).reshape(number_files, 3)
        values, counts = np.unique(file_count_per_observation, return_counts=True)
        for value, count in zip(values, counts):
            source_file_count_histogram[int(value)] = (
                source_file_count_histogram.get(int(value), 0) + int(count)
            )

    file_presence = file_split_counts > 0
    partitions_per_file = file_presence.sum(axis=1)
    file_mask = (
        file_presence[:, 0].astype(np.uint8)
        + 2 * file_presence[:, 1].astype(np.uint8)
        + 4 * file_presence[:, 2].astype(np.uint8)
    )

    observations_any_multisplit_source = 0
    observations_all_sources_all_three = 0
    observations_multiple_source_files = 0
    for bucket_index in range(BUCKETS):
        observations, files, _, file_count_per_observation = effective_file_pairs(
            bucket_index, raw_offsets, raw_file_indptr, raw_file_ids
        )
        multisplit = partitions_per_file[files] >= 2
        all_three = partitions_per_file[files] == 3
        any_multisplit_by_observation = np.zeros(
            len(file_count_per_observation), dtype=bool
        )
        all_three_by_observation = np.ones(
            len(file_count_per_observation), dtype=bool
        )
        np.logical_or.at(any_multisplit_by_observation, observations, multisplit)
        not_all_three = ~all_three
        has_not_all_three = np.zeros(len(file_count_per_observation), dtype=bool)
        np.logical_or.at(has_not_all_three, observations, not_all_three)
        all_three_by_observation &= ~has_not_all_three
        observations_any_multisplit_source += int(any_multisplit_by_observation.sum())
        observations_all_sources_all_three += int(all_three_by_observation.sum())
        observations_multiple_source_files += int(
            np.sum(file_count_per_observation > 1)
        )

    audit = file_map.copy()
    for split_index, split_name in enumerate(SPLIT_NAMES):
        audit[f"effective_observations_{split_name}"] = file_split_counts[:, split_index]
    audit["effective_observation_file_associations"] = file_split_counts.sum(axis=1)
    for split_name in SPLIT_NAMES:
        audit[f"percent_{split_name}_within_source_file"] = np.where(
            audit["effective_observation_file_associations"] > 0,
            100.0
            * audit[f"effective_observations_{split_name}"]
            / audit["effective_observation_file_associations"],
            0.0,
        )
    audit["partitions_present"] = partitions_per_file
    audit["partition_presence_mask"] = file_mask
    audit.to_csv(OUT / "CICIOT2023_SOURCE_FILE_PARTITION_AUDIT.csv", index=False)

    summary = {
        "audit_unit": "effective observation with union of all retained source-file provenance",
        "effective_observations": total_effective_observations,
        "split_observation_counts": {
            name: int(split_observation_counts[index])
            for index, name in enumerate(SPLIT_NAMES)
        },
        "source_files": number_files,
        "source_files_in_exactly_one_partition": int(np.sum(partitions_per_file == 1)),
        "source_files_in_exactly_two_partitions": int(np.sum(partitions_per_file == 2)),
        "source_files_in_all_three_partitions": int(np.sum(partitions_per_file == 3)),
        "effective_observations_with_any_source_file_spanning_multiple_partitions": observations_any_multisplit_source,
        "percent_effective_observations_with_any_source_file_spanning_multiple_partitions": 100.0
        * observations_any_multisplit_source
        / total_effective_observations,
        "effective_observations_for_which_all_source_files_span_all_three_partitions": observations_all_sources_all_three,
        "percent_effective_observations_for_which_all_source_files_span_all_three_partitions": 100.0
        * observations_all_sources_all_three
        / total_effective_observations,
        "effective_observations_with_multiple_source_files": observations_multiple_source_files,
        "percent_effective_observations_with_multiple_source_files": 100.0
        * observations_multiple_source_files
        / total_effective_observations,
        "source_file_count_per_effective_observation_histogram": {
            str(key): value
            for key, value in sorted(source_file_count_histogram.items())
        },
        "source_exact_vector_count_per_effective_observation_histogram": {
            str(key): value
            for key, value in sorted(source_exact_count_histogram.items())
        },
        "available_provenance_fields": [
            "FileID",
            "RelativePath",
            "OriginalFineClass",
        ],
        "unavailable_grouping_fields": [
            "physical device identifier",
            "capture-session identifier",
            "attack-run or episode identifier distinct from source file",
        ],
    }
    (OUT / "CICIOT2023_SOURCE_OVERLAP_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
