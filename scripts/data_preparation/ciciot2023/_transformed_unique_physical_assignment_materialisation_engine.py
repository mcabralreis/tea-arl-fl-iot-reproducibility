import argparse
import csv
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


# ============================================================================
# Immutable current-chain bindings
# ============================================================================

EFFECTIVE_DATASET_SHA = (
    "5708EFE6C08C91CF3637FA8F89F53C4459933F94C7CC0BF819A590CBE9EF8E5D"
)

SPLIT_PROTOCOL_SHA = (
    "C3E938C506BB3342566694A3227FF5B9734E721C5BAF7AB8E9A02E559AD21AD5"
)

SPLIT_ASSIGNMENT_SHA = (
    "DBF8B64C451E5DAB7AFA87FE88B5E9FFE6E7436FFAEAAFB66692076A5C16C276"
)

CLIENT_PARTITION_PROTOCOL_SHA = (
    "7FF24F0A50C283EDC41053F81E7F8DB38A2864411C7E85B97BD8E9BEEEA7BB32"
)

CAPACITY_VECTOR_SHA = (
    "4876137D26ADE608583DE67560E3788E8ED8C3AA8327CB825C8AFCF16F0B11F0"
)

COUNT_PLAN_SET_SHA = (
    "CAD5F8DB2C71CC17614D039E39F792C31F6B1F2C5F8CB3E23EE28FF2BD0BD705"
)

PHYSICAL_ASSIGNMENT_PROTOCOL_SHA = (
    "CF378EB71CDF283C2AB44C4E53CAB0E803672B3902CC92860005FD14635C0435"
)

ORDERING_SEED_MANIFEST_SHA = (
    "ABBABEA1D98D99E0EF3296C7257B75B6B97CA38CD17D53AF8ED7CD468E2EBEC8"
)

SLICE_BOUNDARY_SHA = (
    "2F6A628B3FE1CF3527A95A7B588DDF7123BBA4AFA5E0B57F6DE1EC9F9548ADA8"
)


# ============================================================================
# Protocol identifiers
# ============================================================================

MATERIALISATION_ID = (
    "CICIoT2023_TRANSFORMED_UNIQUE_K30_PHYSICAL_ASSIGNMENT_MATERIALISATION_V1"
)

ASSIGNMENT_LAYER_SET_ID = (
    "CICIoT2023_TASK7_TRANSFORMED_UNIQUE_K30_PHYSICAL_ASSIGNMENT_SET_V2"
)

PHYSICAL_ASSIGNMENT_PROTOCOL_ID = (
    "CICIoT2023_TASK7_TRANSFORMED_UNIQUE_K30_"
    "CONFIG_BOUND_HASH_ORDER_SLICE_ASSIGNMENT_V1"
)

ORDERING_ALGORITHM_ID = (
    "SPLITMIX64_DUAL_KEY_CONFIG_BOUND_LABEL_ORDER_V1"
)

ASSIGNMENT_CONTENT_HASH_ID = (
    "CONFIG_BUCKET_ASCENDING_RAW_UINT8_CLIENT_LAYER_SHA256_V1"
)

ASSIGNMENT_SET_HASH_ID = (
    "SORTED_CONFIG_ID_PLUS_CONFIG_ASSIGNMENT_CONTENT_SHA256_LF_V1"
)

ARTIFACT_MANIFEST_HASH_ID = (
    "SORTED_ROLE_RELATIVEPATH_SIZE_SHA256_LF_V1"
)


# ============================================================================
# Dataset geometry
# ============================================================================

BUCKETS = 256
LABELS = 7
CLIENTS = 30
CONFIGS = 10

EFFECTIVE_OBSERVATIONS = 20_687_272
TRAIN_TOTAL = 16_549_824
VALIDATION_TOTAL = 2_068_724
TEST_TOTAL = 2_068_724

LABEL_NAMES = {
    0: "Benign",
    1: "Brute Force",
    2: "DoS_DDoS",
    3: "Mirai",
    4: "Recon",
    5: "Spoofing",
    6: "Web",
}

TRAIN_COUNTS = {
    0: 874_262,
    1: 10_436,
    2: 12_773_255,
    3: 1_963_693,
    4: 545_764,
    5: 362_726,
    6: 19_688,
}

EXPECTED_CAPACITIES = (
    [551_661] * 24
    +
    [551_660] * 6
)

NON_CLIENT_SENTINEL = np.uint8(255)


# ============================================================================
# Source schemas
# ============================================================================

EFFECTIVE_INDEX_DTYPE = np.dtype(
    [
        ("Task7LabelID", "u1"),
        ("RepresentativeSourceBucket", "u1"),
        ("RepresentativeSourceRowIndex", "<u4"),
        ("TransformedGroupRawExactMultiplicity", "<u4"),
        ("SumSourceRawMultiplicity", "<u8"),
        ("TransformedHash1", "<u8"),
        ("TransformedHash2", "<u8"),
        ("RepresentativeSignatureH1", "<u8"),
        ("RepresentativeSignatureH2", "<u8"),
        ("RepresentativeExactCollisionOrdinal", "<u2"),
        ("OldSplitMask", "u1"),
        ("OldTrainSourceRows", "<u4"),
        ("OldValidationSourceRows", "<u4"),
        ("OldTestSourceRows", "<u4"),
    ],
    align=False,
)

TRAIN_IDENTITY_DTYPE = np.dtype(
    [
        ("TransformedHash1", "<u8"),
        ("TransformedHash2", "<u8"),
        ("EffectiveBucket", "u1"),
        ("EffectiveRowIndex", "<u4"),
    ],
    align=False,
)

HASH_PAIR_DTYPE = np.dtype(
    [
        ("TransformedHash1", "<u8"),
        ("TransformedHash2", "<u8"),
    ],
    align=False,
)


# ============================================================================
# Ordering constants
# ============================================================================

DUAL_KEY_XOR_CONSTANT = np.uint64(
    0xD1B54A32D192ED03
)


# ============================================================================
# Utilities
# ============================================================================

def load_json(path):
    with Path(path).open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        return json.load(handle)


def read_csv(path):
    with Path(path).open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def write_json(path, obj):
    Path(path).write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def write_csv(path, rows, fields):
    with Path(path).open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def require_equal(observed, expected, message):
    if observed != expected:
        raise RuntimeError(
            f"{message}: expected {expected}, observed {observed}"
        )


def require_true(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as handle:
        while True:
            block = handle.read(
                8
                *
                1024
                *
                1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest().upper()


def artifact_manifest_digest(rows):
    digest = hashlib.sha256()

    for row in sorted(
        rows,
        key=lambda item: (
            item["ArtifactRole"],
            item["RelativePath"],
        ),
    ):
        digest.update(
            (
                f"{row['ArtifactRole']}\t"
                f"{row['RelativePath']}\t"
                f"{row['SizeBytes']}\t"
                f"{row['SHA256']}\n"
            ).encode("utf-8")
        )

    return digest.hexdigest().upper()


def close_memmap(array):
    if array is None:
        return

    mmap_object = getattr(
        array,
        "_mmap",
        None,
    )

    if mmap_object is not None:
        mmap_object.close()


def effective_index_path(
    effective_root,
    bucket_id,
):
    return (
        Path(effective_root)
        /
        f"bucket_{bucket_id:03d}"
        /
        "effective_observation_index.npy"
    )


def split_array_path(
    split_root,
    bucket_id,
):
    return (
        Path(split_root)
        /
        f"bucket_{bucket_id:03d}"
        /
        "split_id_u8.npy"
    )


def client_array_path(
    config_root,
    bucket_id,
):
    return (
        Path(config_root)
        /
        f"bucket_{bucket_id:03d}"
        /
        "client_id_u8.npy"
    )


def splitmix64_vectorized(x):
    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        z = np.asarray(
            x,
            dtype=np.uint64,
        ).copy()

        z = (
            z
            +
            np.uint64(
                0x9E3779B97F4A7C15
            )
        )

        z = (
            (
                z
                ^
                (
                    z
                    >>
                    np.uint64(30)
                )
            )
            *
            np.uint64(
                0xBF58476D1CE4E5B9
            )
        )

        z = (
            (
                z
                ^
                (
                    z
                    >>
                    np.uint64(27)
                )
            )
            *
            np.uint64(
                0x94D049BB133111EB
            )
        )

        z = (
            z
            ^
            (
                z
                >>
                np.uint64(31)
            )
        )

    return z.astype(
        np.uint64,
        copy=False,
    )


def rotl64_scalar(value, shift):
    shift = shift % 64

    return (
        (
            (
                value
                <<
                shift
            )
            &
            0xFFFFFFFFFFFFFFFF
        )
        |
        (
            value
            >>
            (
                64
                -
                shift
            )
        )
    )


def derive_order_keys_vectorized(
    transformed_hash1,
    transformed_hash2,
    seed_uint64,
):
    seed = np.uint64(
        int(
            seed_uint64
        )
    )

    rotated_seed = np.uint64(
        rotl64_scalar(
            int(
                seed_uint64
            ),
            32,
        )
    )

    key1 = splitmix64_vectorized(
        np.asarray(
            transformed_hash1,
            dtype=np.uint64,
        )
        ^
        seed
    )

    key2 = splitmix64_vectorized(
        np.asarray(
            transformed_hash2,
            dtype=np.uint64,
        )
        ^
        rotated_seed
        ^
        DUAL_KEY_XOR_CONSTANT
    )

    return (
        key1,
        key2,
    )


def deterministic_order(
    records,
    seed_uint64,
):
    key1, key2 = derive_order_keys_vectorized(
        records[
            "TransformedHash1"
        ],
        records[
            "TransformedHash2"
        ],
        seed_uint64,
    )

    order = np.lexsort(
        (
            records[
                "TransformedHash2"
            ],
            records[
                "TransformedHash1"
            ],
            key2,
            key1,
        )
    )

    return (
        order,
        key1,
        key2,
    )


def order_digest(
    records,
    order,
    chunk_size=1_000_000,
):
    digest = hashlib.sha256()

    for start in range(
        0,
        order.shape[0],
        chunk_size,
    ):
        end = min(
            order.shape[0],
            start
            +
            chunk_size,
        )

        indices = order[
            start:end
        ]

        pairs = np.empty(
            indices.shape[0],
            dtype=HASH_PAIR_DTYPE,
        )

        pairs[
            "TransformedHash1"
        ] = records[
            "TransformedHash1"
        ][
            indices
        ]

        pairs[
            "TransformedHash2"
        ] = records[
            "TransformedHash2"
        ][
            indices
        ]

        digest.update(
            pairs.tobytes(
                order="C"
            )
        )

    return digest.hexdigest().upper()


def seed_manifest_sha256(rows):
    digest = hashlib.sha256()

    for row in sorted(
        rows,
        key=lambda item: item[
            "ConfigID"
        ],
    ):
        digest.update(
            (
                f"{row['ConfigID']}\t"
                f"{row['SeedSHA256']}\t"
                f"{row['SeedUInt64']}\n"
            ).encode("utf-8")
        )

    return digest.hexdigest().upper()


def slice_boundary_sha256(rows):
    digest = hashlib.sha256()

    for row in sorted(
        rows,
        key=lambda item: (
            item["ConfigID"],
            int(item["Task7LabelID"]),
            int(item["ClientID"]),
        ),
    ):
        digest.update(
            (
                f"{row['ConfigID']}\t"
                f"{row['Task7LabelID']}\t"
                f"{row['ClientID']}\t"
                f"{row['StartInclusive']}\t"
                f"{row['EndExclusive']}\t"
                f"{row['Count']}\n"
            ).encode("utf-8")
        )

    return digest.hexdigest().upper()


def matrix_content_sha256(matrix):
    contiguous = np.ascontiguousarray(
        matrix,
        dtype=np.int64,
    )

    return hashlib.sha256(
        contiguous.tobytes(
            order="C"
        )
    ).hexdigest().upper()


def assignment_set_sha256(rows):
    digest = hashlib.sha256()

    for row in sorted(
        rows,
        key=lambda item: item[
            "ConfigID"
        ],
    ):
        digest.update(
            (
                f"{row['ConfigID']}\t"
                f"{row['AssignmentContentSHA256']}\n"
            ).encode("utf-8")
        )

    return digest.hexdigest().upper()


def config_assignment_content_sha256(
    config_root,
):
    digest = hashlib.sha256()

    for bucket_id in range(
        BUCKETS
    ):
        array = None

        try:
            array = np.load(
                client_array_path(
                    config_root,
                    bucket_id,
                ),
                mmap_mode="r",
                allow_pickle=False,
            )

            require_equal(
                array.dtype,
                np.dtype("uint8"),
                (
                    f"Assignment content dtype "
                    f"bucket {bucket_id:03d}"
                ),
            )

            digest.update(
                np.asarray(
                    array,
                    dtype=np.uint8,
                ).tobytes(
                    order="C"
                )
            )

        finally:
            close_memmap(
                array
            )

            array = None

    return digest.hexdigest().upper()


def build_expected_client_vector(
    slice_rows_for_label,
    expected_n,
):
    require_equal(
        len(
            slice_rows_for_label
        ),
        CLIENTS,
        "Slice row count for one config/label",
    )

    expected = np.empty(
        expected_n,
        dtype=np.uint8,
    )

    cursor = 0

    for row in sorted(
        slice_rows_for_label,
        key=lambda item: int(
            item["ClientID"]
        ),
    ):
        client_id = int(
            row[
                "ClientID"
            ]
        )

        start = int(
            row[
                "StartInclusive"
            ]
        )

        end = int(
            row[
                "EndExclusive"
            ]
        )

        count = int(
            row[
                "Count"
            ]
        )

        require_equal(
            start,
            cursor,
            (
                f"Slice start continuity "
                f"client {client_id}"
            ),
        )

        require_equal(
            end
            -
            start,
            count,
            (
                f"Slice length "
                f"client {client_id}"
            ),
        )

        expected[
            start:end
        ] = np.uint8(
            client_id
        )

        cursor = end

    require_equal(
        cursor,
        expected_n,
        "Final slice boundary",
    )

    return expected


def load_count_plan_matrix(
    path,
    capacities,
):
    rows = read_csv(
        path
    )

    require_equal(
        len(rows),
        CLIENTS,
        f"Count-plan row count: {path.name}",
    )

    matrix = np.zeros(
        (
            LABELS,
            CLIENTS,
        ),
        dtype=np.int64,
    )

    for row in rows:
        client_id = int(
            row[
                "ClientID"
            ]
        )

        require_equal(
            int(
                row[
                    "Capacity"
                ]
            ),
            int(
                capacities[
                    client_id
                ]
            ),
            (
                f"Count-plan capacity "
                f"{path.name}, client {client_id}"
            ),
        )

        for label_id in range(
            LABELS
        ):
            field = (
                f"Label{label_id}_"
                f"{LABEL_NAMES[label_id]}"
            )

            matrix[
                label_id,
                client_id
            ] = int(
                row[
                    field
                ]
            )

    return matrix


# ============================================================================
# TRAIN identity stage
# ============================================================================

def build_or_verify_train_identity_stage(
    effective_root,
    split_root,
    stage_root,
):
    stage_root = Path(
        stage_root
    )

    complete_path = (
        stage_root
        /
        "TRAIN_IDENTITY_STAGE_COMPLETE.json"
    )

    if complete_path.exists():
        state = load_json(
            complete_path
        )

        require_equal(
            state.get(
                "status"
            ),
            "COMPLETE",
            "Existing TRAIN identity stage status",
        )

        require_equal(
            state[
                "effective_dataset_fingerprint_sha256"
            ],
            EFFECTIVE_DATASET_SHA,
            "Existing TRAIN identity stage dataset SHA256",
        )

        require_equal(
            state[
                "split_assignment_sha256"
            ],
            SPLIT_ASSIGNMENT_SHA,
            "Existing TRAIN identity stage split SHA256",
        )

        observed_counts = {}

        for label_id in range(
            LABELS
        ):
            artifact = state[
                "label_artifacts"
            ][
                str(label_id)
            ]

            path = (
                stage_root
                /
                artifact[
                    "FileName"
                ]
            )

            require_true(
                path.exists(),
                f"Missing TRAIN identity stage file: {path}",
            )

            require_equal(
                path.stat().st_size,
                artifact[
                    "SizeBytes"
                ],
                f"TRAIN identity stage size: {path}",
            )

            require_equal(
                sha256_file(
                    path
                ),
                artifact[
                    "SHA256"
                ],
                f"TRAIN identity stage SHA256: {path}",
            )

            count = (
                path.stat().st_size
                //
                TRAIN_IDENTITY_DTYPE.itemsize
            )

            require_equal(
                count,
                TRAIN_COUNTS[
                    label_id
                ],
                (
                    f"TRAIN identity stage count "
                    f"label {label_id}"
                ),
            )

            observed_counts[
                label_id
            ] = count

        print(
            "TRAIN identity stage already complete and verified."
        )

        return (
            state,
            observed_counts,
        )

    if stage_root.exists():
        shutil.rmtree(
            stage_root
        )

    stage_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    handles = {}

    paths = {}

    counts = Counter()

    try:
        for label_id in range(
            LABELS
        ):
            path = (
                stage_root
                /
                f"label_{label_id}.bin"
            )

            paths[
                label_id
            ] = path

            handles[
                label_id
            ] = path.open(
                "wb"
            )

        for bucket_id in range(
            BUCKETS
        ):
            index = None
            split_array = None

            try:
                index = np.load(
                    effective_index_path(
                        effective_root,
                        bucket_id,
                    ),
                    mmap_mode="r",
                    allow_pickle=False,
                )

                split_array = np.load(
                    split_array_path(
                        split_root,
                        bucket_id,
                    ),
                    mmap_mode="r",
                    allow_pickle=False,
                )

                require_equal(
                    index.dtype,
                    EFFECTIVE_INDEX_DTYPE,
                    (
                        f"Effective index dtype "
                        f"bucket {bucket_id:03d}"
                    ),
                )

                require_equal(
                    int(
                        index.shape[0]
                    ),
                    int(
                        split_array.shape[0]
                    ),
                    (
                        f"Index/split alignment "
                        f"bucket {bucket_id:03d}"
                    ),
                )

                train_positions = np.flatnonzero(
                    split_array
                    ==
                    np.uint8(0)
                )

                if train_positions.shape[0] == 0:
                    continue

                labels = index[
                    "Task7LabelID"
                ][
                    train_positions
                ]

                h1 = index[
                    "TransformedHash1"
                ][
                    train_positions
                ]

                h2 = index[
                    "TransformedHash2"
                ][
                    train_positions
                ]

                for label_id in np.unique(
                    labels
                ):
                    label_int = int(
                        label_id
                    )

                    local = np.flatnonzero(
                        labels
                        ==
                        label_id
                    )

                    records = np.empty(
                        local.shape[0],
                        dtype=TRAIN_IDENTITY_DTYPE,
                    )

                    records[
                        "TransformedHash1"
                    ] = h1[
                        local
                    ]

                    records[
                        "TransformedHash2"
                    ] = h2[
                        local
                    ]

                    records[
                        "EffectiveBucket"
                    ] = np.uint8(
                        bucket_id
                    )

                    records[
                        "EffectiveRowIndex"
                    ] = train_positions[
                        local
                    ].astype(
                        np.uint32,
                        copy=False,
                    )

                    records.tofile(
                        handles[
                            label_int
                        ]
                    )

                    counts[
                        label_int
                    ] += int(
                        records.shape[0]
                    )

                print(
                    f"[{bucket_id:03d}/255] TRAIN identity stage | "
                    f"train_rows={train_positions.shape[0]}"
                )

            finally:
                close_memmap(
                    split_array
                )

                close_memmap(
                    index
                )

                split_array = None
                index = None

    finally:
        for handle in handles.values():
            handle.close()

    require_equal(
        dict(
            sorted(
                counts.items()
            )
        ),
        TRAIN_COUNTS,
        "TRAIN identity stage per-label counts",
    )

    label_artifacts = {}

    for label_id in range(
        LABELS
    ):
        path = paths[
            label_id
        ]

        expected_size = (
            TRAIN_COUNTS[
                label_id
            ]
            *
            TRAIN_IDENTITY_DTYPE.itemsize
        )

        require_equal(
            path.stat().st_size,
            expected_size,
            (
                f"TRAIN identity stage byte count "
                f"label {label_id}"
            ),
        )

        label_artifacts[
            str(label_id)
        ] = {
            "FileName": (
                path.name
            ),
            "RecordCount": (
                TRAIN_COUNTS[
                    label_id
                ]
            ),
            "SizeBytes": (
                path.stat().st_size
            ),
            "SHA256": (
                sha256_file(
                    path
                )
            ),
        }

    state = {
        "status": (
            "COMPLETE"
        ),
        "effective_dataset_fingerprint_sha256": (
            EFFECTIVE_DATASET_SHA
        ),
        "split_assignment_sha256": (
            SPLIT_ASSIGNMENT_SHA
        ),
        "record_dtype": (
            TRAIN_IDENTITY_DTYPE.descr
        ),
        "scientific_ordering_identity_fields": [
            "TransformedHash1",
            "TransformedHash2",
        ],
        "destination_locator_fields_not_rank_inputs": [
            "EffectiveBucket",
            "EffectiveRowIndex",
        ],
        "train_only": (
            True
        ),
        "label_artifacts": (
            label_artifacts
        ),
    }

    write_json(
        complete_path,
        state,
    )

    print(
        "TRAIN identity stage complete."
    )

    return (
        state,
        dict(counts),
    )


# ============================================================================
# Completed-config verification
# ============================================================================

def verify_completed_config(
    config_root,
    expected_config_id,
):
    complete_path = (
        Path(config_root)
        /
        "CONFIG_ASSIGNMENT_COMPLETE.json"
    )

    if not complete_path.exists():
        return None

    state = load_json(
        complete_path
    )

    require_equal(
        state.get(
            "status"
        ),
        "PASS",
        (
            f"Completed config status "
            f"{expected_config_id}"
        ),
    )

    require_equal(
        state.get(
            "config_id"
        ),
        expected_config_id,
        (
            f"Completed config ID "
            f"{expected_config_id}"
        ),
    )

    require_equal(
        state[
            "physical_assignment_protocol_artifact_manifest_sha256"
        ],
        PHYSICAL_ASSIGNMENT_PROTOCOL_SHA,
        (
            f"Completed config protocol binding "
            f"{expected_config_id}"
        ),
    )

    manifest_path = (
        Path(config_root)
        /
        "CONFIG_ASSIGNMENT_ARTIFACT_MANIFEST.csv"
    )

    rows = read_csv(
        manifest_path
    )

    for row in rows:
        path = (
            Path(config_root)
            /
            row[
                "RelativePath"
            ]
        )

        require_true(
            path.exists(),
            (
                f"Missing completed config artifact "
                f"{expected_config_id}: {path}"
            ),
        )

        require_equal(
            path.stat().st_size,
            int(
                row[
                    "SizeBytes"
                ]
            ),
            (
                f"Completed config artifact size "
                f"{expected_config_id}: {path}"
            ),
        )

        require_equal(
            sha256_file(
                path
            ),
            row[
                "SHA256"
            ],
            (
                f"Completed config artifact SHA256 "
                f"{expected_config_id}: {path}"
            ),
        )

    require_equal(
        artifact_manifest_digest(
            rows
        ),
        state[
            "config_artifact_manifest_sha256"
        ],
        (
            f"Completed config manifest digest "
            f"{expected_config_id}"
        ),
    )

    return state


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--effective-root",
        required=True,
    )

    parser.add_argument(
        "--split-root",
        required=True,
    )

    parser.add_argument(
        "--assignment-root",
        required=True,
    )

    parser.add_argument(
        "--gate76-freeze-json",
        required=True,
    )

    parser.add_argument(
        "--gate76-protocol-json",
        required=True,
    )

    parser.add_argument(
        "--gate76-algorithm-json",
        required=True,
    )

    parser.add_argument(
        "--gate76-seeds-csv",
        required=True,
    )

    parser.add_argument(
        "--gate76-slices-csv",
        required=True,
    )

    parser.add_argument(
        "--gate75-freeze-json",
        required=True,
    )

    parser.add_argument(
        "--gate75-plan-manifest-csv",
        required=True,
    )

    parser.add_argument(
        "--gate75-capacities-csv",
        required=True,
    )

    parser.add_argument(
        "--gate75-plan-root",
        required=True,
    )

    parser.add_argument(
        "--gate71-state-json",
        required=True,
    )

    parser.add_argument(
        "--audit-root",
        required=True,
    )

    args = parser.parse_args()

    effective_root = Path(
        args.effective_root
    )

    split_root = Path(
        args.split_root
    )

    assignment_root = Path(
        args.assignment_root
    )

    audit_root = Path(
        args.audit_root
    )

    assignment_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_state_path = (
        audit_root
        /
        "TRANSFORMED_UNIQUE_PHYSICAL_ASSIGNMENT_MATERIALISATION.json"
    )

    if final_state_path.exists():
        final_state = load_json(
            final_state_path
        )

        if final_state.get(
            "status"
        ) == "PASS":
            raise RuntimeError(
                "A completed Gate-77 materialisation already exists."
            )

    # ------------------------------------------------------------------
    # Gate A - verify frozen current chain.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE A - VERIFY FROZEN CURRENT CHAIN")
    print("=" * 60)
    print("")

    gate71 = load_json(
        args.gate71_state_json
    )

    require_equal(
        gate71.get(
            "status"
        ),
        "PASS",
        "Gate-71 status",
    )

    require_equal(
        gate71[
            "immutable_binding"
        ][
            "effective_dataset_fingerprint_sha256"
        ],
        EFFECTIVE_DATASET_SHA,
        "Gate-71 effective dataset SHA256",
    )

    require_equal(
        gate71[
            "immutable_binding"
        ][
            "split_protocol_artifact_manifest_sha256"
        ],
        SPLIT_PROTOCOL_SHA,
        "Gate-71 split protocol SHA256",
    )

    require_equal(
        gate71[
            "split_assignment_manifest_sha256"
        ],
        SPLIT_ASSIGNMENT_SHA,
        "Gate-71 split assignment SHA256",
    )

    require_equal(
        gate71[
            "counts"
        ][
            "TRAIN"
        ],
        TRAIN_TOTAL,
        "Gate-71 TRAIN total",
    )

    require_equal(
        gate71[
            "counts"
        ][
            "VALIDATION"
        ],
        VALIDATION_TOTAL,
        "Gate-71 VALIDATION total",
    )

    require_equal(
        gate71[
            "counts"
        ][
            "TEST"
        ],
        TEST_TOTAL,
        "Gate-71 TEST total",
    )

    require_true(
        gate71[
            "scientific_boundary"
        ][
            "scientific_training_started"
        ]
        is False,
        "Gate-71 says scientific training started.",
    )

    gate75 = load_json(
        args.gate75_freeze_json
    )

    require_equal(
        gate75.get(
            "status"
        ),
        "FROZEN",
        "Gate-75 status",
    )

    require_equal(
        gate75[
            "capacity_vector_sha256"
        ],
        CAPACITY_VECTOR_SHA,
        "Gate-75 capacity SHA256",
    )

    require_equal(
        gate75[
            "count_plan_set_sha256"
        ],
        COUNT_PLAN_SET_SHA,
        "Gate-75 count-plan set SHA256",
    )

    require_equal(
        gate75[
            "frozen_client_partition_protocol_artifact_manifest_sha256"
        ],
        CLIENT_PARTITION_PROTOCOL_SHA,
        "Gate-75 protocol manifest SHA256",
    )

    require_true(
        gate75[
            "materialization_boundary"
        ][
            "physical_client_assignments_materialized"
        ]
        is False,
        "Gate-75 says physical assignments already exist.",
    )

    gate76 = load_json(
        args.gate76_freeze_json
    )

    require_equal(
        gate76.get(
            "status"
        ),
        "FROZEN",
        "Gate-76 status",
    )

    require_equal(
        gate76.get(
            "physical_assignment_protocol_id"
        ),
        PHYSICAL_ASSIGNMENT_PROTOCOL_ID,
        "Gate-76 physical assignment protocol ID",
    )

    require_equal(
        gate76.get(
            "ordering_algorithm_id"
        ),
        ORDERING_ALGORITHM_ID,
        "Gate-76 ordering algorithm ID",
    )

    require_equal(
        gate76[
            "ordering_seed_manifest_sha256"
        ],
        ORDERING_SEED_MANIFEST_SHA,
        "Gate-76 ordering seed manifest SHA256",
    )

    require_equal(
        gate76[
            "slice_boundary_sha256"
        ],
        SLICE_BOUNDARY_SHA,
        "Gate-76 slice boundary SHA256",
    )

    require_equal(
        gate76[
            "frozen_physical_assignment_protocol_artifact_manifest_sha256"
        ],
        PHYSICAL_ASSIGNMENT_PROTOCOL_SHA,
        "Gate-76 protocol artifact-manifest SHA256",
    )

    require_true(
        gate76[
            "materialization_boundary"
        ][
            "physical_client_assignments_materialized"
        ]
        is False,
        "Gate-76 says assignments already materialized.",
    )

    require_true(
        gate76[
            "materialization_boundary"
        ][
            "scientific_training_started"
        ]
        is False,
        "Gate-76 says scientific training started.",
    )

    protocol76 = load_json(
        args.gate76_protocol_json
    )

    require_equal(
        protocol76.get(
            "status"
        ),
        "FROZEN",
        "Gate-76 protocol JSON status",
    )

    require_equal(
        protocol76[
            "immutable_binding"
        ][
            "effective_dataset_fingerprint_sha256"
        ],
        EFFECTIVE_DATASET_SHA,
        "Gate-76 protocol dataset SHA256",
    )

    require_equal(
        protocol76[
            "immutable_binding"
        ][
            "split_assignment_artifact_manifest_sha256"
        ],
        SPLIT_ASSIGNMENT_SHA,
        "Gate-76 protocol split SHA256",
    )

    require_equal(
        protocol76[
            "immutable_binding"
        ][
            "client_partition_protocol_artifact_manifest_sha256"
        ],
        CLIENT_PARTITION_PROTOCOL_SHA,
        "Gate-76 protocol partition SHA256",
    )

    algorithm76 = load_json(
        args.gate76_algorithm_json
    )

    require_equal(
        algorithm76.get(
            "status"
        ),
        "FROZEN",
        "Gate-76 algorithm status",
    )

    require_equal(
        algorithm76.get(
            "ordering_algorithm_id"
        ),
        ORDERING_ALGORITHM_ID,
        "Gate-76 ordering algorithm JSON ID",
    )

    print(
        "Effective dataset: BOUND"
    )
    print(
        "TRAIN/VALIDATION/TEST split: BOUND"
    )
    print(
        "Gate-75 exact count plans: FROZEN"
    )
    print(
        "Gate-76 physical assignment protocol: FROZEN"
    )
    print(
        "Scientific training started: NO"
    )

    # ------------------------------------------------------------------
    # Gate B - verify seeds, slices, capacities, and count plans.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE B - VERIFY FROZEN SEEDS, SLICES, CAPACITIES, COUNT PLANS")
    print("=" * 60)
    print("")

    seed_rows = read_csv(
        args.gate76_seeds_csv
    )

    require_equal(
        len(seed_rows),
        CONFIGS,
        "Gate-76 ordering seed row count",
    )

    require_equal(
        seed_manifest_sha256(
            seed_rows
        ),
        ORDERING_SEED_MANIFEST_SHA,
        "Gate-76 ordering seed manifest replay",
    )

    seeds_by_config = {
        row[
            "ConfigID"
        ]: row
        for row
        in seed_rows
    }

    require_equal(
        len(
            seeds_by_config
        ),
        CONFIGS,
        "Unique Gate-76 configuration seeds",
    )

    slice_rows = read_csv(
        args.gate76_slices_csv
    )

    require_equal(
        len(slice_rows),
        CONFIGS
        *
        LABELS
        *
        CLIENTS,
        "Gate-76 slice row count",
    )

    require_equal(
        slice_boundary_sha256(
            slice_rows
        ),
        SLICE_BOUNDARY_SHA,
        "Gate-76 slice boundary replay",
    )

    slices_by_config_label = {}

    for row in slice_rows:
        key = (
            row[
                "ConfigID"
            ],
            int(
                row[
                    "Task7LabelID"
                ]
            ),
        )

        slices_by_config_label.setdefault(
            key,
            [],
        ).append(
            row
        )

    capacity_rows = read_csv(
        args.gate75_capacities_csv
    )

    capacities = np.zeros(
        CLIENTS,
        dtype=np.int64,
    )

    for row in capacity_rows:
        capacities[
            int(
                row[
                    "ClientID"
                ]
            )
        ] = int(
            row[
                "ExactCapacity"
            ]
        )

    require_equal(
        capacities.tolist(),
        EXPECTED_CAPACITIES,
        "Gate-75 exact capacity vector",
    )

    plan_manifest_rows = read_csv(
        args.gate75_plan_manifest_csv
    )

    require_equal(
        len(plan_manifest_rows),
        CONFIGS,
        "Gate-75 count-plan manifest row count",
    )

    plan_manifest_by_config = {
        row[
            "ConfigID"
        ]: row
        for row
        in plan_manifest_rows
    }

    require_equal(
        set(
            plan_manifest_by_config
        ),
        set(
            seeds_by_config
        ),
        "Gate-75 / Gate-76 configuration coverage",
    )

    matrices = {}

    for cfg in sorted(
        plan_manifest_by_config
    ):
        row = plan_manifest_by_config[
            cfg
        ]

        plan_path = (
            Path(
                args.gate75_plan_root
            )
            /
            row[
                "FrozenFileName"
            ]
        )

        require_equal(
            sha256_file(
                plan_path
            ),
            row[
                "FrozenFileSHA256"
            ],
            f"Gate-75 frozen plan file SHA256 {cfg}",
        )

        matrix = load_count_plan_matrix(
            plan_path,
            capacities,
        )

        require_equal(
            matrix_content_sha256(
                matrix
            ),
            row[
                "MatrixContentSHA256"
            ],
            f"Gate-75 matrix content SHA256 {cfg}",
        )

        require_true(
            np.array_equal(
                matrix.sum(
                    axis=1
                ),
                np.asarray(
                    [
                        TRAIN_COUNTS[
                            label_id
                        ]
                        for label_id
                        in range(
                            LABELS
                        )
                    ],
                    dtype=np.int64,
                ),
            ),
            f"Gate-75 class totals {cfg}",
        )

        require_true(
            np.array_equal(
                matrix.sum(
                    axis=0
                ),
                capacities,
            ),
            f"Gate-75 client capacities {cfg}",
        )

        matrices[
            cfg
        ] = matrix

        for label_id in range(
            LABELS
        ):
            expected_clients = build_expected_client_vector(
                slices_by_config_label[
                    (
                        cfg,
                        label_id,
                    )
                ],
                TRAIN_COUNTS[
                    label_id
                ],
            )

            observed_counts = np.bincount(
                expected_clients.astype(
                    np.int64,
                    copy=False,
                ),
                minlength=CLIENTS,
            )

            require_true(
                np.array_equal(
                    observed_counts,
                    matrix[
                        label_id,
                        :
                    ],
                ),
                (
                    f"Slice/count-plan mismatch "
                    f"{cfg}, label {label_id}"
                ),
            )

            del expected_clients

    print(
        "Ordering seeds: PASS 10 / 10"
    )
    print(
        "Slice boundaries: PASS 2100 / 2100"
    )
    print(
        "Exact count plans: PASS 10 / 10"
    )
    print(
        "Exact capacities: PASS"
    )

    # ------------------------------------------------------------------
    # Gate C - build or verify reusable TRAIN identity stage.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE C - BUILD OR VERIFY TRAIN IDENTITY STAGE")
    print("=" * 60)
    print("")

    stage_root = (
        audit_root
        /
        "train_identity_stage_v1"
    )

    stage_state, stage_counts = build_or_verify_train_identity_stage(
        effective_root,
        split_root,
        stage_root,
    )

    require_equal(
        dict(
            sorted(
                stage_counts.items()
            )
        ),
        TRAIN_COUNTS,
        "Verified TRAIN identity stage counts",
    )

    # ------------------------------------------------------------------
    # Gate D - materialise / audit / replay each configuration.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE D - MATERIALISE, AUDIT, AND REPLAY TEN CONFIGURATIONS")
    print("=" * 60)
    print("")

    config_summary_rows = []

    for config_index, cfg in enumerate(
        sorted(
            matrices
        ),
        start=1,
    ):
        print("")
        print(
            f"--- CONFIG {config_index}/10: {cfg} ---"
        )
        print("")

        final_config_root = (
            assignment_root
            /
            cfg
        )

        existing_state = None

        if final_config_root.exists():
            try:
                existing_state = verify_completed_config(
                    final_config_root,
                    cfg,
                )
            except Exception:
                existing_state = None

                shutil.rmtree(
                    final_config_root
                )

        if existing_state is not None:
            print(
                f"{cfg}: completed assignment already verified; reusing."
            )

            config_summary_rows.append({
                "ConfigID": (
                    cfg
                ),
                "AssignmentContentSHA256": (
                    existing_state[
                        "assignment_content_sha256"
                    ]
                ),
                "ConfigArtifactManifestSHA256": (
                    existing_state[
                        "config_artifact_manifest_sha256"
                    ]
                ),
                "TrainAssigned": (
                    existing_state[
                        "audit"
                    ][
                        "train_assigned"
                    ]
                ),
                "ValidationAssigned": (
                    existing_state[
                        "audit"
                    ][
                        "validation_assigned"
                    ]
                ),
                "TestAssigned": (
                    existing_state[
                        "audit"
                    ][
                        "test_assigned"
                    ]
                ),
                "ReplayMismatches": (
                    existing_state[
                        "audit"
                    ][
                        "replay_mismatches"
                    ]
                ),
                "ReusedCompletedConfig": (
                    True
                ),
            })

            continue

        temp_config_root = (
            assignment_root
            /
            (
                f".tmp_{cfg}"
            )
        )

        if temp_config_root.exists():
            shutil.rmtree(
                temp_config_root
            )

        temp_config_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        bucket_lengths = {}

        total_slots = 0

        for bucket_id in range(
            BUCKETS
        ):
            index = None

            try:
                index = np.load(
                    effective_index_path(
                        effective_root,
                        bucket_id,
                    ),
                    mmap_mode="r",
                    allow_pickle=False,
                )

                n = int(
                    index.shape[0]
                )

                bucket_lengths[
                    bucket_id
                ] = n

                total_slots += n

            finally:
                close_memmap(
                    index
                )

                index = None

            bucket_root = (
                temp_config_root
                /
                f"bucket_{bucket_id:03d}"
            )

            bucket_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            client_array = np.lib.format.open_memmap(
                bucket_root
                /
                "client_id_u8.npy",
                mode="w+",
                dtype=np.uint8,
                shape=(
                    n,
                ),
            )

            client_array[
                :
            ] = NON_CLIENT_SENTINEL

            client_array.flush()

            close_memmap(
                client_array
            )

            client_array = None

        require_equal(
            total_slots,
            EFFECTIVE_OBSERVATIONS,
            f"Initialised assignment slots {cfg}",
        )

        seed_uint64 = int(
            seeds_by_config[
                cfg
            ][
                "SeedUInt64"
            ]
        )

        first_pass_order_rows = []

        # --------------------------------------------------------------
        # First pass: sort and materialise.
        # --------------------------------------------------------------

        for label_id in range(
            LABELS
        ):
            stage_path = (
                stage_root
                /
                f"label_{label_id}.bin"
            )

            n = TRAIN_COUNTS[
                label_id
            ]

            records = None

            try:
                records = np.memmap(
                    stage_path,
                    mode="r",
                    dtype=TRAIN_IDENTITY_DTYPE,
                    shape=(
                        n,
                    ),
                )

                order, key1, key2 = deterministic_order(
                    records,
                    seed_uint64,
                )

                digest = order_digest(
                    records,
                    order,
                )

                expected_clients = build_expected_client_vector(
                    slices_by_config_label[
                        (
                            cfg,
                            label_id,
                        )
                    ],
                    n,
                )

                sorted_bucket = records[
                    "EffectiveBucket"
                ][
                    order
                ]

                sorted_row = records[
                    "EffectiveRowIndex"
                ][
                    order
                ]

                destination_order = np.argsort(
                    sorted_bucket,
                    kind="stable",
                )

                bucket_sorted = sorted_bucket[
                    destination_order
                ]

                row_sorted = sorted_row[
                    destination_order
                ]

                client_sorted = expected_clients[
                    destination_order
                ]

                if n:
                    starts = np.flatnonzero(
                        np.r_[
                            True,
                            bucket_sorted[
                                1:
                            ]
                            !=
                            bucket_sorted[
                                :-1
                            ],
                        ]
                    )

                    ends = np.r_[
                        starts[
                            1:
                        ],
                        n,
                    ]

                    for start, end in zip(
                        starts,
                        ends,
                    ):
                        bucket_id = int(
                            bucket_sorted[
                                start
                            ]
                        )

                        rows = row_sorted[
                            start:end
                        ].astype(
                            np.int64,
                            copy=False,
                        )

                        values = client_sorted[
                            start:end
                        ]

                        require_true(
                            bool(
                                np.all(
                                    rows
                                    <
                                    bucket_lengths[
                                        bucket_id
                                    ]
                                )
                            ),
                            (
                                f"Destination row out of range "
                                f"{cfg}, label {label_id}, "
                                f"bucket {bucket_id}"
                            ),
                        )

                        assignment = None

                        try:
                            assignment = np.load(
                                client_array_path(
                                    temp_config_root,
                                    bucket_id,
                                ),
                                mmap_mode="r+",
                                allow_pickle=False,
                            )

                            require_true(
                                bool(
                                    np.all(
                                        assignment[
                                            rows
                                        ]
                                        ==
                                        NON_CLIENT_SENTINEL
                                    )
                                ),
                                (
                                    f"Duplicate physical assignment attempt "
                                    f"{cfg}, label {label_id}, "
                                    f"bucket {bucket_id}"
                                ),
                            )

                            assignment[
                                rows
                            ] = values

                            assignment.flush()

                        finally:
                            close_memmap(
                                assignment
                            )

                            assignment = None

                first_pass_order_rows.append({
                    "ConfigID": (
                        cfg
                    ),
                    "Task7LabelID": (
                        label_id
                    ),
                    "Task7Label": (
                        LABEL_NAMES[
                            label_id
                        ]
                    ),
                    "TrainObservationCount": (
                        n
                    ),
                    "SortedTransformedIDOrderSHA256": (
                        digest
                    ),
                })

                print(
                    f"{cfg} | label {label_id} | "
                    f"first-pass order_sha={digest}"
                )

                del destination_order
                del bucket_sorted
                del row_sorted
                del client_sorted
                del sorted_bucket
                del sorted_row
                del expected_clients
                del order
                del key1
                del key2

            finally:
                close_memmap(
                    records
                )

                records = None

                gc.collect()

        # --------------------------------------------------------------
        # Exact coverage / count-plan / capacity audit.
        # --------------------------------------------------------------

        observed_label_client = np.zeros(
            (
                LABELS,
                CLIENTS,
            ),
            dtype=np.int64,
        )

        observed_client_totals = np.zeros(
            CLIENTS,
            dtype=np.int64,
        )

        train_assigned = 0

        validation_assigned = 0

        test_assigned = 0

        invalid_train_values = 0

        for bucket_id in range(
            BUCKETS
        ):
            index = None
            split_array = None
            assignment = None

            try:
                index = np.load(
                    effective_index_path(
                        effective_root,
                        bucket_id,
                    ),
                    mmap_mode="r",
                    allow_pickle=False,
                )

                split_array = np.load(
                    split_array_path(
                        split_root,
                        bucket_id,
                    ),
                    mmap_mode="r",
                    allow_pickle=False,
                )

                assignment = np.load(
                    client_array_path(
                        temp_config_root,
                        bucket_id,
                    ),
                    mmap_mode="r",
                    allow_pickle=False,
                )

                require_equal(
                    int(
                        index.shape[0]
                    ),
                    int(
                        assignment.shape[0]
                    ),
                    (
                        f"Assignment alignment "
                        f"{cfg}, bucket {bucket_id}"
                    ),
                )

                train_mask = (
                    split_array
                    ==
                    np.uint8(0)
                )

                validation_mask = (
                    split_array
                    ==
                    np.uint8(1)
                )

                test_mask = (
                    split_array
                    ==
                    np.uint8(2)
                )

                train_values = assignment[
                    train_mask
                ]

                validation_values = assignment[
                    validation_mask
                ]

                test_values = assignment[
                    test_mask
                ]

                valid_train = (
                    train_values
                    <
                    np.uint8(
                        CLIENTS
                    )
                )

                invalid_train_values += int(
                    np.count_nonzero(
                        ~valid_train
                    )
                )

                train_assigned += int(
                    np.count_nonzero(
                        valid_train
                    )
                )

                validation_assigned += int(
                    np.count_nonzero(
                        validation_values
                        !=
                        NON_CLIENT_SENTINEL
                    )
                )

                test_assigned += int(
                    np.count_nonzero(
                        test_values
                        !=
                        NON_CLIENT_SENTINEL
                    )
                )

                require_true(
                    bool(
                        np.all(
                            validation_values
                            ==
                            NON_CLIENT_SENTINEL
                        )
                    ),
                    (
                        f"VALIDATION assigned to client "
                        f"{cfg}, bucket {bucket_id}"
                    ),
                )

                require_true(
                    bool(
                        np.all(
                            test_values
                            ==
                            NON_CLIENT_SENTINEL
                        )
                    ),
                    (
                        f"TEST assigned to client "
                        f"{cfg}, bucket {bucket_id}"
                    ),
                )

                labels = index[
                    "Task7LabelID"
                ][
                    train_mask
                ].astype(
                    np.int64,
                    copy=False,
                )

                clients = train_values.astype(
                    np.int64,
                    copy=False,
                )

                combined = (
                    labels
                    *
                    CLIENTS
                    +
                    clients
                )

                counts = np.bincount(
                    combined,
                    minlength=(
                        LABELS
                        *
                        CLIENTS
                    ),
                ).reshape(
                    LABELS,
                    CLIENTS,
                )

                observed_label_client += counts

                observed_client_totals += np.bincount(
                    clients,
                    minlength=CLIENTS,
                )

            finally:
                close_memmap(
                    assignment
                )

                close_memmap(
                    split_array
                )

                close_memmap(
                    index
                )

                assignment = None
                split_array = None
                index = None

        require_equal(
            invalid_train_values,
            0,
            f"Invalid/unassigned TRAIN values {cfg}",
        )

        require_equal(
            train_assigned,
            TRAIN_TOTAL,
            f"TRAIN assignment coverage {cfg}",
        )

        require_equal(
            validation_assigned,
            0,
            f"VALIDATION client assignment count {cfg}",
        )

        require_equal(
            test_assigned,
            0,
            f"TEST client assignment count {cfg}",
        )

        require_true(
            np.array_equal(
                observed_label_client,
                matrices[
                    cfg
                ],
            ),
            f"Exact frozen count-plan reproduction {cfg}",
        )

        require_true(
            np.array_equal(
                observed_client_totals,
                capacities,
            ),
            f"Exact client capacity reproduction {cfg}",
        )

        print(
            f"{cfg}: coverage/count/capacity audit PASS"
        )

        # --------------------------------------------------------------
        # Full deterministic replay against materialised arrays.
        # --------------------------------------------------------------

        first_digest_by_label = {
            int(
                row[
                    "Task7LabelID"
                ]
            ): row[
                "SortedTransformedIDOrderSHA256"
            ]
            for row
            in first_pass_order_rows
        }

        replay_rows = []

        total_replay_mismatches = 0

        for label_id in range(
            LABELS
        ):
            stage_path = (
                stage_root
                /
                f"label_{label_id}.bin"
            )

            n = TRAIN_COUNTS[
                label_id
            ]

            records = None

            try:
                records = np.memmap(
                    stage_path,
                    mode="r",
                    dtype=TRAIN_IDENTITY_DTYPE,
                    shape=(
                        n,
                    ),
                )

                replay_order, key1, key2 = deterministic_order(
                    records,
                    seed_uint64,
                )

                replay_digest = order_digest(
                    records,
                    replay_order,
                )

                require_equal(
                    replay_digest,
                    first_digest_by_label[
                        label_id
                    ],
                    (
                        f"Order digest replay "
                        f"{cfg}, label {label_id}"
                    ),
                )

                expected_clients = build_expected_client_vector(
                    slices_by_config_label[
                        (
                            cfg,
                            label_id,
                        )
                    ],
                    n,
                )

                sorted_bucket = records[
                    "EffectiveBucket"
                ][
                    replay_order
                ]

                sorted_row = records[
                    "EffectiveRowIndex"
                ][
                    replay_order
                ]

                destination_order = np.argsort(
                    sorted_bucket,
                    kind="stable",
                )

                bucket_sorted = sorted_bucket[
                    destination_order
                ]

                row_sorted = sorted_row[
                    destination_order
                ]

                expected_sorted = expected_clients[
                    destination_order
                ]

                replay_mismatches = 0

                if n:
                    starts = np.flatnonzero(
                        np.r_[
                            True,
                            bucket_sorted[
                                1:
                            ]
                            !=
                            bucket_sorted[
                                :-1
                            ],
                        ]
                    )

                    ends = np.r_[
                        starts[
                            1:
                        ],
                        n,
                    ]

                    for start, end in zip(
                        starts,
                        ends,
                    ):
                        bucket_id = int(
                            bucket_sorted[
                                start
                            ]
                        )

                        rows = row_sorted[
                            start:end
                        ].astype(
                            np.int64,
                            copy=False,
                        )

                        expected_values = expected_sorted[
                            start:end
                        ]

                        assignment = None

                        try:
                            assignment = np.load(
                                client_array_path(
                                    temp_config_root,
                                    bucket_id,
                                ),
                                mmap_mode="r",
                                allow_pickle=False,
                            )

                            replay_mismatches += int(
                                np.count_nonzero(
                                    assignment[
                                        rows
                                    ]
                                    !=
                                    expected_values
                                )
                            )

                        finally:
                            close_memmap(
                                assignment
                            )

                            assignment = None

                require_equal(
                    replay_mismatches,
                    0,
                    (
                        f"Physical assignment replay mismatches "
                        f"{cfg}, label {label_id}"
                    ),
                )

                total_replay_mismatches += replay_mismatches

                replay_rows.append({
                    "ConfigID": (
                        cfg
                    ),
                    "Task7LabelID": (
                        label_id
                    ),
                    "Task7Label": (
                        LABEL_NAMES[
                            label_id
                        ]
                    ),
                    "TrainObservationCount": (
                        n
                    ),
                    "FirstPassOrderSHA256": (
                        first_digest_by_label[
                            label_id
                        ]
                    ),
                    "ReplayOrderSHA256": (
                        replay_digest
                    ),
                    "OrderDigestExactMatch": (
                        True
                    ),
                    "AssignmentReplayMismatches": (
                        replay_mismatches
                    ),
                })

                print(
                    f"{cfg} | label {label_id} | "
                    f"replay_exact=YES | mismatches=0"
                )

                del destination_order
                del bucket_sorted
                del row_sorted
                del expected_sorted
                del sorted_bucket
                del sorted_row
                del expected_clients
                del replay_order
                del key1
                del key2

            finally:
                close_memmap(
                    records
                )

                records = None

                gc.collect()

        require_equal(
            total_replay_mismatches,
            0,
            f"Total physical assignment replay mismatches {cfg}",
        )

        # --------------------------------------------------------------
        # Write config metadata and immutable artifacts.
        # --------------------------------------------------------------

        assignment_content_sha = config_assignment_content_sha256(
            temp_config_root
        )

        config_schema = {
            "config_id": (
                cfg
            ),
            "status": (
                "MATERIALISED"
            ),
            "assignment_layer_dtype": (
                "uint8"
            ),
            "assignment_values": {
                "0_to_29": (
                    "TRAIN_CLIENT_ID"
                ),
                "255": (
                    "NON_CLIENT_VALIDATION_OR_TEST"
                ),
            },
            "bucket_alignment": (
                "ROW_ALIGNED_WITH_EFFECTIVE_OBSERVATION_INDEX_AND_SPLIT_LAYER"
            ),
            "feature_arrays_copied": (
                False
            ),
            "physical_assignment_protocol_id": (
                PHYSICAL_ASSIGNMENT_PROTOCOL_ID
            ),
            "ordering_algorithm_id": (
                ORDERING_ALGORITHM_ID
            ),
            "assignment_content_hash_id": (
                ASSIGNMENT_CONTENT_HASH_ID
            ),
        }

        config_binding = {
            "config_id": (
                cfg
            ),
            "effective_dataset_fingerprint_sha256": (
                EFFECTIVE_DATASET_SHA
            ),
            "split_assignment_artifact_manifest_sha256": (
                SPLIT_ASSIGNMENT_SHA
            ),
            "client_partition_protocol_artifact_manifest_sha256": (
                CLIENT_PARTITION_PROTOCOL_SHA
            ),
            "count_plan_set_sha256": (
                COUNT_PLAN_SET_SHA
            ),
            "physical_assignment_protocol_artifact_manifest_sha256": (
                PHYSICAL_ASSIGNMENT_PROTOCOL_SHA
            ),
            "ordering_seed_sha256": (
                seeds_by_config[
                    cfg
                ][
                    "SeedSHA256"
                ]
            ),
            "ordering_seed_uint64": (
                int(
                    seeds_by_config[
                        cfg
                    ][
                        "SeedUInt64"
                    ]
                )
            ),
            "frozen_count_plan_matrix_sha256": (
                plan_manifest_by_config[
                    cfg
                ][
                    "MatrixContentSHA256"
                ]
            ),
        }

        schema_path = (
            temp_config_root
            /
            "CONFIG_ASSIGNMENT_SCHEMA.json"
        )

        binding_path = (
            temp_config_root
            /
            "CONFIG_ASSIGNMENT_BINDING.json"
        )

        first_order_path = (
            temp_config_root
            /
            "FIRST_PASS_ORDER_DIGESTS.csv"
        )

        replay_path = (
            temp_config_root
            /
            "DETERMINISTIC_REPLAY_AUDIT.csv"
        )

        counts_path = (
            temp_config_root
            /
            "MATERIALISED_LABEL_CLIENT_COUNTS.csv"
        )

        write_json(
            schema_path,
            config_schema,
        )

        write_json(
            binding_path,
            config_binding,
        )

        write_csv(
            first_order_path,
            first_pass_order_rows,
            [
                "ConfigID",
                "Task7LabelID",
                "Task7Label",
                "TrainObservationCount",
                "SortedTransformedIDOrderSHA256",
            ],
        )

        write_csv(
            replay_path,
            replay_rows,
            [
                "ConfigID",
                "Task7LabelID",
                "Task7Label",
                "TrainObservationCount",
                "FirstPassOrderSHA256",
                "ReplayOrderSHA256",
                "OrderDigestExactMatch",
                "AssignmentReplayMismatches",
            ],
        )

        count_rows = []

        for label_id in range(
            LABELS
        ):
            for client_id in range(
                CLIENTS
            ):
                count_rows.append({
                    "ConfigID": (
                        cfg
                    ),
                    "Task7LabelID": (
                        label_id
                    ),
                    "Task7Label": (
                        LABEL_NAMES[
                            label_id
                        ]
                    ),
                    "ClientID": (
                        client_id
                    ),
                    "Count": (
                        int(
                            observed_label_client[
                                label_id,
                                client_id
                            ]
                        )
                    ),
                })

        write_csv(
            counts_path,
            count_rows,
            [
                "ConfigID",
                "Task7LabelID",
                "Task7Label",
                "ClientID",
                "Count",
            ],
        )

        artifact_rows = []

        for bucket_id in range(
            BUCKETS
        ):
            path = client_array_path(
                temp_config_root,
                bucket_id,
            )

            artifact_rows.append({
                "ArtifactRole": (
                    "BUCKET_CLIENT_IDS"
                ),
                "RelativePath": str(
                    path.relative_to(
                        temp_config_root
                    )
                ),
                "SizeBytes": (
                    path.stat().st_size
                ),
                "SHA256": (
                    sha256_file(
                        path
                    )
                ),
            })

        for role, path in [
            (
                "CONFIG_SCHEMA",
                schema_path,
            ),
            (
                "CONFIG_BINDING",
                binding_path,
            ),
            (
                "FIRST_PASS_ORDER_DIGESTS",
                first_order_path,
            ),
            (
                "DETERMINISTIC_REPLAY_AUDIT",
                replay_path,
            ),
            (
                "MATERIALISED_LABEL_CLIENT_COUNTS",
                counts_path,
            ),
        ]:
            artifact_rows.append({
                "ArtifactRole": (
                    role
                ),
                "RelativePath": str(
                    path.relative_to(
                        temp_config_root
                    )
                ),
                "SizeBytes": (
                    path.stat().st_size
                ),
                "SHA256": (
                    sha256_file(
                        path
                    )
                ),
            })

        artifact_manifest_path = (
            temp_config_root
            /
            "CONFIG_ASSIGNMENT_ARTIFACT_MANIFEST.csv"
        )

        write_csv(
            artifact_manifest_path,
            artifact_rows,
            [
                "ArtifactRole",
                "RelativePath",
                "SizeBytes",
                "SHA256",
            ],
        )

        config_artifact_manifest_sha = artifact_manifest_digest(
            artifact_rows
        )

        complete_state = {
            "status": (
                "PASS"
            ),
            "config_id": (
                cfg
            ),
            "physical_assignment_protocol_artifact_manifest_sha256": (
                PHYSICAL_ASSIGNMENT_PROTOCOL_SHA
            ),
            "assignment_content_hash_id": (
                ASSIGNMENT_CONTENT_HASH_ID
            ),
            "assignment_content_sha256": (
                assignment_content_sha
            ),
            "config_artifact_manifest_sha256": (
                config_artifact_manifest_sha
            ),
            "audit": {
                "train_assigned": (
                    train_assigned
                ),
                "validation_assigned": (
                    validation_assigned
                ),
                "test_assigned": (
                    test_assigned
                ),
                "invalid_or_unassigned_train_values": (
                    invalid_train_values
                ),
                "exact_count_plan_reproduction": (
                    True
                ),
                "exact_client_capacity_reproduction": (
                    True
                ),
                "replay_mismatches": (
                    total_replay_mismatches
                ),
            },
        }

        complete_path = (
            temp_config_root
            /
            "CONFIG_ASSIGNMENT_COMPLETE.json"
        )

        write_json(
            complete_path,
            complete_state,
        )

        # --------------------------------------------------------------
        # Release mappings and atomically commit config directory.
        # --------------------------------------------------------------

        gc.collect()

        committed = False

        last_error = None

        for attempt in range(
            1,
            11,
        ):
            try:
                gc.collect()

                os.replace(
                    temp_config_root,
                    final_config_root,
                )

                committed = True

                break

            except PermissionError as exc:
                last_error = exc

                print(
                    f"{cfg}: commit attempt {attempt}/10 blocked "
                    "by Windows file handle; retrying..."
                )

                time.sleep(
                    1.5
                )

        if not committed:
            raise RuntimeError(
                f"Could not commit {cfg} after 10 retries. "
                f"Last error: {last_error}"
            )

        verified_state = verify_completed_config(
            final_config_root,
            cfg,
        )

        require_equal(
            verified_state[
                "assignment_content_sha256"
            ],
            assignment_content_sha,
            f"Committed assignment content SHA256 {cfg}",
        )

        config_summary_rows.append({
            "ConfigID": (
                cfg
            ),
            "AssignmentContentSHA256": (
                assignment_content_sha
            ),
            "ConfigArtifactManifestSHA256": (
                config_artifact_manifest_sha
            ),
            "TrainAssigned": (
                train_assigned
            ),
            "ValidationAssigned": (
                validation_assigned
            ),
            "TestAssigned": (
                test_assigned
            ),
            "ReplayMismatches": (
                total_replay_mismatches
            ),
            "ReusedCompletedConfig": (
                False
            ),
        })

        print(
            f"{cfg}: MATERIALISED + AUDITED + REPLAYED + COMMITTED"
        )

    require_equal(
        len(
            config_summary_rows
        ),
        CONFIGS,
        "Completed configuration count",
    )

    # ------------------------------------------------------------------
    # Gate E - build immutable assignment-set fingerprint and top metadata.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE E - BUILD IMMUTABLE ASSIGNMENT-SET FINGERPRINT")
    print("=" * 60)
    print("")

    assignment_set_sha = assignment_set_sha256(
        config_summary_rows
    )

    layer_schema = {
        "assignment_layer_set_id": (
            ASSIGNMENT_LAYER_SET_ID
        ),
        "configuration_count": (
            CONFIGS
        ),
        "client_count": (
            CLIENTS
        ),
        "assignment_dtype": (
            "uint8"
        ),
        "assignment_value_semantics": {
            "0_to_29": (
                "TRAIN_CLIENT_ID"
            ),
            "255": (
                "NON_CLIENT_VALIDATION_OR_TEST"
            ),
        },
        "alignment": (
            "ROW_ALIGNED_WITH_EFFECTIVE_OBSERVATION_INDEX_BUCKETS"
        ),
        "feature_arrays_copied": (
            False
        ),
        "assignment_content_hash_id": (
            ASSIGNMENT_CONTENT_HASH_ID
        ),
        "assignment_set_hash_id": (
            ASSIGNMENT_SET_HASH_ID
        ),
    }

    set_binding = {
        "materialisation_id": (
            MATERIALISATION_ID
        ),
        "assignment_layer_set_id": (
            ASSIGNMENT_LAYER_SET_ID
        ),
        "effective_dataset_fingerprint_sha256": (
            EFFECTIVE_DATASET_SHA
        ),
        "split_assignment_artifact_manifest_sha256": (
            SPLIT_ASSIGNMENT_SHA
        ),
        "client_partition_protocol_artifact_manifest_sha256": (
            CLIENT_PARTITION_PROTOCOL_SHA
        ),
        "count_plan_set_sha256": (
            COUNT_PLAN_SET_SHA
        ),
        "physical_assignment_protocol_artifact_manifest_sha256": (
            PHYSICAL_ASSIGNMENT_PROTOCOL_SHA
        ),
        "ordering_seed_manifest_sha256": (
            ORDERING_SEED_MANIFEST_SHA
        ),
        "slice_boundary_sha256": (
            SLICE_BOUNDARY_SHA
        ),
        "assignment_set_content_sha256": (
            assignment_set_sha
        ),
    }

    layer_schema_path = (
        assignment_root
        /
        "ASSIGNMENT_LAYER_SCHEMA.json"
    )

    set_binding_path = (
        assignment_root
        /
        "ASSIGNMENT_SET_BINDING.json"
    )

    config_manifest_path = (
        assignment_root
        /
        "CONFIG_ASSIGNMENT_SET_MANIFEST.csv"
    )

    write_json(
        layer_schema_path,
        layer_schema,
    )

    write_json(
        set_binding_path,
        set_binding,
    )

    write_csv(
        config_manifest_path,
        config_summary_rows,
        [
            "ConfigID",
            "AssignmentContentSHA256",
            "ConfigArtifactManifestSHA256",
            "TrainAssigned",
            "ValidationAssigned",
            "TestAssigned",
            "ReplayMismatches",
            "ReusedCompletedConfig",
        ],
    )

    top_artifact_rows = []

    for cfg in sorted(
        matrices
    ):
        config_root = (
            assignment_root
            /
            cfg
        )

        for role, path in [
            (
                "CONFIG_COMPLETE_STATE",
                config_root
                /
                "CONFIG_ASSIGNMENT_COMPLETE.json",
            ),
            (
                "CONFIG_ARTIFACT_MANIFEST",
                config_root
                /
                "CONFIG_ASSIGNMENT_ARTIFACT_MANIFEST.csv",
            ),
        ]:
            top_artifact_rows.append({
                "ArtifactRole": (
                    role
                ),
                "RelativePath": str(
                    path.relative_to(
                        assignment_root
                    )
                ),
                "SizeBytes": (
                    path.stat().st_size
                ),
                "SHA256": (
                    sha256_file(
                        path
                    )
                ),
            })

    for role, path in [
        (
            "ASSIGNMENT_LAYER_SCHEMA",
            layer_schema_path,
        ),
        (
            "ASSIGNMENT_SET_BINDING",
            set_binding_path,
        ),
        (
            "CONFIG_ASSIGNMENT_SET_MANIFEST",
            config_manifest_path,
        ),
    ]:
        top_artifact_rows.append({
            "ArtifactRole": (
                role
            ),
            "RelativePath": str(
                path.relative_to(
                    assignment_root
                )
            ),
            "SizeBytes": (
                path.stat().st_size
            ),
            "SHA256": (
                sha256_file(
                    path
                )
            ),
        })

    top_artifact_manifest_path = (
        assignment_root
        /
        "ASSIGNMENT_SET_ARTIFACT_MANIFEST.csv"
    )

    write_csv(
        top_artifact_manifest_path,
        top_artifact_rows,
        [
            "ArtifactRole",
            "RelativePath",
            "SizeBytes",
            "SHA256",
        ],
    )

    combined_assignment_artifact_manifest_sha = artifact_manifest_digest(
        top_artifact_rows
    )

    assignment_complete = {
        "status": (
            "PASS"
        ),
        "materialisation_id": (
            MATERIALISATION_ID
        ),
        "assignment_layer_set_id": (
            ASSIGNMENT_LAYER_SET_ID
        ),
        "assignment_set_content_sha256": (
            assignment_set_sha
        ),
        "combined_assignment_artifact_manifest_sha256": (
            combined_assignment_artifact_manifest_sha
        ),
        "configuration_count": (
            CONFIGS
        ),
        "scientific_boundary": {
            "physical_client_assignments_materialized": (
                True
            ),
            "participation_protocol_rebuilt": (
                False
            ),
            "malicious_client_identity_rebuilt": (
                False
            ),
            "attack_protocol_rebound": (
                False
            ),
            "scientific_optimizer_steps_executed": (
                0
            ),
            "scientific_training_started": (
                False
            ),
        },
    }

    write_json(
        assignment_root
        /
        "ASSIGNMENT_SET_COMPLETE.json",
        assignment_complete,
    )

    print(
        f"Assignment-set content SHA256: {assignment_set_sha}"
    )

    print(
        f"Combined assignment artifact-manifest SHA256: "
        f"{combined_assignment_artifact_manifest_sha}"
    )

    # ------------------------------------------------------------------
    # Gate F - write audit state and human-readable report.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE F - WRITE FINAL AUDIT STATE AND REPORT")
    print("=" * 60)
    print("")

    write_csv(
        audit_root
        /
        "CONFIG_MATERIALISATION_SUMMARY.csv",
        config_summary_rows,
        [
            "ConfigID",
            "AssignmentContentSHA256",
            "ConfigArtifactManifestSHA256",
            "TrainAssigned",
            "ValidationAssigned",
            "TestAssigned",
            "ReplayMismatches",
            "ReusedCompletedConfig",
        ],
    )

    final_state = {
        "status": (
            "PASS"
        ),
        "materialisation_id": (
            MATERIALISATION_ID
        ),
        "assignment_layer_set_id": (
            ASSIGNMENT_LAYER_SET_ID
        ),
        "immutable_binding": {
            "effective_dataset_fingerprint_sha256": (
                EFFECTIVE_DATASET_SHA
            ),
            "split_assignment_artifact_manifest_sha256": (
                SPLIT_ASSIGNMENT_SHA
            ),
            "client_partition_protocol_artifact_manifest_sha256": (
                CLIENT_PARTITION_PROTOCOL_SHA
            ),
            "count_plan_set_sha256": (
                COUNT_PLAN_SET_SHA
            ),
            "physical_assignment_protocol_artifact_manifest_sha256": (
                PHYSICAL_ASSIGNMENT_PROTOCOL_SHA
            ),
            "ordering_seed_manifest_sha256": (
                ORDERING_SEED_MANIFEST_SHA
            ),
            "slice_boundary_sha256": (
                SLICE_BOUNDARY_SHA
            ),
        },
        "assignment_set_content_sha256": (
            assignment_set_sha
        ),
        "combined_assignment_artifact_manifest_sha256": (
            combined_assignment_artifact_manifest_sha
        ),
        "audit": {
            "configurations_materialized": (
                CONFIGS
            ),
            "all_train_observations_assigned_once_per_configuration": (
                True
            ),
            "validation_assignments": (
                0
            ),
            "test_assignments": (
                0
            ),
            "exact_count_plan_reproduction_all_configs": (
                True
            ),
            "exact_client_capacity_reproduction_all_configs": (
                True
            ),
            "deterministic_replay_mismatches": (
                0
            ),
        },
        "scientific_boundary": {
            "physical_client_assignments_materialized": (
                True
            ),
            "participation_protocol_rebuilt": (
                False
            ),
            "malicious_client_identity_rebuilt": (
                False
            ),
            "attack_protocol_rebound": (
                False
            ),
            "scientific_optimizer_steps_executed": (
                0
            ),
            "scientific_training_started": (
                False
            ),
        },
    }

    write_json(
        final_state_path,
        final_state,
    )

    report = []

    add = report.append

    add(
        "CICIoT2023 TRANSFORMED-UNIQUE K=30 PHYSICAL CLIENT ASSIGNMENT MATERIALISATION"
    )
    add("=" * 78)
    add("")

    add("STATUS")
    add("-" * 78)
    add(
        "PASS"
    )
    add("")

    add("MATERIALISATION IDENTIFIERS")
    add("-" * 78)
    add(
        f"Materialisation ID: {MATERIALISATION_ID}"
    )
    add(
        f"Assignment-layer set ID: {ASSIGNMENT_LAYER_SET_ID}"
    )
    add("")

    add("IMMUTABLE BINDING")
    add("-" * 78)
    add(
        f"Effective dataset fingerprint SHA256: {EFFECTIVE_DATASET_SHA}"
    )
    add(
        f"Split assignment artifact-manifest SHA256: {SPLIT_ASSIGNMENT_SHA}"
    )
    add(
        f"Client-partition protocol artifact-manifest SHA256: "
        f"{CLIENT_PARTITION_PROTOCOL_SHA}"
    )
    add(
        f"Count-plan set SHA256: {COUNT_PLAN_SET_SHA}"
    )
    add(
        f"Physical-assignment protocol artifact-manifest SHA256: "
        f"{PHYSICAL_ASSIGNMENT_PROTOCOL_SHA}"
    )
    add(
        f"Ordering-seed manifest SHA256: {ORDERING_SEED_MANIFEST_SHA}"
    )
    add(
        f"Slice-boundary SHA256: {SLICE_BOUNDARY_SHA}"
    )
    add("")

    add("MATERIALISED LAYERS")
    add("-" * 78)
    add(
        "Configurations materialized: 10 / 10"
    )
    add(
        "Clients per configuration: 30"
    )
    add(
        "Assignment dtype: uint8"
    )
    add(
        "TRAIN values: client IDs 0..29"
    )
    add(
        "VALIDATION / TEST value: 255"
    )
    add(
        "Feature arrays copied: NO"
    )
    add("")

    add("COMPLETE COVERAGE AUDIT")
    add("-" * 78)
    add(
        "TRAIN observations assigned per configuration: 16549824 / 16549824"
    )
    add(
        "Every TRAIN effective observation assigned exactly once: YES"
    )
    add(
        "Invalid or unassigned TRAIN observations: 0"
    )
    add(
        "VALIDATION observations assigned to clients: 0"
    )
    add(
        "TEST observations assigned to clients: 0"
    )
    add("")

    add("COUNT-PLAN / CAPACITY AUDIT")
    add("-" * 78)
    add(
        "Exact frozen 7x30 count-plan reproduction: PASS ALL 10"
    )
    add(
        "Exact frozen client-capacity reproduction: PASS ALL 10"
    )
    add("")

    add("DETERMINISTIC REPLAY")
    add("-" * 78)
    add(
        "Configuration/label order digests replayed: EXACT MATCH ALL 70"
    )
    add(
        "Physical assignment replay mismatches: 0"
    )
    add("")

    add("CONFIGURATION ASSIGNMENT CONTENT FINGERPRINTS")
    add("-" * 78)

    for row in sorted(
        config_summary_rows,
        key=lambda item: item[
            "ConfigID"
        ],
    ):
        add(
            f"{row['ConfigID']}: {row['AssignmentContentSHA256']}"
        )

    add("")

    add("IMMUTABLE ASSIGNMENT SET")
    add("-" * 78)
    add(
        f"Assignment content hash ID: {ASSIGNMENT_CONTENT_HASH_ID}"
    )
    add(
        f"Assignment-set hash ID: {ASSIGNMENT_SET_HASH_ID}"
    )
    add(
        f"Assignment-set content SHA256: {assignment_set_sha}"
    )
    add(
        f"Combined assignment artifact-manifest SHA256: "
        f"{combined_assignment_artifact_manifest_sha}"
    )
    add("")

    add("SCIENTIFIC BOUNDARY")
    add("-" * 78)
    add(
        "Physical client assignments materialized: YES"
    )
    add(
        "Participation protocol rebuilt: NO"
    )
    add(
        "Malicious-client identity rebuilt: NO"
    )
    add(
        "Attack protocol rebound: NO"
    )
    add(
        "Scientific optimizer steps executed: 0"
    )
    add(
        "Scientific training started: NO"
    )
    add("")

    add("NEXT GATE")
    add("-" * 78)
    add(
        "Audit the completed physical assignment set as an immutable layer, "
        "then revalidate and rebuild the 30-client participation schedule and "
        "fixed nested malicious-client rankings against the new assignment-set "
        "fingerprint. Do not rebind attacks or training until participation and "
        "malicious identities are frozen on the rebuilt branch."
    )

    (
        audit_root
        /
        "TRANSFORMED_UNIQUE_PHYSICAL_ASSIGNMENT_MATERIALISATION_REPORT.txt"
    ).write_text(
        "\n".join(
            report
        )
        +
        "\n",
        encoding="utf-8",
    )

    print("")
    print("=" * 60)
    print("STATUS: PASS")
    print(
        f"CONFIGURATIONS MATERIALISED: {CONFIGS} / {CONFIGS}"
    )
    print(
        f"ASSIGNMENT-SET CONTENT SHA256: {assignment_set_sha}"
    )
    print(
        f"COMBINED ASSIGNMENT ARTIFACT MANIFEST: "
        f"{combined_assignment_artifact_manifest_sha}"
    )
    print(
        "VALIDATION / TEST ASSIGNMENTS: 0 / 0"
    )
    print(
        "DETERMINISTIC REPLAY MISMATCHES: 0"
    )
    print(
        "PARTICIPATION PROTOCOL REBUILT: NO"
    )
    print(
        "SCIENTIFIC TRAINING STARTED: NO"
    )
    print("=" * 60)
    print("")

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )
