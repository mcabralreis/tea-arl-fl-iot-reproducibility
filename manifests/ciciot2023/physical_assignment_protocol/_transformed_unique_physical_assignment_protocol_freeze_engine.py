import argparse
import csv
import hashlib
import json
import sys
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

TRAIN_WEIGHT_POLICY_SHA = (
    "F77939B96EF739F6A298B821CB01BE4142248FFB517A600D64D482D525DD2E47"
)

CANONICAL_WEIGHT_VECTOR_SHA = (
    "C869BE9CA3CA092EBD86C1C86C99B24CA7668D99AEB4A9AEF3E8F8D46A818A69"
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


# ============================================================================
# Gate-75 identifiers
# ============================================================================

GATE75_PROTOCOL_ID = (
    "CICIoT2023_TASK7_TRANSFORMED_UNIQUE_K30_"
    "CAPACITY_BALANCED_DIRICHLET_ALPHA_0P1_1P0_5SEEDS_V2"
)

GATE75_COUNT_PLAN_SET_ID = (
    "CICIoT2023_TASK7_TRANSFORMED_UNIQUE_K30_EXACT_COUNT_PLAN_SET_V2"
)


# ============================================================================
# Gate-76 freeze identifiers
# ============================================================================

FREEZE_ID = (
    "CICIoT2023_TRANSFORMED_UNIQUE_PHYSICAL_ASSIGNMENT_PROTOCOL_FREEZE_V1"
)

PHYSICAL_ASSIGNMENT_PROTOCOL_ID = (
    "CICIoT2023_TASK7_TRANSFORMED_UNIQUE_K30_"
    "CONFIG_BOUND_HASH_ORDER_SLICE_ASSIGNMENT_V1"
)

ORDERING_ALGORITHM_ID = (
    "SPLITMIX64_DUAL_KEY_CONFIG_BOUND_LABEL_ORDER_V1"
)

ORDERING_SEED_DERIVATION_ID = (
    "SHA256_CONFIG_BOUND_PHYSICAL_ASSIGNMENT_ORDER_SEED_V1"
)

ORDERING_SEED_UINT64_DERIVATION = (
    "UNSIGNED_BIG_ENDIAN_INTEGER_FROM_FIRST_8_BYTES_OF_SEED_SHA256"
)

SLICE_ALLOCATION_ID = (
    "ASCENDING_CLIENT_ID_CONTIGUOUS_FROZEN_COUNT_PLAN_SLICES_V1"
)

ASSIGNMENT_DOMAIN_ID = (
    "TRAIN_ONLY_EFFECTIVE_TRANSFORMED_UNIQUE_OBSERVATIONS_V1"
)

SEED_MANIFEST_HASH_ID = (
    "SORTED_CONFIG_ID_PLUS_SEED_SHA256_PLUS_SEED_UINT64_LF_V1"
)

SLICE_BOUNDARY_HASH_ID = (
    "SORTED_CONFIG_LABEL_CLIENT_START_END_COUNT_LF_V1"
)


# ============================================================================
# Dataset geometry
# ============================================================================

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

TRAIN_TOTAL = 16_549_824
CLIENT_COUNT = 30
CONFIG_COUNT = 10

EXPECTED_CAPACITIES = (
    [551_661] * 24
    +
    [551_660] * 6
)


# ============================================================================
# Ordering constants
# ============================================================================

UINT64_MASK = 0xFFFFFFFFFFFFFFFF

SPLITMIX64_INCREMENT = (
    0x9E3779B97F4A7C15
)

SPLITMIX64_MULTIPLIER_1 = (
    0xBF58476D1CE4E5B9
)

SPLITMIX64_MULTIPLIER_2 = (
    0x94D049BB133111EB
)

DUAL_KEY_XOR_CONSTANT = (
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
            item[
                "ArtifactRole"
            ],
            item[
                "FileName"
            ],
        ),
    ):
        digest.update(
            (
                f"{row['ArtifactRole']}\t"
                f"{row['FileName']}\t"
                f"{row['SizeBytes']}\t"
                f"{row['SHA256']}\n"
            ).encode(
                "utf-8"
            )
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


def capacity_vector_sha256(capacities):
    contiguous = np.ascontiguousarray(
        capacities,
        dtype=np.int64,
    )

    return hashlib.sha256(
        contiguous.tobytes(
            order="C"
        )
    ).hexdigest().upper()


def rotl64(value, shift):
    shift = shift % 64

    return (
        (
            (
                value
                <<
                shift
            )
            &
            UINT64_MASK
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


def splitmix64_scalar(value):
    z = (
        value
        +
        SPLITMIX64_INCREMENT
    ) & UINT64_MASK

    z = (
        (
            z
            ^
            (
                z
                >>
                30
            )
        )
        *
        SPLITMIX64_MULTIPLIER_1
    ) & UINT64_MASK

    z = (
        (
            z
            ^
            (
                z
                >>
                27
            )
        )
        *
        SPLITMIX64_MULTIPLIER_2
    ) & UINT64_MASK

    z = (
        z
        ^
        (
            z
            >>
            31
        )
    ) & UINT64_MASK

    return z


def derive_order_keys(
    transformed_hash1,
    transformed_hash2,
    seed_uint64,
):
    key1 = splitmix64_scalar(
        transformed_hash1
        ^
        seed_uint64
    )

    key2 = splitmix64_scalar(
        transformed_hash2
        ^
        rotl64(
            seed_uint64,
            32,
        )
        ^
        DUAL_KEY_XOR_CONSTANT
    )

    return (
        key1,
        key2,
    )


def derive_assignment_seed(
    config_id,
    matrix_content_sha256,
):
    material = (
        "CICIoT2023_TASK7_TRANSFORMED_UNIQUE_K30_"
        "PHYSICAL_ASSIGNMENT_ORDER_V1"
        "|"
        f"EFFECTIVE_DATASET_SHA256={EFFECTIVE_DATASET_SHA}"
        "|"
        f"SPLIT_ASSIGNMENT_SHA256={SPLIT_ASSIGNMENT_SHA}"
        "|"
        f"COUNT_PLAN_SET_SHA256={COUNT_PLAN_SET_SHA}"
        "|"
        f"CONFIG_ID={config_id}"
        "|"
        f"MATRIX_CONTENT_SHA256={matrix_content_sha256}"
    )

    seed_sha256 = hashlib.sha256(
        material.encode(
            "utf-8"
        )
    ).hexdigest().upper()

    seed_uint64 = int.from_bytes(
        bytes.fromhex(
            seed_sha256[
                :16
            ]
        ),
        byteorder="big",
        signed=False,
    )

    return (
        material,
        seed_sha256,
        seed_uint64,
    )


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
            ).encode(
                "utf-8"
            )
        )

    return digest.hexdigest().upper()


def slice_boundary_sha256(rows):
    digest = hashlib.sha256()

    for row in sorted(
        rows,
        key=lambda item: (
            item[
                "ConfigID"
            ],
            int(
                item[
                    "Task7LabelID"
                ]
            ),
            int(
                item[
                    "ClientID"
                ]
            ),
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
            ).encode(
                "utf-8"
            )
        )

    return digest.hexdigest().upper()


def load_count_plan_matrix(
    path,
    capacities,
):
    rows = read_csv(
        path
    )

    require_equal(
        len(
            rows
        ),
        CLIENT_COUNT,
        f"Count-plan client row count: {path.name}",
    )

    matrix = np.zeros(
        (
            len(
                TRAIN_COUNTS
            ),
            CLIENT_COUNT,
        ),
        dtype=np.int64,
    )

    observed_client_ids = []

    for row in rows:
        client_id = int(
            row[
                "ClientID"
            ]
        )

        observed_client_ids.append(
            client_id
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
            f"Count-plan capacity {path.name}, client {client_id}",
        )

        for label_id in range(
            len(
                TRAIN_COUNTS
            )
        ):
            field = (
                f"Label{label_id}_"
                f"{LABEL_NAMES[label_id]}"
            )

            value = int(
                row[
                    field
                ]
            )

            require_true(
                value
                >=
                0,
                (
                    f"Negative count in {path.name}, "
                    f"label {label_id}, client {client_id}"
                ),
            )

            matrix[
                label_id,
                client_id
            ] = value

    require_equal(
        sorted(
            observed_client_ids
        ),
        list(
            range(
                CLIENT_COUNT
            )
        ),
        f"ClientID coverage: {path.name}",
    )

    return matrix


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gate75-freeze-json",
        required=True,
    )

    parser.add_argument(
        "--gate75-protocol-json",
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
        "--output",
        required=True,
    )

    args = parser.parse_args()

    output_root = Path(
        args.output
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence_paths = {
        "GATE75_PARTITION_PROTOCOL_FREEZE": Path(
            args.gate75_freeze_json
        ),
        "GATE75_PARTITION_PROTOCOL": Path(
            args.gate75_protocol_json
        ),
        "GATE75_COUNT_PLAN_MANIFEST": Path(
            args.gate75_plan_manifest_csv
        ),
        "GATE75_CAPACITIES": Path(
            args.gate75_capacities_csv
        ),
        "GATE71_SPLIT_MATERIALISATION": Path(
            args.gate71_state_json
        ),
    }

    plan_root = Path(
        args.gate75_plan_root
    )

    # ------------------------------------------------------------------
    # Gate A - verify immutable current chain.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE A - VERIFY CURRENT CHAIN AND GATE-75 FREEZE")
    print("=" * 60)
    print("")

    gate71 = load_json(
        evidence_paths[
            "GATE71_SPLIT_MATERIALISATION"
        ]
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

    require_true(
        gate71[
            "scientific_boundary"
        ][
            "client_assignments_materialized"
        ]
        is False,
        "Gate-71 says client assignments already exist.",
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
        evidence_paths[
            "GATE75_PARTITION_PROTOCOL_FREEZE"
        ]
    )

    require_equal(
        gate75.get(
            "status"
        ),
        "FROZEN",
        "Gate-75 status",
    )

    require_equal(
        gate75.get(
            "protocol_id"
        ),
        GATE75_PROTOCOL_ID,
        "Gate-75 protocol ID",
    )

    require_equal(
        gate75.get(
            "count_plan_set_id"
        ),
        GATE75_COUNT_PLAN_SET_ID,
        "Gate-75 count-plan set ID",
    )

    require_equal(
        gate75[
            "capacity_vector_sha256"
        ],
        CAPACITY_VECTOR_SHA,
        "Gate-75 capacity vector SHA256",
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
        "Gate-75 protocol artifact-manifest SHA256",
    )

    require_true(
        gate75[
            "materialization_boundary"
        ][
            "physical_assignment_algorithm_frozen"
        ]
        is False,
        "Gate-75 says physical assignment algorithm already frozen.",
    )

    require_true(
        gate75[
            "materialization_boundary"
        ][
            "physical_client_assignments_materialized"
        ]
        is False,
        "Gate-75 says physical clients already materialized.",
    )

    require_true(
        gate75[
            "materialization_boundary"
        ][
            "scientific_training_started"
        ]
        is False,
        "Gate-75 says scientific training started.",
    )

    protocol75 = load_json(
        evidence_paths[
            "GATE75_PARTITION_PROTOCOL"
        ]
    )

    require_equal(
        protocol75.get(
            "status"
        ),
        "FROZEN",
        "Gate-75 protocol JSON status",
    )

    require_equal(
        protocol75.get(
            "protocol_id"
        ),
        GATE75_PROTOCOL_ID,
        "Gate-75 protocol JSON ID",
    )

    for key, expected in [
        (
            "effective_dataset_fingerprint_sha256",
            EFFECTIVE_DATASET_SHA,
        ),
        (
            "split_protocol_artifact_manifest_sha256",
            SPLIT_PROTOCOL_SHA,
        ),
        (
            "split_assignment_artifact_manifest_sha256",
            SPLIT_ASSIGNMENT_SHA,
        ),
        (
            "train_weight_policy_artifact_manifest_sha256",
            TRAIN_WEIGHT_POLICY_SHA,
        ),
        (
            "canonical_weight_vector_sha256",
            CANONICAL_WEIGHT_VECTOR_SHA,
        ),
    ]:
        require_equal(
            protocol75[
                "immutable_binding"
            ][
                key
            ],
            expected,
            f"Gate-75 protocol binding {key}",
        )

    print(
        "Effective dataset: BOUND"
    )
    print(
        "Split assignment: BOUND"
    )
    print(
        "Gate-75 partition protocol: FROZEN"
    )
    print(
        "Gate-75 exact count plans: FROZEN"
    )
    print(
        "Physical client assignments: NO"
    )
    print(
        "Scientific training started: NO"
    )

    # ------------------------------------------------------------------
    # Gate B - verify capacities and all ten frozen count plans.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE B - VERIFY FROZEN CAPACITIES AND COUNT PLANS")
    print("=" * 60)
    print("")

    capacity_rows = read_csv(
        evidence_paths[
            "GATE75_CAPACITIES"
        ]
    )

    require_equal(
        len(
            capacity_rows
        ),
        CLIENT_COUNT,
        "Gate-75 capacity row count",
    )

    capacities = np.zeros(
        CLIENT_COUNT,
        dtype=np.int64,
    )

    for row in capacity_rows:
        client_id = int(
            row[
                "ClientID"
            ]
        )

        capacities[
            client_id
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

    require_equal(
        capacity_vector_sha256(
            capacities
        ),
        CAPACITY_VECTOR_SHA,
        "Gate-75 exact capacity vector replay",
    )

    plan_manifest_rows = read_csv(
        evidence_paths[
            "GATE75_COUNT_PLAN_MANIFEST"
        ]
    )

    require_equal(
        len(
            plan_manifest_rows
        ),
        CONFIG_COUNT,
        "Gate-75 count-plan manifest row count",
    )

    seen_configs = set()

    matrices = {}

    for row in plan_manifest_rows:
        cfg = row[
            "ConfigID"
        ]

        require_true(
            cfg
            not in seen_configs,
            f"Duplicate Gate-75 ConfigID: {cfg}",
        )

        seen_configs.add(
            cfg
        )

        plan_path = (
            plan_root
            /
            row[
                "FrozenFileName"
            ]
        )

        require_true(
            plan_path.exists(),
            f"Missing frozen Gate-75 count plan: {plan_path}",
        )

        require_equal(
            sha256_file(
                plan_path
            ),
            row[
                "FrozenFileSHA256"
            ],
            f"Frozen count-plan file SHA256: {cfg}",
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
            f"Frozen count-plan matrix SHA256: {cfg}",
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
                            len(
                                TRAIN_COUNTS
                            )
                        )
                    ],
                    dtype=np.int64,
                ),
            ),
            f"Frozen count-plan class totals: {cfg}",
        )

        require_true(
            np.array_equal(
                matrix.sum(
                    axis=0
                ),
                capacities,
            ),
            f"Frozen count-plan capacities: {cfg}",
        )

        matrices[
            cfg
        ] = matrix

    require_equal(
        len(
            matrices
        ),
        CONFIG_COUNT,
        "Verified frozen count-plan count",
    )

    print(
        "Exact client capacities: PASS"
    )
    print(
        "Frozen count plans verified: 10 / 10"
    )
    print(
        "Exact class totals: PASS ALL 10"
    )
    print(
        "Exact client capacities: PASS ALL 10"
    )

    # ------------------------------------------------------------------
    # Gate C - derive and freeze exact configuration ordering seeds.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE C - DERIVE CONFIGURATION-SPECIFIC ORDERING SEEDS")
    print("=" * 60)
    print("")

    seed_rows = []

    seen_seed_sha = set()

    seen_seed_uint64 = set()

    manifest_by_config = {
        row[
            "ConfigID"
        ]: row
        for row
        in plan_manifest_rows
    }

    for cfg in sorted(
        matrices
    ):
        matrix_sha = manifest_by_config[
            cfg
        ][
            "MatrixContentSHA256"
        ]

        (
            seed_material,
            seed_sha256,
            seed_uint64,
        ) = derive_assignment_seed(
            cfg,
            matrix_sha,
        )

        require_true(
            seed_sha256
            not in seen_seed_sha,
            f"Duplicate assignment seed SHA256: {cfg}",
        )

        require_true(
            seed_uint64
            not in seen_seed_uint64,
            f"Duplicate assignment seed uint64: {cfg}",
        )

        replay = derive_assignment_seed(
            cfg,
            matrix_sha,
        )

        require_equal(
            replay,
            (
                seed_material,
                seed_sha256,
                seed_uint64,
            ),
            f"Assignment seed replay: {cfg}",
        )

        seen_seed_sha.add(
            seed_sha256
        )

        seen_seed_uint64.add(
            seed_uint64
        )

        seed_rows.append({
            "ConfigID": (
                cfg
            ),
            "Alpha": (
                manifest_by_config[
                    cfg
                ][
                    "Alpha"
                ]
            ),
            "ExperimentalSeed": (
                manifest_by_config[
                    cfg
                ][
                    "ExperimentalSeed"
                ]
            ),
            "MatrixContentSHA256": (
                matrix_sha
            ),
            "SeedDerivationID": (
                ORDERING_SEED_DERIVATION_ID
            ),
            "SeedMaterial": (
                seed_material
            ),
            "SeedSHA256": (
                seed_sha256
            ),
            "SeedUInt64": (
                seed_uint64
            ),
            "SeedUInt64Derivation": (
                ORDERING_SEED_UINT64_DERIVATION
            ),
        })

        print(
            f"{cfg} | seed_sha={seed_sha256} | "
            f"seed_uint64={seed_uint64}"
        )

    require_equal(
        len(
            seed_rows
        ),
        CONFIG_COUNT,
        "Frozen assignment seed count",
    )

    seed_manifest_sha = seed_manifest_sha256(
        seed_rows
    )

    print(
        f"Ordering seed manifest SHA256: {seed_manifest_sha}"
    )

    # ------------------------------------------------------------------
    # Gate D - derive exact per-label/client slice boundaries.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE D - DERIVE EXACT FROZEN SLICE BOUNDARIES")
    print("=" * 60)
    print("")

    slice_rows = []

    for cfg in sorted(
        matrices
    ):
        matrix = matrices[
            cfg
        ]

        for label_id in range(
            len(
                TRAIN_COUNTS
            )
        ):
            cursor = 0

            for client_id in range(
                CLIENT_COUNT
            ):
                count = int(
                    matrix[
                        label_id,
                        client_id
                    ]
                )

                start = cursor

                end = (
                    start
                    +
                    count
                )

                slice_rows.append({
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
                    "StartInclusive": (
                        start
                    ),
                    "EndExclusive": (
                        end
                    ),
                    "Count": (
                        count
                    ),
                })

                cursor = end

            require_equal(
                cursor,
                TRAIN_COUNTS[
                    label_id
                ],
                (
                    f"Final frozen slice boundary "
                    f"{cfg}, label {label_id}"
                ),
            )

    require_equal(
        len(
            slice_rows
        ),
        CONFIG_COUNT
        *
        len(
            TRAIN_COUNTS
        )
        *
        CLIENT_COUNT,
        "Frozen slice-boundary row count",
    )

    slice_boundary_sha = slice_boundary_sha256(
        slice_rows
    )

    print(
        f"Slice boundary rows: {len(slice_rows)}"
    )
    print(
        f"Slice boundary SHA256: {slice_boundary_sha}"
    )

    # ------------------------------------------------------------------
    # Gate E - freeze exact ordering algorithm and assignment semantics.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE E - FREEZE ORDERING ALGORITHM AND ASSIGNMENT SEMANTICS")
    print("=" * 60)
    print("")

    algorithm = {
        "status": (
            "FROZEN"
        ),
        "ordering_algorithm_id": (
            ORDERING_ALGORITHM_ID
        ),
        "assignment_domain_id": (
            ASSIGNMENT_DOMAIN_ID
        ),
        "ordering_scope": (
            "INDEPENDENTLY_WITHIN_EACH_CONFIGURATION_AND_TASK7_LABEL"
        ),
        "eligible_observations": (
            "EFFECTIVE_OBSERVATIONS_WITH_SPLIT_ID_TRAIN_ONLY"
        ),
        "effective_observation_identity": [
            "TransformedHash1",
            "TransformedHash2",
        ],
        "seed_scope": (
            "ONE_FROZEN_SEED_PER_CONFIGURATION"
        ),
        "task7_label_role": (
            "ORDERING_SCOPE_ONLY_NOT_A_RANK_KEY"
        ),
        "order_key_1": (
            "splitmix64(TransformedHash1 XOR SeedU64)"
        ),
        "order_key_2": (
            "splitmix64(TransformedHash2 XOR ROTL64(SeedU64,32) "
            "XOR 0xD1B54A32D192ED03)"
        ),
        "ascending_lexicographic_sort_tuple": [
            "OrderKey1",
            "OrderKey2",
            "TransformedHash1",
            "TransformedHash2",
        ],
        "splitmix64_definition": {
            "modulus": (
                "2^64"
            ),
            "increment_hex": (
                "0x9E3779B97F4A7C15"
            ),
            "xor_shift_1": (
                30
            ),
            "multiplier_1_hex": (
                "0xBF58476D1CE4E5B9"
            ),
            "xor_shift_2": (
                27
            ),
            "multiplier_2_hex": (
                "0x94D049BB133111EB"
            ),
            "xor_shift_3": (
                31
            ),
            "dual_key_xor_constant_hex": (
                "0xD1B54A32D192ED03"
            ),
        },
        "slice_allocation_id": (
            SLICE_ALLOCATION_ID
        ),
        "slice_allocation": {
            "client_order": (
                "ASCENDING_CLIENT_ID_0_TO_29"
            ),
            "slice_lengths": (
                "EXACT_FROZEN_CONFIGURATION_COUNT_PLAN_ROW"
            ),
            "slice_interpretation": (
                "CONTIGUOUS_HALF_OPEN_INTERVALS_IN_SORTED_LABEL_ORDER"
            ),
            "zero_count_client": (
                "EMPTY_SLICE"
            ),
        },
        "required_per_configuration_properties": {
            "every_train_effective_observation_assigned_exactly_once": (
                True
            ),
            "assigned_client_id_range": (
                "0_TO_29"
            ),
            "validation_observations_assigned_to_clients": (
                False
            ),
            "test_observations_assigned_to_clients": (
                False
            ),
            "exact_count_plan_reproduction": (
                True
            ),
            "exact_client_capacity_reproduction": (
                True
            ),
        },
        "prohibited_rank_or_assignment_inputs": [
            "OldSplitID",
            "OldPhysicalClientAssignment",
            "RepresentativeSourceBucket",
            "RepresentativeSourceRowIndex",
            "RawMultiplicity",
            "TransformedGroupRawExactMultiplicity",
            "SumSourceRawMultiplicity",
            "Provenance",
            "FeatureValues",
            "TRAINWeightVector",
            "ModelResults",
            "AttackResults",
            "VALIDATIONObservations",
            "TESTObservations",
        ],
    }

    protocol = {
        "status": (
            "FROZEN"
        ),
        "freeze_id": (
            FREEZE_ID
        ),
        "physical_assignment_protocol_id": (
            PHYSICAL_ASSIGNMENT_PROTOCOL_ID
        ),
        "ordering_algorithm_id": (
            ORDERING_ALGORITHM_ID
        ),
        "ordering_seed_derivation_id": (
            ORDERING_SEED_DERIVATION_ID
        ),
        "slice_allocation_id": (
            SLICE_ALLOCATION_ID
        ),
        "immutable_binding": {
            "effective_dataset_fingerprint_sha256": (
                EFFECTIVE_DATASET_SHA
            ),
            "split_protocol_artifact_manifest_sha256": (
                SPLIT_PROTOCOL_SHA
            ),
            "split_assignment_artifact_manifest_sha256": (
                SPLIT_ASSIGNMENT_SHA
            ),
            "train_weight_policy_artifact_manifest_sha256": (
                TRAIN_WEIGHT_POLICY_SHA
            ),
            "canonical_weight_vector_sha256": (
                CANONICAL_WEIGHT_VECTOR_SHA
            ),
            "client_partition_protocol_artifact_manifest_sha256": (
                CLIENT_PARTITION_PROTOCOL_SHA
            ),
            "capacity_vector_sha256": (
                CAPACITY_VECTOR_SHA
            ),
            "count_plan_set_sha256": (
                COUNT_PLAN_SET_SHA
            ),
        },
        "frozen_configuration_count": (
            CONFIG_COUNT
        ),
        "frozen_seed_manifest_hash_id": (
            SEED_MANIFEST_HASH_ID
        ),
        "frozen_seed_manifest_sha256": (
            seed_manifest_sha
        ),
        "frozen_slice_boundary_hash_id": (
            SLICE_BOUNDARY_HASH_ID
        ),
        "frozen_slice_boundary_sha256": (
            slice_boundary_sha
        ),
        "assignment_semantics": {
            "configuration_independence": (
                "EACH_CONFIG_HAS_ITS_OWN_COMPLETE_ASSIGNMENT_LAYER"
            ),
            "train_only": (
                True
            ),
            "validation_and_test_global_not_client_partitioned": (
                True
            ),
            "one_client_per_train_observation_per_configuration": (
                True
            ),
            "count_plan_is_primary_assignment_target": (
                True
            ),
        },
        "scientific_boundary": {
            "physical_assignment_algorithm_frozen": (
                True
            ),
            "ordering_seeds_frozen": (
                True
            ),
            "slice_boundaries_frozen": (
                True
            ),
            "physical_client_assignments_materialized": (
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

    # Deterministic algorithm self-check.
    probe = [
        (
            0x0000000000000000,
            0x0000000000000000,
        ),
        (
            0x0123456789ABCDEF,
            0xFEDCBA9876543210,
        ),
        (
            0xFFFFFFFFFFFFFFFF,
            0x1111111111111111,
        ),
    ]

    probe_seed = int(
        seed_rows[
            0
        ][
            "SeedUInt64"
        ]
    )

    replay_1 = [
        derive_order_keys(
            h1,
            h2,
            probe_seed,
        )
        for h1, h2
        in probe
    ]

    replay_2 = [
        derive_order_keys(
            h1,
            h2,
            probe_seed,
        )
        for h1, h2
        in probe
    ]

    require_equal(
        replay_1,
        replay_2,
        "Ordering algorithm deterministic self-check",
    )

    require_equal(
        len(
            set(
                replay_1
            )
        ),
        len(
            replay_1
        ),
        "Ordering algorithm probe distinctness",
    )

    print(
        f"Ordering algorithm ID: {ORDERING_ALGORITHM_ID}"
    )
    print(
        "Configuration-specific ordering seeds: 10 / 10"
    )
    print(
        "TASK7 label role: ordering scope only"
    )
    print(
        "Assignment slices: client IDs 0..29 in ascending order"
    )
    print(
        "Algorithm deterministic self-check: PASS"
    )

    # ------------------------------------------------------------------
    # Gate F - write frozen artifacts and immutable manifest.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE F - WRITE FROZEN ASSIGNMENT PROTOCOL ARTIFACTS")
    print("=" * 60)
    print("")

    algorithm_path = (
        output_root
        /
        "FROZEN_PHYSICAL_ASSIGNMENT_ORDERING_ALGORITHM.json"
    )

    protocol_path = (
        output_root
        /
        "FROZEN_PHYSICAL_ASSIGNMENT_PROTOCOL.json"
    )

    seeds_path = (
        output_root
        /
        "FROZEN_CONFIGURATION_ORDERING_SEEDS.csv"
    )

    slices_path = (
        output_root
        /
        "FROZEN_LABEL_CLIENT_SLICE_BOUNDARIES.csv"
    )

    write_json(
        algorithm_path,
        algorithm,
    )

    write_json(
        protocol_path,
        protocol,
    )

    write_csv(
        seeds_path,
        seed_rows,
        [
            "ConfigID",
            "Alpha",
            "ExperimentalSeed",
            "MatrixContentSHA256",
            "SeedDerivationID",
            "SeedMaterial",
            "SeedSHA256",
            "SeedUInt64",
            "SeedUInt64Derivation",
        ],
    )

    write_csv(
        slices_path,
        slice_rows,
        [
            "ConfigID",
            "Task7LabelID",
            "Task7Label",
            "ClientID",
            "StartInclusive",
            "EndExclusive",
            "Count",
        ],
    )

    artifact_rows = []

    for role, path in [
        (
            "PHYSICAL_ASSIGNMENT_PROTOCOL",
            protocol_path,
        ),
        (
            "ORDERING_ALGORITHM",
            algorithm_path,
        ),
        (
            "CONFIGURATION_ORDERING_SEEDS",
            seeds_path,
        ),
        (
            "LABEL_CLIENT_SLICE_BOUNDARIES",
            slices_path,
        ),
    ]:
        artifact_rows.append({
            "ArtifactRole": (
                role
            ),
            "FileName": (
                path.name
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
        output_root
        /
        "FROZEN_PHYSICAL_ASSIGNMENT_PROTOCOL_ARTIFACT_MANIFEST.csv"
    )

    write_csv(
        artifact_manifest_path,
        artifact_rows,
        [
            "ArtifactRole",
            "FileName",
            "SizeBytes",
            "SHA256",
        ],
    )

    combined_manifest_sha256 = artifact_manifest_digest(
        artifact_rows
    )

    freeze_state = {
        "status": (
            "FROZEN"
        ),
        "freeze_id": (
            FREEZE_ID
        ),
        "physical_assignment_protocol_id": (
            PHYSICAL_ASSIGNMENT_PROTOCOL_ID
        ),
        "ordering_algorithm_id": (
            ORDERING_ALGORITHM_ID
        ),
        "ordering_seed_manifest_sha256": (
            seed_manifest_sha
        ),
        "slice_boundary_sha256": (
            slice_boundary_sha
        ),
        "frozen_physical_assignment_protocol_artifact_manifest_sha256": (
            combined_manifest_sha256
        ),
        "materialization_boundary": {
            "physical_client_assignments_materialized": (
                False
            ),
            "scientific_training_started": (
                False
            ),
        },
    }

    freeze_state_path = (
        output_root
        /
        "TRANSFORMED_UNIQUE_PHYSICAL_ASSIGNMENT_PROTOCOL_FREEZE.json"
    )

    write_json(
        freeze_state_path,
        freeze_state,
    )

    # Evidence manifest.
    evidence_rows = []

    for role, path in evidence_paths.items():
        evidence_rows.append({
            "EvidenceRole": (
                role
            ),
            "Path": str(
                path
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

    write_csv(
        output_root
        /
        "FREEZE_EVIDENCE_MANIFEST.csv",
        evidence_rows,
        [
            "EvidenceRole",
            "Path",
            "SizeBytes",
            "SHA256",
        ],
    )

    # ------------------------------------------------------------------
    # Human-readable report.
    # ------------------------------------------------------------------

    report = []

    add = report.append

    add(
        "CICIoT2023 TRANSFORMED-UNIQUE PHYSICAL CLIENT ASSIGNMENT PROTOCOL FREEZE"
    )
    add("=" * 78)
    add("")

    add("STATUS")
    add("-" * 78)
    add(
        "FROZEN"
    )
    add("")

    add("PROTOCOL IDENTIFIERS")
    add("-" * 78)
    add(
        f"Freeze ID: {FREEZE_ID}"
    )
    add(
        f"Physical assignment protocol ID: {PHYSICAL_ASSIGNMENT_PROTOCOL_ID}"
    )
    add(
        f"Ordering algorithm ID: {ORDERING_ALGORITHM_ID}"
    )
    add(
        f"Ordering seed derivation ID: {ORDERING_SEED_DERIVATION_ID}"
    )
    add(
        f"Slice allocation ID: {SLICE_ALLOCATION_ID}"
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
        f"Capacity vector SHA256: {CAPACITY_VECTOR_SHA}"
    )
    add(
        f"Count-plan set SHA256: {COUNT_PLAN_SET_SHA}"
    )
    add("")

    add("FROZEN ASSIGNMENT DOMAIN")
    add("-" * 78)
    add(
        "Eligible observations: TRAIN effective transformed-unique observations only"
    )
    add(
        "VALIDATION observations assigned to clients: NO"
    )
    add(
        "TEST observations assigned to clients: NO"
    )
    add(
        "Each configuration has an independent complete assignment layer: YES"
    )
    add(
        "Exactly one client per TRAIN observation per configuration: REQUIRED"
    )
    add("")

    add("FROZEN CONFIGURATION ORDERING SEEDS")
    add("-" * 78)
    add(
        "Seeds: 10"
    )
    add(
        f"Seed manifest hash ID: {SEED_MANIFEST_HASH_ID}"
    )
    add(
        f"Seed manifest SHA256: {seed_manifest_sha}"
    )
    add("")

    for row in seed_rows:
        add(
            f"{row['ConfigID']}:"
        )
        add(
            f"  Seed SHA256: {row['SeedSHA256']}"
        )
        add(
            f"  Seed uint64: {row['SeedUInt64']}"
        )

    add("")

    add("FROZEN ORDERING ALGORITHM")
    add("-" * 78)
    add(
        "Scope: independently within each configuration and TASK7 label"
    )
    add(
        "Effective observation identity:"
    )
    add(
        "  TransformedHash1"
    )
    add(
        "  TransformedHash2"
    )
    add("")
    add(
        "OrderKey1 = splitmix64(TransformedHash1 XOR SeedU64)"
    )
    add(
        "OrderKey2 = splitmix64(TransformedHash2 XOR ROTL64(SeedU64,32) "
        "XOR 0xD1B54A32D192ED03)"
    )
    add("")
    add(
        "Ascending lexicographic sort tuple:"
    )
    add(
        "  OrderKey1"
    )
    add(
        "  OrderKey2"
    )
    add(
        "  TransformedHash1"
    )
    add(
        "  TransformedHash2"
    )
    add(
        "TASK7 label used as rank-key input: NO"
    )
    add("")

    add("FROZEN SLICE ALLOCATION")
    add("-" * 78)
    add(
        "Client order: ascending ClientID 0..29"
    )
    add(
        "Slice lengths: exact frozen configuration count-plan row"
    )
    add(
        "Slice type: contiguous half-open intervals in sorted label order"
    )
    add(
        "Zero-count client: empty slice"
    )
    add(
        f"Frozen slice-boundary rows: {len(slice_rows)}"
    )
    add(
        f"Slice-boundary hash ID: {SLICE_BOUNDARY_HASH_ID}"
    )
    add(
        f"Slice-boundary SHA256: {slice_boundary_sha}"
    )
    add("")

    add("PROHIBITED ASSIGNMENT INPUTS")
    add("-" * 78)
    add(
        "Old split ID"
    )
    add(
        "Old physical client assignments"
    )
    add(
        "Representative source bucket / row"
    )
    add(
        "RawMultiplicity"
    )
    add(
        "Transformed-group multiplicity"
    )
    add(
        "SumSourceRawMultiplicity"
    )
    add(
        "Provenance"
    )
    add(
        "Feature values"
    )
    add(
        "TRAIN weight vector"
    )
    add(
        "Model outcomes"
    )
    add(
        "Attack outcomes"
    )
    add(
        "VALIDATION observations"
    )
    add(
        "TEST observations"
    )
    add("")

    add("IMMUTABLE PROTOCOL ARTIFACT")
    add("-" * 78)

    for row in artifact_rows:
        add(
            f"{row['ArtifactRole']}: {row['FileName']}"
        )
        add(
            f"  SHA256: {row['SHA256']}"
        )

    add("")
    add(
        f"Combined physical-assignment protocol artifact-manifest SHA256: "
        f"{combined_manifest_sha256}"
    )
    add("")

    add("SCIENTIFIC STATE")
    add("-" * 78)
    add(
        "Physical assignment algorithm frozen: YES"
    )
    add(
        "Configuration ordering seeds frozen: YES"
    )
    add(
        "Label/client slice boundaries frozen: YES"
    )
    add(
        "Physical client assignments materialized: NO"
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
        "Materialise the ten physical TRAIN client-assignment layers from the "
        "immutable transformed hashes, TRAIN split membership, TASK7 labels, "
        "configuration-specific frozen ordering seeds, and frozen slice "
        "boundaries. Then independently audit exact one-client coverage, exact "
        "count-plan reproduction, exact client capacities, deterministic replay, "
        "and zero assignment of VALIDATION/TEST observations before rebuilding "
        "participation or attack protocols."
    )

    (
        output_root
        /
        "TRANSFORMED_UNIQUE_PHYSICAL_ASSIGNMENT_PROTOCOL_FREEZE_REPORT.txt"
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
    print("STATUS: FROZEN")
    print(
        f"PHYSICAL ASSIGNMENT PROTOCOL: {PHYSICAL_ASSIGNMENT_PROTOCOL_ID}"
    )
    print(
        f"ORDERING SEED MANIFEST: {seed_manifest_sha}"
    )
    print(
        f"SLICE BOUNDARY SHA256: {slice_boundary_sha}"
    )
    print(
        f"COMBINED PROTOCOL MANIFEST: {combined_manifest_sha256}"
    )
    print(
        "PHYSICAL CLIENT ASSIGNMENTS MATERIALIZED: NO"
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
