import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


# ============================================================================
# Immutable scientific bindings
# ============================================================================

RAW_EXACT_UNIQUE_DATASET_SHA = (
    "D86976FD34A72E4E60249C536505165490F878D71A21301AC6F6FB7D387D6C8D"
)

SOURCE_FEATURE_LAYER_SHA = (
    "3BCCF823E11D0088970EC9BD26D411C73213D4E22C4469EBD0BF1D92F255944A"
)

TRANSFORMED_COLLISION_POLICY_SHA = (
    "04B6F4D4DCDDA139BF2814CD9A3FE15146847738F5F75D1C1E6E851A372B5515"
)

EFFECTIVE_DATASET_SHA = (
    "5708EFE6C08C91CF3637FA8F89F53C4459933F94C7CC0BF819A590CBE9EF8E5D"
)

EFFECTIVE_DATASET_ID = (
    "CICIoT2023_TASK7_TRANSFORMED_UNIQUE_F32_INDEX_V1"
)

EFFECTIVE_AUDIT_ID = (
    "CICIoT2023_TASK7_TRANSFORMED_UNIQUE_F32_INDEX_AUDIT_V1"
)


# ============================================================================
# Split protocol identifiers
# ============================================================================

FREEZE_ID = (
    "CICIoT2023_TRANSFORMED_UNIQUE_SPLIT_PROTOCOL_FREEZE_V1"
)

SPLIT_PROTOCOL_ID = (
    "CICIoT2023_TASK7_TRANSFORMED_UNIQUE_F32_SPLIT_80_10_10_V1"
)

ALLOCATION_ALGORITHM_ID = (
    "SPLITMIX64_DUAL_KEY_EFFECTIVE_TVT_V1"
)

SPLIT_UNIT_ID = (
    "ONE_BITWISE_EXACT_39_FLOAT32_MODEL_INPUT_VECTOR"
)

SEED_MATERIAL = (
    "CICIoT2023_TASK7_TRANSFORMED_UNIQUE_F32_INDEX_V1"
    "|"
    "5708EFE6C08C91CF3637FA8F89F53C4459933F94C7CC0BF819A590CBE9EF8E5D"
    "|"
    "TVT_SPLIT_V1"
)

EXPECTED_SEED_SHA256 = (
    "57F86F49720C8236885F1BF612C1B0AB4B937DD7D7BF73EE1FC087917212831B"
)

EXPECTED_SEED_UINT64 = 6_338_938_836_760_691_254


# ============================================================================
# Effective dataset counts
# ============================================================================

EFFECTIVE_OBSERVATIONS = 20_687_272

LABEL_NAMES = {
    0: "Benign",
    1: "Brute Force",
    2: "DoS_DDoS",
    3: "Mirai",
    4: "Recon",
    5: "Spoofing",
    6: "Web",
}

EXPECTED_LABEL_COUNTS = {
    0: 1_092_826,
    1: 13_044,
    2: 15_966_567,
    3: 2_454_615,
    4: 682_204,
    5: 453_406,
    6: 24_610,
}

EXPECTED_TARGETS = {
    0: {
        "TRAIN": 874_262,
        "VALIDATION": 109_282,
        "TEST": 109_282,
    },
    1: {
        "TRAIN": 10_436,
        "VALIDATION": 1_304,
        "TEST": 1_304,
    },
    2: {
        "TRAIN": 12_773_255,
        "VALIDATION": 1_596_656,
        "TEST": 1_596_656,
    },
    3: {
        "TRAIN": 1_963_693,
        "VALIDATION": 245_461,
        "TEST": 245_461,
    },
    4: {
        "TRAIN": 545_764,
        "VALIDATION": 68_220,
        "TEST": 68_220,
    },
    5: {
        "TRAIN": 362_726,
        "VALIDATION": 45_340,
        "TEST": 45_340,
    },
    6: {
        "TRAIN": 19_688,
        "VALIDATION": 2_461,
        "TEST": 2_461,
    },
}

EXPECTED_GLOBAL_TARGETS = {
    "TRAIN": 16_549_824,
    "VALIDATION": 2_068_724,
    "TEST": 2_068_724,
}


# ============================================================================
# Frozen algorithm constants
# ============================================================================

UINT64_MASK = (
    0xFFFFFFFFFFFFFFFF
)

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
        key=lambda item: item[
            "ArtifactRole"
        ],
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


def rotl64(value, shift):
    shift = (
        shift
        %
        64
    )

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


def derive_seed():
    seed_sha = hashlib.sha256(
        SEED_MATERIAL.encode(
            "utf-8"
        )
    ).hexdigest().upper()

    seed_uint64 = int.from_bytes(
        bytes.fromhex(
            seed_sha[
                :16
            ]
        ),
        byteorder="big",
        signed=False,
    )

    return (
        seed_sha,
        seed_uint64,
    )


def derive_rank_keys(
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


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gate68-json",
        required=True,
    )

    parser.add_argument(
        "--gate69-audit-json",
        required=True,
    )

    parser.add_argument(
        "--gate67-freeze-json",
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
        "GATE68_EFFECTIVE_MATERIALISATION": Path(
            args.gate68_json
        ),
        "GATE69_EFFECTIVE_AUDIT": Path(
            args.gate69_audit_json
        ),
        "GATE67_COLLISION_POLICY_FREEZE": Path(
            args.gate67_freeze_json
        ),
    }

    # ------------------------------------------------------------------
    # Gate A - verify effective dataset and audit state.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE A - VERIFY AUDITED EFFECTIVE DATASET")
    print("=" * 60)
    print("")

    gate68 = load_json(
        evidence_paths[
            "GATE68_EFFECTIVE_MATERIALISATION"
        ]
    )

    require_equal(
        gate68.get(
            "status"
        ),
        "PASS",
        "Gate-68 status",
    )

    require_equal(
        gate68.get(
            "dataset_id"
        ),
        EFFECTIVE_DATASET_ID,
        "Gate-68 dataset ID",
    )

    require_equal(
        gate68[
            "effective_dataset_fingerprint_sha256"
        ],
        EFFECTIVE_DATASET_SHA,
        "Gate-68 effective dataset fingerprint",
    )

    require_equal(
        gate68[
            "counts"
        ][
            "effective_observations"
        ],
        EFFECTIVE_OBSERVATIONS,
        "Gate-68 effective observations",
    )

    require_true(
        gate68[
            "scientific_boundary"
        ][
            "new_split_materialized"
        ]
        is False,
        "Gate-68 says a new split is already materialized.",
    )

    require_true(
        gate68[
            "scientific_boundary"
        ][
            "scientific_training_started"
        ]
        is False,
        "Gate-68 says scientific training has started.",
    )

    gate69 = load_json(
        evidence_paths[
            "GATE69_EFFECTIVE_AUDIT"
        ]
    )

    require_equal(
        gate69.get(
            "status"
        ),
        "PASS",
        "Gate-69 status",
    )

    require_equal(
        gate69.get(
            "audit_id"
        ),
        EFFECTIVE_AUDIT_ID,
        "Gate-69 audit ID",
    )

    require_equal(
        gate69[
            "immutable_binding"
        ][
            "effective_dataset_fingerprint_sha256"
        ],
        EFFECTIVE_DATASET_SHA,
        "Gate-69 effective dataset fingerprint",
    )

    require_equal(
        gate69[
            "counts"
        ][
            "effective_observations"
        ],
        EFFECTIVE_OBSERVATIONS,
        "Gate-69 effective observations",
    )

    require_equal(
        gate69[
            "artifact_fingerprint_replay"
        ],
        "EXACT_MATCH",
        "Gate-69 artifact fingerprint replay",
    )

    require_equal(
        gate69[
            "counts"
        ][
            "missing_source_rows"
        ],
        0,
        "Gate-69 missing source rows",
    )

    require_true(
        gate69[
            "scientific_boundary"
        ][
            "new_split_frozen"
        ]
        is False,
        "Gate-69 says a new split is already frozen.",
    )

    require_true(
        gate69[
            "scientific_boundary"
        ][
            "new_split_materialized"
        ]
        is False,
        "Gate-69 says a new split is already materialized.",
    )

    require_true(
        gate69[
            "scientific_boundary"
        ][
            "scientific_training_started"
        ]
        is False,
        "Gate-69 says scientific training has started.",
    )

    gate67 = load_json(
        evidence_paths[
            "GATE67_COLLISION_POLICY_FREEZE"
        ]
    )

    require_equal(
        gate67.get(
            "status"
        ),
        "FROZEN",
        "Gate-67 status",
    )

    require_equal(
        gate67[
            "frozen_transformed_collision_policy_artifact_manifest_sha256"
        ],
        TRANSFORMED_COLLISION_POLICY_SHA,
        "Gate-67 policy SHA256",
    )

    print(
        "Gate-68 effective materialisation: PASS"
    )
    print(
        "Gate-69 independent audit: PASS"
    )
    print(
        "Effective dataset fingerprint: BOUND"
    )
    print(
        "Scientific training started: NO"
    )

    # ------------------------------------------------------------------
    # Gate B - verify exact per-label counts and targets.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE B - VERIFY EXACT PER-LABEL SPLIT TARGETS")
    print("=" * 60)
    print("")

    observed_label_counts = {
        int(
            label_id
        ): int(
            count
        )
        for label_id, count
        in gate69[
            "per_label_effective_observations"
        ].items()
    }

    require_equal(
        observed_label_counts,
        EXPECTED_LABEL_COUNTS,
        "Gate-69 per-label effective counts",
    )

    require_equal(
        sum(
            observed_label_counts.values()
        ),
        EFFECTIVE_OBSERVATIONS,
        "Effective per-label count total",
    )

    target_rows = []

    computed_global_targets = {
        "TRAIN": 0,
        "VALIDATION": 0,
        "TEST": 0,
    }

    for label_id in range(
        7
    ):
        n = observed_label_counts[
            label_id
        ]

        validation = (
            n
            //
            10
        )

        test = (
            n
            //
            10
        )

        train = (
            n
            -
            validation
            -
            test
        )

        computed = {
            "TRAIN": (
                train
            ),
            "VALIDATION": (
                validation
            ),
            "TEST": (
                test
            ),
        }

        require_equal(
            computed,
            EXPECTED_TARGETS[
                label_id
            ],
            f"Per-label targets for {LABEL_NAMES[label_id]}",
        )

        for split_name, count in computed.items():
            computed_global_targets[
                split_name
            ] += count

        target_rows.append({
            "Task7LabelID": (
                label_id
            ),
            "Task7Label": (
                LABEL_NAMES[
                    label_id
                ]
            ),
            "EffectiveObservations": (
                n
            ),
            "TRAIN": (
                train
            ),
            "VALIDATION": (
                validation
            ),
            "TEST": (
                test
            ),
        })

    require_equal(
        computed_global_targets,
        EXPECTED_GLOBAL_TARGETS,
        "Global split targets",
    )

    print(
        "Per-label counts: EXACT"
    )
    print(
        "Per-label integer targets: EXACT"
    )
    print(
        f"TRAIN target: {computed_global_targets['TRAIN']}"
    )
    print(
        f"VALIDATION target: {computed_global_targets['VALIDATION']}"
    )
    print(
        f"TEST target: {computed_global_targets['TEST']}"
    )

    # ------------------------------------------------------------------
    # Gate C - derive and verify immutable split seed.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE C - DERIVE AND VERIFY SPLIT SEED")
    print("=" * 60)
    print("")

    seed_sha256, seed_uint64 = derive_seed()

    require_equal(
        seed_sha256,
        EXPECTED_SEED_SHA256,
        "Derived seed SHA256",
    )

    require_equal(
        seed_uint64,
        EXPECTED_SEED_UINT64,
        "Derived seed uint64",
    )

    print(
        f"Seed material: {SEED_MATERIAL}"
    )
    print(
        f"Seed SHA256: {seed_sha256}"
    )
    print(
        f"Seed uint64: {seed_uint64}"
    )

    # ------------------------------------------------------------------
    # Gate D - freeze exact deterministic allocation algorithm.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE D - FREEZE EXACT SPLIT ALLOCATION ALGORITHM")
    print("=" * 60)
    print("")

    algorithm = {
        "algorithm_id": (
            ALLOCATION_ALGORITHM_ID
        ),
        "split_unit_id": (
            SPLIT_UNIT_ID
        ),
        "scope": (
            "INDEPENDENTLY_WITHIN_EACH_TASK7_LABEL"
        ),
        "effective_observation_id": [
            "TransformedHash1",
            "TransformedHash2",
        ],
        "seed_material": (
            SEED_MATERIAL
        ),
        "seed_sha256": (
            seed_sha256
        ),
        "seed_uint64_derivation": (
            "UNSIGNED_BIG_ENDIAN_INTEGER_FROM_FIRST_8_BYTES_OF_SEED_SHA256"
        ),
        "seed_uint64": (
            seed_uint64
        ),
        "rank_key_1": (
            "splitmix64(TransformedHash1 XOR SeedU64)"
        ),
        "rank_key_2": (
            "splitmix64(TransformedHash2 XOR ROTL64(SeedU64,32) "
            "XOR 0xD1B54A32D192ED03)"
        ),
        "sort_tuple": [
            "RankKey1",
            "RankKey2",
            "TransformedHash1",
            "TransformedHash2",
        ],
        "sort_direction": (
            "ASCENDING_LEXICOGRAPHIC"
        ),
        "allocation_order_after_sort": [
            "TRAIN",
            "VALIDATION",
            "TEST",
        ],
        "allocation_boundaries": {
            "TRAIN": (
                "first TRAIN_target observations"
            ),
            "VALIDATION": (
                "next VALIDATION_target observations"
            ),
            "TEST": (
                "remaining TEST_target observations"
            ),
        },
        "integer_target_rule_per_label": {
            "VALIDATION": (
                "floor(N_label / 10)"
            ),
            "TEST": (
                "floor(N_label / 10)"
            ),
            "TRAIN": (
                "N_label - VALIDATION - TEST"
            ),
        },
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
        },
        "rank_inputs_prohibited": [
            "OldSplitID",
            "RepresentativeSourceBucket",
            "RepresentativeSourceRowIndex",
            "RawMultiplicity",
            "TransformedGroupRawExactMultiplicity",
            "SumSourceRawMultiplicity",
            "Provenance",
            "ModelResults",
        ],
        "task7_label_usage": (
            "STRATIFICATION_SCOPE_ONLY_NOT_A_RANK_KEY"
        ),
    }

    # Small deterministic algorithm self-check.
    test_vectors = [
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

    replay_1 = [
        derive_rank_keys(
            h1,
            h2,
            seed_uint64,
        )
        for h1, h2
        in test_vectors
    ]

    replay_2 = [
        derive_rank_keys(
            h1,
            h2,
            seed_uint64,
        )
        for h1, h2
        in test_vectors
    ]

    require_equal(
        replay_1,
        replay_2,
        "Deterministic rank-key replay",
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
        "Synthetic rank-key distinctness",
    )

    print(
        f"Algorithm ID: {ALLOCATION_ALGORITHM_ID}"
    )
    print(
        "Allocation scope: within each TASK7 label"
    )
    print(
        "Sort tuple:"
    )
    print(
        "  RankKey1"
    )
    print(
        "  RankKey2"
    )
    print(
        "  TransformedHash1"
    )
    print(
        "  TransformedHash2"
    )
    print(
        "Allocation order: TRAIN -> VALIDATION -> TEST"
    )
    print(
        "Deterministic synthetic replay: PASS"
    )

    # ------------------------------------------------------------------
    # Gate E - write frozen protocol artifacts.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE E - WRITE FROZEN SPLIT PROTOCOL ARTIFACTS")
    print("=" * 60)
    print("")

    protocol = {
        "status": (
            "FROZEN"
        ),
        "freeze_id": (
            FREEZE_ID
        ),
        "split_protocol_id": (
            SPLIT_PROTOCOL_ID
        ),
        "immutable_binding": {
            "raw_exact_unique_dataset_manifest_sha256": (
                RAW_EXACT_UNIQUE_DATASET_SHA
            ),
            "source_transformed_feature_layer_manifest_sha256": (
                SOURCE_FEATURE_LAYER_SHA
            ),
            "transformed_collision_policy_artifact_manifest_sha256": (
                TRANSFORMED_COLLISION_POLICY_SHA
            ),
            "effective_dataset_fingerprint_sha256": (
                EFFECTIVE_DATASET_SHA
            ),
            "effective_dataset_id": (
                EFFECTIVE_DATASET_ID
            ),
            "effective_audit_id": (
                EFFECTIVE_AUDIT_ID
            ),
        },
        "frozen_split_unit": {
            "split_unit_id": (
                SPLIT_UNIT_ID
            ),
            "definition": (
                "One effective observation = one bitwise-exact 39-float32 model-input vector"
            ),
        },
        "frozen_ratios": {
            "TRAIN": (
                0.8
            ),
            "VALIDATION": (
                0.1
            ),
            "TEST": (
                0.1
            ),
        },
        "effective_observations": (
            EFFECTIVE_OBSERVATIONS
        ),
        "global_targets": (
            computed_global_targets
        ),
        "seed": {
            "material": (
                SEED_MATERIAL
            ),
            "sha256": (
                seed_sha256
            ),
            "uint64": (
                seed_uint64
            ),
            "uint64_derivation": (
                "UNSIGNED_BIG_ENDIAN_INTEGER_FROM_FIRST_8_BYTES_OF_SEED_SHA256"
            ),
        },
        "allocation_algorithm": (
            algorithm
        ),
        "scientific_boundary": {
            "split_protocol_frozen": (
                True
            ),
            "split_materialized": (
                False
            ),
            "client_assignments_materialized": (
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

    protocol_path = (
        output_root
        /
        "FROZEN_TRANSFORMED_UNIQUE_SPLIT_PROTOCOL.json"
    )

    algorithm_path = (
        output_root
        /
        "FROZEN_SPLIT_ALLOCATION_ALGORITHM.json"
    )

    targets_path = (
        output_root
        /
        "FROZEN_SPLIT_TARGETS.csv"
    )

    seed_path = (
        output_root
        /
        "FROZEN_SPLIT_SEED.json"
    )

    write_json(
        protocol_path,
        protocol,
    )

    write_json(
        algorithm_path,
        algorithm,
    )

    write_csv(
        targets_path,
        target_rows,
        [
            "Task7LabelID",
            "Task7Label",
            "EffectiveObservations",
            "TRAIN",
            "VALIDATION",
            "TEST",
        ],
    )

    write_json(
        seed_path,
        {
            "seed_material": (
                SEED_MATERIAL
            ),
            "seed_sha256": (
                seed_sha256
            ),
            "seed_uint64": (
                seed_uint64
            ),
            "seed_uint64_derivation": (
                "UNSIGNED_BIG_ENDIAN_INTEGER_FROM_FIRST_8_BYTES_OF_SEED_SHA256"
            ),
        },
    )

    artifact_rows = []

    for role, path in [
        (
            "SPLIT_PROTOCOL",
            protocol_path,
        ),
        (
            "ALLOCATION_ALGORITHM",
            algorithm_path,
        ),
        (
            "SPLIT_TARGETS",
            targets_path,
        ),
        (
            "SPLIT_SEED",
            seed_path,
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

    manifest_path = (
        output_root
        /
        "FROZEN_SPLIT_PROTOCOL_ARTIFACT_MANIFEST.csv"
    )

    write_csv(
        manifest_path,
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
        "split_protocol_id": (
            SPLIT_PROTOCOL_ID
        ),
        "effective_dataset_fingerprint_sha256": (
            EFFECTIVE_DATASET_SHA
        ),
        "frozen_split_protocol_artifact_manifest_sha256": (
            combined_manifest_sha256
        ),
        "materialization_boundary": {
            "split_materialized": (
                False
            ),
            "client_assignments_materialized": (
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
        "TRANSFORMED_UNIQUE_SPLIT_PROTOCOL_FREEZE.json"
    )

    write_json(
        freeze_state_path,
        freeze_state,
    )

    # ------------------------------------------------------------------
    # Evidence manifest.
    # ------------------------------------------------------------------

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
        "CICIoT2023 TRANSFORMED-UNIQUE 80/10/10 SPLIT PROTOCOL FREEZE"
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
        f"Split protocol ID: {SPLIT_PROTOCOL_ID}"
    )
    add(
        f"Allocation algorithm ID: {ALLOCATION_ALGORITHM_ID}"
    )
    add("")

    add("IMMUTABLE BINDING")
    add("-" * 78)
    add(
        f"Raw exact-unique dataset SHA256: {RAW_EXACT_UNIQUE_DATASET_SHA}"
    )
    add(
        f"Source transformed feature-layer SHA256: {SOURCE_FEATURE_LAYER_SHA}"
    )
    add(
        f"Transformed-collision policy SHA256: {TRANSFORMED_COLLISION_POLICY_SHA}"
    )
    add(
        f"Effective dataset fingerprint SHA256: {EFFECTIVE_DATASET_SHA}"
    )
    add("")

    add("AUDITED SPLIT UNIT")
    add("-" * 78)
    add(
        "One effective observation = one bitwise-exact 39-float32 model-input vector"
    )
    add(
        f"Effective observations: {EFFECTIVE_OBSERVATIONS}"
    )
    add("")

    add("FROZEN RATIOS")
    add("-" * 78)
    add(
        "TRAIN: 80%"
    )
    add(
        "VALIDATION: 10%"
    )
    add(
        "TEST: 10%"
    )
    add("")

    add("FROZEN INTEGER TARGET RULE")
    add("-" * 78)
    add(
        "Within each TASK7 label:"
    )
    add(
        "  VALIDATION = floor(N_label / 10)"
    )
    add(
        "  TEST = floor(N_label / 10)"
    )
    add(
        "  TRAIN = N_label - VALIDATION - TEST"
    )
    add("")

    add("PER-LABEL TARGETS")
    add("-" * 78)

    for row in target_rows:
        add(
            f"{row['Task7LabelID']} | {row['Task7Label']}: "
            f"N={row['EffectiveObservations']} | "
            f"TRAIN={row['TRAIN']} | "
            f"VALIDATION={row['VALIDATION']} | "
            f"TEST={row['TEST']}"
        )

    add("")

    add("GLOBAL TARGETS")
    add("-" * 78)
    add(
        f"TRAIN: {computed_global_targets['TRAIN']}"
    )
    add(
        f"VALIDATION: {computed_global_targets['VALIDATION']}"
    )
    add(
        f"TEST: {computed_global_targets['TEST']}"
    )
    add("")

    add("FROZEN SEED")
    add("-" * 78)
    add(
        f"Seed material: {SEED_MATERIAL}"
    )
    add(
        f"Seed SHA256: {seed_sha256}"
    )
    add(
        f"Seed uint64: {seed_uint64}"
    )
    add(
        "Seed uint64 derivation: unsigned big-endian integer from first 8 SHA256 bytes"
    )
    add("")

    add("FROZEN ALLOCATION ALGORITHM")
    add("-" * 78)
    add(
        f"Algorithm ID: {ALLOCATION_ALGORITHM_ID}"
    )
    add(
        "Scope: independently within each TASK7 label"
    )
    add(
        "Effective observation ID:"
    )
    add(
        "  TransformedHash1"
    )
    add(
        "  TransformedHash2"
    )
    add("")
    add(
        "RankKey1 = splitmix64(TransformedHash1 XOR SeedU64)"
    )
    add(
        "RankKey2 = splitmix64(TransformedHash2 XOR ROTL64(SeedU64,32) "
        "XOR 0xD1B54A32D192ED03)"
    )
    add("")
    add(
        "Ascending lexicographic sort tuple:"
    )
    add(
        "  RankKey1"
    )
    add(
        "  RankKey2"
    )
    add(
        "  TransformedHash1"
    )
    add(
        "  TransformedHash2"
    )
    add("")
    add(
        "Allocation after sort:"
    )
    add(
        "  1. first TRAIN_target observations -> TRAIN"
    )
    add(
        "  2. next VALIDATION_target observations -> VALIDATION"
    )
    add(
        "  3. remaining TEST_target observations -> TEST"
    )
    add("")

    add("PROHIBITED RANK INPUTS")
    add("-" * 78)
    add(
        "Old split ID"
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
        "Model results"
    )
    add(
        "TASK7 label is used only to define the stratification scope, not as a rank key"
    )
    add("")

    add("IMMUTABLE SPLIT PROTOCOL ARTIFACT")
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
        f"Combined split-protocol artifact-manifest SHA256: "
        f"{combined_manifest_sha256}"
    )
    add("")

    add("SCIENTIFIC STATE")
    add("-" * 78)
    add(
        "Split protocol frozen: YES"
    )
    add(
        "Split materialized: NO"
    )
    add(
        "Client assignments materialized: NO"
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
        "Materialize the exact split IDs for all 20687272 effective observations "
        "using only the frozen transformed hashes, TASK7 stratification scope, "
        "seed, dual-key SplitMix64 ranking, sort tuple, and per-label targets. "
        "Then audit exact counts, coverage, deterministic replay, and zero "
        "dependency on superseded old split IDs before any client reconstruction."
    )

    (
        output_root
        /
        "TRANSFORMED_UNIQUE_SPLIT_PROTOCOL_FREEZE_REPORT.txt"
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
        f"SPLIT PROTOCOL: {SPLIT_PROTOCOL_ID}"
    )
    print(
        f"SEED SHA256: {seed_sha256}"
    )
    print(
        f"SEED UINT64: {seed_uint64}"
    )
    print(
        f"TRAIN / VAL / TEST: "
        f"{computed_global_targets['TRAIN']} / "
        f"{computed_global_targets['VALIDATION']} / "
        f"{computed_global_targets['TEST']}"
    )
    print(
        f"COMBINED SPLIT-PROTOCOL MANIFEST: {combined_manifest_sha256}"
    )
    print("SPLIT MATERIALIZED: NO")
    print("SCIENTIFIC TRAINING STARTED: NO")
    print("=" * 60)
    print("")

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )
