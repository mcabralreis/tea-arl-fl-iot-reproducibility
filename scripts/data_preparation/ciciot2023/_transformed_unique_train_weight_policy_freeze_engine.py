import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path


# ============================================================================
# Immutable scientific bindings
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

SPLIT_ID = (
    "CICIoT2023_TASK7_TRANSFORMED_UNIQUE_F32_SPLIT_80_10_10_V1"
)

GATE72_DIAGNOSTIC_ID = (
    "CICIoT2023_TRANSFORMED_UNIQUE_TRAIN_IMBALANCE_DIAGNOSTIC_V1"
)

GATE72_DISPOSITION = (
    "POLICY_PRINCIPLE_REVALIDATED_EXACT_WEIGHT_VECTOR_MUST_BE_RECOMPUTED"
)


# ============================================================================
# Freeze identifiers
# ============================================================================

FREEZE_ID = (
    "CICIoT2023_TRANSFORMED_UNIQUE_TRAIN_WEIGHT_POLICY_FREEZE_V1"
)

PRIMARY_POLICY_ID = (
    "KEEP_ALL_TRANSFORMED_UNIQUE_TRAIN_PLUS_"
    "SQRT_INVERSE_FREQUENCY_WEIGHTED_CE_V1"
)

WEIGHT_VECTOR_ID = (
    "CICIoT2023_TASK7_TRANSFORMED_UNIQUE_GLOBAL_TRAIN_"
    "SQRT_INVFREQ_WEIGHTS_V1"
)

SECONDARY_POLICY_ID = (
    "NATURAL_UNWEIGHTED_CE"
)

CANONICAL_VECTOR_SERIALISATION_ID = (
    "TASK7_LABEL_ASCENDING_COUNT_DECIMAL17_FLOATHEX_V1"
)


# ============================================================================
# Exact new TRAIN geometry
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
CLASS_COUNT = 7

EXPECTED_IMBALANCE_RATIO = (
    1223.960808738980
)


# ============================================================================
# Expected exact binary64 values
#
# The formula and counts are the primary definition.
# These constants are regression guards only.
# ============================================================================

EXPECTED_WEIGHTS = {
    0: 2.361947655626154,
    1: 21.61842421005929,
    2: 0.6179314203594863,
    3: 1.5759929623559756,
    4: 2.989431643370104,
    5: 3.6669224591635454,
    6: 15.7394759911818,
}


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


def require_close(
    observed,
    expected,
    message,
    tolerance=1e-12,
):
    if abs(
        float(observed)
        -
        float(expected)
    ) > tolerance:
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


def compute_weights(counts):
    total = sum(
        counts.values()
    )

    labels = sorted(
        counts
    )

    k = len(
        labels
    )

    raw = {
        label_id: math.sqrt(
            total
            /
            (
                k
                *
                counts[
                    label_id
                ]
            )
        )
        for label_id
        in labels
    }

    normalisation_factor = (
        sum(
            counts[
                label_id
            ]
            *
            raw[
                label_id
            ]
            for label_id
            in labels
        )
        /
        total
    )

    weights = {
        label_id: (
            raw[
                label_id
            ]
            /
            normalisation_factor
        )
        for label_id
        in labels
    }

    weighted_mean = (
        sum(
            counts[
                label_id
            ]
            *
            weights[
                label_id
            ]
            for label_id
            in labels
        )
        /
        total
    )

    return {
        "total": (
            total
        ),
        "class_count": (
            k
        ),
        "raw_weights": (
            raw
        ),
        "normalisation_factor": (
            normalisation_factor
        ),
        "weights": (
            weights
        ),
        "sample_weighted_mean": (
            weighted_mean
        ),
    }


def canonical_vector_text(
    counts,
    weights,
):
    lines = []

    for label_id in sorted(
        counts
    ):
        weight = weights[
            label_id
        ]

        lines.append(
            f"{label_id}\t"
            f"{counts[label_id]}\t"
            f"{format(weight, '.17g')}\t"
            f"{weight.hex()}\n"
        )

    return "".join(
        lines
    )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gate72-diagnostic-json",
        required=True,
    )

    parser.add_argument(
        "--gate71-state-json",
        required=True,
    )

    parser.add_argument(
        "--gate71-counts-csv",
        required=True,
    )

    parser.add_argument(
        "--gate70-freeze-json",
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
        "GATE72_IMBALANCE_DIAGNOSTIC": Path(
            args.gate72_diagnostic_json
        ),
        "GATE71_SPLIT_MATERIALISATION": Path(
            args.gate71_state_json
        ),
        "GATE71_SPLIT_COUNTS": Path(
            args.gate71_counts_csv
        ),
        "GATE70_SPLIT_PROTOCOL_FREEZE": Path(
            args.gate70_freeze_json
        ),
    }

    # ------------------------------------------------------------------
    # Gate A - verify immutable new TRAIN binding.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE A - VERIFY NEW TRAIN AND DIAGNOSTIC BINDINGS")
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
        gate71.get(
            "split_id"
        ),
        SPLIT_ID,
        "Gate-71 split ID",
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
            "audit"
        ][
            "split_assignment_replay_mismatches"
        ],
        0,
        "Gate-71 replay mismatches",
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

    gate70 = load_json(
        evidence_paths[
            "GATE70_SPLIT_PROTOCOL_FREEZE"
        ]
    )

    require_equal(
        gate70.get(
            "status"
        ),
        "FROZEN",
        "Gate-70 status",
    )

    require_equal(
        gate70[
            "frozen_split_protocol_artifact_manifest_sha256"
        ],
        SPLIT_PROTOCOL_SHA,
        "Gate-70 split protocol SHA256",
    )

    gate72 = load_json(
        evidence_paths[
            "GATE72_IMBALANCE_DIAGNOSTIC"
        ]
    )

    require_equal(
        gate72.get(
            "status"
        ),
        "DIAGNOSIS COMPLETE",
        "Gate-72 status",
    )

    require_equal(
        gate72.get(
            "diagnostic_id"
        ),
        GATE72_DIAGNOSTIC_ID,
        "Gate-72 diagnostic ID",
    )

    require_equal(
        gate72.get(
            "disposition"
        ),
        GATE72_DISPOSITION,
        "Gate-72 disposition",
    )

    require_equal(
        gate72[
            "immutable_binding"
        ][
            "effective_dataset_fingerprint_sha256"
        ],
        EFFECTIVE_DATASET_SHA,
        "Gate-72 effective dataset SHA256",
    )

    require_equal(
        gate72[
            "immutable_binding"
        ][
            "split_protocol_artifact_manifest_sha256"
        ],
        SPLIT_PROTOCOL_SHA,
        "Gate-72 split protocol SHA256",
    )

    require_equal(
        gate72[
            "immutable_binding"
        ][
            "split_assignment_manifest_sha256"
        ],
        SPLIT_ASSIGNMENT_SHA,
        "Gate-72 split assignment SHA256",
    )

    require_true(
        gate72[
            "policy_revalidation"
        ][
            "principle_revalidated"
        ]
        is True,
        "Gate-72 did not revalidate the policy principle.",
    )

    require_true(
        gate72[
            "policy_revalidation"
        ][
            "legacy_exact_vector_reusable_unchanged"
        ]
        is False,
        "Gate-72 unexpectedly allows legacy vector reuse.",
    )

    require_true(
        gate72[
            "scientific_boundary"
        ][
            "scientific_training_started"
        ]
        is False,
        "Gate-72 says scientific training started.",
    )

    print(
        "Effective dataset: BOUND"
    )
    print(
        "Split protocol: BOUND"
    )
    print(
        "Split assignment: BOUND"
    )
    print(
        "Gate-72 policy diagnosis: BOUND"
    )
    print(
        "Scientific training started: NO"
    )

    # ------------------------------------------------------------------
    # Gate B - verify exact new TRAIN counts.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE B - VERIFY EXACT NEW TRAIN COUNTS")
    print("=" * 60)
    print("")

    split_count_rows = read_csv(
        evidence_paths[
            "GATE71_SPLIT_COUNTS"
        ]
    )

    require_equal(
        len(
            split_count_rows
        ),
        21,
        "Gate-71 split-count row count",
    )

    observed_train_counts = {}

    for row in split_count_rows:
        if row[
            "Split"
        ] != "TRAIN":
            continue

        label_id = int(
            row[
                "Task7LabelID"
            ]
        )

        observed_train_counts[
            label_id
        ] = int(
            row[
                "Count"
            ]
        )

    require_equal(
        observed_train_counts,
        TRAIN_COUNTS,
        "Exact new TRAIN counts",
    )

    require_equal(
        sum(
            observed_train_counts.values()
        ),
        TRAIN_TOTAL,
        "Exact new TRAIN total",
    )

    require_equal(
        {
            int(
                label_id
            ): int(
                count
            )
            for label_id, count
            in gate72[
                "new_train_geometry"
            ][
                "per_label_counts"
            ].items()
        },
        TRAIN_COUNTS,
        "Gate-72 TRAIN counts replay",
    )

    imbalance_ratio = (
        max(
            TRAIN_COUNTS.values()
        )
        /
        min(
            TRAIN_COUNTS.values()
        )
    )

    require_close(
        imbalance_ratio,
        EXPECTED_IMBALANCE_RATIO,
        "TRAIN imbalance ratio",
        tolerance=1e-12,
    )

    print(
        "TRAIN per-label counts: EXACT MATCH"
    )
    print(
        f"TRAIN total: {TRAIN_TOTAL}"
    )
    print(
        f"Largest / smallest ratio: {imbalance_ratio:.12f}:1"
    )

    # ------------------------------------------------------------------
    # Gate C - recompute and verify exact global weight vector.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE C - RECOMPUTE EXACT GLOBAL TASK7 WEIGHT VECTOR")
    print("=" * 60)
    print("")

    result = compute_weights(
        TRAIN_COUNTS
    )

    require_equal(
        result[
            "total"
        ],
        TRAIN_TOTAL,
        "Weight derivation TRAIN total",
    )

    require_equal(
        result[
            "class_count"
        ],
        CLASS_COUNT,
        "Weight derivation class count",
    )

    for label_id in range(
        CLASS_COUNT
    ):
        require_close(
            result[
                "weights"
            ][
                label_id
            ],
            EXPECTED_WEIGHTS[
                label_id
            ],
            f"Exact weight regression label {label_id}",
            tolerance=1e-15,
        )

        require_close(
            result[
                "weights"
            ][
                label_id
            ],
            gate72[
                "policy_revalidation"
            ][
                "candidate_new_weights"
            ][
                str(
                    label_id
                )
            ],
            f"Gate-72 candidate weight replay label {label_id}",
            tolerance=1e-12,
        )

    require_close(
        result[
            "sample_weighted_mean"
        ],
        1.0,
        "Final sample-weighted mean",
        tolerance=1e-15,
    )

    canonical_text = canonical_vector_text(
        TRAIN_COUNTS,
        result[
            "weights"
        ],
    )

    weight_vector_sha256 = hashlib.sha256(
        canonical_text.encode(
            "utf-8"
        )
    ).hexdigest().upper()

    print(
        "Weight basis: exact new global TRAIN counts"
    )
    print(
        "Formula replay: PASS"
    )
    print(
        f"Normalisation factor: "
        f"{result['normalisation_factor']:.17g}"
    )
    print(
        f"Sample-weighted mean: "
        f"{result['sample_weighted_mean']:.17g}"
    )
    print(
        f"Canonical weight-vector SHA256: "
        f"{weight_vector_sha256}"
    )

    # ------------------------------------------------------------------
    # Gate D - freeze policy and prohibitions.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE D - FREEZE TRAIN IMBALANCE / LOSS POLICY")
    print("=" * 60)
    print("")

    policy = {
        "status": (
            "FROZEN"
        ),
        "freeze_id": (
            FREEZE_ID
        ),
        "primary_policy_id": (
            PRIMARY_POLICY_ID
        ),
        "weight_vector_id": (
            WEIGHT_VECTOR_ID
        ),
        "secondary_policy_id": (
            SECONDARY_POLICY_ID
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
            "train_total": (
                TRAIN_TOTAL
            ),
        },
        "frozen_train_handling": {
            "keep_all_transformed_unique_train_observations": (
                True
            ),
            "resampling": (
                False
            ),
            "undersampling": (
                False
            ),
            "oversampling": (
                False
            ),
            "synthetic_sample_generation": (
                False
            ),
        },
        "frozen_primary_loss": {
            "loss": (
                "CROSS_ENTROPY"
            ),
            "reduction": (
                "MEAN"
            ),
            "class_weighting": (
                "GLOBAL_TASK7_CLASS_VECTOR"
            ),
            "same_global_weight_vector_for_all_clients": (
                True
            ),
            "client_specific_class_weights": (
                False
            ),
            "weight_basis": (
                "EXACT_GLOBAL_TRANSFORMED_UNIQUE_TRAIN_COUNTS_ONLY"
            ),
            "raw_weight_formula": (
                "sqrt(N_train / (K * n_c))"
            ),
            "normalisation": (
                "divide by sample-weighted mean of raw weights"
            ),
            "required_property": (
                "sum(n_c * w_c) / N_train = 1"
            ),
            "canonical_computation_precision": (
                "IEEE754_BINARY64"
            ),
            "canonical_vector_serialisation_id": (
                CANONICAL_VECTOR_SERIALISATION_ID
            ),
            "canonical_weight_vector_sha256": (
                weight_vector_sha256
            ),
        },
        "frozen_prohibited_weight_inputs": [
            "VAL_COUNTS",
            "TEST_COUNTS",
            "CLIENT_SPECIFIC_COUNTS",
            "CLIENT_PARTITION_RESULTS",
            "RawMultiplicity",
            "TransformedGroupRawExactMultiplicity",
            "SumSourceRawMultiplicity",
            "Provenance",
            "ModelResults",
            "AttackResults",
        ],
        "frozen_secondary_policy": {
            "policy_id": (
                SECONDARY_POLICY_ID
            ),
            "loss": (
                "CROSS_ENTROPY"
            ),
            "class_weights": (
                None
            ),
            "role": (
                "PREDEFINED_SECONDARY_SENSITIVITY_REFERENCE"
            ),
        },
        "legacy_gate50_disposition": {
            "mathematical_principle": (
                "REVALIDATED"
            ),
            "exact_legacy_vector": (
                "SUPERSEDED_NOT_REUSABLE_UNCHANGED"
            ),
        },
        "scientific_boundary": {
            "new_weight_policy_frozen": (
                True
            ),
            "client_assignments_rebuilt": (
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

    print(
        f"Primary policy ID: {PRIMARY_POLICY_ID}"
    )
    print(
        "Keep all TRAIN observations: YES"
    )
    print(
        "Resampling: NO"
    )
    print(
        "Same global vector for all future clients: YES"
    )
    print(
        "Client-specific class weights: NO"
    )
    print(
        "Legacy exact vector reusable unchanged: NO"
    )

    # ------------------------------------------------------------------
    # Gate E - write frozen weight artifacts.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE E - WRITE FROZEN WEIGHT ARTIFACTS")
    print("=" * 60)
    print("")

    weight_rows = []

    for label_id in range(
        CLASS_COUNT
    ):
        weight = result[
            "weights"
        ][
            label_id
        ]

        weight_rows.append({
            "Task7LabelID": (
                label_id
            ),
            "Task7Label": (
                LABEL_NAMES[
                    label_id
                ]
            ),
            "TrainCount": (
                TRAIN_COUNTS[
                    label_id
                ]
            ),
            "RawWeightDecimal17": (
                format(
                    result[
                        "raw_weights"
                    ][
                        label_id
                    ],
                    ".17g",
                )
            ),
            "FinalWeightDecimal17": (
                format(
                    weight,
                    ".17g",
                )
            ),
            "FinalWeightFloat64Hex": (
                weight.hex()
            ),
        })

    policy_path = (
        output_root
        /
        "FROZEN_TRANSFORMED_UNIQUE_TRAIN_WEIGHT_POLICY.json"
    )

    weights_path = (
        output_root
        /
        "FROZEN_TASK7_GLOBAL_CLASS_WEIGHTS.csv"
    )

    derivation_path = (
        output_root
        /
        "FROZEN_WEIGHT_DERIVATION.json"
    )

    canonical_vector_path = (
        output_root
        /
        "FROZEN_WEIGHT_VECTOR_CANONICAL.txt"
    )

    write_json(
        policy_path,
        policy,
    )

    write_csv(
        weights_path,
        weight_rows,
        [
            "Task7LabelID",
            "Task7Label",
            "TrainCount",
            "RawWeightDecimal17",
            "FinalWeightDecimal17",
            "FinalWeightFloat64Hex",
        ],
    )

    write_json(
        derivation_path,
        {
            "weight_vector_id": (
                WEIGHT_VECTOR_ID
            ),
            "train_total": (
                TRAIN_TOTAL
            ),
            "class_count": (
                CLASS_COUNT
            ),
            "train_counts": {
                str(
                    label_id
                ): TRAIN_COUNTS[
                    label_id
                ]
                for label_id
                in range(
                    CLASS_COUNT
                )
            },
            "raw_weight_formula": (
                "sqrt(N_train / (K * n_c))"
            ),
            "normalisation_factor_decimal17": (
                format(
                    result[
                        "normalisation_factor"
                    ],
                    ".17g",
                )
            ),
            "sample_weighted_mean_decimal17": (
                format(
                    result[
                        "sample_weighted_mean"
                    ],
                    ".17g",
                )
            ),
            "canonical_vector_serialisation_id": (
                CANONICAL_VECTOR_SERIALISATION_ID
            ),
            "canonical_weight_vector_sha256": (
                weight_vector_sha256
            ),
        },
    )

    # IMPORTANT:
    # Write canonical bytes directly.  On Windows, Path.write_text() with the
    # default newline=None translates LF to CRLF, which changes the on-disk
    # SHA256 relative to the in-memory canonical UTF-8 LF byte sequence.
    canonical_vector_path.write_bytes(
        canonical_text.encode(
            "utf-8"
        )
    )

    require_equal(
        sha256_file(
            canonical_vector_path
        ),
        weight_vector_sha256,
        "Canonical weight-vector file SHA256",
    )

    artifact_rows = []

    for role, path in [
        (
            "TRAIN_WEIGHT_POLICY",
            policy_path,
        ),
        (
            "TASK7_GLOBAL_CLASS_WEIGHTS",
            weights_path,
        ),
        (
            "WEIGHT_DERIVATION",
            derivation_path,
        ),
        (
            "CANONICAL_WEIGHT_VECTOR",
            canonical_vector_path,
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
        "FROZEN_TRAIN_WEIGHT_POLICY_ARTIFACT_MANIFEST.csv"
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
        "primary_policy_id": (
            PRIMARY_POLICY_ID
        ),
        "weight_vector_id": (
            WEIGHT_VECTOR_ID
        ),
        "canonical_weight_vector_sha256": (
            weight_vector_sha256
        ),
        "frozen_train_weight_policy_artifact_manifest_sha256": (
            combined_manifest_sha256
        ),
        "materialization_boundary": {
            "client_assignments_rebuilt": (
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
        "TRANSFORMED_UNIQUE_TRAIN_WEIGHT_POLICY_FREEZE.json"
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
        "CICIoT2023 TRANSFORMED-UNIQUE TRAIN IMBALANCE / LOSS POLICY FREEZE"
    )
    add("=" * 78)
    add("")

    add("STATUS")
    add("-" * 78)
    add(
        "FROZEN"
    )
    add("")

    add("POLICY IDENTIFIERS")
    add("-" * 78)
    add(
        f"Freeze ID: {FREEZE_ID}"
    )
    add(
        f"Primary policy ID: {PRIMARY_POLICY_ID}"
    )
    add(
        f"Weight vector ID: {WEIGHT_VECTOR_ID}"
    )
    add(
        f"Secondary policy ID: {SECONDARY_POLICY_ID}"
    )
    add("")

    add("IMMUTABLE BINDING")
    add("-" * 78)
    add(
        f"Effective dataset fingerprint SHA256: {EFFECTIVE_DATASET_SHA}"
    )
    add(
        f"Split protocol artifact-manifest SHA256: {SPLIT_PROTOCOL_SHA}"
    )
    add(
        f"Split assignment artifact-manifest SHA256: {SPLIT_ASSIGNMENT_SHA}"
    )
    add("")

    add("FROZEN TRAIN GEOMETRY")
    add("-" * 78)
    add(
        f"TRAIN observations: {TRAIN_TOTAL}"
    )
    add(
        f"Largest / smallest class ratio: {imbalance_ratio:.12f}:1"
    )
    add("")

    for label_id in range(
        CLASS_COUNT
    ):
        add(
            f"{label_id} | {LABEL_NAMES[label_id]}: "
            f"{TRAIN_COUNTS[label_id]}"
        )

    add("")

    add("FROZEN PRIMARY POLICY")
    add("-" * 78)
    add(
        "Keep all transformed-unique TRAIN observations: YES"
    )
    add(
        "Resampling: NO"
    )
    add(
        "Primary loss: weighted cross-entropy"
    )
    add(
        "Loss reduction: mean"
    )
    add(
        "Weight basis: exact global transformed-unique TRAIN counts only"
    )
    add(
        "Raw formula: sqrt(N_train / (K * n_c))"
    )
    add(
        "Normalisation: divide by sample-weighted mean"
    )
    add(
        "Required property: sum(n_c * w_c) / N_train = 1"
    )
    add(
        "Same global weight vector for every future client: YES"
    )
    add(
        "Client-specific class weights: NO"
    )
    add("")

    add("FROZEN EXACT TASK7 WEIGHT VECTOR")
    add("-" * 78)

    for label_id in range(
        CLASS_COUNT
    ):
        weight = result[
            "weights"
        ][
            label_id
        ]

        add(
            f"{label_id} | {LABEL_NAMES[label_id]}: "
            f"{format(weight, '.17g')}"
        )
        add(
            f"  float64 hex: {weight.hex()}"
        )

    add("")
    add(
        f"Normalisation factor: "
        f"{format(result['normalisation_factor'], '.17g')}"
    )
    add(
        f"Sample-weighted mean: "
        f"{format(result['sample_weighted_mean'], '.17g')}"
    )
    add(
        f"Canonical weight-vector SHA256: {weight_vector_sha256}"
    )
    add("")

    add("PROHIBITED WEIGHT INPUTS")
    add("-" * 78)
    add(
        "VAL counts"
    )
    add(
        "TEST counts"
    )
    add(
        "Client-specific counts"
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
        "Model outcomes"
    )
    add(
        "Attack outcomes"
    )
    add("")

    add("LEGACY GATE-50 DISPOSITION")
    add("-" * 78)
    add(
        "Mathematical principle: REVALIDATED"
    )
    add(
        "Exact legacy weight vector: SUPERSEDED"
    )
    add(
        "Exact legacy vector may be reused unchanged: NO"
    )
    add("")

    add("SECONDARY POLICY")
    add("-" * 78)
    add(
        SECONDARY_POLICY_ID
    )
    add(
        "Role: predefined secondary sensitivity/reference"
    )
    add("")

    add("IMMUTABLE POLICY ARTIFACT")
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
        f"Combined TRAIN-weight-policy artifact-manifest SHA256: "
        f"{combined_manifest_sha256}"
    )
    add("")

    add("SCIENTIFIC STATE")
    add("-" * 78)
    add(
        "New TRAIN weight policy frozen: YES"
    )
    add(
        "Client assignments rebuilt: NO"
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
        "Diagnose and revalidate the 30-client transformed-unique TRAIN "
        "partition protocol. Reuse prior design principles only after binding "
        "them to the new TRAIN counts, split assignment fingerprint, and frozen "
        "global class-weight policy. Do not materialise new clients until the "
        "partition protocol is revalidated and frozen."
    )

    (
        output_root
        /
        "TRANSFORMED_UNIQUE_TRAIN_WEIGHT_POLICY_FREEZE_REPORT.txt"
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
        f"PRIMARY POLICY: {PRIMARY_POLICY_ID}"
    )
    print(
        f"WEIGHT VECTOR SHA256: {weight_vector_sha256}"
    )
    print(
        f"COMBINED POLICY MANIFEST: {combined_manifest_sha256}"
    )
    print(
        "CLIENT ASSIGNMENTS REBUILT: NO"
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
