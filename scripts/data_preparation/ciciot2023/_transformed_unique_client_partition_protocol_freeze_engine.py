import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


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

GATE74_DIAGNOSTIC_ID = (
    "CICIoT2023_TRANSFORMED_UNIQUE_K30_PARTITION_PROTOCOL_DIAGNOSTIC_V1"
)

GATE74_DISPOSITION = (
    "PRIOR_30_CLIENT_CAPACITY_BALANCED_DIRICHLET_IPF_PRINCIPLE_"
    "REVALIDATED_NEW_COUNT_PLANS_REQUIRED"
)

GATE74_CANDIDATE_PROTOCOL_ID = (
    "CICIoT2023_TASK7_TRANSFORMED_UNIQUE_K30_"
    "CAPACITY_BALANCED_DIRICHLET_ALPHA_0P1_1P0_5SEEDS_V2_CANDIDATE"
)

FREEZE_ID = (
    "CICIoT2023_TRANSFORMED_UNIQUE_K30_PARTITION_PROTOCOL_FREEZE_V2"
)

PROTOCOL_ID = (
    "CICIoT2023_TASK7_TRANSFORMED_UNIQUE_K30_"
    "CAPACITY_BALANCED_DIRICHLET_ALPHA_0P1_1P0_5SEEDS_V2"
)

COUNT_PLAN_SET_ID = (
    "CICIoT2023_TASK7_TRANSFORMED_UNIQUE_K30_EXACT_COUNT_PLAN_SET_V2"
)

LEGACY_DESIGN_PRINCIPLE_ID = (
    "CAPACITY_BALANCED_DIRICHLET_IPF_V1"
)

RNG_BINDING_ID = (
    "SHA256_BOUND_PCG64_DIRICHLET_V1"
)

CAPACITY_RULE_ID = (
    "FIRST_REMAINDER_CLIENT_IDS_GET_BASE_PLUS_ONE_V1"
)

CONTINUOUS_BALANCING_ID = (
    "CLASSWISE_DIRICHLET_PLUS_IPF_RAS_V1"
)

INTEGERISATION_ID = (
    "DETERMINISTIC_MIN_COST_BIPARTITE_RESIDUAL_ROUNDING_V1"
)

MATRIX_CONTENT_HASH_ID = (
    "INT64_C_ORDER_7X30_SHA256_V1"
)

COUNT_PLAN_SET_HASH_ID = (
    "SORTED_CONFIG_ID_PLUS_MATRIX_CONTENT_SHA256_LF_V1"
)

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

EXPECTED_CAPACITIES = (
    [551_661] * 24
    +
    [551_660] * 6
)

ALPHAS = (
    0.1,
    1.0,
)

EXPERIMENTAL_SEEDS = (
    42,
    123,
    456,
    789,
    999,
)

EXPECTED_CONFIGS = [
    (
        alpha,
        seed,
    )
    for alpha
    in ALPHAS
    for seed
    in EXPERIMENTAL_SEEDS
]

IPF_ABS_TOLERANCE = 1e-6
IPF_MAX_ITERATIONS = 100_000
FRACTION_SCORE_SCALE = 10**12


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
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
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


def alpha_token(alpha):
    if alpha == 0.1:
        return "0p1"
    if alpha == 1.0:
        return "1p0"
    raise RuntimeError(f"Unsupported alpha: {alpha}")


def config_id(alpha, seed):
    return f"alpha_{alpha_token(alpha)}_seed_{seed}"


def matrix_content_sha256(matrix):
    contiguous = np.ascontiguousarray(
        matrix,
        dtype=np.int64,
    )
    return hashlib.sha256(
        contiguous.tobytes(order="C")
    ).hexdigest().upper()


def count_plan_set_sha256(rows):
    digest = hashlib.sha256()
    for row in sorted(
        rows,
        key=lambda item: item["ConfigID"],
    ):
        digest.update(
            (
                f"{row['ConfigID']}\t"
                f"{row['MatrixContentSHA256']}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest().upper()


def capacity_vector_sha256(capacities):
    array = np.ascontiguousarray(
        capacities,
        dtype=np.int64,
    )
    return hashlib.sha256(
        array.tobytes(order="C")
    ).hexdigest().upper()


def parse_bool(value):
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def load_matrix_from_csv(path, expected_capacities):
    rows = read_csv(path)

    require_equal(
        len(rows),
        CLIENT_COUNT,
        f"Client row count in {path.name}",
    )

    matrix = np.zeros(
        (
            len(TRAIN_COUNTS),
            CLIENT_COUNT,
        ),
        dtype=np.int64,
    )

    observed_client_ids = []

    for row in rows:
        client_id = int(row["ClientID"])
        observed_client_ids.append(client_id)

        require_true(
            0 <= client_id < CLIENT_COUNT,
            f"Invalid ClientID in {path.name}: {client_id}",
        )

        require_equal(
            int(row["Capacity"]),
            int(expected_capacities[client_id]),
            f"Capacity in {path.name}, client {client_id}",
        )

        for label_id in range(len(TRAIN_COUNTS)):
            field = f"Label{label_id}_{LABEL_NAMES[label_id]}"
            value = int(row[field])

            require_true(
                value >= 0,
                (
                    f"Negative count in {path.name}, "
                    f"client {client_id}, label {label_id}"
                ),
            )

            matrix[label_id, client_id] = value

    require_equal(
        sorted(observed_client_ids),
        list(range(CLIENT_COUNT)),
        f"ClientID coverage in {path.name}",
    )

    return matrix, rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gate74-diagnostic-json",
        required=True,
    )
    parser.add_argument(
        "--gate74-config-summary-csv",
        required=True,
    )
    parser.add_argument(
        "--gate74-alpha-summary-csv",
        required=True,
    )
    parser.add_argument(
        "--gate74-coverage-csv",
        required=True,
    )
    parser.add_argument(
        "--gate74-capacities-csv",
        required=True,
    )
    parser.add_argument(
        "--gate74-matrix-root",
        required=True,
    )
    parser.add_argument(
        "--gate73-freeze-json",
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

    output_root = Path(args.output)
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    frozen_matrix_root = (
        output_root / "frozen_count_plans"
    )

    frozen_matrix_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence_paths = {
        "GATE74_PARTITION_DIAGNOSTIC": Path(
            args.gate74_diagnostic_json
        ),
        "GATE74_CONFIG_SUMMARY": Path(
            args.gate74_config_summary_csv
        ),
        "GATE74_ALPHA_SUMMARY": Path(
            args.gate74_alpha_summary_csv
        ),
        "GATE74_LABEL_COVERAGE": Path(
            args.gate74_coverage_csv
        ),
        "GATE74_CAPACITIES": Path(
            args.gate74_capacities_csv
        ),
        "GATE73_TRAIN_WEIGHT_POLICY_FREEZE": Path(
            args.gate73_freeze_json
        ),
        "GATE71_SPLIT_MATERIALISATION": Path(
            args.gate71_state_json
        ),
    }

    matrix_root = Path(
        args.gate74_matrix_root
    )

    print("")
    print("=" * 60)
    print("GATE A - VERIFY CURRENT CHAIN AND GATE-74 EVIDENCE")
    print("=" * 60)
    print("")

    gate71 = load_json(
        evidence_paths["GATE71_SPLIT_MATERIALISATION"]
    )

    require_equal(
        gate71.get("status"),
        "PASS",
        "Gate-71 status",
    )

    require_equal(
        gate71["immutable_binding"][
            "effective_dataset_fingerprint_sha256"
        ],
        EFFECTIVE_DATASET_SHA,
        "Gate-71 effective dataset SHA256",
    )

    require_equal(
        gate71["immutable_binding"][
            "split_protocol_artifact_manifest_sha256"
        ],
        SPLIT_PROTOCOL_SHA,
        "Gate-71 split protocol SHA256",
    )

    require_equal(
        gate71["split_assignment_manifest_sha256"],
        SPLIT_ASSIGNMENT_SHA,
        "Gate-71 split assignment SHA256",
    )

    require_true(
        gate71["scientific_boundary"][
            "client_assignments_materialized"
        ] is False,
        "Gate-71 says client assignments already exist.",
    )

    require_true(
        gate71["scientific_boundary"][
            "scientific_training_started"
        ] is False,
        "Gate-71 says scientific training started.",
    )

    gate73 = load_json(
        evidence_paths["GATE73_TRAIN_WEIGHT_POLICY_FREEZE"]
    )

    require_equal(
        gate73.get("status"),
        "FROZEN",
        "Gate-73 status",
    )

    require_equal(
        gate73[
            "frozen_train_weight_policy_artifact_manifest_sha256"
        ],
        TRAIN_WEIGHT_POLICY_SHA,
        "Gate-73 policy manifest SHA256",
    )

    require_equal(
        gate73["canonical_weight_vector_sha256"],
        CANONICAL_WEIGHT_VECTOR_SHA,
        "Gate-73 canonical weight-vector SHA256",
    )

    require_true(
        gate73["materialization_boundary"][
            "client_assignments_rebuilt"
        ] is False,
        "Gate-73 says client assignments already exist.",
    )

    require_true(
        gate73["materialization_boundary"][
            "scientific_training_started"
        ] is False,
        "Gate-73 says scientific training started.",
    )

    gate74 = load_json(
        evidence_paths["GATE74_PARTITION_DIAGNOSTIC"]
    )

    require_equal(
        gate74.get("status"),
        "DIAGNOSIS COMPLETE",
        "Gate-74 status",
    )

    require_equal(
        gate74.get("diagnostic_id"),
        GATE74_DIAGNOSTIC_ID,
        "Gate-74 diagnostic ID",
    )

    require_equal(
        gate74.get("disposition"),
        GATE74_DISPOSITION,
        "Gate-74 disposition",
    )

    require_equal(
        gate74.get("candidate_protocol_id_not_frozen"),
        GATE74_CANDIDATE_PROTOCOL_ID,
        "Gate-74 candidate protocol ID",
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
            gate74["immutable_binding"][key],
            expected,
            f"Gate-74 immutable binding {key}",
        )

    require_true(
        gate74["diagnostic_results"][
            "all_10_matrices_exact_row_totals"
        ] is True,
        "Gate-74 row-total evidence failed.",
    )

    require_true(
        gate74["diagnostic_results"][
            "all_10_matrices_exact_column_capacities"
        ] is True,
        "Gate-74 capacity evidence failed.",
    )

    require_true(
        gate74["diagnostic_results"][
            "all_10_matrices_exact_replay"
        ] is True,
        "Gate-74 matrix replay evidence failed.",
    )

    require_true(
        gate74["diagnostic_results"][
            "aggregate_alpha_separation_pass"
        ] is True,
        "Gate-74 alpha separation did not pass.",
    )

    require_true(
        gate74["scientific_boundary"][
            "count_plans_frozen"
        ] is False,
        "Gate-74 says count plans were already frozen.",
    )

    require_true(
        gate74["scientific_boundary"][
            "physical_client_assignments_materialized"
        ] is False,
        "Gate-74 says physical clients already exist.",
    )

    require_true(
        gate74["scientific_boundary"][
            "scientific_training_started"
        ] is False,
        "Gate-74 says scientific training started.",
    )

    print("Effective dataset: BOUND")
    print("Split assignment: BOUND")
    print("TRAIN weight policy: BOUND")
    print("Gate-74 diagnostic disposition: ACCEPTED")
    print("Physical client assignments: NO")
    print("Scientific training started: NO")

    print("")
    print("=" * 60)
    print("GATE B - VERIFY AND FREEZE EXACT CLIENT CAPACITIES")
    print("=" * 60)
    print("")

    capacity_rows = read_csv(
        evidence_paths["GATE74_CAPACITIES"]
    )

    require_equal(
        len(capacity_rows),
        CLIENT_COUNT,
        "Gate-74 capacity row count",
    )

    capacities = np.zeros(
        CLIENT_COUNT,
        dtype=np.int64,
    )

    observed_ids = []

    for row in capacity_rows:
        client_id = int(row["ClientID"])
        observed_ids.append(client_id)
        capacities[client_id] = int(
            row["ExactCapacity"]
        )

    require_equal(
        sorted(observed_ids),
        list(range(CLIENT_COUNT)),
        "Capacity ClientID coverage",
    )

    require_equal(
        capacities.tolist(),
        EXPECTED_CAPACITIES,
        "Exact capacity vector",
    )

    require_equal(
        int(capacities.sum()),
        TRAIN_TOTAL,
        "Capacity total",
    )

    require_equal(
        int(capacities.max() - capacities.min()),
        1,
        "Capacity spread",
    )

    capacity_sha = capacity_vector_sha256(
        capacities
    )

    frozen_capacity_rows = [
        {
            "ClientID": client_id,
            "ExactCapacity": int(
                capacities[client_id]
            ),
        }
        for client_id
        in range(CLIENT_COUNT)
    ]

    frozen_capacities_path = (
        output_root
        / "FROZEN_EXACT_CLIENT_CAPACITIES.csv"
    )

    write_csv(
        frozen_capacities_path,
        frozen_capacity_rows,
        [
            "ClientID",
            "ExactCapacity",
        ],
    )

    print("Clients 0..23: 551661")
    print("Clients 24..29: 551660")
    print(f"Capacity vector SHA256: {capacity_sha}")

    print("")
    print("=" * 60)
    print("GATE C - RE-AUDIT AND FREEZE TEN EXACT COUNT PLANS")
    print("=" * 60)
    print("")

    config_summary_rows = read_csv(
        evidence_paths["GATE74_CONFIG_SUMMARY"]
    )

    require_equal(
        len(config_summary_rows),
        10,
        "Gate-74 config-summary row count",
    )

    summary_by_config = {}
    observed_config_pairs = []

    for row in config_summary_rows:
        alpha = float(row["Alpha"])
        experimental_seed = int(
            row["ExperimentalSeed"]
        )
        cfg = row["ConfigID"]

        require_equal(
            cfg,
            config_id(
                alpha,
                experimental_seed,
            ),
            "Gate-74 ConfigID replay",
        )

        observed_config_pairs.append(
            (
                alpha,
                experimental_seed,
            )
        )

        require_true(
            parse_bool(row["ExactReplay"]),
            f"Gate-74 ExactReplay false for {cfg}",
        )

        require_true(
            float(row["IPFMaxRowError"])
            <=
            IPF_ABS_TOLERANCE,
            f"IPF row error too large for {cfg}",
        )

        require_true(
            float(row["IPFMaxColumnError"])
            <=
            IPF_ABS_TOLERANCE,
            f"IPF column error too large for {cfg}",
        )

        summary_by_config[cfg] = row

    require_equal(
        sorted(observed_config_pairs),
        sorted(EXPECTED_CONFIGS),
        "Expected alpha/seed configuration grid",
    )

    row_targets = np.asarray(
        [
            TRAIN_COUNTS[label_id]
            for label_id
            in range(len(TRAIN_COUNTS))
        ],
        dtype=np.int64,
    )

    frozen_plan_rows = []
    frozen_matrix_artifacts = []

    for alpha, experimental_seed in EXPECTED_CONFIGS:
        cfg = config_id(
            alpha,
            experimental_seed,
        )

        summary = summary_by_config[cfg]

        source_matrix_path = (
            matrix_root
            / f"{cfg}_candidate_count_matrix.csv"
        )

        require_true(
            source_matrix_path.exists(),
            f"Missing Gate-74 matrix: {source_matrix_path}",
        )

        matrix, source_rows = load_matrix_from_csv(
            source_matrix_path,
            capacities,
        )

        require_true(
            np.array_equal(
                matrix.sum(axis=1),
                row_targets,
            ),
            f"Class-total mismatch for {cfg}",
        )

        require_true(
            np.array_equal(
                matrix.sum(axis=0),
                capacities,
            ),
            f"Capacity mismatch for {cfg}",
        )

        require_true(
            bool(np.all(matrix >= 0)),
            f"Negative count for {cfg}",
        )

        matrix_sha = matrix_content_sha256(
            matrix
        )

        require_equal(
            matrix_sha,
            summary["CandidateMatrixSHA256"],
            f"Gate-74 matrix content SHA256 for {cfg}",
        )

        frozen_file_name = (
            f"{cfg}_FROZEN_COUNT_PLAN_V2.csv"
        )

        frozen_path = (
            frozen_matrix_root
            / frozen_file_name
        )

        write_csv(
            frozen_path,
            source_rows,
            list(source_rows[0].keys()),
        )

        frozen_matrix, _ = load_matrix_from_csv(
            frozen_path,
            capacities,
        )

        require_equal(
            matrix_content_sha256(
                frozen_matrix
            ),
            matrix_sha,
            f"Frozen matrix content replay for {cfg}",
        )

        frozen_file_sha = sha256_file(
            frozen_path
        )

        frozen_plan_rows.append(
            {
                "ConfigID": cfg,
                "Alpha": alpha,
                "ExperimentalSeed": experimental_seed,
                "DerivedRNGSeedSHA256": summary[
                    "DerivedRNGSeedSHA256"
                ],
                "DerivedRNGSeedUInt64": summary[
                    "DerivedRNGSeedUInt64"
                ],
                "IPFIterations": summary[
                    "IPFIterations"
                ],
                "IPFMaxRowError": summary[
                    "IPFMaxRowError"
                ],
                "IPFMaxColumnError": summary[
                    "IPFMaxColumnError"
                ],
                "ResidualIntegerisationUnits": summary[
                    "ResidualIntegerisationUnits"
                ],
                "MatrixContentHashID": MATRIX_CONTENT_HASH_ID,
                "MatrixContentSHA256": matrix_sha,
                "FrozenFileName": frozen_file_name,
                "FrozenFileSHA256": frozen_file_sha,
                "ExactClassTotals": True,
                "ExactClientCapacities": True,
                "NegativeCounts": 0,
            }
        )

        frozen_matrix_artifacts.append(
            {
                "ArtifactRole": "FROZEN_COUNT_PLAN",
                "RelativePath": str(
                    frozen_path.relative_to(
                        output_root
                    )
                ),
                "SizeBytes": frozen_path.stat().st_size,
                "SHA256": frozen_file_sha,
            }
        )

        print(
            f"{cfg} | matrix_sha={matrix_sha} | FROZEN"
        )

    require_equal(
        len(frozen_plan_rows),
        10,
        "Frozen count-plan count",
    )

    count_plan_set_sha = count_plan_set_sha256(
        frozen_plan_rows
    )

    print(
        f"Count-plan set SHA256: {count_plan_set_sha}"
    )

    print("")
    print("=" * 60)
    print("GATE D - VERIFY HETEROGENEITY EVIDENCE")
    print("=" * 60)
    print("")

    alpha_rows = read_csv(
        evidence_paths["GATE74_ALPHA_SUMMARY"]
    )

    require_equal(
        len(alpha_rows),
        2,
        "Gate-74 alpha-summary row count",
    )

    alpha_summary = {
        float(row["Alpha"]): row
        for row
        in alpha_rows
    }

    require_equal(
        set(alpha_summary),
        set(ALPHAS),
        "Alpha-summary coverage",
    )

    low = alpha_summary[0.1]
    high = alpha_summary[1.0]

    require_true(
        float(low["MeanActiveClassesPerClient"])
        <
        float(high["MeanActiveClassesPerClient"]),
        "Alpha active-class separation failed.",
    )

    require_true(
        float(low["MeanNormalisedEntropy"])
        <
        float(high["MeanNormalisedEntropy"]),
        "Alpha entropy separation failed.",
    )

    require_true(
        float(low["MeanJSDivergenceFromGlobal"])
        >
        float(high["MeanJSDivergenceFromGlobal"]),
        "Alpha JS-divergence separation failed.",
    )

    require_true(
        float(low["MeanMaxClassShare"])
        >
        float(high["MeanMaxClassShare"]),
        "Alpha max-class-share separation failed.",
    )

    frozen_alpha_path = (
        output_root
        / "FROZEN_ALPHA_HETEROGENEITY_EVIDENCE.csv"
    )

    write_csv(
        frozen_alpha_path,
        alpha_rows,
        list(alpha_rows[0].keys()),
    )

    print(
        "alpha=0.1 more heterogeneous than alpha=1.0: PASS"
    )

    print("")
    print("=" * 60)
    print("GATE E - WRITE FROZEN PROTOCOL AND COUNT-PLAN MANIFEST")
    print("=" * 60)
    print("")

    frozen_plan_manifest_path = (
        output_root
        / "FROZEN_EXACT_COUNT_PLAN_MANIFEST.csv"
    )

    write_csv(
        frozen_plan_manifest_path,
        frozen_plan_rows,
        [
            "ConfigID",
            "Alpha",
            "ExperimentalSeed",
            "DerivedRNGSeedSHA256",
            "DerivedRNGSeedUInt64",
            "IPFIterations",
            "IPFMaxRowError",
            "IPFMaxColumnError",
            "ResidualIntegerisationUnits",
            "MatrixContentHashID",
            "MatrixContentSHA256",
            "FrozenFileName",
            "FrozenFileSHA256",
            "ExactClassTotals",
            "ExactClientCapacities",
            "NegativeCounts",
        ],
    )

    protocol = {
        "status": "FROZEN",
        "freeze_id": FREEZE_ID,
        "protocol_id": PROTOCOL_ID,
        "count_plan_set_id": COUNT_PLAN_SET_ID,
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
        },
        "frozen_design": {
            "legacy_design_principle_id": (
                LEGACY_DESIGN_PRINCIPLE_ID
            ),
            "client_count": CLIENT_COUNT,
            "alphas": list(ALPHAS),
            "experimental_seeds": list(
                EXPERIMENTAL_SEEDS
            ),
            "configuration_count": 10,
            "capacity_rule_id": CAPACITY_RULE_ID,
            "exact_capacities": capacities.tolist(),
            "capacity_vector_sha256": capacity_sha,
            "rng_binding_id": RNG_BINDING_ID,
            "continuous_balancing_id": (
                CONTINUOUS_BALANCING_ID
            ),
            "ipf_absolute_tolerance": (
                IPF_ABS_TOLERANCE
            ),
            "ipf_max_iterations": (
                IPF_MAX_ITERATIONS
            ),
            "integerisation_id": INTEGERISATION_ID,
            "fraction_score_scale": (
                FRACTION_SCORE_SCALE
            ),
            "matrix_content_hash_id": (
                MATRIX_CONTENT_HASH_ID
            ),
            "count_plan_set_hash_id": (
                COUNT_PLAN_SET_HASH_ID
            ),
            "count_plan_set_sha256": (
                count_plan_set_sha
            ),
        },
        "frozen_count_plan_semantics": {
            "matrix_shape": (
                "7_TASK7_LABELS_X_30_CLIENTS"
            ),
            "row_semantics": (
                "EXACT_GLOBAL_TRANSFORMED_UNIQUE_TRAIN_CLASS_TOTALS"
            ),
            "column_semantics": (
                "EXACT_BALANCED_CLIENT_CAPACITIES"
            ),
            "matrix_values": (
                "NONNEGATIVE_INTEGER_OBSERVATION_COUNTS"
            ),
            "count_plans_primary_for_future_assignment": True,
            "regeneration_algorithm_role": "AUDIT_PROVENANCE",
        },
        "frozen_heterogeneity_interpretation": {
            "alpha_0p1_role": (
                "HIGHER_NON_IID_HETEROGENEITY"
            ),
            "alpha_1p0_role": (
                "MODERATE_NON_IID_HETEROGENEITY"
            ),
            "aggregate_separation_pass": True,
        },
        "prohibited_inputs": [
            "VALIDATION_OBSERVATIONS",
            "TEST_OBSERVATIONS",
            "RawMultiplicity",
            "TransformedGroupRawExactMultiplicity",
            "SumSourceRawMultiplicity",
            "Provenance",
            "ModelResults",
            "AttackResults",
        ],
        "supersession": {
            "prior_physical_client_assignments": (
                "SUPERSEDED_NOT_REUSABLE"
            ),
            "prior_count_plans": (
                "SUPERSEDED_NOT_REUSABLE"
            ),
        },
        "scientific_boundary": {
            "partition_protocol_frozen": True,
            "exact_count_plans_frozen": True,
            "physical_assignment_algorithm_frozen": False,
            "physical_client_assignments_materialized": False,
            "scientific_optimizer_steps_executed": 0,
            "scientific_training_started": False,
        },
    }

    protocol_path = (
        output_root
        / "FROZEN_TRANSFORMED_UNIQUE_CLIENT_PARTITION_PROTOCOL.json"
    )

    write_json(
        protocol_path,
        protocol,
    )

    generation_provenance = {
        "status": "FROZEN_PROVENANCE",
        "protocol_id": PROTOCOL_ID,
        "count_plan_set_id": COUNT_PLAN_SET_ID,
        "generator_environment_observed_at_gate74": (
            gate74["environment"]
        ),
        "rng_binding_id": RNG_BINDING_ID,
        "continuous_balancing_id": (
            CONTINUOUS_BALANCING_ID
        ),
        "ipf_absolute_tolerance": (
            IPF_ABS_TOLERANCE
        ),
        "ipf_max_iterations": IPF_MAX_ITERATIONS,
        "integerisation_id": INTEGERISATION_ID,
        "fraction_score_scale": FRACTION_SCORE_SCALE,
        "count_plans_are_primary_immutable_outputs": True,
        "regeneration_required_for_scientific_use": False,
    }

    provenance_path = (
        output_root
        / "FROZEN_COUNT_PLAN_GENERATION_PROVENANCE.json"
    )

    write_json(
        provenance_path,
        generation_provenance,
    )

    print("")
    print("=" * 60)
    print("GATE F - WRITE IMMUTABLE ARTIFACT MANIFEST")
    print("=" * 60)
    print("")

    artifact_rows = list(
        frozen_matrix_artifacts
    )

    for role, path in [
        (
            "CLIENT_PARTITION_PROTOCOL",
            protocol_path,
        ),
        (
            "EXACT_CLIENT_CAPACITIES",
            frozen_capacities_path,
        ),
        (
            "EXACT_COUNT_PLAN_MANIFEST",
            frozen_plan_manifest_path,
        ),
        (
            "COUNT_PLAN_GENERATION_PROVENANCE",
            provenance_path,
        ),
        (
            "ALPHA_HETEROGENEITY_EVIDENCE",
            frozen_alpha_path,
        ),
    ]:
        artifact_rows.append(
            {
                "ArtifactRole": role,
                "RelativePath": str(
                    path.relative_to(
                        output_root
                    )
                ),
                "SizeBytes": path.stat().st_size,
                "SHA256": sha256_file(path),
            }
        )

    artifact_manifest_path = (
        output_root
        / "FROZEN_CLIENT_PARTITION_PROTOCOL_ARTIFACT_MANIFEST.csv"
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

    combined_manifest_sha256 = artifact_manifest_digest(
        artifact_rows
    )

    freeze_state = {
        "status": "FROZEN",
        "freeze_id": FREEZE_ID,
        "protocol_id": PROTOCOL_ID,
        "count_plan_set_id": COUNT_PLAN_SET_ID,
        "capacity_vector_sha256": capacity_sha,
        "count_plan_set_sha256": count_plan_set_sha,
        "frozen_client_partition_protocol_artifact_manifest_sha256": (
            combined_manifest_sha256
        ),
        "materialization_boundary": {
            "physical_assignment_algorithm_frozen": False,
            "physical_client_assignments_materialized": False,
            "scientific_training_started": False,
        },
    }

    freeze_state_path = (
        output_root
        / "TRANSFORMED_UNIQUE_CLIENT_PARTITION_PROTOCOL_FREEZE.json"
    )

    write_json(
        freeze_state_path,
        freeze_state,
    )

    evidence_rows = []

    for role, path in evidence_paths.items():
        evidence_rows.append(
            {
                "EvidenceRole": role,
                "Path": str(path),
                "SizeBytes": path.stat().st_size,
                "SHA256": sha256_file(path),
            }
        )

    write_csv(
        output_root / "FREEZE_EVIDENCE_MANIFEST.csv",
        evidence_rows,
        [
            "EvidenceRole",
            "Path",
            "SizeBytes",
            "SHA256",
        ],
    )

    report = []

    def add(value=""):
        report.append(value)

    add(
        "CICIoT2023 TRANSFORMED-UNIQUE K=30 CLIENT PARTITION PROTOCOL FREEZE"
    )
    add("=" * 78)
    add("")
    add("STATUS")
    add("-" * 78)
    add("FROZEN")
    add("")
    add("PROTOCOL IDENTIFIERS")
    add("-" * 78)
    add(f"Freeze ID: {FREEZE_ID}")
    add(f"Protocol ID: {PROTOCOL_ID}")
    add(f"Count-plan set ID: {COUNT_PLAN_SET_ID}")
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
    add(
        f"TRAIN weight-policy artifact-manifest SHA256: {TRAIN_WEIGHT_POLICY_SHA}"
    )
    add(
        f"Canonical weight-vector SHA256: {CANONICAL_WEIGHT_VECTOR_SHA}"
    )
    add("")
    add("FROZEN DESIGN")
    add("-" * 78)
    add("Clients: 30")
    add("Alpha values: 0.1, 1.0")
    add("Experimental seeds: 42, 123, 456, 789, 999")
    add("Configurations: 10")
    add(
        f"Legacy design principle ID: {LEGACY_DESIGN_PRINCIPLE_ID}"
    )
    add(f"RNG binding ID: {RNG_BINDING_ID}")
    add(
        f"Continuous balancing ID: {CONTINUOUS_BALANCING_ID}"
    )
    add(f"Exact integerisation ID: {INTEGERISATION_ID}")
    add("")
    add("FROZEN EXACT CAPACITIES")
    add("-" * 78)
    add("Clients 0..23: 551661 observations each")
    add("Clients 24..29: 551660 observations each")
    add("Maximum capacity difference: 1")
    add(f"Capacity total: {int(capacities.sum())}")
    add(f"Capacity vector SHA256: {capacity_sha}")
    add("")
    add("FROZEN EXACT COUNT PLANS")
    add("-" * 78)
    add("Count plans: 10")
    add("Matrix shape: 7 TASK7 labels x 30 clients")
    add("Exact global class totals: PASS ALL 10")
    add("Exact client capacities: PASS ALL 10")
    add("Negative counts: 0")
    add(f"Matrix content hash ID: {MATRIX_CONTENT_HASH_ID}")
    add(f"Count-plan set hash ID: {COUNT_PLAN_SET_HASH_ID}")
    add(f"Count-plan set SHA256: {count_plan_set_sha}")
    add("")

    for row in frozen_plan_rows:
        add(
            f"{row['ConfigID']}: {row['MatrixContentSHA256']}"
        )

    add("")
    add("FROZEN HETEROGENEITY INTERPRETATION")
    add("-" * 78)
    add("alpha=0.1: higher non-IID heterogeneity")
    add("alpha=1.0: moderate non-IID heterogeneity")
    add("Aggregate alpha separation: PASS")
    add("")
    add("SUPERSESSION")
    add("-" * 78)
    add("Prior count plans: SUPERSEDED")
    add("Prior physical client assignments: SUPERSEDED")
    add("Prior count plans may be reused: NO")
    add("Prior physical assignments may be reused: NO")
    add("")
    add("IMMUTABLE ARTIFACT")
    add("-" * 78)
    add(
        "Combined client-partition protocol artifact-manifest SHA256: "
        f"{combined_manifest_sha256}"
    )
    add("")
    add("SCIENTIFIC STATE")
    add("-" * 78)
    add("Partition protocol frozen: YES")
    add("Exact count plans frozen: YES")
    add("Physical assignment algorithm frozen: NO")
    add("Physical client assignments materialized: NO")
    add("Scientific optimizer steps executed: 0")
    add("Scientific training started: NO")
    add("")
    add("NEXT GATE")
    add("-" * 78)
    add(
        "Design and freeze the deterministic physical TRAIN-observation "
        "assignment algorithm. The algorithm must bind only the new TRAIN "
        "effective observation identity, label, configuration-specific frozen "
        "count plan, and a configuration-specific deterministic ordering seed. "
        "Do not materialise physical client assignments until that assignment "
        "algorithm and all ordering seeds are frozen."
    )

    (
        output_root
        / "TRANSFORMED_UNIQUE_CLIENT_PARTITION_PROTOCOL_FREEZE_REPORT.txt"
    ).write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("")
    print("=" * 60)
    print("STATUS: FROZEN")
    print(f"PROTOCOL ID: {PROTOCOL_ID}")
    print(f"CAPACITY VECTOR SHA256: {capacity_sha}")
    print(f"COUNT-PLAN SET SHA256: {count_plan_set_sha}")
    print(
        f"COMBINED PROTOCOL MANIFEST: {combined_manifest_sha256}"
    )
    print("PHYSICAL ASSIGNMENT ALGORITHM FROZEN: NO")
    print("PHYSICAL CLIENT ASSIGNMENTS: NO")
    print("SCIENTIFIC TRAINING STARTED: NO")
    print("=" * 60)
    print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
