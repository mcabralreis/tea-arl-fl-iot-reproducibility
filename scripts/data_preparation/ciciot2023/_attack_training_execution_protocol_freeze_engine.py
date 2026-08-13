import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path


# ============================================================================
# Immutable branch bindings
# ============================================================================

EFFECTIVE_DATASET_SHA = (
    "5708EFE6C08C91CF3637FA8F89F53C4459933F94C7CC0BF819A590CBE9EF8E5D"
)

FEATURE_LAYER_SHA = (
    "3BCCF823E11D0088970EC9BD26D411C73213D4E22C4469EBD0BF1D92F255944A"
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

ASSIGNMENT_SET_CONTENT_SHA = (
    "6161DDCA8874F3A079B3D61C19C62A0C1A60389ECFF2B2028E043FFC38B36E98"
)

PHYSICAL_ASSIGNMENT_AUDIT_SHA = (
    "E04148B061539804DF6928544707E47DFB84A3FCAE5544D55428719569DA67D5"
)

PREATTACK_IDENTITY_SHA = (
    "81057E395366BA10088F4455897446945556325756CA18BBF39B5F5A6A1198E2"
)

PARTICIPATION_MALICIOUS_IDENTITY_ARTIFACT_SHA = (
    "94C65F3EE7EDDCF19CC35FE1324AC595C706D472F3C4C898EC562747B8EA6031"
)


# ============================================================================
# Gate-81 accepted candidate fingerprints
# ============================================================================

GATE81_DIAGNOSTIC_ID = (
    "CICIoT2023_TRANSFORMED_UNIQUE_ATTACK_CORE_TRAINING_REBIND_DIAGNOSTIC_V1"
)

GATE81_DISPOSITION = (
    "ATTACK_AND_CORE_TRAINING_PRINCIPLES_REVALIDATED_"
    "DETERMINISTIC_SEED_AND_SAMPLE_ORDER_FREEZE_REQUIRED"
)

SCENARIO_PLAN_SHA = (
    "B660EC4C1D007C580E1451575E9192AFF68E0AC261E0FA7216080947D2DA8208"
)

GLOBAL_INIT_SEED_SET_SHA = (
    "1A5D810307F46F54CC329E1F255B2E72AD36647E72189BFAA01C8372C3DFC30C"
)

LOCAL_ORDER_SEED_SET_SHA = (
    "0EFBAA3CA5AE95B102950DB32F84C7719BDCA65FDE8471FC4524E1C178CEC9C3"
)

LOCAL_TRAIN_RNG_SEED_SET_SHA = (
    "0792A7ED915B655D76776136DEB602EF789AE7C2510354112DAC522AD960CC59"
)

CAMPAIGN_MATRIX_SHA = (
    "D67D6501935D5987BF0D3CDD9FBB8FC23FCCE611046BC59262C500F777B7DFEB"
)

GATE81_COMBINED_BINDING_SHA = (
    "C3189C36A8E7BD9F1667663D0E5DEDF3B5B9D4306E796D0E2B45FB43FC2F4F59"
)


# ============================================================================
# Gate-82 accepted execution-detail fingerprints
# ============================================================================

GATE82_DIAGNOSTIC_ID = (
    "CICIoT2023_TRANSFORMED_UNIQUE_EXACT_EXECUTION_ORDER_DIAGNOSTIC_V1"
)

GATE82_DISPOSITION = (
    "EXACT_LOCAL_SAMPLE_ORDER_BLOCK_EXECUTION_AND_CHECKPOINT_RULES_"
    "REVALIDATED_FREEZE_READY"
)

LOCAL_ORDER_ALGORITHM_SHA = (
    "0C757D687DE35822211B8E2AC21FE1BC01FC8CFC70B5A3C3D6953931DCB4B21F"
)

BLOCK_BOUNDARY_SET_SHA = (
    "7846A18811A29B5EE4876B85CF31E0DA49061AE6CC84127E96204C1B028B0607"
)

ROUND_EXECUTION_SET_SHA = (
    "028C4D8072D0B3B9BDF2A45FD81AE3A2F8C47BB9D0331EE70DB8AEE1E74AFFAA"
)

CHECKPOINT_COMPARATOR_SHA = (
    "17C600042D27918CF2D754FC3A6E1C72224A48882B06E44254D41A8E517ED60B"
)

GATE82_COMBINED_EXECUTION_DETAIL_SHA = (
    "02C7E853FF2453AE6DB681CA29EAD6DE131CBD9A2F334DCBC9E9A5AE9EB6C0FD"
)


# ============================================================================
# Final frozen identifiers
# ============================================================================

FREEZE_ID = (
    "CICIoT2023_TRANSFORMED_UNIQUE_ATTACK_TRAINING_EXECUTION_PROTOCOL_FREEZE_V1"
)

COMMON_PROTOCOL_ID = (
    "CICIoT2023_TASK7_COMMON_ATTACK_TRAINING_EXECUTION_PROTOCOL_V1"
)

ATTACK_PROTOCOL_ID = (
    "CICIoT2023_SIGNFLIP_NEG5_R5_TO_R20_MAIN_MU0P2_0P4_V2"
)

CORE_TRAINING_PROTOCOL_ID = (
    "CICIoT2023_MLP256_128_CPU_ADAM1E3_B256_S200_DISJOINT6BLOCKS_V2"
)

SEED_PROTOCOL_ID = (
    "CICIoT2023_PAIRED_GLOBAL_INIT_LOCAL_ORDER_LOCAL_RNG_SEEDS_V1"
)

LOCAL_SAMPLE_ORDER_ALGORITHM_ID = (
    "SPLITMIX64_DUAL_KEY_CONFIG_CLIENT_LOCAL_SAMPLE_ORDER_V1"
)

LOCAL_BLOCK_ALLOCATION_ID = (
    "SIX_DISJOINT_CONTIGUOUS_51200_BLOCKS_BY_PARTICIPATION_ORDINAL_V1"
)

ROUND_CLIENT_EXECUTION_ID = (
    "FROZEN_SCHEDULE_SLOT_ASCENDING_CLIENT_EXECUTION_V1"
)

LOCAL_RNG_APPLICATION_ID = (
    "EVENT_SEED_MOD_2POW63_MINUS1_BEFORE_STOCHASTIC_LOCAL_TRAINING_V1"
)

CHECKPOINT_COMPARATOR_ID = (
    "LEXICOGRAPHIC_MAX_MACROF1_BALACC_NEGROUND_FLOAT64_NO_TOLERANCE_V1"
)

MAIN_CAMPAIGN_SET_ID = (
    "CICIoT2023_MAIN_CAMPAIGN_10CONFIG_6METHOD_3SCENARIO_180RUN_V1"
)

COMMON_PROTOCOL_BINDING_HASH_ID = (
    "COMMON_ATTACK_TRAINING_EXECUTION_BINDINGS_SHA256_LF_V1"
)


# ============================================================================
# Geometry
# ============================================================================

EXPECTED_CONFIGS = 10
EXPECTED_METHODS = 6
EXPECTED_MAIN_SCENARIOS = 3
EXPECTED_MAIN_RUNS = 180

EXPECTED_METHOD_IDS = {
    "FEDAVG",
    "FEDPROX",
    "RANDOM_TRIMMED_MEAN",
    "FEDLE_ADAPTED",
    "TEA_FL",
    "ARL_FL",
}

EXPECTED_MAIN_SCENARIO_IDS = {
    "CLEAN",
    "SIGNFLIP_MU0P2",
    "SIGNFLIP_MU0P4",
}

EXPECTED_ALL_SCENARIO_IDS = {
    "CLEAN",
    "SIGNFLIP_MU0P2",
    "SIGNFLIP_MU0P4",
    "SIGNFLIP_MU0P1_DIAGNOSTIC",
}

EXPECTED_GLOBAL_INIT_ROWS = 5
EXPECTED_LOCAL_ORDER_SEED_ROWS = 300
EXPECTED_LOCAL_TRAIN_RNG_ROWS = 900
EXPECTED_BLOCK_ROWS = 1800
EXPECTED_ROUND_EXECUTION_ROWS = 900
EXPECTED_RNG_APPLICATION_ROWS = 900


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
        return list(csv.DictReader(handle))


def write_json(path, obj):
    Path(path).write_bytes(
        (
            json.dumps(
                obj,
                indent=2,
                ensure_ascii=False,
            )
            +
            "\n"
        ).encode("utf-8")
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


def sha256_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest().upper()


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
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


def canonical_text_from_file(path):
    text = Path(path).read_text(
        encoding="utf-8-sig"
    )

    return (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def copy_bytes(source, destination):
    shutil.copyfile(
        source,
        destination,
    )


def canonical_seed_set_sha(
    rows,
    sort_fields,
):
    digest = hashlib.sha256()

    integer_fields = {
        "ExperimentalSeed",
        "ClientID",
        "Round",
        "ParticipationOrdinal",
    }

    for row in sorted(
        rows,
        key=lambda item: tuple(
            (
                int(item[field])
                if field in integer_fields
                else item[field]
            )
            for field
            in sort_fields
        ),
    ):
        prefix = "\t".join(
            str(row[field])
            for field
            in sort_fields
        )

        digest.update(
            (
                f"{prefix}\t"
                f"{row['SeedSHA256']}\t"
                f"{row['SeedUInt64']}\n"
            ).encode("utf-8")
        )

    return digest.hexdigest().upper()


def scenario_plan_sha(rows):
    digest = hashlib.sha256()

    for row in sorted(
        rows,
        key=lambda item: int(item["ScenarioOrder"]),
    ):
        digest.update(
            (
                f"{row['ScenarioID']}\t"
                f"{row['MaliciousFraction']}\t"
                f"{row['MaliciousPrefixSize']}\t"
                f"{row['AttackActiveRounds']}\t"
                f"{row['AttackOperator']}\n"
            ).encode("utf-8")
        )

    return digest.hexdigest().upper()


def campaign_matrix_sha(rows):
    digest = hashlib.sha256()

    for row in sorted(
        rows,
        key=lambda item: (
            item["ConfigID"],
            item["MethodID"],
            item["ScenarioID"],
        ),
    ):
        digest.update(
            (
                f"{row['ConfigID']}\t"
                f"{row['MethodID']}\t"
                f"{row['ScenarioID']}\n"
            ).encode("utf-8")
        )

    return digest.hexdigest().upper()


def block_boundary_sha(rows):
    digest = hashlib.sha256()

    for row in sorted(
        rows,
        key=lambda item: (
            item["ConfigID"],
            int(item["ClientID"]),
            int(item["ParticipationOrdinal"]),
        ),
    ):
        digest.update(
            (
                f"{row['ConfigID']}\t"
                f"{row['ClientID']}\t"
                f"{row['ParticipationOrdinal']}\t"
                f"{row['StartInclusive']}\t"
                f"{row['EndExclusive']}\t"
                f"{row['Count']}\n"
            ).encode("utf-8")
        )

    return digest.hexdigest().upper()


def round_execution_sha(rows):
    digest = hashlib.sha256()

    for row in sorted(
        rows,
        key=lambda item: (
            int(item["ExperimentalSeed"]),
            int(item["Round"]),
            int(item["Slot"]),
        ),
    ):
        digest.update(
            (
                f"{row['ExperimentalSeed']}\t"
                f"{row['Round']}\t"
                f"{row['Slot']}\t"
                f"{row['ClientID']}\n"
            ).encode("utf-8")
        )

    return digest.hexdigest().upper()


def artifact_manifest_digest(rows):
    digest = hashlib.sha256()

    for row in sorted(
        rows,
        key=lambda item: (
            item["ArtifactRole"],
            item["FileName"],
        ),
    ):
        digest.update(
            (
                f"{row['ArtifactRole']}\t"
                f"{row['FileName']}\t"
                f"{row['SizeBytes']}\t"
                f"{row['SHA256']}\n"
            ).encode("utf-8")
        )

    return digest.hexdigest().upper()


def combined_gate81_binding_sha(
    scenario_sha,
    global_init_sha,
    local_order_seed_sha,
    local_train_rng_sha,
    campaign_sha,
):
    text = (
        f"ASSIGNMENT_SET_SHA256={ASSIGNMENT_SET_CONTENT_SHA}\n"
        f"PHYSICAL_ASSIGNMENT_AUDIT_SHA256={PHYSICAL_ASSIGNMENT_AUDIT_SHA}\n"
        f"PREATTACK_IDENTITY_SHA256={PREATTACK_IDENTITY_SHA}\n"
        f"PARTICIPATION_MALICIOUS_IDENTITY_ARTIFACT_SHA256="
        f"{PARTICIPATION_MALICIOUS_IDENTITY_ARTIFACT_SHA}\n"
        f"TRAIN_WEIGHT_POLICY_SHA256={TRAIN_WEIGHT_POLICY_SHA}\n"
        f"CANONICAL_WEIGHT_VECTOR_SHA256={CANONICAL_WEIGHT_VECTOR_SHA}\n"
        f"SCENARIO_PLAN_SHA256={scenario_sha}\n"
        f"GLOBAL_INIT_SEED_SET_SHA256={global_init_sha}\n"
        f"LOCAL_ORDER_SEED_SET_SHA256={local_order_seed_sha}\n"
        f"LOCAL_TRAIN_RNG_SEED_SET_SHA256={local_train_rng_sha}\n"
        f"CAMPAIGN_MATRIX_SHA256={campaign_sha}\n"
    )

    return sha256_text(text)


def combined_gate82_execution_sha(
    local_order_algorithm_sha,
    block_sha,
    round_execution_sha256,
    checkpoint_sha,
):
    text = (
        f"GATE81_COMBINED_BINDING_SHA256={GATE81_COMBINED_BINDING_SHA}\n"
        f"LOCAL_ORDER_ALGORITHM_SHA256={local_order_algorithm_sha}\n"
        f"BLOCK_BOUNDARY_SET_SHA256={block_sha}\n"
        f"ROUND_EXECUTION_SET_SHA256={round_execution_sha256}\n"
        f"CHECKPOINT_COMPARATOR_SHA256={checkpoint_sha}\n"
    )

    return sha256_text(text)


def common_protocol_binding_sha(
    local_order_algorithm_sha,
    block_sha,
    round_execution_sha256,
    checkpoint_sha,
):
    text = (
        f"EFFECTIVE_DATASET_SHA256={EFFECTIVE_DATASET_SHA}\n"
        f"FEATURE_LAYER_SHA256={FEATURE_LAYER_SHA}\n"
        f"SPLIT_ASSIGNMENT_SHA256={SPLIT_ASSIGNMENT_SHA}\n"
        f"TRAIN_WEIGHT_POLICY_SHA256={TRAIN_WEIGHT_POLICY_SHA}\n"
        f"CANONICAL_WEIGHT_VECTOR_SHA256={CANONICAL_WEIGHT_VECTOR_SHA}\n"
        f"ASSIGNMENT_SET_CONTENT_SHA256={ASSIGNMENT_SET_CONTENT_SHA}\n"
        f"PHYSICAL_ASSIGNMENT_AUDIT_SHA256={PHYSICAL_ASSIGNMENT_AUDIT_SHA}\n"
        f"PREATTACK_IDENTITY_SHA256={PREATTACK_IDENTITY_SHA}\n"
        f"PARTICIPATION_MALICIOUS_IDENTITY_ARTIFACT_SHA256="
        f"{PARTICIPATION_MALICIOUS_IDENTITY_ARTIFACT_SHA}\n"
        f"SCENARIO_PLAN_SHA256={SCENARIO_PLAN_SHA}\n"
        f"GLOBAL_INIT_SEED_SET_SHA256={GLOBAL_INIT_SEED_SET_SHA}\n"
        f"LOCAL_ORDER_SEED_SET_SHA256={LOCAL_ORDER_SEED_SET_SHA}\n"
        f"LOCAL_TRAIN_RNG_SEED_SET_SHA256={LOCAL_TRAIN_RNG_SEED_SET_SHA}\n"
        f"CAMPAIGN_MATRIX_SHA256={CAMPAIGN_MATRIX_SHA}\n"
        f"LOCAL_ORDER_ALGORITHM_SHA256={local_order_algorithm_sha}\n"
        f"BLOCK_BOUNDARY_SET_SHA256={block_sha}\n"
        f"ROUND_EXECUTION_SET_SHA256={round_execution_sha256}\n"
        f"CHECKPOINT_COMPARATOR_SHA256={checkpoint_sha}\n"
        f"GATE81_COMBINED_BINDING_SHA256={GATE81_COMBINED_BINDING_SHA}\n"
        f"GATE82_COMBINED_EXECUTION_DETAIL_SHA256="
        f"{GATE82_COMBINED_EXECUTION_DETAIL_SHA}\n"
    )

    return sha256_text(text)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gate81-diagnostic-json", required=True)
    parser.add_argument("--gate81-attack-json", required=True)
    parser.add_argument("--gate81-training-json", required=True)
    parser.add_argument("--gate81-scenario-csv", required=True)
    parser.add_argument("--gate81-exposure-csv", required=True)
    parser.add_argument("--gate81-global-init-csv", required=True)
    parser.add_argument("--gate81-local-order-seeds-csv", required=True)
    parser.add_argument("--gate81-local-train-rng-csv", required=True)
    parser.add_argument("--gate81-campaign-csv", required=True)

    parser.add_argument("--gate82-diagnostic-json", required=True)
    parser.add_argument("--gate82-algorithm-txt", required=True)
    parser.add_argument("--gate82-block-csv", required=True)
    parser.add_argument("--gate82-round-execution-csv", required=True)
    parser.add_argument("--gate82-rng-application-csv", required=True)
    parser.add_argument("--gate82-comparator-txt", required=True)
    parser.add_argument("--gate82-probe-csv", required=True)

    parser.add_argument("--gate80-freeze-json", required=True)
    parser.add_argument("--gate78-audit-json", required=True)
    parser.add_argument("--gate73-freeze-json", required=True)

    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    output_root = Path(args.output)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Gate A - verify prior frozen branch.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE A - VERIFY PRIOR FROZEN BRANCH")
    print("=" * 60)
    print("")

    gate80 = load_json(
        args.gate80_freeze_json
    )

    require_equal(
        gate80.get("status"),
        "FROZEN",
        "Gate-80 status",
    )

    require_equal(
        gate80[
            "combined_preattack_identity_sha256"
        ],
        PREATTACK_IDENTITY_SHA,
        "Gate-80 pre-attack identity SHA256",
    )

    require_equal(
        gate80[
            "frozen_participation_malicious_identity_artifact_manifest_sha256"
        ],
        PARTICIPATION_MALICIOUS_IDENTITY_ARTIFACT_SHA,
        "Gate-80 artifact-manifest SHA256",
    )

    require_true(
        gate80[
            "scientific_boundary"
        ][
            "attack_protocol_rebound"
        ]
        is False,
        "Gate-80 says attack already rebound.",
    )

    require_true(
        gate80[
            "scientific_boundary"
        ][
            "scientific_training_started"
        ]
        is False,
        "Gate-80 says scientific training started.",
    )

    gate78 = load_json(
        args.gate78_audit_json
    )

    require_equal(
        gate78.get("status"),
        "PASS",
        "Gate-78 status",
    )

    require_equal(
        gate78[
            "assignment_set_content_sha256"
        ],
        ASSIGNMENT_SET_CONTENT_SHA,
        "Gate-78 assignment-set SHA256",
    )

    require_equal(
        gate78[
            "physical_assignment_audit_artifact_manifest_sha256"
        ],
        PHYSICAL_ASSIGNMENT_AUDIT_SHA,
        "Gate-78 audit SHA256",
    )

    gate73 = load_json(
        args.gate73_freeze_json
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
        "Gate-73 weight-policy SHA256",
    )

    require_equal(
        gate73[
            "canonical_weight_vector_sha256"
        ],
        CANONICAL_WEIGHT_VECTOR_SHA,
        "Gate-73 canonical weight-vector SHA256",
    )

    print("Gate-73 weight policy: FROZEN")
    print("Gate-78 physical assignment audit: PASS")
    print("Gate-80 participation / malicious identity: FROZEN")
    print("Scientific training started: NO")

    # ------------------------------------------------------------------
    # Gate B - verify and replay Gate-81 candidate.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE B - VERIFY / REPLAY GATE-81 CANDIDATE")
    print("=" * 60)
    print("")

    gate81 = load_json(
        args.gate81_diagnostic_json
    )

    require_equal(
        gate81.get("status"),
        "DIAGNOSIS COMPLETE",
        "Gate-81 status",
    )

    require_equal(
        gate81.get("diagnostic_id"),
        GATE81_DIAGNOSTIC_ID,
        "Gate-81 diagnostic ID",
    )

    require_equal(
        gate81.get("disposition"),
        GATE81_DISPOSITION,
        "Gate-81 disposition",
    )

    gate81_fp = gate81[
        "candidate_fingerprints_not_frozen"
    ]

    require_equal(
        gate81_fp[
            "scenario_plan_sha256"
        ],
        SCENARIO_PLAN_SHA,
        "Gate-81 scenario-plan SHA256",
    )

    require_equal(
        gate81_fp[
            "global_init_seed_set_sha256"
        ],
        GLOBAL_INIT_SEED_SET_SHA,
        "Gate-81 global-init seed-set SHA256",
    )

    require_equal(
        gate81_fp[
            "local_order_seed_set_sha256"
        ],
        LOCAL_ORDER_SEED_SET_SHA,
        "Gate-81 local-order seed-set SHA256",
    )

    require_equal(
        gate81_fp[
            "local_train_rng_seed_set_sha256"
        ],
        LOCAL_TRAIN_RNG_SEED_SET_SHA,
        "Gate-81 local-train RNG seed-set SHA256",
    )

    require_equal(
        gate81_fp[
            "campaign_matrix_sha256"
        ],
        CAMPAIGN_MATRIX_SHA,
        "Gate-81 campaign-matrix SHA256",
    )

    require_equal(
        gate81_fp[
            "combined_candidate_binding_sha256"
        ],
        GATE81_COMBINED_BINDING_SHA,
        "Gate-81 combined binding SHA256",
    )

    require_true(
        gate81[
            "scientific_boundary"
        ][
            "scientific_training_started"
        ]
        is False,
        "Gate-81 says scientific training started.",
    )

    attack_candidate = load_json(
        args.gate81_attack_json
    )

    training_candidate = load_json(
        args.gate81_training_json
    )

    require_equal(
        attack_candidate.get("status"),
        "CANDIDATE_NOT_FROZEN",
        "Gate-81 attack candidate status",
    )

    require_equal(
        training_candidate.get("status"),
        "CANDIDATE_NOT_FROZEN",
        "Gate-81 training candidate status",
    )

    scenario_rows = read_csv(
        args.gate81_scenario_csv
    )

    global_init_rows = read_csv(
        args.gate81_global_init_csv
    )

    local_order_seed_rows = read_csv(
        args.gate81_local_order_seeds_csv
    )

    local_train_rng_rows = read_csv(
        args.gate81_local_train_rng_csv
    )

    campaign_rows = read_csv(
        args.gate81_campaign_csv
    )

    require_equal(
        scenario_plan_sha(
            scenario_rows
        ),
        SCENARIO_PLAN_SHA,
        "Scenario-plan SHA256 replay",
    )

    require_equal(
        canonical_seed_set_sha(
            global_init_rows,
            (
                "ExperimentalSeed",
            ),
        ),
        GLOBAL_INIT_SEED_SET_SHA,
        "Global-init seed-set SHA256 replay",
    )

    require_equal(
        canonical_seed_set_sha(
            local_order_seed_rows,
            (
                "ConfigID",
                "ClientID",
            ),
        ),
        LOCAL_ORDER_SEED_SET_SHA,
        "Local-order seed-set SHA256 replay",
    )

    require_equal(
        canonical_seed_set_sha(
            local_train_rng_rows,
            (
                "ExperimentalSeed",
                "Round",
                "ClientID",
                "ParticipationOrdinal",
            ),
        ),
        LOCAL_TRAIN_RNG_SEED_SET_SHA,
        "Local-train RNG seed-set SHA256 replay",
    )

    require_equal(
        campaign_matrix_sha(
            campaign_rows
        ),
        CAMPAIGN_MATRIX_SHA,
        "Campaign-matrix SHA256 replay",
    )

    require_equal(
        combined_gate81_binding_sha(
            SCENARIO_PLAN_SHA,
            GLOBAL_INIT_SEED_SET_SHA,
            LOCAL_ORDER_SEED_SET_SHA,
            LOCAL_TRAIN_RNG_SEED_SET_SHA,
            CAMPAIGN_MATRIX_SHA,
        ),
        GATE81_COMBINED_BINDING_SHA,
        "Gate-81 combined binding replay",
    )

    print("Scenario-plan fingerprint replay: EXACT MATCH")
    print("Seed-set fingerprints replay: EXACT MATCH")
    print("Campaign-matrix fingerprint replay: EXACT MATCH")
    print("Gate-81 combined binding replay: EXACT MATCH")

    # ------------------------------------------------------------------
    # Gate C - verify Gate-81 scientific geometry.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE C - VERIFY GATE-81 SCIENTIFIC GEOMETRY")
    print("=" * 60)
    print("")

    require_equal(
        len(scenario_rows),
        4,
        "Scenario-plan row count",
    )

    require_equal(
        {
            row[
                "ScenarioID"
            ]
            for row
            in scenario_rows
        },
        EXPECTED_ALL_SCENARIO_IDS,
        "Scenario ID set",
    )

    main_scenario_rows = [
        row
        for row
        in scenario_rows
        if str(
            row[
                "MainCampaign"
            ]
        ).strip().lower()
        in {
            "true",
            "1",
            "yes",
        }
    ]

    require_equal(
        len(main_scenario_rows),
        EXPECTED_MAIN_SCENARIOS,
        "Main scenario count",
    )

    require_equal(
        {
            row[
                "ScenarioID"
            ]
            for row
            in main_scenario_rows
        },
        EXPECTED_MAIN_SCENARIO_IDS,
        "Main scenario ID set",
    )

    require_equal(
        len(global_init_rows),
        EXPECTED_GLOBAL_INIT_ROWS,
        "Global-init seed row count",
    )

    require_equal(
        len(local_order_seed_rows),
        EXPECTED_LOCAL_ORDER_SEED_ROWS,
        "Local-order seed row count",
    )

    require_equal(
        len(local_train_rng_rows),
        EXPECTED_LOCAL_TRAIN_RNG_ROWS,
        "Local-train RNG row count",
    )

    require_equal(
        len(campaign_rows),
        EXPECTED_MAIN_RUNS,
        "Main campaign run count",
    )

    require_equal(
        len(
            {
                (
                    row[
                        "ConfigID"
                    ],
                    row[
                        "MethodID"
                    ],
                    row[
                        "ScenarioID"
                    ],
                )
                for row
                in campaign_rows
            }
        ),
        EXPECTED_MAIN_RUNS,
        "Unique campaign triple count",
    )

    require_equal(
        len(
            {
                row[
                    "ConfigID"
                ]
                for row
                in campaign_rows
            }
        ),
        EXPECTED_CONFIGS,
        "Campaign configuration count",
    )

    require_equal(
        {
            row[
                "MethodID"
            ]
            for row
            in campaign_rows
        },
        EXPECTED_METHOD_IDS,
        "Campaign method ID set",
    )

    require_equal(
        {
            row[
                "ScenarioID"
            ]
            for row
            in campaign_rows
        },
        EXPECTED_MAIN_SCENARIO_IDS,
        "Campaign scenario ID set",
    )

    print("Main scenarios: CLEAN / MU0P2 / MU0P4")
    print("Diagnostic-only scenario: MU0P1")
    print("Global-init seed families: 5")
    print("Local-order seeds: 300")
    print("Local-train RNG event seeds: 900")
    print("Main campaign rows: 180")
    print("Methods represented: 6")
    print("Method-specific parameter rebinding performed: NO")

    # ------------------------------------------------------------------
    # Gate D - verify and replay Gate-82 execution details.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE D - VERIFY / REPLAY GATE-82 EXECUTION DETAILS")
    print("=" * 60)
    print("")

    gate82 = load_json(
        args.gate82_diagnostic_json
    )

    require_equal(
        gate82.get("status"),
        "DIAGNOSIS COMPLETE",
        "Gate-82 status",
    )

    require_equal(
        gate82.get("diagnostic_id"),
        GATE82_DIAGNOSTIC_ID,
        "Gate-82 diagnostic ID",
    )

    require_equal(
        gate82.get("disposition"),
        GATE82_DISPOSITION,
        "Gate-82 disposition",
    )

    gate82_fp = gate82[
        "candidate_exact_execution_details_not_frozen"
    ]

    require_equal(
        gate82_fp[
            "local_order_algorithm_sha256"
        ],
        LOCAL_ORDER_ALGORITHM_SHA,
        "Gate-82 local-order algorithm SHA256",
    )

    require_equal(
        gate82_fp[
            "block_boundary_set_sha256"
        ],
        BLOCK_BOUNDARY_SET_SHA,
        "Gate-82 block-boundary set SHA256",
    )

    require_equal(
        gate82_fp[
            "round_execution_set_sha256"
        ],
        ROUND_EXECUTION_SET_SHA,
        "Gate-82 round-execution set SHA256",
    )

    require_equal(
        gate82_fp[
            "checkpoint_comparator_sha256"
        ],
        CHECKPOINT_COMPARATOR_SHA,
        "Gate-82 checkpoint comparator SHA256",
    )

    require_equal(
        gate82_fp[
            "combined_execution_detail_sha256"
        ],
        GATE82_COMBINED_EXECUTION_DETAIL_SHA,
        "Gate-82 combined execution-detail SHA256",
    )

    require_true(
        gate82[
            "scientific_boundary"
        ][
            "scientific_training_started"
        ]
        is False,
        "Gate-82 says scientific training started.",
    )

    algorithm_text = canonical_text_from_file(
        args.gate82_algorithm_txt
    )

    comparator_text = canonical_text_from_file(
        args.gate82_comparator_txt
    )

    block_rows = read_csv(
        args.gate82_block_csv
    )

    round_execution_rows = read_csv(
        args.gate82_round_execution_csv
    )

    rng_application_rows = read_csv(
        args.gate82_rng_application_csv
    )

    require_equal(
        sha256_text(
            algorithm_text
        ),
        LOCAL_ORDER_ALGORITHM_SHA,
        "Local-order algorithm SHA256 replay",
    )

    require_equal(
        block_boundary_sha(
            block_rows
        ),
        BLOCK_BOUNDARY_SET_SHA,
        "Block-boundary set SHA256 replay",
    )

    require_equal(
        round_execution_sha(
            round_execution_rows
        ),
        ROUND_EXECUTION_SET_SHA,
        "Round-execution set SHA256 replay",
    )

    require_equal(
        sha256_text(
            comparator_text
        ),
        CHECKPOINT_COMPARATOR_SHA,
        "Checkpoint comparator SHA256 replay",
    )

    require_equal(
        combined_gate82_execution_sha(
            LOCAL_ORDER_ALGORITHM_SHA,
            BLOCK_BOUNDARY_SET_SHA,
            ROUND_EXECUTION_SET_SHA,
            CHECKPOINT_COMPARATOR_SHA,
        ),
        GATE82_COMBINED_EXECUTION_DETAIL_SHA,
        "Gate-82 combined execution-detail replay",
    )

    require_equal(
        len(block_rows),
        EXPECTED_BLOCK_ROWS,
        "Block-boundary row count",
    )

    require_equal(
        len(round_execution_rows),
        EXPECTED_ROUND_EXECUTION_ROWS,
        "Round-execution row count",
    )

    require_equal(
        len(rng_application_rows),
        EXPECTED_RNG_APPLICATION_ROWS,
        "RNG-application row count",
    )

    print("Local-order formula fingerprint replay: EXACT MATCH")
    print("Block-boundary set replay: EXACT MATCH")
    print("Round-execution set replay: EXACT MATCH")
    print("Checkpoint comparator replay: EXACT MATCH")
    print("Gate-82 combined execution-detail replay: EXACT MATCH")

    # ------------------------------------------------------------------
    # Gate E - compute final common-protocol binding.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE E - COMPUTE FINAL COMMON-PROTOCOL BINDING")
    print("=" * 60)
    print("")

    common_binding_sha = common_protocol_binding_sha(
        LOCAL_ORDER_ALGORITHM_SHA,
        BLOCK_BOUNDARY_SET_SHA,
        ROUND_EXECUTION_SET_SHA,
        CHECKPOINT_COMPARATOR_SHA,
    )

    print(
        f"Common protocol binding SHA256: {common_binding_sha}"
    )

    # ------------------------------------------------------------------
    # Gate F - freeze exact scientific artifacts.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE F - WRITE FROZEN SCIENTIFIC ARTIFACTS")
    print("=" * 60)
    print("")

    frozen_files = [
        (
            "SCENARIO_PLAN",
            Path(args.gate81_scenario_csv),
            output_root
            /
            "FROZEN_SCENARIO_PLAN.csv",
        ),
        (
            "ATTACK_EXPOSURE_REPLAY",
            Path(args.gate81_exposure_csv),
            output_root
            /
            "FROZEN_ATTACK_EXPOSURE_REPLAY.csv",
        ),
        (
            "GLOBAL_INIT_SEEDS",
            Path(args.gate81_global_init_csv),
            output_root
            /
            "FROZEN_GLOBAL_INIT_SEEDS.csv",
        ),
        (
            "LOCAL_ORDER_SEEDS",
            Path(args.gate81_local_order_seeds_csv),
            output_root
            /
            "FROZEN_LOCAL_ORDER_SEEDS.csv",
        ),
        (
            "LOCAL_TRAIN_RNG_SEEDS",
            Path(args.gate81_local_train_rng_csv),
            output_root
            /
            "FROZEN_LOCAL_TRAIN_RNG_SEEDS.csv",
        ),
        (
            "MAIN_CAMPAIGN_MATRIX",
            Path(args.gate81_campaign_csv),
            output_root
            /
            "FROZEN_MAIN_CAMPAIGN_MATRIX.csv",
        ),
        (
            "EXACT_LOCAL_SAMPLE_ORDER_ALGORITHM",
            Path(args.gate82_algorithm_txt),
            output_root
            /
            "FROZEN_EXACT_LOCAL_SAMPLE_ORDER_ALGORITHM.txt",
        ),
        (
            "LOCAL_PARTICIPATION_BLOCK_BOUNDARIES",
            Path(args.gate82_block_csv),
            output_root
            /
            "FROZEN_LOCAL_PARTICIPATION_BLOCK_BOUNDARIES.csv",
        ),
        (
            "ROUND_CLIENT_EXECUTION_ORDER",
            Path(args.gate82_round_execution_csv),
            output_root
            /
            "FROZEN_ROUND_CLIENT_EXECUTION_ORDER.csv",
        ),
        (
            "LOCAL_RNG_APPLICATION",
            Path(args.gate82_rng_application_csv),
            output_root
            /
            "FROZEN_LOCAL_RNG_APPLICATION.csv",
        ),
        (
            "EXACT_CHECKPOINT_COMPARATOR",
            Path(args.gate82_comparator_txt),
            output_root
            /
            "FROZEN_EXACT_CHECKPOINT_COMPARATOR.txt",
        ),
        (
            "LOCAL_ORDER_TECHNICAL_PROBES",
            Path(args.gate82_probe_csv),
            output_root
            /
            "LOCAL_ORDER_TECHNICAL_PROBES.csv",
        ),
    ]

    for _role, source, destination in frozen_files:
        copy_bytes(
            source,
            destination,
        )

    # Final protocol JSON.
    attack_frozen = dict(
        attack_candidate
    )

    attack_frozen[
        "status"
    ] = "FROZEN"

    attack_frozen[
        "attack_protocol_id"
    ] = ATTACK_PROTOCOL_ID

    attack_frozen.pop(
        "candidate_attack_protocol_id",
        None,
    )

    training_frozen = dict(
        training_candidate
    )

    training_frozen[
        "status"
    ] = "FROZEN"

    training_frozen[
        "core_training_protocol_id"
    ] = CORE_TRAINING_PROTOCOL_ID

    training_frozen[
        "seed_protocol_id"
    ] = SEED_PROTOCOL_ID

    training_frozen[
        "local_sample_order_algorithm_id"
    ] = LOCAL_SAMPLE_ORDER_ALGORITHM_ID

    training_frozen[
        "local_block_allocation_id"
    ] = LOCAL_BLOCK_ALLOCATION_ID

    training_frozen[
        "round_client_execution_id"
    ] = ROUND_CLIENT_EXECUTION_ID

    training_frozen[
        "local_rng_application_id"
    ] = LOCAL_RNG_APPLICATION_ID

    training_frozen[
        "checkpoint_comparator_id"
    ] = CHECKPOINT_COMPARATOR_ID

    training_frozen.pop(
        "candidate_core_training_protocol_id",
        None,
    )

    training_frozen.pop(
        "candidate_seed_protocol_id",
        None,
    )

    training_frozen.pop(
        "candidate_local_order_id",
        None,
    )

    final_protocol = {
        "status": (
            "FROZEN"
        ),
        "freeze_id": (
            FREEZE_ID
        ),
        "common_protocol_id": (
            COMMON_PROTOCOL_ID
        ),
        "attack_protocol_id": (
            ATTACK_PROTOCOL_ID
        ),
        "core_training_protocol_id": (
            CORE_TRAINING_PROTOCOL_ID
        ),
        "seed_protocol_id": (
            SEED_PROTOCOL_ID
        ),
        "local_sample_order_algorithm_id": (
            LOCAL_SAMPLE_ORDER_ALGORITHM_ID
        ),
        "local_block_allocation_id": (
            LOCAL_BLOCK_ALLOCATION_ID
        ),
        "round_client_execution_id": (
            ROUND_CLIENT_EXECUTION_ID
        ),
        "local_rng_application_id": (
            LOCAL_RNG_APPLICATION_ID
        ),
        "checkpoint_comparator_id": (
            CHECKPOINT_COMPARATOR_ID
        ),
        "main_campaign_set_id": (
            MAIN_CAMPAIGN_SET_ID
        ),
        "immutable_binding": {
            "effective_dataset_fingerprint_sha256": (
                EFFECTIVE_DATASET_SHA
            ),
            "feature_layer_fingerprint_sha256": (
                FEATURE_LAYER_SHA
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
            "assignment_set_content_sha256": (
                ASSIGNMENT_SET_CONTENT_SHA
            ),
            "physical_assignment_audit_artifact_manifest_sha256": (
                PHYSICAL_ASSIGNMENT_AUDIT_SHA
            ),
            "combined_preattack_identity_sha256": (
                PREATTACK_IDENTITY_SHA
            ),
            "participation_malicious_identity_artifact_manifest_sha256": (
                PARTICIPATION_MALICIOUS_IDENTITY_ARTIFACT_SHA
            ),
        },
        "frozen_scientific_fingerprints": {
            "scenario_plan_sha256": (
                SCENARIO_PLAN_SHA
            ),
            "global_init_seed_set_sha256": (
                GLOBAL_INIT_SEED_SET_SHA
            ),
            "local_order_seed_set_sha256": (
                LOCAL_ORDER_SEED_SET_SHA
            ),
            "local_train_rng_seed_set_sha256": (
                LOCAL_TRAIN_RNG_SEED_SET_SHA
            ),
            "campaign_matrix_sha256": (
                CAMPAIGN_MATRIX_SHA
            ),
            "local_order_algorithm_sha256": (
                LOCAL_ORDER_ALGORITHM_SHA
            ),
            "block_boundary_set_sha256": (
                BLOCK_BOUNDARY_SET_SHA
            ),
            "round_execution_set_sha256": (
                ROUND_EXECUTION_SET_SHA
            ),
            "checkpoint_comparator_sha256": (
                CHECKPOINT_COMPARATOR_SHA
            ),
            "gate81_combined_candidate_binding_sha256": (
                GATE81_COMBINED_BINDING_SHA
            ),
            "gate82_combined_execution_detail_sha256": (
                GATE82_COMBINED_EXECUTION_DETAIL_SHA
            ),
            "common_protocol_binding_hash_id": (
                COMMON_PROTOCOL_BINDING_HASH_ID
            ),
            "common_protocol_binding_sha256": (
                common_binding_sha
            ),
        },
        "frozen_attack_protocol": (
            attack_frozen
        ),
        "frozen_core_training_protocol": (
            training_frozen
        ),
        "frozen_execution_details": {
            "local_order": {
                "formula_artifact": (
                    "FROZEN_EXACT_LOCAL_SAMPLE_ORDER_ALGORITHM.txt"
                ),
                "formula_sha256": (
                    LOCAL_ORDER_ALGORITHM_SHA
                ),
            },
            "six_disjoint_blocks": {
                "boundary_artifact": (
                    "FROZEN_LOCAL_PARTICIPATION_BLOCK_BOUNDARIES.csv"
                ),
                "boundary_set_sha256": (
                    BLOCK_BOUNDARY_SET_SHA
                ),
                "rows": (
                    EXPECTED_BLOCK_ROWS
                ),
                "block_size": (
                    51_200
                ),
                "participation_ordinals": (
                    "0_TO_5"
                ),
                "replacement": (
                    False
                ),
                "wrap": (
                    False
                ),
            },
            "round_client_execution": {
                "artifact": (
                    "FROZEN_ROUND_CLIENT_EXECUTION_ORDER.csv"
                ),
                "set_sha256": (
                    ROUND_EXECUTION_SET_SHA
                ),
                "rule": (
                    "FROZEN_SCHEDULE_SLOT_ASCENDING"
                ),
            },
            "local_rng_application": {
                "artifact": (
                    "FROZEN_LOCAL_RNG_APPLICATION.csv"
                ),
                "torch_seed_mapping": (
                    "SeedUInt64 mod (2^63 - 1)"
                ),
                "application_point": (
                    "IMMEDIATELY_BEFORE_FIRST_STOCHASTIC_LOCAL_TRAINING_OPERATION"
                ),
                "dataloader_shuffle": (
                    False
                ),
                "dataloader_workers": (
                    0
                ),
            },
            "checkpoint_comparator": {
                "artifact": (
                    "FROZEN_EXACT_CHECKPOINT_COMPARATOR.txt"
                ),
                "sha256": (
                    CHECKPOINT_COMPARATOR_SHA
                ),
                "rule": (
                    "MAX_LEXICOGRAPHIC_MACROF1_BALACC_NEGROUND_FLOAT64_NO_TOLERANCE"
                ),
                "nonfinite_metrics_prohibited": (
                    True
                ),
            },
        },
        "frozen_main_campaign": {
            "configuration_count": (
                EXPECTED_CONFIGS
            ),
            "method_count": (
                EXPECTED_METHODS
            ),
            "scenario_count": (
                EXPECTED_MAIN_SCENARIOS
            ),
            "run_count": (
                EXPECTED_MAIN_RUNS
            ),
            "method_ids": (
                sorted(
                    EXPECTED_METHOD_IDS
                )
            ),
            "scenario_ids": (
                sorted(
                    EXPECTED_MAIN_SCENARIO_IDS
                )
            ),
            "campaign_matrix_artifact": (
                "FROZEN_MAIN_CAMPAIGN_MATRIX.csv"
            ),
            "campaign_matrix_sha256": (
                CAMPAIGN_MATRIX_SHA
            ),
        },
        "scientific_boundary": {
            "common_attack_protocol_frozen": (
                True
            ),
            "common_core_training_protocol_frozen": (
                True
            ),
            "deterministic_seed_protocol_frozen": (
                True
            ),
            "exact_execution_detail_protocol_frozen": (
                True
            ),
            "main_campaign_matrix_frozen": (
                True
            ),
            "method_specific_algorithm_parameters_rebound": (
                False
            ),
            "end_to_end_method_validation_started": (
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
        "FROZEN_ATTACK_TRAINING_EXECUTION_PROTOCOL.json"
    )

    write_json(
        protocol_path,
        final_protocol,
    )

    print("Frozen scenario plan: WRITTEN")
    print("Frozen seed hierarchy: WRITTEN")
    print("Frozen exact local-order / block / execution rules: WRITTEN")
    print("Frozen checkpoint comparator: WRITTEN")
    print("Frozen 180-run campaign matrix: WRITTEN")

    # ------------------------------------------------------------------
    # Gate G - write artifact manifest and freeze state.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE G - WRITE IMMUTABLE FREEZE MANIFEST")
    print("=" * 60)
    print("")

    artifact_rows = []

    for role, _source, destination in frozen_files:
        artifact_rows.append({
            "ArtifactRole": (
                role
            ),
            "FileName": (
                destination.name
            ),
            "SizeBytes": (
                destination.stat().st_size
            ),
            "SHA256": (
                sha256_file(
                    destination
                )
            ),
        })

    artifact_rows.append({
        "ArtifactRole": (
            "FROZEN_COMMON_PROTOCOL"
        ),
        "FileName": (
            protocol_path.name
        ),
        "SizeBytes": (
            protocol_path.stat().st_size
        ),
        "SHA256": (
            sha256_file(
                protocol_path
            )
        ),
    })

    artifact_manifest_path = (
        output_root
        /
        "FROZEN_ATTACK_TRAINING_EXECUTION_ARTIFACT_MANIFEST.csv"
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

    combined_artifact_manifest_sha = artifact_manifest_digest(
        artifact_rows
    )

    freeze_state = {
        "status": (
            "FROZEN"
        ),
        "freeze_id": (
            FREEZE_ID
        ),
        "common_protocol_id": (
            COMMON_PROTOCOL_ID
        ),
        "common_protocol_binding_sha256": (
            common_binding_sha
        ),
        "frozen_attack_training_execution_artifact_manifest_sha256": (
            combined_artifact_manifest_sha
        ),
        "scientific_boundary": {
            "common_attack_protocol_frozen": (
                True
            ),
            "common_core_training_protocol_frozen": (
                True
            ),
            "deterministic_seed_protocol_frozen": (
                True
            ),
            "exact_execution_detail_protocol_frozen": (
                True
            ),
            "main_campaign_matrix_frozen": (
                True
            ),
            "method_specific_algorithm_parameters_rebound": (
                False
            ),
            "end_to_end_method_validation_started": (
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

    freeze_state_path = (
        output_root
        /
        "ATTACK_TRAINING_EXECUTION_PROTOCOL_FREEZE.json"
    )

    write_json(
        freeze_state_path,
        freeze_state,
    )

    evidence_paths = {
        "GATE81_DIAGNOSTIC": Path(
            args.gate81_diagnostic_json
        ),
        "GATE81_ATTACK_CANDIDATE": Path(
            args.gate81_attack_json
        ),
        "GATE81_TRAINING_CANDIDATE": Path(
            args.gate81_training_json
        ),
        "GATE82_DIAGNOSTIC": Path(
            args.gate82_diagnostic_json
        ),
        "GATE80_FREEZE": Path(
            args.gate80_freeze_json
        ),
        "GATE78_AUDIT": Path(
            args.gate78_audit_json
        ),
        "GATE73_FREEZE": Path(
            args.gate73_freeze_json
        ),
    }

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
        "CICIoT2023 ATTACK / TRAINING / EXECUTION COMMON PROTOCOL FREEZE"
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
        f"Common protocol ID: {COMMON_PROTOCOL_ID}"
    )
    add(
        f"Attack protocol ID: {ATTACK_PROTOCOL_ID}"
    )
    add(
        f"Core training protocol ID: {CORE_TRAINING_PROTOCOL_ID}"
    )
    add(
        f"Seed protocol ID: {SEED_PROTOCOL_ID}"
    )
    add(
        f"Main campaign set ID: {MAIN_CAMPAIGN_SET_ID}"
    )
    add("")

    add("IMMUTABLE BRANCH BINDING")
    add("-" * 78)
    add(
        f"Effective dataset fingerprint SHA256: {EFFECTIVE_DATASET_SHA}"
    )
    add(
        f"Feature-layer fingerprint SHA256: {FEATURE_LAYER_SHA}"
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
    add(
        f"Assignment-set content SHA256: {ASSIGNMENT_SET_CONTENT_SHA}"
    )
    add(
        f"Physical-assignment audit artifact-manifest SHA256: "
        f"{PHYSICAL_ASSIGNMENT_AUDIT_SHA}"
    )
    add(
        f"Combined pre-attack identity SHA256: {PREATTACK_IDENTITY_SHA}"
    )
    add(
        f"Participation/malicious-identity artifact-manifest SHA256: "
        f"{PARTICIPATION_MALICIOUS_IDENTITY_ARTIFACT_SHA}"
    )
    add("")

    add("FROZEN ATTACK PROTOCOL")
    add("-" * 78)
    add(
        "Main scenarios: CLEAN, SIGNFLIP_MU0P2, SIGNFLIP_MU0P4"
    )
    add(
        "Diagnostic-only reference: SIGNFLIP_MU0P1"
    )
    add(
        "Honest delta: Delta_i = w_i_local - w_t"
    )
    add(
        "Malicious delta: Delta_i_tilde = -5 * Delta_i"
    )
    add(
        "Application point: after honest local training, before aggregation"
    )
    add(
        "Warm-up rounds: 1-4"
    )
    add(
        "Attack-active rounds: 5-20"
    )
    add(
        "Server blind to malicious identity: YES"
    )
    add(
        f"Scenario-plan SHA256: {SCENARIO_PLAN_SHA}"
    )
    add("")

    add("FROZEN CORE TRAINING PROTOCOL")
    add("-" * 78)
    add(
        "Model: 39 -> 256 -> 128 -> 7"
    )
    add(
        "Activation: ReLU"
    )
    add(
        "Dropout: 0.2"
    )
    add(
        "Trainable parameters: 44039"
    )
    add(
        "Optimizer: Adam, lr=0.001, betas=(0.9,0.999), eps=1e-8"
    )
    add(
        "Fresh optimizer per selected client participation: YES"
    )
    add(
        "Batch size: 256"
    )
    add(
        "Local optimizer steps per participation: 200"
    )
    add(
        "Samples per participation: 51200"
    )
    add(
        "Participations per client: 6"
    )
    add(
        "Samples per client per run: 307200"
    )
    add(
        "Primary loss: frozen global TASK7 weighted cross-entropy"
    )
    add(
        "Secondary sensitivity: NATURAL_UNWEIGHTED_CE"
    )
    add(
        "CPU deterministic primary: YES"
    )
    add("")

    add("FROZEN DETERMINISTIC SEED HIERARCHY")
    add("-" * 78)
    add(
        f"Global-init seed-set SHA256: {GLOBAL_INIT_SEED_SET_SHA}"
    )
    add(
        f"Local-order seed-set SHA256: {LOCAL_ORDER_SEED_SET_SHA}"
    )
    add(
        f"Local-train RNG seed-set SHA256: {LOCAL_TRAIN_RNG_SEED_SET_SHA}"
    )
    add(
        "Global init shared across alpha pair, methods, and scenarios: YES"
    )
    add(
        "Local-train RNG shared across alpha pair, methods, and scenarios: YES"
    )
    add(
        "Local-order seed configuration/client-specific: YES"
    )
    add("")

    add("FROZEN EXACT EXECUTION DETAILS")
    add("-" * 78)
    add(
        f"Local sample-order algorithm ID: {LOCAL_SAMPLE_ORDER_ALGORITHM_ID}"
    )
    add(
        f"Local-order algorithm SHA256: {LOCAL_ORDER_ALGORITHM_SHA}"
    )
    add(
        f"Local block allocation ID: {LOCAL_BLOCK_ALLOCATION_ID}"
    )
    add(
        f"Block-boundary set SHA256: {BLOCK_BOUNDARY_SET_SHA}"
    )
    add(
        f"Round client execution ID: {ROUND_CLIENT_EXECUTION_ID}"
    )
    add(
        f"Round-execution set SHA256: {ROUND_EXECUTION_SET_SHA}"
    )
    add(
        f"Local RNG application ID: {LOCAL_RNG_APPLICATION_ID}"
    )
    add(
        "DataLoader shuffle: NO"
    )
    add(
        "DataLoader workers: 0"
    )
    add(
        f"Checkpoint comparator ID: {CHECKPOINT_COMPARATOR_ID}"
    )
    add(
        f"Checkpoint comparator SHA256: {CHECKPOINT_COMPARATOR_SHA}"
    )
    add(
        "Checkpoint rule: max lexicographic "
        "(MacroF1, BalancedAccuracy, -Round)"
    )
    add(
        "NaN / +/-Inf metrics prohibited: YES"
    )
    add("")

    add("FROZEN MAIN CAMPAIGN MATRIX")
    add("-" * 78)
    add(
        "Configurations: 10"
    )
    add(
        "Methods: 6"
    )
    add(
        "Scenarios: 3"
    )
    add(
        "Main scientific runs: 180"
    )
    add(
        f"Campaign-matrix SHA256: {CAMPAIGN_MATRIX_SHA}"
    )
    add("")

    add("FROZEN SCIENTIFIC FINGERPRINTS")
    add("-" * 78)
    add(
        f"Gate-81 combined candidate binding SHA256: "
        f"{GATE81_COMBINED_BINDING_SHA}"
    )
    add(
        f"Gate-82 combined execution-detail SHA256: "
        f"{GATE82_COMBINED_EXECUTION_DETAIL_SHA}"
    )
    add(
        f"Common protocol binding SHA256: {common_binding_sha}"
    )
    add("")

    add("IMMUTABLE FREEZE ARTIFACT")
    add("-" * 78)
    add(
        f"Combined attack/training/execution artifact-manifest SHA256: "
        f"{combined_artifact_manifest_sha}"
    )
    add("")

    add("SCIENTIFIC BOUNDARY")
    add("-" * 78)
    add(
        "Common attack protocol frozen: YES"
    )
    add(
        "Common core training protocol frozen: YES"
    )
    add(
        "Deterministic seed protocol frozen: YES"
    )
    add(
        "Exact execution-detail protocol frozen: YES"
    )
    add(
        "Main campaign matrix frozen: YES"
    )
    add(
        "Method-specific algorithm parameters rebound: NO"
    )
    add(
        "End-to-end method validation started: NO"
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
        "Rebind the six method implementations to this frozen common protocol. "
        "Audit each method's exact aggregation/update semantics and fixed "
        "parameters, then execute a minimal end-to-end technical validation "
        "before any scientific campaign run."
    )

    (
        output_root
        /
        "ATTACK_TRAINING_EXECUTION_PROTOCOL_FREEZE_REPORT.txt"
    ).write_bytes(
        (
            "\n".join(report)
            +
            "\n"
        ).encode("utf-8")
    )

    print("")
    print("=" * 60)
    print("STATUS: FROZEN")
    print(
        f"COMMON PROTOCOL BINDING SHA256: "
        f"{common_binding_sha}"
    )
    print(
        f"COMBINED FREEZE ARTIFACT MANIFEST: "
        f"{combined_artifact_manifest_sha}"
    )
    print(
        "METHOD-SPECIFIC PARAMETERS REBOUND: NO"
    )
    print(
        "END-TO-END METHOD VALIDATION STARTED: NO"
    )
    print(
        "SCIENTIFIC TRAINING STARTED: NO"
    )
    print("=" * 60)
    print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
