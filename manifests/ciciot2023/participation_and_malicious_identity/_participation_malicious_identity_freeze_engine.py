import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path


# ============================================================================
# Immutable bindings from the rebuilt branch
# ============================================================================

ASSIGNMENT_SET_CONTENT_SHA = (
    "6161DDCA8874F3A079B3D61C19C62A0C1A60389ECFF2B2028E043FFC38B36E98"
)

ASSIGNMENT_ARTIFACT_MANIFEST_SHA = (
    "61A3B010B58CB6BEF199C6EA521D6014D68310DB95B867C66B2183226CA2EC5D"
)

PHYSICAL_ASSIGNMENT_AUDIT_SHA = (
    "E04148B061539804DF6928544707E47DFB84A3FCAE5544D55428719569DA67D5"
)


# ============================================================================
# Gate-79 accepted evidence
# ============================================================================

GATE79_DIAGNOSTIC_ID = (
    "CICIoT2023_TRANSFORMED_UNIQUE_PARTICIPATION_"
    "MALICIOUS_IDENTITY_DIAGNOSTIC_V1"
)

GATE79_DISPOSITION = (
    "SEED_PAIRED_SIX_EPOCH_PARTICIPATION_AND_NESTED_HASH_RANKING_"
    "REVALIDATED_FREEZE_REQUIRED"
)

EXPECTED_CONFIG_MAPPING_SHA = (
    "444E14F821630F3814E939A3B9A0F493B86D7E25BA245B8BDCDD7EE32A77FF9E"
)

EXPECTED_SCHEDULE_SET_SHA = (
    "072DE9493641E92D8A450A61E02E5C211AC818F80309041CD7F4CA5C6D50CE66"
)

EXPECTED_RANKING_SET_SHA = (
    "65ADA6C60BCF1D5E7FCCF4A3874BD296AEAF1CF69B2D3C50087D5BC449CE491D"
)


# ============================================================================
# Freeze identifiers
# ============================================================================

FREEZE_ID = (
    "CICIoT2023_TRANSFORMED_UNIQUE_PARTICIPATION_"
    "MALICIOUS_IDENTITY_FREEZE_V2"
)

PARTICIPATION_PROTOCOL_ID = (
    "CICIoT2023_K30_R20_M9_SEED_PAIRED_SIX_EPOCH_"
    "EXACT_EXPOSURE_V2"
)

MALICIOUS_IDENTITY_ID = (
    "CICIoT2023_K30_SEED_PAIRED_FIXED_HASH_RANK_"
    "NESTED_PREFIX_V2"
)

PAIRING_POLICY_ID = (
    "SAME_EXPERIMENTAL_SEED_SHARED_ACROSS_ALPHA_0P1_AND_1P0_V1"
)

CROSS_METHOD_SCENARIO_PAIRING_ID = (
    "SAME_SEED_SCHEDULE_AND_RANKING_SHARED_ACROSS_METHODS_AND_SCENARIOS_V1"
)

PARTICIPATION_ALGORITHM_ID = (
    "SIX_EPOCH_HASH_PERMUTATION_WITH_DETERMINISTIC_BOUNDARY_REPAIR_V2"
)

MALICIOUS_RANKING_ALGORITHM_ID = (
    "HASH_RANK_FIXED_CLIENTS_NESTED_PREFIX_SEED_PAIRED_V2"
)

CONFIG_MAPPING_HASH_ID = (
    "SORTED_CONFIG_ID_PLUS_EXPERIMENTAL_SEED_LF_V1"
)

SCHEDULE_SET_HASH_ID = (
    "SORTED_EXPERIMENTAL_SEED_PLUS_SCHEDULE_SHA256_LF_V1"
)

RANKING_SET_HASH_ID = (
    "SORTED_EXPERIMENTAL_SEED_PLUS_RANKING_SHA256_LF_V1"
)

COMBINED_PREATTACK_IDENTITY_HASH_ID = (
    "CONFIG_MAPPING_PLUS_SCHEDULE_SET_PLUS_RANKING_SET_SHA256_LF_V1"
)


# ============================================================================
# Geometry
# ============================================================================

CLIENTS = 30
ROUNDS = 20
SELECTED_PER_ROUND = 9
TOTAL_SELECTIONS = 180
EXPOSURES_PER_CLIENT = 6

EXPERIMENTAL_SEEDS = (
    42,
    123,
    456,
    789,
    999,
)

MALICIOUS_PREFIXES = {
    0.1: 3,
    0.2: 6,
    0.4: 12,
}

REFERENCE_ATTACK_START_ROUND = 5
REFERENCE_ATTACK_END_ROUND = 20

RANDOM_ADJACENT_OVERLAP_REFERENCE = (
    SELECTED_PER_ROUND
    *
    SELECTED_PER_ROUND
    /
    CLIENTS
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


def sha256_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest().upper()


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


def canonical_set_sha(rows, key_name, sha_name):
    digest = hashlib.sha256()

    for row in sorted(
        rows,
        key=lambda item: int(
            item[key_name]
        ),
    ):
        digest.update(
            (
                f"{row[key_name]}\t"
                f"{row[sha_name]}\n"
            ).encode("utf-8")
        )

    return digest.hexdigest().upper()


def config_mapping_sha(rows):
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
                f"{row['ExperimentalSeed']}\n"
            ).encode("utf-8")
        )

    return digest.hexdigest().upper()


def combined_preattack_identity_sha(
    config_mapping_sha256,
    schedule_set_sha256,
    ranking_set_sha256,
):
    text = (
        f"CONFIG_MAPPING_SHA256={config_mapping_sha256}\n"
        f"SCHEDULE_SET_SHA256={schedule_set_sha256}\n"
        f"RANKING_SET_SHA256={ranking_set_sha256}\n"
    )

    return sha256_text(
        text
    )


def parse_bool(value):
    return str(
        value
    ).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def schedule_sha_for_seed(
    rows,
    experimental_seed,
):
    selected = [
        row
        for row
        in rows
        if int(
            row[
                "ExperimentalSeed"
            ]
        )
        ==
        experimental_seed
    ]

    selected.sort(
        key=lambda row: (
            int(
                row[
                    "Round"
                ]
            ),
            int(
                row[
                    "Slot"
                ]
            ),
        )
    )

    require_equal(
        len(
            selected
        ),
        TOTAL_SELECTIONS,
        (
            "Schedule row count seed "
            f"{experimental_seed}"
        ),
    )

    text = "".join(
        (
            f"{experimental_seed}\t"
            f"{int(row['Round'])}\t"
            f"{int(row['Slot'])}\t"
            f"{int(row['ClientID'])}\n"
        )
        for row
        in selected
    )

    return sha256_text(
        text
    )


def ranking_sha_for_seed(
    rows,
    experimental_seed,
):
    selected = [
        row
        for row
        in rows
        if int(
            row[
                "ExperimentalSeed"
            ]
        )
        ==
        experimental_seed
    ]

    selected.sort(
        key=lambda row: int(
            row[
                "RankPosition"
            ]
        )
    )

    require_equal(
        len(
            selected
        ),
        CLIENTS,
        (
            "Ranking row count seed "
            f"{experimental_seed}"
        ),
    )

    text = "".join(
        (
            f"{experimental_seed}\t"
            f"{int(row['RankPosition'])}\t"
            f"{int(row['ClientID'])}\n"
        )
        for row
        in selected
    )

    return sha256_text(
        text
    )


def schedule_rounds(
    schedule_rows,
    experimental_seed,
):
    rows = [
        row
        for row
        in schedule_rows
        if int(
            row[
                "ExperimentalSeed"
            ]
        )
        ==
        experimental_seed
    ]

    rounds = []

    for round_id in range(
        1,
        ROUNDS
        +
        1,
    ):
        selected = [
            row
            for row
            in rows
            if int(
                row[
                    "Round"
                ]
            )
            ==
            round_id
        ]

        selected.sort(
            key=lambda row: int(
                row[
                    "Slot"
                ]
            )
        )

        require_equal(
            len(
                selected
            ),
            SELECTED_PER_ROUND,
            (
                f"Selected client count seed "
                f"{experimental_seed}, round {round_id}"
            ),
        )

        clients = [
            int(
                row[
                    "ClientID"
                ]
            )
            for row
            in selected
        ]

        require_equal(
            len(
                set(
                    clients
                )
            ),
            SELECTED_PER_ROUND,
            (
                f"Unique client count seed "
                f"{experimental_seed}, round {round_id}"
            ),
        )

        rounds.append(
            clients
        )

    return rounds


def maximum_consecutive_absence(
    rounds,
):
    maximum = 0

    for client_id in range(
        CLIENTS
    ):
        streak = 0
        client_maximum = 0

        for selected in rounds:
            if client_id in selected:
                streak = 0
            else:
                streak += 1

                client_maximum = max(
                    client_maximum,
                    streak,
                )

        maximum = max(
            maximum,
            client_maximum,
        )

    return maximum


def malicious_exposure(
    rounds,
    ranking,
    prefix_size,
):
    malicious = set(
        ranking[
            :prefix_size
        ]
    )

    counts = [
        len(
            malicious
            &
            set(
                rounds[
                    round_id
                    -
                    1
                ]
            )
        )
        for round_id
        in range(
            REFERENCE_ATTACK_START_ROUND,
            REFERENCE_ATTACK_END_ROUND
            +
            1,
        )
    ]

    return counts


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gate79-diagnostic-json",
        required=True,
    )

    parser.add_argument(
        "--gate79-mapping-csv",
        required=True,
    )

    parser.add_argument(
        "--gate79-schedule-csv",
        required=True,
    )

    parser.add_argument(
        "--gate79-schedule-summary-csv",
        required=True,
    )

    parser.add_argument(
        "--gate79-repair-csv",
        required=True,
    )

    parser.add_argument(
        "--gate79-ranking-csv",
        required=True,
    )

    parser.add_argument(
        "--gate79-ranking-summary-csv",
        required=True,
    )

    parser.add_argument(
        "--gate79-exposure-csv",
        required=True,
    )

    parser.add_argument(
        "--gate79-exposure-summary-csv",
        required=True,
    )

    parser.add_argument(
        "--gate78-state-json",
        required=True,
    )

    parser.add_argument(
        "--gate77-state-json",
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

    # ------------------------------------------------------------------
    # Gate A - verify current branch and accepted diagnostic.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE A - VERIFY AUDITED BRANCH AND GATE-79 DIAGNOSTIC")
    print("=" * 60)
    print("")

    gate78 = load_json(
        args.gate78_state_json
    )

    require_equal(
        gate78.get(
            "status"
        ),
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
            "gate77_assignment_artifact_manifest_sha256"
        ],
        ASSIGNMENT_ARTIFACT_MANIFEST_SHA,
        "Gate-78 Gate-77 artifact SHA256",
    )

    require_equal(
        gate78[
            "physical_assignment_audit_artifact_manifest_sha256"
        ],
        PHYSICAL_ASSIGNMENT_AUDIT_SHA,
        "Gate-78 audit SHA256",
    )

    require_true(
        gate78[
            "scientific_boundary"
        ][
            "participation_protocol_rebuilt"
        ]
        is False,
        "Gate-78 says participation already rebuilt.",
    )

    require_true(
        gate78[
            "scientific_boundary"
        ][
            "malicious_client_identity_rebuilt"
        ]
        is False,
        "Gate-78 says malicious identity already rebuilt.",
    )

    require_true(
        gate78[
            "scientific_boundary"
        ][
            "attack_protocol_rebound"
        ]
        is False,
        "Gate-78 says attack already rebound.",
    )

    require_true(
        gate78[
            "scientific_boundary"
        ][
            "scientific_training_started"
        ]
        is False,
        "Gate-78 says scientific training started.",
    )

    gate77 = load_json(
        args.gate77_state_json
    )

    require_equal(
        gate77.get(
            "status"
        ),
        "PASS",
        "Gate-77 status",
    )

    require_equal(
        gate77[
            "assignment_set_content_sha256"
        ],
        ASSIGNMENT_SET_CONTENT_SHA,
        "Gate-77 assignment-set SHA256",
    )

    gate79 = load_json(
        args.gate79_diagnostic_json
    )

    require_equal(
        gate79.get(
            "status"
        ),
        "DIAGNOSIS COMPLETE",
        "Gate-79 status",
    )

    require_equal(
        gate79.get(
            "diagnostic_id"
        ),
        GATE79_DIAGNOSTIC_ID,
        "Gate-79 diagnostic ID",
    )

    require_equal(
        gate79.get(
            "disposition"
        ),
        GATE79_DISPOSITION,
        "Gate-79 disposition",
    )

    require_equal(
        gate79[
            "immutable_binding"
        ][
            "assignment_set_content_sha256"
        ],
        ASSIGNMENT_SET_CONTENT_SHA,
        "Gate-79 assignment-set binding",
    )

    require_equal(
        gate79[
            "immutable_binding"
        ][
            "assignment_artifact_manifest_sha256"
        ],
        ASSIGNMENT_ARTIFACT_MANIFEST_SHA,
        "Gate-79 assignment artifact binding",
    )

    require_equal(
        gate79[
            "immutable_binding"
        ][
            "physical_assignment_audit_artifact_manifest_sha256"
        ],
        PHYSICAL_ASSIGNMENT_AUDIT_SHA,
        "Gate-79 audit binding",
    )

    require_equal(
        gate79[
            "candidate_fingerprints_not_frozen"
        ][
            "config_mapping_sha256"
        ],
        EXPECTED_CONFIG_MAPPING_SHA,
        "Gate-79 config mapping SHA256",
    )

    require_equal(
        gate79[
            "candidate_fingerprints_not_frozen"
        ][
            "schedule_set_sha256"
        ],
        EXPECTED_SCHEDULE_SET_SHA,
        "Gate-79 schedule-set SHA256",
    )

    require_equal(
        gate79[
            "candidate_fingerprints_not_frozen"
        ][
            "ranking_set_sha256"
        ],
        EXPECTED_RANKING_SET_SHA,
        "Gate-79 ranking-set SHA256",
    )

    require_true(
        gate79[
            "scientific_boundary"
        ][
            "participation_protocol_frozen"
        ]
        is False,
        "Gate-79 says participation already frozen.",
    )

    require_true(
        gate79[
            "scientific_boundary"
        ][
            "malicious_client_identity_frozen"
        ]
        is False,
        "Gate-79 says malicious identity already frozen.",
    )

    require_true(
        gate79[
            "scientific_boundary"
        ][
            "attack_protocol_rebound"
        ]
        is False,
        "Gate-79 says attack already rebound.",
    )

    print(
        "Audited assignment branch: BOUND"
    )
    print(
        "Gate-79 diagnostic: ACCEPTED"
    )
    print(
        "Attack rebound: NO"
    )
    print(
        "Scientific training started: NO"
    )

    # ------------------------------------------------------------------
    # Gate B - independently replay mapping / schedule / ranking fingerprints.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE B - REPLAY CANDIDATE SCIENTIFIC FINGERPRINTS")
    print("=" * 60)
    print("")

    mapping_rows = read_csv(
        args.gate79_mapping_csv
    )

    schedule_rows = read_csv(
        args.gate79_schedule_csv
    )

    schedule_summary_rows = read_csv(
        args.gate79_schedule_summary_csv
    )

    repair_rows = read_csv(
        args.gate79_repair_csv
    )

    ranking_rows = read_csv(
        args.gate79_ranking_csv
    )

    ranking_summary_rows = read_csv(
        args.gate79_ranking_summary_csv
    )

    exposure_rows = read_csv(
        args.gate79_exposure_csv
    )

    exposure_summary_rows = read_csv(
        args.gate79_exposure_summary_csv
    )

    require_equal(
        len(
            mapping_rows
        ),
        10,
        "Configuration mapping row count",
    )

    require_equal(
        len(
            schedule_rows
        ),
        5
        *
        TOTAL_SELECTIONS,
        "Schedule row count",
    )

    require_equal(
        len(
            ranking_rows
        ),
        5
        *
        CLIENTS,
        "Ranking row count",
    )

    replayed_mapping_sha = config_mapping_sha(
        mapping_rows
    )

    require_equal(
        replayed_mapping_sha,
        EXPECTED_CONFIG_MAPPING_SHA,
        "Config mapping SHA256 replay",
    )

    schedule_set_rows = []

    ranking_set_rows = []

    schedule_sha_by_seed = {}

    ranking_sha_by_seed = {}

    for experimental_seed in EXPERIMENTAL_SEEDS:
        schedule_sha = schedule_sha_for_seed(
            schedule_rows,
            experimental_seed,
        )

        schedule_sha_by_seed[
            experimental_seed
        ] = schedule_sha

        schedule_set_rows.append({
            "ExperimentalSeed": (
                experimental_seed
            ),
            "ScheduleSHA256": (
                schedule_sha
            ),
        })

        ranking_sha = ranking_sha_for_seed(
            ranking_rows,
            experimental_seed,
        )

        ranking_sha_by_seed[
            experimental_seed
        ] = ranking_sha

        ranking_set_rows.append({
            "ExperimentalSeed": (
                experimental_seed
            ),
            "RankingSHA256": (
                ranking_sha
            ),
        })

    replayed_schedule_set_sha = canonical_set_sha(
        schedule_set_rows,
        "ExperimentalSeed",
        "ScheduleSHA256",
    )

    replayed_ranking_set_sha = canonical_set_sha(
        ranking_set_rows,
        "ExperimentalSeed",
        "RankingSHA256",
    )

    require_equal(
        replayed_schedule_set_sha,
        EXPECTED_SCHEDULE_SET_SHA,
        "Schedule-set SHA256 replay",
    )

    require_equal(
        replayed_ranking_set_sha,
        EXPECTED_RANKING_SET_SHA,
        "Ranking-set SHA256 replay",
    )

    require_equal(
        len(
            set(
                schedule_sha_by_seed.values()
            )
        ),
        5,
        "Unique schedule family count",
    )

    require_equal(
        len(
            set(
                ranking_sha_by_seed.values()
            )
        ),
        5,
        "Unique ranking family count",
    )

    combined_identity_sha = combined_preattack_identity_sha(
        replayed_mapping_sha,
        replayed_schedule_set_sha,
        replayed_ranking_set_sha,
    )

    print(
        f"Config mapping SHA256: {replayed_mapping_sha}"
    )
    print(
        f"Schedule-set SHA256: {replayed_schedule_set_sha}"
    )
    print(
        f"Ranking-set SHA256: {replayed_ranking_set_sha}"
    )
    print(
        f"Combined pre-attack identity SHA256: {combined_identity_sha}"
    )

    # ------------------------------------------------------------------
    # Gate C - re-audit participation geometry and pairing.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE C - RE-AUDIT PARTICIPATION GEOMETRY AND PAIRING")
    print("=" * 60)
    print("")

    schedule_summary_by_seed = {
        int(
            row[
                "ExperimentalSeed"
            ]
        ): row
        for row
        in schedule_summary_rows
    }

    require_equal(
        set(
            schedule_summary_by_seed
        ),
        set(
            EXPERIMENTAL_SEEDS
        ),
        "Schedule summary seed coverage",
    )

    mapping_by_seed = {
        seed: []
        for seed
        in EXPERIMENTAL_SEEDS
    }

    for row in mapping_rows:
        experimental_seed = int(
            row[
                "ExperimentalSeed"
            ]
        )

        mapping_by_seed[
            experimental_seed
        ].append(
            row
        )

        require_equal(
            int(
                row[
                    "ScheduleFamilyKey"
                ]
            ),
            experimental_seed,
            "Schedule family key",
        )

        require_equal(
            int(
                row[
                    "MaliciousRankingFamilyKey"
                ]
            ),
            experimental_seed,
            "Malicious ranking family key",
        )

        require_equal(
            row[
                "PairingPolicyID"
            ],
            PAIRING_POLICY_ID,
            "Pairing policy ID",
        )

    participation_audit_rows = []

    for experimental_seed in EXPERIMENTAL_SEEDS:
        require_equal(
            len(
                mapping_by_seed[
                    experimental_seed
                ]
            ),
            2,
            (
                "Configuration count for seed "
                f"{experimental_seed}"
            ),
        )

        require_equal(
            {
                float(
                    row[
                        "Alpha"
                    ]
                )
                for row
                in mapping_by_seed[
                    experimental_seed
                ]
            },
            {
                0.1,
                1.0,
            },
            (
                "Alpha pair for seed "
                f"{experimental_seed}"
            ),
        )

        rounds = schedule_rounds(
            schedule_rows,
            experimental_seed,
        )

        exposure = {
            client_id: 0
            for client_id
            in range(
                CLIENTS
            )
        }

        for selected in rounds:
            for client_id in selected:
                require_true(
                    0
                    <=
                    client_id
                    <
                    CLIENTS,
                    (
                        "Invalid participation ClientID "
                        f"seed {experimental_seed}"
                    ),
                )

                exposure[
                    client_id
                ] += 1

        require_equal(
            set(
                exposure.values()
            ),
            {
                EXPOSURES_PER_CLIENT
            },
            (
                "Exact participation exposure "
                f"seed {experimental_seed}"
            ),
        )

        max_absence = maximum_consecutive_absence(
            rounds
        )

        require_true(
            max_absence
            <=
            6,
            (
                "Maximum consecutive absence "
                f"seed {experimental_seed}"
            ),
        )

        adjacent_overlaps = [
            len(
                set(
                    rounds[
                        round_index
                    ]
                )
                &
                set(
                    rounds[
                        round_index
                        +
                        1
                    ]
                )
            )
            for round_index
            in range(
                ROUNDS
                -
                1
            )
        ]

        mean_adjacent_overlap = statistics.mean(
            adjacent_overlaps
        )

        require_true(
            mean_adjacent_overlap
            <
            RANDOM_ADJACENT_OVERLAP_REFERENCE,
            (
                "Mean adjacent overlap "
                f"seed {experimental_seed}"
            ),
        )

        identical_round_pairs = sum(
            1
            for first
            in range(
                ROUNDS
            )
            for second
            in range(
                first
                +
                1,
                ROUNDS,
            )
            if set(
                rounds[
                    first
                ]
            )
            ==
            set(
                rounds[
                    second
                ]
            )
        )

        require_equal(
            identical_round_pairs,
            0,
            (
                "Identical participation round pairs "
                f"seed {experimental_seed}"
            ),
        )

        summary = schedule_summary_by_seed[
            experimental_seed
        ]

        require_equal(
            summary[
                "ScheduleSHA256"
            ],
            schedule_sha_by_seed[
                experimental_seed
            ],
            (
                "Schedule summary SHA256 "
                f"seed {experimental_seed}"
            ),
        )

        require_equal(
            int(
                summary[
                    "MaximumConsecutiveAbsence"
                ]
            ),
            max_absence,
            (
                "Schedule summary max absence "
                f"seed {experimental_seed}"
            ),
        )

        participation_audit_rows.append({
            "ExperimentalSeed": (
                experimental_seed
            ),
            "ScheduleSHA256": (
                schedule_sha_by_seed[
                    experimental_seed
                ]
            ),
            "Rounds": (
                ROUNDS
            ),
            "SelectedPerRound": (
                SELECTED_PER_ROUND
            ),
            "ExactParticipationsPerClient": (
                EXPOSURES_PER_CLIENT
            ),
            "MaximumConsecutiveAbsence": (
                max_absence
            ),
            "MeanAdjacentRoundOverlap": (
                f"{mean_adjacent_overlap:.12f}"
            ),
            "IdenticalRoundPairs": (
                identical_round_pairs
            ),
        })

    print(
        "Exact 6 participations/client: PASS ALL 5"
    )
    print(
        "Unique 9-client rounds: PASS ALL 5"
    )
    print(
        "Maximum consecutive absence <= 6: PASS ALL 5"
    )
    print(
        "Mean adjacent overlap below random reference: PASS ALL 5"
    )
    print(
        "Alpha-pair schedule sharing: PASS ALL 5"
    )

    # ------------------------------------------------------------------
    # Gate D - re-audit fixed nested malicious identities and exposure.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE D - RE-AUDIT NESTED MALICIOUS IDENTITIES")
    print("=" * 60)
    print("")

    ranking_summary_by_seed = {
        int(
            row[
                "ExperimentalSeed"
            ]
        ): row
        for row
        in ranking_summary_rows
    }

    require_equal(
        set(
            ranking_summary_by_seed
        ),
        set(
            EXPERIMENTAL_SEEDS
        ),
        "Ranking summary seed coverage",
    )

    malicious_audit_rows = []

    exposure_replay_rows = []

    for experimental_seed in EXPERIMENTAL_SEEDS:
        selected = [
            row
            for row
            in ranking_rows
            if int(
                row[
                    "ExperimentalSeed"
                ]
            )
            ==
            experimental_seed
        ]

        selected.sort(
            key=lambda row: int(
                row[
                    "RankPosition"
                ]
            )
        )

        ranking = [
            int(
                row[
                    "ClientID"
                ]
            )
            for row
            in selected
        ]

        require_equal(
            sorted(
                ranking
            ),
            list(
                range(
                    CLIENTS
                )
            ),
            (
                "Malicious ranking client coverage "
                f"seed {experimental_seed}"
            ),
        )

        prefix3 = set(
            ranking[
                :3
            ]
        )

        prefix6 = set(
            ranking[
                :6
            ]
        )

        prefix12 = set(
            ranking[
                :12
            ]
        )

        require_true(
            prefix3
            <
            prefix6,
            (
                "3-client prefix not strict subset "
                f"seed {experimental_seed}"
            ),
        )

        require_true(
            prefix6
            <
            prefix12,
            (
                "6-client prefix not strict subset "
                f"seed {experimental_seed}"
            ),
        )

        for row in selected:
            rank_position = int(
                row[
                    "RankPosition"
                ]
            )

            require_equal(
                parse_bool(
                    row[
                        "InMu0p1Prefix"
                    ]
                ),
                rank_position
                <=
                3,
                "Mu0p1 prefix flag",
            )

            require_equal(
                parse_bool(
                    row[
                        "InMu0p2Prefix"
                    ]
                ),
                rank_position
                <=
                6,
                "Mu0p2 prefix flag",
            )

            require_equal(
                parse_bool(
                    row[
                        "InMu0p4Prefix"
                    ]
                ),
                rank_position
                <=
                12,
                "Mu0p4 prefix flag",
            )

        summary = ranking_summary_by_seed[
            experimental_seed
        ]

        require_equal(
            summary[
                "RankingSHA256"
            ],
            ranking_sha_by_seed[
                experimental_seed
            ],
            (
                "Ranking summary SHA256 "
                f"seed {experimental_seed}"
            ),
        )

        rounds = schedule_rounds(
            schedule_rows,
            experimental_seed,
        )

        severe_counts = malicious_exposure(
            rounds,
            ranking,
            12,
        )

        require_equal(
            sum(
                1
                for value
                in severe_counts
                if value == 0
            ),
            0,
            (
                "Mu=0.4 zero-malicious reference rounds "
                f"seed {experimental_seed}"
            ),
        )

        for malicious_fraction, prefix_size in MALICIOUS_PREFIXES.items():
            counts = malicious_exposure(
                rounds,
                ranking,
                prefix_size,
            )

            exposure_replay_rows.append({
                "ExperimentalSeed": (
                    experimental_seed
                ),
                "MaliciousFraction": (
                    malicious_fraction
                ),
                "PrefixSize": (
                    prefix_size
                ),
                "ReferenceWindow": (
                    "ROUNDS_5_TO_20_DIAGNOSTIC_ONLY_ATTACK_NOT_REBOUND"
                ),
                "TotalSelectedMalicious": (
                    sum(
                        counts
                    )
                ),
                "MinPerRound": (
                    min(
                        counts
                    )
                ),
                "MaxPerRound": (
                    max(
                        counts
                    )
                ),
                "MeanPerRound": (
                    f"{statistics.mean(counts):.12f}"
                ),
                "ZeroMaliciousRounds": (
                    sum(
                        1
                        for value
                        in counts
                        if value == 0
                    )
                ),
            })

        malicious_audit_rows.append({
            "ExperimentalSeed": (
                experimental_seed
            ),
            "RankingSHA256": (
                ranking_sha_by_seed[
                    experimental_seed
                ]
            ),
            "Mu0p1Clients": (
                ",".join(
                    str(
                        client_id
                    )
                    for client_id
                    in ranking[
                        :3
                    ]
                )
            ),
            "Mu0p2Clients": (
                ",".join(
                    str(
                        client_id
                    )
                    for client_id
                    in ranking[
                        :6
                    ]
                )
            ),
            "Mu0p4Clients": (
                ",".join(
                    str(
                        client_id
                    )
                    for client_id
                    in ranking[
                        :12
                    ]
                )
            ),
            "NestedPrefixesExact": (
                True
            ),
            "Mu0p4ZeroReferenceRounds": (
                0
            ),
        })

    print(
        "Fixed 30-client rankings: PASS ALL 5"
    )
    print(
        "Nested prefixes 3 < 6 < 12: PASS ALL 5"
    )
    print(
        "Same ranking family across alpha pair: PASS ALL 5"
    )
    print(
        "Mu=0.4 malicious presence in every reference round 5-20: PASS ALL 5"
    )
    print(
        "Attack timing frozen by this gate: NO"
    )

    # ------------------------------------------------------------------
    # Gate E - write frozen scientific artifacts.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE E - WRITE FROZEN PARTICIPATION / IDENTITY ARTIFACTS")
    print("=" * 60)
    print("")

    frozen_mapping_path = (
        output_root
        /
        "FROZEN_CONFIG_SEED_FAMILY_MAPPING.csv"
    )

    frozen_schedule_path = (
        output_root
        /
        "FROZEN_PARTICIPATION_SCHEDULES.csv"
    )

    frozen_schedule_summary_path = (
        output_root
        /
        "FROZEN_PARTICIPATION_AUDIT_SUMMARY.csv"
    )

    frozen_repair_path = (
        output_root
        /
        "FROZEN_PARTICIPATION_BOUNDARY_REPAIR_PROVENANCE.csv"
    )

    frozen_ranking_path = (
        output_root
        /
        "FROZEN_MALICIOUS_CLIENT_RANKINGS.csv"
    )

    frozen_ranking_summary_path = (
        output_root
        /
        "FROZEN_MALICIOUS_IDENTITY_AUDIT_SUMMARY.csv"
    )

    exposure_diagnostic_path = (
        output_root
        /
        "REFERENCE_WINDOW_MALICIOUS_EXPOSURE_DIAGNOSTIC.csv"
    )

    protocol_path = (
        output_root
        /
        "FROZEN_PARTICIPATION_MALICIOUS_IDENTITY_PROTOCOL.json"
    )

    write_csv(
        frozen_mapping_path,
        mapping_rows,
        list(
            mapping_rows[
                0
            ].keys()
        ),
    )

    write_csv(
        frozen_schedule_path,
        schedule_rows,
        list(
            schedule_rows[
                0
            ].keys()
        ),
    )

    write_csv(
        frozen_schedule_summary_path,
        participation_audit_rows,
        [
            "ExperimentalSeed",
            "ScheduleSHA256",
            "Rounds",
            "SelectedPerRound",
            "ExactParticipationsPerClient",
            "MaximumConsecutiveAbsence",
            "MeanAdjacentRoundOverlap",
            "IdenticalRoundPairs",
        ],
    )

    repair_fields = (
        list(
            repair_rows[
                0
            ].keys()
        )
        if repair_rows
        else
        [
            "ExperimentalSeed",
            "EpochID",
            "PositionA",
            "PositionB",
            "ClientAtPositionA",
            "ClientAtPositionB",
        ]
    )

    write_csv(
        frozen_repair_path,
        repair_rows,
        repair_fields,
    )

    write_csv(
        frozen_ranking_path,
        ranking_rows,
        list(
            ranking_rows[
                0
            ].keys()
        ),
    )

    write_csv(
        frozen_ranking_summary_path,
        malicious_audit_rows,
        [
            "ExperimentalSeed",
            "RankingSHA256",
            "Mu0p1Clients",
            "Mu0p2Clients",
            "Mu0p4Clients",
            "NestedPrefixesExact",
            "Mu0p4ZeroReferenceRounds",
        ],
    )

    write_csv(
        exposure_diagnostic_path,
        exposure_replay_rows,
        [
            "ExperimentalSeed",
            "MaliciousFraction",
            "PrefixSize",
            "ReferenceWindow",
            "TotalSelectedMalicious",
            "MinPerRound",
            "MaxPerRound",
            "MeanPerRound",
            "ZeroMaliciousRounds",
        ],
    )

    protocol = {
        "status": (
            "FROZEN"
        ),
        "freeze_id": (
            FREEZE_ID
        ),
        "participation_protocol_id": (
            PARTICIPATION_PROTOCOL_ID
        ),
        "malicious_identity_id": (
            MALICIOUS_IDENTITY_ID
        ),
        "immutable_binding": {
            "assignment_set_content_sha256": (
                ASSIGNMENT_SET_CONTENT_SHA
            ),
            "assignment_artifact_manifest_sha256": (
                ASSIGNMENT_ARTIFACT_MANIFEST_SHA
            ),
            "physical_assignment_audit_artifact_manifest_sha256": (
                PHYSICAL_ASSIGNMENT_AUDIT_SHA
            ),
        },
        "frozen_pairing": {
            "alpha_pairing_policy_id": (
                PAIRING_POLICY_ID
            ),
            "cross_method_scenario_pairing_id": (
                CROSS_METHOD_SCENARIO_PAIRING_ID
            ),
            "same_seed_schedule_shared_across_alpha_0p1_and_1p0": (
                True
            ),
            "same_seed_malicious_ranking_shared_across_alpha_0p1_and_1p0": (
                True
            ),
            "same_seed_schedule_to_be_shared_across_all_methods": (
                True
            ),
            "same_seed_schedule_to_be_shared_across_clean_and_attack_scenarios": (
                True
            ),
            "same_seed_malicious_ranking_to_be_shared_across_all_methods": (
                True
            ),
            "same_seed_malicious_ranking_to_be_shared_across_attack_severities": (
                True
            ),
        },
        "frozen_participation_geometry": {
            "clients": (
                CLIENTS
            ),
            "rounds": (
                ROUNDS
            ),
            "selected_per_round": (
                SELECTED_PER_ROUND
            ),
            "total_selections_per_schedule": (
                TOTAL_SELECTIONS
            ),
            "exact_participations_per_client": (
                EXPOSURES_PER_CLIENT
            ),
            "experimental_seeds": (
                list(
                    EXPERIMENTAL_SEEDS
                )
            ),
            "schedule_family_count": (
                5
            ),
            "participation_algorithm_id": (
                PARTICIPATION_ALGORITHM_ID
            ),
        },
        "frozen_malicious_identity_geometry": {
            "ranking_family_count": (
                5
            ),
            "ranking_fixed_across_rounds": (
                True
            ),
            "malicious_ranking_algorithm_id": (
                MALICIOUS_RANKING_ALGORITHM_ID
            ),
            "nested_prefixes": {
                "0.1": (
                    3
                ),
                "0.2": (
                    6
                ),
                "0.4": (
                    12
                ),
            },
        },
        "frozen_scientific_fingerprints": {
            "config_mapping_hash_id": (
                CONFIG_MAPPING_HASH_ID
            ),
            "config_mapping_sha256": (
                replayed_mapping_sha
            ),
            "schedule_set_hash_id": (
                SCHEDULE_SET_HASH_ID
            ),
            "schedule_set_sha256": (
                replayed_schedule_set_sha
            ),
            "ranking_set_hash_id": (
                RANKING_SET_HASH_ID
            ),
            "ranking_set_sha256": (
                replayed_ranking_set_sha
            ),
            "combined_preattack_identity_hash_id": (
                COMBINED_PREATTACK_IDENTITY_HASH_ID
            ),
            "combined_preattack_identity_sha256": (
                combined_identity_sha
            ),
        },
        "reference_exposure_diagnostic": {
            "status": (
                "DIAGNOSTIC_ONLY_ATTACK_NOT_REBOUND"
            ),
            "reference_rounds": (
                "5_TO_20"
            ),
            "mu_0p4_presence_every_reference_round_all_seeds": (
                True
            ),
            "mu_0p2_zero_malicious_rounds_allowed_under_partial_participation": (
                True
            ),
            "mu_0p1_zero_malicious_rounds_allowed_under_partial_participation": (
                True
            ),
        },
        "prohibited_identity_inputs": [
            "ModelResults",
            "AttackResults",
            "ValidationMetrics",
            "TestMetrics",
            "FeatureValues",
            "RawMultiplicity",
            "Provenance",
        ],
        "scientific_boundary": {
            "participation_protocol_frozen": (
                True
            ),
            "malicious_client_identity_frozen": (
                True
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
        protocol_path,
        protocol,
    )

    # ------------------------------------------------------------------
    # Gate F - immutable artifact manifest and freeze state.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE F - WRITE IMMUTABLE FREEZE MANIFEST")
    print("=" * 60)
    print("")

    artifact_rows = []

    for role, path in [
        (
            "PARTICIPATION_MALICIOUS_IDENTITY_PROTOCOL",
            protocol_path,
        ),
        (
            "CONFIG_SEED_FAMILY_MAPPING",
            frozen_mapping_path,
        ),
        (
            "PARTICIPATION_SCHEDULES",
            frozen_schedule_path,
        ),
        (
            "PARTICIPATION_AUDIT_SUMMARY",
            frozen_schedule_summary_path,
        ),
        (
            "PARTICIPATION_BOUNDARY_REPAIR_PROVENANCE",
            frozen_repair_path,
        ),
        (
            "MALICIOUS_CLIENT_RANKINGS",
            frozen_ranking_path,
        ),
        (
            "MALICIOUS_IDENTITY_AUDIT_SUMMARY",
            frozen_ranking_summary_path,
        ),
        (
            "REFERENCE_EXPOSURE_DIAGNOSTIC",
            exposure_diagnostic_path,
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
        "FROZEN_PARTICIPATION_MALICIOUS_IDENTITY_ARTIFACT_MANIFEST.csv"
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
        "participation_protocol_id": (
            PARTICIPATION_PROTOCOL_ID
        ),
        "malicious_identity_id": (
            MALICIOUS_IDENTITY_ID
        ),
        "config_mapping_sha256": (
            replayed_mapping_sha
        ),
        "schedule_set_sha256": (
            replayed_schedule_set_sha
        ),
        "ranking_set_sha256": (
            replayed_ranking_set_sha
        ),
        "combined_preattack_identity_sha256": (
            combined_identity_sha
        ),
        "frozen_participation_malicious_identity_artifact_manifest_sha256": (
            combined_artifact_manifest_sha
        ),
        "scientific_boundary": {
            "participation_protocol_frozen": (
                True
            ),
            "malicious_client_identity_frozen": (
                True
            ),
            "attack_protocol_rebound": (
                False
            ),
            "scientific_training_started": (
                False
            ),
        },
    }

    write_json(
        output_root
        /
        "PARTICIPATION_MALICIOUS_IDENTITY_FREEZE.json",
        freeze_state,
    )

    # Evidence manifest.
    evidence_paths = {
        "GATE79_DIAGNOSTIC": Path(
            args.gate79_diagnostic_json
        ),
        "GATE79_MAPPING": Path(
            args.gate79_mapping_csv
        ),
        "GATE79_SCHEDULE": Path(
            args.gate79_schedule_csv
        ),
        "GATE79_SCHEDULE_SUMMARY": Path(
            args.gate79_schedule_summary_csv
        ),
        "GATE79_REPAIRS": Path(
            args.gate79_repair_csv
        ),
        "GATE79_RANKING": Path(
            args.gate79_ranking_csv
        ),
        "GATE79_RANKING_SUMMARY": Path(
            args.gate79_ranking_summary_csv
        ),
        "GATE79_EXPOSURE": Path(
            args.gate79_exposure_csv
        ),
        "GATE79_EXPOSURE_SUMMARY": Path(
            args.gate79_exposure_summary_csv
        ),
        "GATE78_STATE": Path(
            args.gate78_state_json
        ),
        "GATE77_STATE": Path(
            args.gate77_state_json
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
        "CICIoT2023 SEED-PAIRED PARTICIPATION / MALICIOUS-IDENTITY FREEZE"
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
        f"Participation protocol ID: {PARTICIPATION_PROTOCOL_ID}"
    )
    add(
        f"Malicious identity ID: {MALICIOUS_IDENTITY_ID}"
    )
    add(
        f"Alpha-pairing policy ID: {PAIRING_POLICY_ID}"
    )
    add(
        f"Cross-method/scenario pairing ID: "
        f"{CROSS_METHOD_SCENARIO_PAIRING_ID}"
    )
    add("")

    add("IMMUTABLE BINDING")
    add("-" * 78)
    add(
        f"Assignment-set content SHA256: {ASSIGNMENT_SET_CONTENT_SHA}"
    )
    add(
        f"Assignment artifact-manifest SHA256: "
        f"{ASSIGNMENT_ARTIFACT_MANIFEST_SHA}"
    )
    add(
        f"Physical-assignment audit artifact-manifest SHA256: "
        f"{PHYSICAL_ASSIGNMENT_AUDIT_SHA}"
    )
    add("")

    add("FROZEN CONFIGURATION PAIRING")
    add("-" * 78)
    add(
        "Configurations: 10"
    )
    add(
        "Experimental seed families: 5"
    )
    add(
        "Each experimental seed shared across alpha=0.1 and alpha=1.0: YES"
    )
    add(
        "Same seed-family participation schedule across alpha pair: YES"
    )
    add(
        "Same seed-family malicious ranking across alpha pair: YES"
    )
    add(
        "Same schedule to be reused across all methods: YES"
    )
    add(
        "Same schedule to be reused across clean and attack scenarios: YES"
    )
    add(
        "Same malicious ranking to be reused across all methods/severities: YES"
    )
    add("")

    add("FROZEN PARTICIPATION GEOMETRY")
    add("-" * 78)
    add(
        "Clients: 30"
    )
    add(
        "Rounds: 20"
    )
    add(
        "Selected clients per round: 9"
    )
    add(
        "Total selections per schedule: 180"
    )
    add(
        "Exact participations per client: 6"
    )
    add(
        "Schedule families: 5"
    )
    add(
        f"Participation algorithm ID: {PARTICIPATION_ALGORITHM_ID}"
    )
    add("")

    for row in participation_audit_rows:
        add(
            f"Seed {row['ExperimentalSeed']}: "
            f"schedule_sha={row['ScheduleSHA256']}, "
            f"max_absence={row['MaximumConsecutiveAbsence']}, "
            f"mean_adjacent_overlap={row['MeanAdjacentRoundOverlap']}"
        )

    add("")
    add(
        "Exact 6 participations/client: PASS ALL 5"
    )
    add(
        "Unique clients within every round: PASS ALL 5"
    )
    add(
        "Maximum consecutive absence <= 6: PASS ALL 5"
    )
    add(
        "Mean adjacent overlap below random reference 2.7: PASS ALL 5"
    )
    add("")

    add("FROZEN FIXED NESTED MALICIOUS IDENTITIES")
    add("-" * 78)
    add(
        "10% malicious fraction: first 3 ranked clients"
    )
    add(
        "20% malicious fraction: first 6 ranked clients"
    )
    add(
        "40% malicious fraction: first 12 ranked clients"
    )
    add(
        "Nested prefixes 3 subset 6 subset 12: PASS ALL 5"
    )
    add(
        "Ranking fixed across rounds: YES"
    )
    add("")

    for row in malicious_audit_rows:
        add(
            f"Seed {row['ExperimentalSeed']}: "
            f"ranking_sha={row['RankingSHA256']}"
        )
        add(
            f"  mu=0.1: {row['Mu0p1Clients']}"
        )
        add(
            f"  mu=0.2: {row['Mu0p2Clients']}"
        )
        add(
            f"  mu=0.4: {row['Mu0p4Clients']}"
        )

    add("")

    add("FROZEN SCIENTIFIC FINGERPRINTS")
    add("-" * 78)
    add(
        f"Config mapping SHA256: {replayed_mapping_sha}"
    )
    add(
        f"Schedule-set SHA256: {replayed_schedule_set_sha}"
    )
    add(
        f"Ranking-set SHA256: {replayed_ranking_set_sha}"
    )
    add(
        f"Combined pre-attack identity SHA256: {combined_identity_sha}"
    )
    add("")

    add("REFERENCE ROUNDS 5-20 EXPOSURE - DIAGNOSTIC ONLY")
    add("-" * 78)
    add(
        "Attack timing frozen by this gate: NO"
    )
    add(
        "mu=0.4: at least one selected malicious client every reference round, all 5 seeds"
    )
    add(
        "mu=0.2: zero-malicious rounds are allowed under natural partial participation"
    )
    add(
        "mu=0.1: zero-malicious rounds are allowed under natural partial participation"
    )
    add("")

    add("IMMUTABLE ARTIFACT")
    add("-" * 78)
    add(
        f"Combined participation/malicious-identity artifact-manifest SHA256: "
        f"{combined_artifact_manifest_sha}"
    )
    add("")

    add("SCIENTIFIC BOUNDARY")
    add("-" * 78)
    add(
        "Participation protocol frozen: YES"
    )
    add(
        "Malicious-client identity frozen: YES"
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
        "Rebind the sign-flip attack semantics and timing to the frozen "
        "participation / malicious-identity branch, then rebind the deterministic "
        "training/checkpoint protocol. Do not start scientific training until "
        "the combined attack-and-training protocol has been validated and frozen."
    )

    (
        output_root
        /
        "PARTICIPATION_MALICIOUS_IDENTITY_FREEZE_REPORT.txt"
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
        f"CONFIG MAPPING SHA256: {replayed_mapping_sha}"
    )
    print(
        f"SCHEDULE-SET SHA256: {replayed_schedule_set_sha}"
    )
    print(
        f"RANKING-SET SHA256: {replayed_ranking_set_sha}"
    )
    print(
        f"COMBINED PRE-ATTACK IDENTITY SHA256: {combined_identity_sha}"
    )
    print(
        f"COMBINED FREEZE ARTIFACT MANIFEST: "
        f"{combined_artifact_manifest_sha}"
    )
    print(
        "ATTACK PROTOCOL REBOUND: NO"
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
