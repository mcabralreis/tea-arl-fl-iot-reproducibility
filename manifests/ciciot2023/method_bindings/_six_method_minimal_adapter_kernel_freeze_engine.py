import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path


# ============================================================================
# Immutable upstream bindings
# ============================================================================

COMMON_PROTOCOL_SHA = (
    "0DA3146E8195CA630EE9F4DEDDC3D665F2BE9C08E3906677E5F79E6D1C602570"
)

GATE83_ARTIFACT_MANIFEST_SHA = (
    "DB39208DA367B5DC53DD33832D14746517EE82058C8DB07117806922BB8F86CA"
)

SIX_METHOD_BINDING_SHA = (
    "8F0B7E875885A2ECAA6E5BC4F683D5517C0265C7AF92FFD49D41BE6EDC44D5A7"
)

PARAMETER_RECORD_SET_SHA = (
    "DA93C652FA17D99B7F70F64284CE98DAC32E1D2EAAF206BE0BED33DC2EDE8A29"
)

GATE89_ARTIFACT_MANIFEST_SHA = (
    "8D3D94AF6BFE7A27C817F4D62DD43E7AFAAD2D3EE17DC7D2D19C07EE2375927B"
)

GATE90_EXECUTION_CONTRACT_SHA = (
    "9112F809049681A23955D21C32BE1888F45EFC93691B1D2EC8610495B230DE91"
)

GATE91_EXECUTION_UNIT_MAP_SHA = (
    "0FABFEF73B1B29E73E016A3EF8A6BE0ACC2E017335D63DC1B40ECB85AE605283"
)

GATE92_ADAPTER_CHAIN_MAP_SHA = (
    "C56E10D75892526F66F9DA668A01080D4D9144CECCCD92FA288DCA070AE3B151"
)

GATE93_CANDIDATE_ADAPTER_MAP_SHA = (
    "67E576309FF615A93B6CA6C91EFA83BE2FD5FBC6ED14CFE7095AAB2DD75A4F13"
)

GATE94_FEDLE_TRANSITIVE_FLOW_PROOF_SHA = (
    "99AFF1D42A0565310574D20AB8C266F3074891ED47D0F60543194E2C37C97F04"
)

GATE95_FEDLE_SECOND_OUTPUT_PROOF_SHA = (
    "A91E366D9AEC5BCD027AAA62C72BDC323E26D8E5158A5B86DDAA506FAF521872"
)


# ============================================================================
# Frozen protocol identifiers
# ============================================================================

FREEZE_ID = (
    "CICIoT2023_SIX_METHOD_MINIMAL_ADAPTER_KERNEL_MAP_FREEZE_V1"
)

ADAPTER_PROTOCOL_ID = (
    "CICIoT2023_COMMON_SCHEDULE_SOURCE_BOUND_SIX_METHOD_ADAPTER_V1"
)

PARTICIPATION_AUTHORITY = (
    "GATE80_GATE83_FROZEN_COMMON_SCHEDULE"
)

PARTICIPATION_RULE = (
    "METHOD_SPECIFIC_SELECTION_MAY_NOT_REPLACE_FROZEN_PARTICIPANT_IDENTITIES"
)

COMMON_LOCAL_TRAINING_RULE = (
    "GATE83_COMMON_LOCAL_TRAINING_FOR_ALL_METHODS_EXCEPT_FEDPROX_PROXIMAL_OVERRIDE"
)

METHODS = (
    "FEDAVG",
    "FEDPROX",
    "RANDOM_TRIMMED_MEAN",
    "FEDLE_ADAPTED",
    "TEA_FL",
    "ARL_FL",
)

EXPECTED_KERNELS = {
    "FEDAVG": {
        ("LOCAL_TRAINING", "train_one_client"):
            "COMMON_GATE83_LOCAL_TRAINING_SOURCE_SEMANTICS_REFERENCE_ONLY",
        ("SERVER_AGGREGATION", "fedavg_weighted_state"):
            "EXECUTE",
    },
    "FEDPROX": {
        ("LOCAL_TRAINING", "train_one_client_method"):
            "EXECUTE",
        ("SERVER_AGGREGATION", "fedavg_weighted_state"):
            "EXECUTE",
    },
    "RANDOM_TRIMMED_MEAN": {
        ("LOCAL_TRAINING", "train_one_client_method"):
            "COMMON_GATE83_LOCAL_TRAINING_SOURCE_SEMANTICS_REFERENCE_ONLY",
        ("SERVER_AGGREGATION", "trimmed_mean_state"):
            "EXECUTE",
    },
    "FEDLE_ADAPTED": {
        ("LOCAL_TRAINING", "train_one_client"):
            "COMMON_GATE83_LOCAL_TRAINING_SOURCE_SEMANTICS_REFERENCE_ONLY",
        ("SERVER_AGGREGATION", "weighted_fedavg_state"):
            "EXECUTE",
    },
    "TEA_FL": {
        ("LOCAL_TRAINING", "train_one_client"):
            "COMMON_GATE83_LOCAL_TRAINING_SOURCE_SEMANTICS_REFERENCE_ONLY",
        ("STATE_UPDATE", "update_trust_from_round_reference"):
            "EXECUTE",
        ("SERVER_AGGREGATION", "tea_weighted_aggregation"):
            "EXECUTE",
    },
    "ARL_FL": {
        ("LOCAL_TRAINING", "train_one_client"):
            "COMMON_GATE83_LOCAL_TRAINING_SOURCE_SEMANTICS_REFERENCE_ONLY",
        ("STATE_UPDATE", "compute_round_risk"):
            "EXECUTE",
        ("SERVER_AGGREGATION", "adaptive_robust_aggregation"):
            "EXECUTE",
    },
}

EXPECTED_SELECTION_CALLABLE = {
    "FEDAVG": "",
    "FEDPROX": "",
    "RANDOM_TRIMMED_MEAN": "",
    "FEDLE_ADAPTED": "select_fedle_clients",
    "TEA_FL": "select_tea_clients",
    "ARL_FL": "select_arl_clients",
}

EXPECTED_FINAL_SUBSTITUTION_STATUS = {
    "FEDAVG": "NO_METHOD_SELECTION_CALLABLE",
    "FEDPROX": "NO_METHOD_SELECTION_CALLABLE",
    "RANDOM_TRIMMED_MEAN": "NO_METHOD_SELECTION_CALLABLE",
    "FEDLE_ADAPTED": "FEDLE_COMMON_SCHEDULE_SUBSTITUTION_FULLY_PROVEN",
    "TEA_FL": "STATIC_COMMON_SELECTED_SUBSTITUTION_SUPPORTED",
    "ARL_FL": "STATIC_COMMON_SELECTED_SUBSTITUTION_SUPPORTED",
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


def candidate_adapter_map_sha(rows):
    digest = hashlib.sha256()

    digest.update(
        (
            f"COMMON_PROTOCOL_SHA256={COMMON_PROTOCOL_SHA}\n"
            f"SIX_METHOD_BINDING_SHA256={SIX_METHOD_BINDING_SHA}\n"
            f"PARAMETER_SET_SHA256={PARAMETER_RECORD_SET_SHA}\n"
            f"GATE90_SHA256={GATE90_EXECUTION_CONTRACT_SHA}\n"
            f"GATE91_SHA256={GATE91_EXECUTION_UNIT_MAP_SHA}\n"
            f"GATE92_SHA256={GATE92_ADAPTER_CHAIN_MAP_SHA}\n"
            f"PARTICIPATION_AUTHORITY={PARTICIPATION_AUTHORITY}\n"
        ).encode("utf-8")
    )

    for row in sorted(
        rows,
        key=lambda item: (
            item[
                "MethodID"
            ],
            item[
                "Role"
            ],
            item[
                "CallableName"
            ],
        ),
    ):
        digest.update(
            (
                f"{row['MethodID']}\t"
                f"{row['Role']}\t"
                f"{row['CallableName']}\t"
                f"{row['ResolutionScope']}\t"
                f"{row['ResolvedSHA256']}\t"
                f"{row['CallableSignature']}\t"
                f"{row['ExecutionDisposition']}\n"
            ).encode("utf-8")
        )

    return digest.hexdigest().upper()


def selection_policy_sha(rows):
    digest = hashlib.sha256()

    digest.update(
        (
            f"PARTICIPATION_AUTHORITY={PARTICIPATION_AUTHORITY}\n"
            f"PARTICIPATION_RULE={PARTICIPATION_RULE}\n"
        ).encode("utf-8")
    )

    for row in sorted(
        rows,
        key=lambda item: item[
            "MethodID"
        ],
    ):
        digest.update(
            (
                f"{row['MethodID']}\t"
                f"{row['MethodSelectionCallable']}\t"
                f"{row['OriginalSelectionAssignedTargets']}\t"
                f"{row['FinalCommonScheduleSubstitutionStatus']}\t"
                f"{row['SelectionExecutionDisposition']}\t"
                f"{row['AuxiliaryOutputDisposition']}\n"
            ).encode("utf-8")
        )

    return digest.hexdigest().upper()


def final_adapter_binding_sha(
    kernel_rows,
    selection_rows,
):
    kernel_sha = candidate_adapter_map_sha(
        kernel_rows
    )

    selection_sha = selection_policy_sha(
        selection_rows
    )

    digest = hashlib.sha256()

    digest.update(
        (
            f"COMMON_PROTOCOL_SHA256={COMMON_PROTOCOL_SHA}\n"
            f"GATE83_ARTIFACT_MANIFEST_SHA256="
            f"{GATE83_ARTIFACT_MANIFEST_SHA}\n"
            f"SIX_METHOD_BINDING_SHA256={SIX_METHOD_BINDING_SHA}\n"
            f"PARAMETER_RECORD_SET_SHA256={PARAMETER_RECORD_SET_SHA}\n"
            f"GATE89_ARTIFACT_MANIFEST_SHA256="
            f"{GATE89_ARTIFACT_MANIFEST_SHA}\n"
            f"GATE90_EXECUTION_CONTRACT_SHA256="
            f"{GATE90_EXECUTION_CONTRACT_SHA}\n"
            f"GATE91_EXECUTION_UNIT_MAP_SHA256="
            f"{GATE91_EXECUTION_UNIT_MAP_SHA}\n"
            f"GATE92_ADAPTER_CHAIN_MAP_SHA256="
            f"{GATE92_ADAPTER_CHAIN_MAP_SHA}\n"
            f"GATE93_CANDIDATE_ADAPTER_MAP_SHA256="
            f"{GATE93_CANDIDATE_ADAPTER_MAP_SHA}\n"
            f"GATE94_FEDLE_TRANSITIVE_FLOW_PROOF_SHA256="
            f"{GATE94_FEDLE_TRANSITIVE_FLOW_PROOF_SHA}\n"
            f"GATE95_FEDLE_SECOND_OUTPUT_PROOF_SHA256="
            f"{GATE95_FEDLE_SECOND_OUTPUT_PROOF_SHA}\n"
            f"PARTICIPATION_AUTHORITY={PARTICIPATION_AUTHORITY}\n"
            f"PARTICIPATION_RULE={PARTICIPATION_RULE}\n"
            f"COMMON_LOCAL_TRAINING_RULE="
            f"{COMMON_LOCAL_TRAINING_RULE}\n"
            f"KERNEL_MAP_SHA256={kernel_sha}\n"
            f"SELECTION_POLICY_SHA256={selection_sha}\n"
        ).encode("utf-8")
    )

    return (
        digest.hexdigest().upper(),
        kernel_sha,
        selection_sha,
    )


def artifact_manifest_sha(rows):
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
            ).encode("utf-8")
        )

    return digest.hexdigest().upper()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gate83-freeze-json",
        required=True,
    )

    parser.add_argument(
        "--gate89-freeze-json",
        required=True,
    )

    parser.add_argument(
        "--gate90-diagnostic-json",
        required=True,
    )

    parser.add_argument(
        "--gate91-diagnostic-json",
        required=True,
    )

    parser.add_argument(
        "--gate92-diagnostic-json",
        required=True,
    )

    parser.add_argument(
        "--gate93-diagnostic-json",
        required=True,
    )

    parser.add_argument(
        "--gate93-kernel-map-csv",
        required=True,
    )

    parser.add_argument(
        "--gate93-selection-audit-csv",
        required=True,
    )

    parser.add_argument(
        "--gate94-diagnostic-json",
        required=True,
    )

    parser.add_argument(
        "--gate95-diagnostic-json",
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
    # Gate A - verify immutable chain.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE A - VERIFY GATE-83 / 89 / 90 / 91 / 92 / 93 / 94 / 95")
    print("=" * 60)
    print("")

    gate83 = load_json(
        args.gate83_freeze_json
    )

    require_equal(
        gate83.get(
            "status"
        ),
        "FROZEN",
        "Gate-83 status",
    )

    require_equal(
        gate83.get(
            "common_protocol_binding_sha256"
        ),
        COMMON_PROTOCOL_SHA,
        "Gate-83 common protocol SHA256",
    )

    require_equal(
        gate83.get(
            "frozen_attack_training_execution_artifact_manifest_sha256"
        ),
        GATE83_ARTIFACT_MANIFEST_SHA,
        "Gate-83 artifact manifest SHA256",
    )

    gate89 = load_json(
        args.gate89_freeze_json
    )

    require_equal(
        gate89.get(
            "status"
        ),
        "FROZEN",
        "Gate-89 status",
    )

    require_equal(
        gate89.get(
            "frozen_six_method_binding_sha256"
        ),
        SIX_METHOD_BINDING_SHA,
        "Gate-89 six-method binding SHA256",
    )

    require_equal(
        gate89.get(
            "frozen_parameter_record_set_sha256"
        ),
        PARAMETER_RECORD_SET_SHA,
        "Gate-89 parameter set SHA256",
    )

    require_equal(
        gate89.get(
            "frozen_six_method_binding_artifact_manifest_sha256"
        ),
        GATE89_ARTIFACT_MANIFEST_SHA,
        "Gate-89 artifact manifest SHA256",
    )

    gate90 = load_json(
        args.gate90_diagnostic_json
    )

    require_equal(
        gate90[
            "candidate_execution_contract"
        ][
            "candidate_execution_contract_sha256"
        ],
        GATE90_EXECUTION_CONTRACT_SHA,
        "Gate-90 execution contract SHA256",
    )

    gate91 = load_json(
        args.gate91_diagnostic_json
    )

    require_equal(
        gate91[
            "candidate_execution_unit_map"
        ][
            "candidate_execution_unit_map_sha256"
        ],
        GATE91_EXECUTION_UNIT_MAP_SHA,
        "Gate-91 execution-unit map SHA256",
    )

    gate92 = load_json(
        args.gate92_diagnostic_json
    )

    require_equal(
        gate92[
            "candidate_adapter_chain_map"
        ][
            "candidate_adapter_chain_map_sha256"
        ],
        GATE92_ADAPTER_CHAIN_MAP_SHA,
        "Gate-92 adapter-chain map SHA256",
    )

    gate93 = load_json(
        args.gate93_diagnostic_json
    )

    require_equal(
        gate93.get(
            "status"
        ),
        "DIAGNOSIS COMPLETE",
        "Gate-93 status",
    )

    require_equal(
        gate93[
            "provisional_adapter_map"
        ][
            "candidate_minimal_adapter_map_sha256"
        ],
        GATE93_CANDIDATE_ADAPTER_MAP_SHA,
        "Gate-93 candidate adapter map SHA256",
    )

    require_equal(
        gate93[
            "provisional_adapter_map"
        ][
            "unresolved_methods"
        ],
        [
            "FEDLE_ADAPTED"
        ],
        "Gate-93 unresolved method list",
    )

    gate94 = load_json(
        args.gate94_diagnostic_json
    )

    require_equal(
        gate94.get(
            "status"
        ),
        "DIAGNOSIS COMPLETE",
        "Gate-94 status",
    )

    require_equal(
        gate94[
            "fedle_resolution"
        ][
            "fedle_transitive_flow_proof_sha256"
        ],
        GATE94_FEDLE_TRANSITIVE_FLOW_PROOF_SHA,
        "Gate-94 FedLE flow proof SHA256",
    )

    require_true(
        gate94[
            "fedle_resolution"
        ][
            "selected_to_training_to_aggregation_proof"
        ]
        is True,
        "Gate-94 selected->training->aggregation proof is not true.",
    )

    gate95 = load_json(
        args.gate95_diagnostic_json
    )

    require_equal(
        gate95.get(
            "status"
        ),
        "DIAGNOSIS COMPLETE",
        "Gate-95 status",
    )

    require_equal(
        gate95.get(
            "disposition"
        ),
        (
            "FEDLE_SECOND_OUTPUT_PROVEN_AUDIT_ONLY_"
            "SIX_METHOD_ADAPTER_MAP_FREEZE_READY"
        ),
        "Gate-95 disposition",
    )

    require_equal(
        gate95[
            "fedle_second_output_resolution"
        ][
            "fedle_second_output_independence_proof_sha256"
        ],
        GATE95_FEDLE_SECOND_OUTPUT_PROOF_SHA,
        "Gate-95 FedLE second-output proof SHA256",
    )

    require_true(
        gate95[
            "fedle_second_output_resolution"
        ][
            "second_output_audit_only_proof"
        ]
        is True,
        "Gate-95 FedLE second output is not audit-only.",
    )

    require_equal(
        gate95[
            "fedle_second_output_resolution"
        ][
            "forbidden_scientific_use_count"
        ],
        0,
        "Gate-95 forbidden scientific use count",
    )

    require_equal(
        gate95[
            "fedle_second_output_resolution"
        ][
            "final_common_schedule_substitution_status"
        ],
        "FEDLE_COMMON_SCHEDULE_SUBSTITUTION_FULLY_PROVEN",
        "Gate-95 final FedLE substitution status",
    )

    for gate_name, gate in (
        (
            "Gate-93",
            gate93,
        ),
        (
            "Gate-94",
            gate94,
        ),
        (
            "Gate-95",
            gate95,
        ),
    ):
        require_equal(
            gate[
                "scientific_boundary"
            ][
                "technical_optimizer_steps_executed"
            ],
            0,
            f"{gate_name} technical optimizer steps",
        )

        require_true(
            gate[
                "scientific_boundary"
            ][
                "scientific_training_started"
            ]
            is False,
            f"{gate_name} says scientific training started.",
        )

    print(
        "Gate-83 common protocol: FROZEN"
    )
    print(
        "Gate-89 six-method binding: FROZEN"
    )
    print(
        "Gate-90/91/92/93 diagnostics: BOUND"
    )
    print(
        "Gate-94 FedLE transitive scientific path: PASS"
    )
    print(
        "Gate-95 FedLE second-output audit-only proof: PASS"
    )
    print(
        "Technical optimizer steps: 0"
    )
    print(
        "Scientific training: NO"
    )

    # ------------------------------------------------------------------
    # Gate B - verify exact 14-row kernel map.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE B - VERIFY EXACT SIX-METHOD KERNEL MAP")
    print("=" * 60)
    print("")

    kernel_rows = read_csv(
        args.gate93_kernel_map_csv
    )

    require_equal(
        len(
            kernel_rows
        ),
        14,
        "Kernel map row count",
    )

    require_equal(
        candidate_adapter_map_sha(
            kernel_rows
        ),
        GATE93_CANDIDATE_ADAPTER_MAP_SHA,
        "Gate-93 candidate adapter-map replay",
    )

    observed_by_method = {}

    for row in kernel_rows:
        method_id = row[
            "MethodID"
        ]

        require_true(
            method_id in EXPECTED_KERNELS,
            f"Unexpected method in kernel map: {method_id}",
        )

        key = (
            row[
                "Role"
            ],
            row[
                "CallableName"
            ],
        )

        require_true(
            key in EXPECTED_KERNELS[
                method_id
            ],
            (
                f"Unexpected kernel for {method_id}: "
                f"{key}"
            ),
        )

        require_equal(
            row[
                "ExecutionDisposition"
            ],
            EXPECTED_KERNELS[
                method_id
            ][
                key
            ],
            (
                f"Execution disposition for "
                f"{method_id} {key}"
            ),
        )

        resolved_path = Path(
            row[
                "ResolvedPath"
            ]
        )

        require_true(
            resolved_path.exists(),
            (
                f"Resolved source/dependency missing for "
                f"{method_id} {key}: {resolved_path}"
            ),
        )

        require_equal(
            sha256_file(
                resolved_path
            ),
            row[
                "ResolvedSHA256"
            ],
            (
                f"Resolved source/dependency SHA256 for "
                f"{method_id} {key}"
            ),
        )

        observed_by_method.setdefault(
            method_id,
            set(),
        ).add(
            key
        )

        print(
            f"{method_id}: {key[0]} -> {key[1]} PASS"
        )

    require_equal(
        set(
            observed_by_method
        ),
        set(
            METHODS
        ),
        "Kernel-map method set",
    )

    for method_id in METHODS:
        require_equal(
            observed_by_method[
                method_id
            ],
            set(
                EXPECTED_KERNELS[
                    method_id
                ]
            ),
            f"Exact kernel set for {method_id}",
        )

    # ------------------------------------------------------------------
    # Gate C - finalize selection-substitution policy.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE C - FINALIZE COMMON-SCHEDULE SELECTION POLICY")
    print("=" * 60)
    print("")

    gate93_selection_rows = read_csv(
        args.gate93_selection_audit_csv
    )

    require_equal(
        len(
            gate93_selection_rows
        ),
        6,
        "Gate-93 selection-audit row count",
    )

    selection_by_method = {
        row[
            "MethodID"
        ]: row
        for row in gate93_selection_rows
    }

    require_equal(
        set(
            selection_by_method
        ),
        set(
            METHODS
        ),
        "Selection-audit method set",
    )

    final_selection_rows = []

    for method_id in METHODS:
        source = selection_by_method[
            method_id
        ]

        require_equal(
            source[
                "ParticipationAuthority"
            ],
            PARTICIPATION_AUTHORITY,
            f"Participation authority for {method_id}",
        )

        require_equal(
            source[
                "SelectionCallable"
            ],
            EXPECTED_SELECTION_CALLABLE[
                method_id
            ],
            f"Selection callable for {method_id}",
        )

        if EXPECTED_SELECTION_CALLABLE[
            method_id
        ]:
            require_equal(
                source[
                    "SelectionExecutionDisposition"
                ],
                (
                    "DO_NOT_EXECUTE_FOR_PARTICIPANT_IDENTITY_"
                    "INJECT_FROZEN_SELECTED_LIST"
                ),
                f"Selection execution disposition for {method_id}",
            )

        if method_id == "FEDLE_ADAPTED":
            final_status = (
                "FEDLE_COMMON_SCHEDULE_SUBSTITUTION_FULLY_PROVEN"
            )

            auxiliary_disposition = (
                "AUDIT_ONLY_SECOND_OUTPUT_OMITTED_FROM_SCIENTIFIC_PATH"
            )

        else:
            final_status = source[
                "CommonScheduleSubstitutionStatus"
            ]

            auxiliary_disposition = (
                "NOT_APPLICABLE"
            )

        require_equal(
            final_status,
            EXPECTED_FINAL_SUBSTITUTION_STATUS[
                method_id
            ],
            f"Final substitution status for {method_id}",
        )

        final_row = {
            "MethodID": (
                method_id
            ),
            "ParticipationAuthority": (
                PARTICIPATION_AUTHORITY
            ),
            "MethodSelectionCallable": (
                source[
                    "SelectionCallable"
                ]
            ),
            "OriginalSelectionAssignedTargets": (
                source[
                    "SelectionAssignedTargets"
                ]
            ),
            "FinalCommonScheduleSubstitutionStatus": (
                final_status
            ),
            "SelectionExecutionDisposition": (
                source[
                    "SelectionExecutionDisposition"
                ]
            ),
            "AuxiliaryOutputDisposition": (
                auxiliary_disposition
            ),
        }

        final_selection_rows.append(
            final_row
        )

        print(
            f"{method_id}: {final_status}"
        )

    # ------------------------------------------------------------------
    # Gate D - compute frozen adapter binding.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE D - COMPUTE FROZEN ADAPTER BINDING")
    print("=" * 60)
    print("")

    (
        final_binding_sha,
        kernel_map_sha,
        selection_policy_sha_value,
    ) = final_adapter_binding_sha(
        kernel_rows,
        final_selection_rows,
    )

    require_equal(
        kernel_map_sha,
        GATE93_CANDIDATE_ADAPTER_MAP_SHA,
        "Frozen kernel-map SHA256",
    )

    print(
        f"Frozen kernel-map SHA256: "
        f"{kernel_map_sha}"
    )

    print(
        f"Frozen selection-policy SHA256: "
        f"{selection_policy_sha_value}"
    )

    print(
        f"Final adapter protocol binding SHA256: "
        f"{final_binding_sha}"
    )

    # ------------------------------------------------------------------
    # Gate E - write immutable frozen artifacts.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE E - WRITE IMMUTABLE ADAPTER FREEZE ARTIFACTS")
    print("=" * 60)
    print("")

    frozen_kernel_path = (
        output_root
        /
        "FROZEN_MINIMAL_ADAPTER_KERNEL_MAP.csv"
    )

    frozen_selection_path = (
        output_root
        /
        "FROZEN_COMMON_SCHEDULE_SELECTION_POLICY.csv"
    )

    write_csv(
        frozen_kernel_path,
        kernel_rows,
        list(
            kernel_rows[
                0
            ].keys()
        ),
    )

    write_csv(
        frozen_selection_path,
        final_selection_rows,
        [
            "MethodID",
            "ParticipationAuthority",
            "MethodSelectionCallable",
            "OriginalSelectionAssignedTargets",
            "FinalCommonScheduleSubstitutionStatus",
            "SelectionExecutionDisposition",
            "AuxiliaryOutputDisposition",
        ],
    )

    provenance_dir = (
        output_root
        /
        "frozen_adapter_source_provenance"
    )

    provenance_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    unique_sources = {}

    for row in kernel_rows:
        key = (
            row[
                "ResolvedSHA256"
            ],
            row[
                "ResolvedPath"
            ],
        )

        unique_sources[
            key
        ] = Path(
            row[
                "ResolvedPath"
            ]
        )

    provenance_rows = []

    for index, (
        (
            source_sha,
            source_path_text,
        ),
        source_path,
    ) in enumerate(
        sorted(
            unique_sources.items(),
            key=lambda item: (
                item[
                    0
                ][
                    0
                ],
                item[
                    0
                ][
                    1
                ],
            ),
        ),
        start=1,
    ):
        destination_name = (
            f"{index:02d}__"
            f"{source_sha[:12]}__"
            f"{source_path.name}"
        )

        destination = (
            provenance_dir
            /
            destination_name
        )

        shutil.copyfile(
            source_path,
            destination,
        )

        require_equal(
            sha256_file(
                destination
            ),
            source_sha,
            (
                f"Frozen adapter source copy SHA256 "
                f"for {source_path}"
            ),
        )

        methods_roles = sorted(
            {
                (
                    row[
                        "MethodID"
                    ]
                    +
                    ":"
                    +
                    row[
                        "Role"
                    ]
                    +
                    ":"
                    +
                    row[
                        "CallableName"
                    ]
                )
                for row in kernel_rows
                if row[
                    "ResolvedSHA256"
                ]
                ==
                source_sha
                and
                row[
                    "ResolvedPath"
                ]
                ==
                source_path_text
            }
        )

        provenance_rows.append(
            {
                "FrozenFileName": (
                    destination_name
                ),
                "OriginalResolvedPath": (
                    source_path_text
                ),
                "SHA256": (
                    source_sha
                ),
                "BoundMethodRoleCallables": (
                    ";".join(
                        methods_roles
                    )
                ),
            }
        )

    provenance_index_path = (
        output_root
        /
        "FROZEN_ADAPTER_SOURCE_PROVENANCE.csv"
    )

    write_csv(
        provenance_index_path,
        provenance_rows,
        [
            "FrozenFileName",
            "OriginalResolvedPath",
            "SHA256",
            "BoundMethodRoleCallables",
        ],
    )

    protocol = {
        "status": (
            "FROZEN"
        ),
        "freeze_id": (
            FREEZE_ID
        ),
        "adapter_protocol_id": (
            ADAPTER_PROTOCOL_ID
        ),
        "immutable_binding": {
            "common_protocol_binding_sha256": (
                COMMON_PROTOCOL_SHA
            ),
            "gate83_artifact_manifest_sha256": (
                GATE83_ARTIFACT_MANIFEST_SHA
            ),
            "six_method_binding_sha256": (
                SIX_METHOD_BINDING_SHA
            ),
            "parameter_record_set_sha256": (
                PARAMETER_RECORD_SET_SHA
            ),
            "gate89_artifact_manifest_sha256": (
                GATE89_ARTIFACT_MANIFEST_SHA
            ),
            "gate90_execution_contract_sha256": (
                GATE90_EXECUTION_CONTRACT_SHA
            ),
            "gate91_execution_unit_map_sha256": (
                GATE91_EXECUTION_UNIT_MAP_SHA
            ),
            "gate92_adapter_chain_map_sha256": (
                GATE92_ADAPTER_CHAIN_MAP_SHA
            ),
            "gate93_candidate_adapter_map_sha256": (
                GATE93_CANDIDATE_ADAPTER_MAP_SHA
            ),
            "gate94_fedle_transitive_flow_proof_sha256": (
                GATE94_FEDLE_TRANSITIVE_FLOW_PROOF_SHA
            ),
            "gate95_fedle_second_output_proof_sha256": (
                GATE95_FEDLE_SECOND_OUTPUT_PROOF_SHA
            ),
            "frozen_kernel_map_sha256": (
                kernel_map_sha
            ),
            "frozen_selection_policy_sha256": (
                selection_policy_sha_value
            ),
            "final_adapter_protocol_binding_sha256": (
                final_binding_sha
            ),
        },
        "participation_authority": (
            PARTICIPATION_AUTHORITY
        ),
        "participation_rule": (
            PARTICIPATION_RULE
        ),
        "common_local_training_rule": (
            COMMON_LOCAL_TRAINING_RULE
        ),
        "method_count": (
            6
        ),
        "kernel_row_count": (
            14
        ),
        "method_specific_selection_executed_for_participant_identity": (
            False
        ),
        "fedle_auxiliary_output": (
            "AUDIT_ONLY_SECOND_OUTPUT_OMITTED_FROM_SCIENTIFIC_PATH"
        ),
        "scientific_boundary": {
            "execution_adapter_frozen": (
                True
            ),
            "method_implementations_modified": (
                False
            ),
            "technical_optimizer_steps_executed": (
                0
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
        "FROZEN_SIX_METHOD_ADAPTER_PROTOCOL.json"
    )

    write_json(
        protocol_path,
        protocol,
    )

    # ------------------------------------------------------------------
    # Gate F - immutable artifact manifest.
    # ------------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE F - WRITE IMMUTABLE ARTIFACT MANIFEST")
    print("=" * 60)
    print("")

    artifact_rows = []

    fixed_artifacts = [
        (
            "FROZEN_KERNEL_MAP",
            frozen_kernel_path,
        ),
        (
            "FROZEN_SELECTION_POLICY",
            frozen_selection_path,
        ),
        (
            "FROZEN_SOURCE_PROVENANCE_INDEX",
            provenance_index_path,
        ),
        (
            "FROZEN_ADAPTER_PROTOCOL",
            protocol_path,
        ),
    ]

    for role, path in fixed_artifacts:
        artifact_rows.append(
            {
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
            }
        )

    for row in provenance_rows:
        path = (
            provenance_dir
            /
            row[
                "FrozenFileName"
            ]
        )

        artifact_rows.append(
            {
                "ArtifactRole": (
                    "ADAPTER_SOURCE_PROVENANCE"
                ),
                "FileName": (
                    str(
                        Path(
                            "frozen_adapter_source_provenance"
                        )
                        /
                        path.name
                    ).replace(
                        "\\",
                        "/",
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
            }
        )

    artifact_manifest_path = (
        output_root
        /
        "FROZEN_SIX_METHOD_ADAPTER_ARTIFACT_MANIFEST.csv"
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

    artifact_manifest_sha_value = artifact_manifest_sha(
        artifact_rows
    )

    freeze_state = {
        "status": (
            "FROZEN"
        ),
        "freeze_id": (
            FREEZE_ID
        ),
        "adapter_protocol_id": (
            ADAPTER_PROTOCOL_ID
        ),
        "frozen_kernel_map_sha256": (
            kernel_map_sha
        ),
        "frozen_selection_policy_sha256": (
            selection_policy_sha_value
        ),
        "final_adapter_protocol_binding_sha256": (
            final_binding_sha
        ),
        "frozen_six_method_adapter_artifact_manifest_sha256": (
            artifact_manifest_sha_value
        ),
        "scientific_boundary": {
            "execution_adapter_frozen": (
                True
            ),
            "method_implementations_modified": (
                False
            ),
            "technical_optimizer_steps_executed": (
                0
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

    write_json(
        output_root
        /
        "SIX_METHOD_MINIMAL_ADAPTER_KERNEL_FREEZE.json",
        freeze_state,
    )

    # ------------------------------------------------------------------
    # Human-readable report.
    # ------------------------------------------------------------------

    report = []

    add = report.append

    add(
        "CICIoT2023 SIX-METHOD MINIMAL ADAPTER-KERNEL MAP FREEZE"
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
        f"Adapter protocol ID: {ADAPTER_PROTOCOL_ID}"
    )
    add("")

    add("IMMUTABLE UPSTREAM BINDING")
    add("-" * 78)
    add(
        f"Common protocol binding SHA256: "
        f"{COMMON_PROTOCOL_SHA}"
    )
    add(
        f"Gate-83 artifact-manifest SHA256: "
        f"{GATE83_ARTIFACT_MANIFEST_SHA}"
    )
    add(
        f"Six-method binding SHA256: "
        f"{SIX_METHOD_BINDING_SHA}"
    )
    add(
        f"Parameter-record set SHA256: "
        f"{PARAMETER_RECORD_SET_SHA}"
    )
    add(
        f"Gate-89 artifact-manifest SHA256: "
        f"{GATE89_ARTIFACT_MANIFEST_SHA}"
    )
    add(
        f"Gate-90 execution-contract SHA256: "
        f"{GATE90_EXECUTION_CONTRACT_SHA}"
    )
    add(
        f"Gate-91 execution-unit map SHA256: "
        f"{GATE91_EXECUTION_UNIT_MAP_SHA}"
    )
    add(
        f"Gate-92 adapter-chain map SHA256: "
        f"{GATE92_ADAPTER_CHAIN_MAP_SHA}"
    )
    add(
        f"Gate-93 candidate adapter-map SHA256: "
        f"{GATE93_CANDIDATE_ADAPTER_MAP_SHA}"
    )
    add(
        f"Gate-94 FedLE transitive flow-proof SHA256: "
        f"{GATE94_FEDLE_TRANSITIVE_FLOW_PROOF_SHA}"
    )
    add(
        f"Gate-95 FedLE second-output proof SHA256: "
        f"{GATE95_FEDLE_SECOND_OUTPUT_PROOF_SHA}"
    )
    add("")

    add("FROZEN PARTICIPATION AUTHORITY")
    add("-" * 78)
    add(
        f"Participant identities: {PARTICIPATION_AUTHORITY}"
    )
    add(
        f"Rule: {PARTICIPATION_RULE}"
    )
    add(
        "Method-specific selection executed for participant identity: NO"
    )
    add(
        "Frozen selected list injected where legacy method selection "
        "previously supplied participant identities: YES"
    )
    add("")

    add("FROZEN LOCAL-TRAINING RULE")
    add("-" * 78)
    add(
        COMMON_LOCAL_TRAINING_RULE
    )
    add(
        "FedProx local proximal override: EXECUTE"
    )
    add(
        "Other methods use Gate-83 common local-training semantics: YES"
    )
    add("")

    add("FROZEN SIX-METHOD ADAPTER MAP")
    add("-" * 78)

    for method_id in METHODS:
        add(
            f"{method_id}: FROZEN"
        )

        for row in sorted(
            (
                item
                for item in kernel_rows
                if item[
                    "MethodID"
                ]
                ==
                method_id
            ),
            key=lambda item: (
                item[
                    "Role"
                ],
                item[
                    "CallableName"
                ],
            ),
        ):
            add(
                f"  {row['Role']}: "
                f"{row['CallableName']}"
            )
            add(
                f"    resolution_scope="
                f"{row['ResolutionScope']}"
            )
            add(
                f"    source_sha256="
                f"{row['ResolvedSHA256']}"
            )
            add(
                f"    execution_disposition="
                f"{row['ExecutionDisposition']}"
            )

        policy = [
            row
            for row in final_selection_rows
            if row[
                "MethodID"
            ]
            ==
            method_id
        ][
            0
        ]

        add(
            f"  selection_callable="
            f"{policy['MethodSelectionCallable']}"
        )
        add(
            f"  final_substitution_status="
            f"{policy['FinalCommonScheduleSubstitutionStatus']}"
        )
        add(
            f"  selection_execution="
            f"{policy['SelectionExecutionDisposition']}"
        )
        add(
            f"  auxiliary_output="
            f"{policy['AuxiliaryOutputDisposition']}"
        )

    add("")

    add("FROZEN SCIENTIFIC FINGERPRINTS")
    add("-" * 78)
    add(
        f"Frozen kernel-map SHA256: "
        f"{kernel_map_sha}"
    )
    add(
        f"Frozen selection-policy SHA256: "
        f"{selection_policy_sha_value}"
    )
    add(
        f"Final adapter protocol binding SHA256: "
        f"{final_binding_sha}"
    )
    add("")

    add("IMMUTABLE FREEZE ARTIFACT")
    add("-" * 78)
    add(
        f"Combined six-method adapter artifact-manifest SHA256: "
        f"{artifact_manifest_sha_value}"
    )
    add("")

    add("SCIENTIFIC BOUNDARY")
    add("-" * 78)
    add(
        "Execution adapter frozen: YES"
    )
    add(
        "Method implementations modified: NO"
    )
    add(
        "Technical optimizer steps executed: 0"
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
        "Build the minimal deterministic CICIoT2023 six-method end-to-end "
        "technical validator bound to the Gate-83 common protocol, Gate-89 "
        "method bindings, and this frozen adapter protocol. The validator "
        "must remain technical-only and minimal before any scientific run."
    )

    (
        output_root
        /
        "SIX_METHOD_MINIMAL_ADAPTER_KERNEL_FREEZE_REPORT.txt"
    ).write_bytes(
        (
            "\n".join(
                report
            )
            +
            "\n"
        ).encode("utf-8")
    )

    print("")
    print("=" * 60)
    print(
        "STATUS: FROZEN"
    )
    print(
        f"FROZEN KERNEL-MAP SHA256: "
        f"{kernel_map_sha}"
    )
    print(
        f"FROZEN SELECTION-POLICY SHA256: "
        f"{selection_policy_sha_value}"
    )
    print(
        f"FINAL ADAPTER PROTOCOL BINDING SHA256: "
        f"{final_binding_sha}"
    )
    print(
        f"ARTIFACT-MANIFEST SHA256: "
        f"{artifact_manifest_sha_value}"
    )
    print(
        "EXECUTION ADAPTER FROZEN: YES"
    )
    print(
        "TECHNICAL OPTIMIZER STEPS EXECUTED: 0"
    )
    print(
        "END-TO-END VALIDATION STARTED: NO"
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
