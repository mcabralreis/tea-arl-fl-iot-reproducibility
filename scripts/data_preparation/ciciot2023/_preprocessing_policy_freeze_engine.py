import argparse
import csv
import hashlib
import json
import math
import struct
import sys
from pathlib import Path


EXPECTED_DATASET_ID = (
    "CICIoT2023_TASK7_UNIQUE_EXACT_V1"
)

EXPECTED_DATASET_MANIFEST_SHA256 = (
    "D86976FD34A72E4E60249C536505165490F878D71A21301AC6F6FB7D387D6C8D"
)

EXPECTED_SPLIT_ID = (
    "CICIoT2023_TASK7_UNIQUE_EXACT_SPLIT_80_10_10_V1"
)

EXPECTED_SPLIT_MANIFEST_SHA256 = (
    "67A95B6665C44169513001D8997708D51948BF626F03EF0D293FCE76072BD2EA"
)

EXPECTED_TRAIN_ROWS = 16_843_095
EXPECTED_FEATURES = 39

EXPECTED_GATE50_POLICY_ID = (
    "KEEP_ALL_UNIQUE_TRAIN_PLUS_SQRT_INVERSE_FREQUENCY_WEIGHTED_CE"
)

EXPECTED_GATE51_SAMPLE_ROWS = 499_727

EXPECTED_GATE51_SAMPLE_ID_SHA256 = (
    "BED1D61A5B44233699B7DC16D1DBE7AFD1DF8BDD8A5BB31C90682B162D85F19B"
)

POLICY_ID = (
    "SIGNED_LOG1P_TRAIN_MAXABS_FLOAT32_V1"
)


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        return json.load(handle)


def read_csv_rows(path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def write_csv(
    path,
    rows,
    fieldnames,
):
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            block = handle.read(
                8 * 1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest().upper()


def require_equal(
    observed,
    expected,
    message,
):
    if observed != expected:
        raise RuntimeError(
            f"{message}: expected {expected}, observed {observed}"
        )


def require_true(
    condition,
    message,
):
    if not condition:
        raise RuntimeError(
            message
        )


def load_feature_names(path):
    rows = read_csv_rows(
        path
    )

    ordered = sorted(
        (
            int(
                row[
                    "IndexOneBased"
                ]
            ),
            row[
                "ColumnName"
            ],
        )
        for row
        in rows
    )

    names = [
        name
        for _, name
        in ordered
    ]

    require_equal(
        len(
            names
        ),
        EXPECTED_FEATURES,
        "Feature-name count",
    )

    return names


def signed_log1p(value):
    if value > 0.0:
        return math.log1p(
            value
        )

    if value < 0.0:
        return -math.log1p(
            -value
        )

    return 0.0


def float64_hex(value):
    return struct.pack(
        ">d",
        float(
            value
        ),
    ).hex().upper()


def float32_hex(value):
    return struct.pack(
        ">f",
        float(
            value
        ),
    ).hex().upper()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gate46-json",
        required=True,
    )

    parser.add_argument(
        "--gate48-json",
        required=True,
    )

    parser.add_argument(
        "--gate50-json",
        required=True,
    )

    parser.add_argument(
        "--gate51-json",
        required=True,
    )

    parser.add_argument(
        "--gate51-report",
        required=True,
    )

    parser.add_argument(
        "--gate51-profile",
        required=True,
    )

    parser.add_argument(
        "--gate51-quantiles",
        required=True,
    )

    parser.add_argument(
        "--expected-columns",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    paths = {
        "Gate46Json": Path(
            args.gate46_json
        ),
        "Gate48Json": Path(
            args.gate48_json
        ),
        "Gate50Json": Path(
            args.gate50_json
        ),
        "Gate51Json": Path(
            args.gate51_json
        ),
        "Gate51Report": Path(
            args.gate51_report
        ),
        "Gate51ProfileCsv": Path(
            args.gate51_profile
        ),
        "Gate51QuantilesCsv": Path(
            args.gate51_quantiles
        ),
        "ExpectedColumnsCsv": Path(
            args.expected_columns
        ),
    }

    output_root = Path(
        args.output
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Gate A - verify immutable scientific state
    # --------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE A - VERIFY DATASET, SPLIT, TRAINING, AND PROFILE STATE")
    print("=" * 60)
    print("")

    gate46 = load_json(
        paths[
            "Gate46Json"
        ]
    )

    require_equal(
        gate46.get(
            "status"
        ),
        "PASS",
        "Gate-46 status",
    )

    require_equal(
        gate46.get(
            "dataset_id"
        ),
        EXPECTED_DATASET_ID,
        "Gate-46 dataset ID",
    )

    require_equal(
        gate46[
            "integrity"
        ][
            "dataset_manifest_sha256"
        ],
        EXPECTED_DATASET_MANIFEST_SHA256,
        "Gate-46 dataset fingerprint",
    )

    gate48 = load_json(
        paths[
            "Gate48Json"
        ]
    )

    require_equal(
        gate48.get(
            "status"
        ),
        "PASS",
        "Gate-48 status",
    )

    require_equal(
        gate48.get(
            "split_dataset_id"
        ),
        EXPECTED_SPLIT_ID,
        "Gate-48 split ID",
    )

    require_equal(
        gate48[
            "split_assignment_manifest_sha256"
        ],
        EXPECTED_SPLIT_MANIFEST_SHA256,
        "Gate-48 split fingerprint",
    )

    gate50 = load_json(
        paths[
            "Gate50Json"
        ]
    )

    require_equal(
        gate50.get(
            "status"
        ),
        "FROZEN",
        "Gate-50 status",
    )

    require_equal(
        gate50[
            "primary_training_imbalance_policy"
        ][
            "policy_id"
        ],
        EXPECTED_GATE50_POLICY_ID,
        "Gate-50 primary policy",
    )

    gate51 = load_json(
        paths[
            "Gate51Json"
        ]
    )

    require_equal(
        gate51.get(
            "status"
        ),
        "DIAGNOSIS_COMPLETE",
        "Gate-51 status",
    )

    require_equal(
        gate51[
            "scope"
        ][
            "split"
        ],
        "TRAIN_ONLY",
        "Gate-51 profile scope",
    )

    require_equal(
        int(
            gate51[
                "scope"
            ][
                "train_rows"
            ]
        ),
        EXPECTED_TRAIN_ROWS,
        "Gate-51 TRAIN row count",
    )

    require_equal(
        int(
            gate51[
                "scope"
            ][
                "features"
            ]
        ),
        EXPECTED_FEATURES,
        "Gate-51 feature count",
    )

    require_equal(
        int(
            gate51[
                "deterministic_quantile_sample"
            ][
                "observed_rows"
            ]
        ),
        EXPECTED_GATE51_SAMPLE_ROWS,
        "Gate-51 sample row count",
    )

    require_equal(
        gate51[
            "deterministic_quantile_sample"
        ][
            "sample_vector_id_sha256"
        ],
        EXPECTED_GATE51_SAMPLE_ID_SHA256,
        "Gate-51 sample ID SHA256",
    )

    findings = gate51[
        "findings"
    ]

    require_equal(
        findings[
            "constant_train_features"
        ],
        [],
        "Gate-51 constant TRAIN features",
    )

    require_equal(
        findings[
            "features_with_negative_train_values"
        ],
        [
            "IAT"
        ],
        "Gate-51 negative-valued feature list",
    )

    require_equal(
        int(
            findings[
                "all_nonnegative_feature_count"
            ]
        ),
        38,
        "Gate-51 nonnegative feature count",
    )

    require_equal(
        len(
            findings[
                "raw_exact_max_abs_z_ge_100"
            ]
        ),
        10,
        "Gate-51 raw max-z >= 100 count",
    )

    require_equal(
        len(
            findings[
                "raw_exact_max_abs_z_ge_1000"
            ]
        ),
        1,
        "Gate-51 raw max-z >= 1000 count",
    )

    require_true(
        findings[
            "all_features_finite_float32_range_safe"
        ],
        "Gate-51 float32 finite-range safety failed.",
    )

    print(
        "Gate-46 immutable dataset: PASS"
    )
    print(
        "Gate-48 immutable split: PASS"
    )
    print(
        "Gate-50 training policy: PASS"
    )
    print(
        "Gate-51 TRAIN-only profile: PASS"
    )

    # --------------------------------------------------------------
    # Gate B - verify detailed feature evidence
    # --------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE B - VERIFY DETAILED TRAIN FEATURE EVIDENCE")
    print("=" * 60)
    print("")

    feature_names = load_feature_names(
        paths[
            "ExpectedColumnsCsv"
        ]
    )

    profile_rows = read_csv_rows(
        paths[
            "Gate51ProfileCsv"
        ]
    )

    require_equal(
        len(
            profile_rows
        ),
        EXPECTED_FEATURES,
        "Gate-51 profile row count",
    )

    require_equal(
        [
            row[
                "FeatureName"
            ]
            for row
            in profile_rows
        ],
        feature_names,
        "Gate-51 feature order",
    )

    profile_by_name = {
        row[
            "FeatureName"
        ]: row
        for row
        in profile_rows
    }

    require_true(
        float(
            profile_by_name[
                "IAT"
            ][
                "RawExactMaximumAbsoluteZScore"
            ]
        )
        >
        2000.0,
        "IAT raw extreme-z evidence is weaker than expected.",
    )

    require_true(
        float(
            profile_by_name[
                "IRC"
            ][
                "Log1pExactMaximumAbsoluteZScore"
            ]
        )
        >
        400.0,
        "IRC log1p-z evidence is weaker than expected.",
    )

    require_true(
        float(
            profile_by_name[
                "cwr_flag_number"
            ][
                "Log1pExactMaximumAbsoluteZScore"
            ]
        )
        >
        300.0,
        "cwr_flag_number log1p-z evidence is weaker than expected.",
    )

    require_true(
        float(
            profile_by_name[
                "ece_flag_number"
            ][
                "Log1pExactMaximumAbsoluteZScore"
            ]
        )
        >
        200.0,
        "ece_flag_number log1p-z evidence is weaker than expected.",
    )

    print(
        "Raw z-score extreme-tail evidence: PASS"
    )
    print(
        "Residual sparse-feature extremes after log1p+zscore: PASS"
    )
    print(
        "Zero TRAIN constant features: PASS"
    )

    # --------------------------------------------------------------
    # Gate C - compute exact frozen preprocessing parameters
    # --------------------------------------------------------------

    print("")
    print("=" * 60)
    print("GATE C - COMPUTE FROZEN SIGNED-LOG1P MAXABS PARAMETERS")
    print("=" * 60)
    print("")

    parameter_rows = []

    active_zero_scale_features = []

    for feature_index, row in enumerate(
        profile_rows
    ):
        feature_name = row[
            "FeatureName"
        ]

        minimum = float(
            row[
                "Minimum"
            ]
        )

        maximum = float(
            row[
                "Maximum"
            ]
        )

        transformed_minimum = (
            signed_log1p(
                minimum
            )
        )

        transformed_maximum = (
            signed_log1p(
                maximum
            )
        )

        fitted_scale = max(
            abs(
                transformed_minimum
            ),
            abs(
                transformed_maximum
            ),
        )

        zero_variance_fallback_active = (
            fitted_scale
            ==
            0.0
        )

        if zero_variance_fallback_active:
            active_zero_scale_features.append(
                feature_name
            )

            effective_denominator = 1.0

            train_output_min = 0.0
            train_output_max = 0.0

        else:
            effective_denominator = (
                fitted_scale
            )

            train_output_min = (
                transformed_minimum
                /
                effective_denominator
            )

            train_output_max = (
                transformed_maximum
                /
                effective_denominator
            )

        require_true(
            math.isfinite(
                fitted_scale
            ),
            "Non-finite fitted scale for "
            + feature_name,
        )

        require_true(
            math.isfinite(
                train_output_min
            ),
            "Non-finite TRAIN output minimum for "
            + feature_name,
        )

        require_true(
            math.isfinite(
                train_output_max
            ),
            "Non-finite TRAIN output maximum for "
            + feature_name,
        )

        require_true(
            train_output_min
            >=
            -1.0
            -
            1e-15,
            "TRAIN transformed minimum below -1 for "
            + feature_name,
        )

        require_true(
            train_output_max
            <=
            1.0
            +
            1e-15,
            "TRAIN transformed maximum above 1 for "
            + feature_name,
        )

        scale32 = struct.unpack(
            ">f",
            struct.pack(
                ">f",
                effective_denominator,
            ),
        )[0]

        parameter_rows.append({
            "FeatureIndexOneBased": (
                feature_index
                +
                1
            ),
            "FeatureName": (
                feature_name
            ),
            "Transform": (
                "SIGNED_LOG1P"
            ),
            "TrainMinimumRawFloat64": (
                format(
                    minimum,
                    ".17g",
                )
            ),
            "TrainMaximumRawFloat64": (
                format(
                    maximum,
                    ".17g",
                )
            ),
            "TrainMinimumSignedLog1pFloat64": (
                format(
                    transformed_minimum,
                    ".17g",
                )
            ),
            "TrainMaximumSignedLog1pFloat64": (
                format(
                    transformed_maximum,
                    ".17g",
                )
            ),
            "FrozenMaxAbsScaleFloat64": (
                format(
                    fitted_scale,
                    ".17g",
                )
            ),
            "FrozenMaxAbsScaleFloat64HexBE": (
                float64_hex(
                    fitted_scale
                )
            ),
            "ReferenceEffectiveScaleFloat32": (
                format(
                    scale32,
                    ".9g",
                )
            ),
            "ReferenceEffectiveScaleFloat32HexBE": (
                float32_hex(
                    scale32
                )
            ),
            "ZeroVarianceFallbackActive": (
                zero_variance_fallback_active
            ),
            "FrozenEffectiveDenominatorFloat64": (
                format(
                    effective_denominator,
                    ".17g",
                )
            ),
            "ExpectedTrainOutputMinimum": (
                format(
                    train_output_min,
                    ".17g",
                )
            ),
            "ExpectedTrainOutputMaximum": (
                format(
                    train_output_max,
                    ".17g",
                )
            ),
            "OutputStorageDtype": (
                "float32"
            ),
        })

    require_equal(
        active_zero_scale_features,
        [],
        "Active zero-scale TRAIN features",
    )

    write_csv(
        output_root
        / "FROZEN_PREPROCESSING_PARAMETERS.csv",
        parameter_rows,
        [
            "FeatureIndexOneBased",
            "FeatureName",
            "Transform",
            "TrainMinimumRawFloat64",
            "TrainMaximumRawFloat64",
            "TrainMinimumSignedLog1pFloat64",
            "TrainMaximumSignedLog1pFloat64",
            "FrozenMaxAbsScaleFloat64",
            "FrozenMaxAbsScaleFloat64HexBE",
            "ReferenceEffectiveScaleFloat32",
            "ReferenceEffectiveScaleFloat32HexBE",
            "ZeroVarianceFallbackActive",
            "FrozenEffectiveDenominatorFloat64",
            "ExpectedTrainOutputMinimum",
            "ExpectedTrainOutputMaximum",
            "OutputStorageDtype",
        ],
    )

    parameter_file_sha256 = sha256_file(
        output_root
        / "FROZEN_PREPROCESSING_PARAMETERS.csv"
    )

    print(
        "All 39 feature parameters computed: YES"
    )
    print(
        "Active zero-variance fallbacks: 0"
    )
    print(
        f"Parameter-file SHA256: {parameter_file_sha256}"
    )

    # --------------------------------------------------------------
    # Evidence manifest
    # --------------------------------------------------------------

    evidence_rows = []

    for role, path in (
        paths.items()
    ):
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
        / "FREEZE_EVIDENCE_MANIFEST.csv",
        evidence_rows,
        [
            "EvidenceRole",
            "Path",
            "SizeBytes",
            "SHA256",
        ],
    )

    # --------------------------------------------------------------
    # Freeze state
    # --------------------------------------------------------------

    freeze_state = {
        "status": "FROZEN",
        "policy_id": (
            POLICY_ID
        ),
        "dataset_binding": {
            "dataset_id": (
                EXPECTED_DATASET_ID
            ),
            "dataset_manifest_sha256": (
                EXPECTED_DATASET_MANIFEST_SHA256
            ),
        },
        "split_binding": {
            "split_dataset_id": (
                EXPECTED_SPLIT_ID
            ),
            "split_assignment_manifest_sha256": (
                EXPECTED_SPLIT_MANIFEST_SHA256
            ),
        },
        "fit_scope": {
            "split": (
                "TRAIN_ONLY"
            ),
            "exact_vectors": (
                EXPECTED_TRAIN_ROWS
            ),
            "raw_multiplicity_used": (
                False
            ),
            "validation_used": (
                False
            ),
            "test_used": (
                False
            ),
            "per_client_fitting": (
                False
            ),
        },
        "feature_policy": {
            "input_features": (
                EXPECTED_FEATURES
            ),
            "output_features": (
                EXPECTED_FEATURES
            ),
            "feature_order_preserved": (
                True
            ),
            "features_removed": [],
            "one_hot_expansion": (
                False
            ),
            "rarity_based_feature_removal": (
                False
            ),
        },
        "transformation": {
            "formula": (
                "g(x) = sign(x) * log1p(abs(x)); "
                "z_j = g(x_j) / s_j"
            ),
            "scale_formula": (
                "s_j = max over frozen TRAIN of abs(g(x_j))"
            ),
            "parameter_computation_dtype": (
                "float64"
            ),
            "output_dtype": (
                "float32"
            ),
            "centering": (
                False
            ),
            "clipping": (
                False
            ),
            "quantile_transform": (
                False
            ),
            "standard_deviation_scaling": (
                False
            ),
        },
        "zero_variance_policy": {
            "rule": (
                "If fitted s_j == 0, retain feature and output zero "
                "using effective denominator 1."
            ),
            "active_features": [],
        },
        "application_scope": {
            "same_parameters_for_train_validation_test": (
                True
            ),
            "same_parameters_for_all_future_clients": (
                True
            ),
            "client_specific_preprocessing": (
                False
            ),
            "refit_after_client_construction": (
                False
            ),
            "validation_or_test_values_may_exceed_train_range": (
                True
            ),
            "out_of_train_range_clipping": (
                False
            ),
        },
        "frozen_parameter_artifact": {
            "path": str(
                output_root
                / "FROZEN_PREPROCESSING_PARAMETERS.csv"
            ),
            "sha256": (
                parameter_file_sha256
            ),
        },
        "decision_evidence": {
            "constant_train_features": 0,
            "negative_train_features": [
                "IAT"
            ],
            "all_nonnegative_train_features": 38,
            "raw_features_with_exact_max_abs_z_ge_100": 10,
            "raw_features_with_exact_max_abs_z_ge_1000": 1,
            "raw_iat_exact_max_abs_z": float(
                profile_by_name[
                    "IAT"
                ][
                    "RawExactMaximumAbsoluteZScore"
                ]
            ),
            "log1p_irc_exact_max_abs_z": float(
                profile_by_name[
                    "IRC"
                ][
                    "Log1pExactMaximumAbsoluteZScore"
                ]
            ),
            "log1p_cwr_exact_max_abs_z": float(
                profile_by_name[
                    "cwr_flag_number"
                ][
                    "Log1pExactMaximumAbsoluteZScore"
                ]
            ),
            "log1p_ece_exact_max_abs_z": float(
                profile_by_name[
                    "ece_flag_number"
                ][
                    "Log1pExactMaximumAbsoluteZScore"
                ]
            ),
        },
        "rejected_as_primary": {
            "RAW_ZSCORE": (
                "Ten features have exact TRAIN max|z| >= 100 and IAT exceeds 2271."
            ),
            "SIGNED_LOG1P_ZSCORE": (
                "Magnitude-heavy tails improve, but prevalence-driven sparse "
                "features retain extreme z-scores; e.g. IRC > 400, "
                "cwr_flag_number > 300, ece_flag_number > 200."
            ),
            "RAW_MINMAX": (
                "Does not compress heavy-tailed magnitude before scaling."
            ),
            "QUANTILE_TRANSFORM": (
                "Avoided as a more complex distribution-warping transformation "
                "when a deterministic analytic bounded-TRAIN transform is available."
            ),
        },
        "scientific_boundary": {
            "transformed_dataset_materialised": (
                False
            ),
            "federated_clients_constructed": (
                False
            ),
            "training_started": (
                False
            ),
        },
    }

    with (
        output_root
        / "PREPROCESSING_POLICY_FREEZE.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            freeze_state,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------------
    # Human-readable report
    # --------------------------------------------------------------

    report = []

    report.append(
        "CICIoT2023 PREPROCESSING POLICY FREEZE"
    )
    report.append("=" * 78)
    report.append("")

    report.append("STATUS")
    report.append("-" * 78)
    report.append(
        "FROZEN"
    )
    report.append("")

    report.append("DATASET AND SPLIT BINDING")
    report.append("-" * 78)
    report.append(
        f"Dataset ID: {EXPECTED_DATASET_ID}"
    )
    report.append(
        f"Dataset manifest SHA256: "
        f"{EXPECTED_DATASET_MANIFEST_SHA256}"
    )
    report.append(
        f"Split dataset ID: {EXPECTED_SPLIT_ID}"
    )
    report.append(
        f"Split assignment manifest SHA256: "
        f"{EXPECTED_SPLIT_MANIFEST_SHA256}"
    )
    report.append("")

    report.append("FROZEN PREPROCESSING POLICY")
    report.append("-" * 78)
    report.append(
        f"Policy ID: {POLICY_ID}"
    )
    report.append("")
    report.append(
        "For every feature j:"
    )
    report.append(
        "  g(x) = sign(x) * log1p(abs(x))"
    )
    report.append(
        "  s_j  = max over frozen TRAIN of abs(g(x_j))"
    )
    report.append(
        "  z_j  = g(x_j) / s_j"
    )
    report.append("")

    report.append("FIT SCOPE")
    report.append("-" * 78)
    report.append(
        f"Fitted on exact unique-vector TRAIN only: {EXPECTED_TRAIN_ROWS} vectors"
    )
    report.append(
        "RawMultiplicity used: NO"
    )
    report.append(
        "VALIDATION used: NO"
    )
    report.append(
        "TEST used: NO"
    )
    report.append(
        "Per-client fitting: NO"
    )
    report.append("")

    report.append("FEATURE POLICY")
    report.append("-" * 78)
    report.append(
        "Input features:  39"
    )
    report.append(
        "Output features: 39"
    )
    report.append(
        "Feature order preserved: YES"
    )
    report.append(
        "Features removed: 0"
    )
    report.append(
        "One-hot expansion: NO"
    )
    report.append(
        "Rarity-based feature removal: NO"
    )
    report.append("")

    report.append("NUMERIC POLICY")
    report.append("-" * 78)
    report.append(
        "Parameter fitting/computation dtype: float64"
    )
    report.append(
        "Materialised/model-input dtype:     float32"
    )
    report.append(
        "Centering: NO"
    )
    report.append(
        "Standard-deviation scaling: NO"
    )
    report.append(
        "Clipping: NO"
    )
    report.append(
        "Quantile transformation: NO"
    )
    report.append("")

    report.append("ZERO-VARIANCE POLICY")
    report.append("-" * 78)
    report.append(
        "If fitted s_j == 0:"
    )
    report.append(
        "  retain the feature"
    )
    report.append(
        "  use effective denominator 1"
    )
    report.append(
        "  output zero"
    )
    report.append(
        "Active zero-variance features in frozen TRAIN: 0"
    )
    report.append("")

    report.append("APPLICATION RULE")
    report.append("-" * 78)
    report.append(
        "Same frozen parameters for TRAIN / VALIDATION / TEST: YES"
    )
    report.append(
        "Same frozen parameters for every future client: YES"
    )
    report.append(
        "Client-specific preprocessing: NO"
    )
    report.append(
        "Refit after client construction: NO"
    )
    report.append(
        "Out-of-TRAIN-range clipping: NO"
    )
    report.append("")
    report.append(
        "VALIDATION or TEST values outside the TRAIN range may therefore "
        "transform beyond [-1,1]; this is allowed and will be audited without "
        "re-fitting or changing the policy."
    )
    report.append("")

    report.append("DECISION EVIDENCE")
    report.append("-" * 78)
    report.append(
        "Constant TRAIN features: 0"
    )
    report.append(
        "Negative-valued TRAIN features: IAT only"
    )
    report.append(
        "Raw features with exact max|z| >= 100: 10"
    )
    report.append(
        "Raw features with exact max|z| >= 1000: 1"
    )
    report.append(
        f"IAT raw exact max|z|: "
        f"{float(profile_by_name['IAT']['RawExactMaximumAbsoluteZScore']):.9f}"
    )
    report.append("")
    report.append(
        "Even after log1p, sparse prevalence-driven features remain extreme "
        "under standard-deviation scaling:"
    )
    report.append(
        f"  IRC max|z|: "
        f"{float(profile_by_name['IRC']['Log1pExactMaximumAbsoluteZScore']):.9f}"
    )
    report.append(
        f"  cwr_flag_number max|z|: "
        f"{float(profile_by_name['cwr_flag_number']['Log1pExactMaximumAbsoluteZScore']):.9f}"
    )
    report.append(
        f"  ece_flag_number max|z|: "
        f"{float(profile_by_name['ece_flag_number']['Log1pExactMaximumAbsoluteZScore']):.9f}"
    )
    report.append("")

    report.append("WHY THIS POLICY")
    report.append("-" * 78)
    report.append(
        "The signed-log1p stage compresses magnitude-heavy tails and handles "
        "the negative IAT feature using one deterministic formula."
    )
    report.append(
        "The TRAIN-fitted max-absolute stage avoids dividing rare features by "
        "tiny standard deviations."
    )
    report.append(
        "The transform preserves zero exactly, retains all features, introduces "
        "no arbitrary clipping threshold, and bounds every TRAIN feature to [-1,1]."
    )
    report.append("")

    report.append("FROZEN PARAMETER ARTIFACT")
    report.append("-" * 78)
    report.append(
        "File: FROZEN_PREPROCESSING_PARAMETERS.csv"
    )
    report.append(
        f"SHA256: {parameter_file_sha256}"
    )
    report.append(
        "Feature parameters: 39 / 39"
    )
    report.append("")

    report.append("SCIENTIFIC BOUNDARY")
    report.append("-" * 78)
    report.append(
        "Transformed dataset materialised: NO"
    )
    report.append(
        "Federated clients constructed: NO"
    )
    report.append(
        "Training started: NO"
    )
    report.append("")

    report.append("NEXT GATE")
    report.append("-" * 78)
    report.append(
        "Materialise the frozen preprocessing transformation for all exact vectors, "
        "preserving bucket/row alignment and split assignments, then audit finite "
        "float32 outputs, exact TRAIN range, held-out range exceedances, and "
        "deterministic replay before client construction."
    )

    (
        output_root
        / "PREPROCESSING_POLICY_FREEZE_REPORT.txt"
    ).write_text(
        "\n".join(
            report
        )
        + "\n",
        encoding="utf-8",
    )

    print("")
    print("=" * 60)
    print("STATUS: FROZEN")
    print(
        f"POLICY: {POLICY_ID}"
    )
    print(
        f"PARAMETER SHA256: {parameter_file_sha256}"
    )
    print(
        "FEATURES RETAINED: 39 / 39"
    )
    print(
        "TRANSFORMED DATASET MATERIALISED: NO"
    )
    print("=" * 60)
    print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
