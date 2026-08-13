from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "ERROR: numpy and pandas are required in the project environment."
    ) from exc


# =============================================================================
# Frozen campaign design
# =============================================================================

OUTER_FOLDS = (1, 2, 3, 4, 5)
ALPHAS = (1.0, 0.1)
FL_SEEDS = (123, 456)

METHODS = (
    "fedavg",
    "fedprox",
    "random_trimmed_mean",
    "fedle_adapted",
    "tea_fl",
    "arl_fl",
)

SCENARIOS = (
    {
        "scenario": "clean",
        "attack": "none",
        "malicious_fraction_nominal": 0.0,
        "malicious_count": 0,
    },
    {
        "scenario": "signflip_mu0p2",
        "attack": "sign_flip",
        "malicious_fraction_nominal": 0.2,
        "malicious_count": 6,
    },
    {
        "scenario": "signflip_mu0p4",
        "attack": "sign_flip",
        "malicious_fraction_nominal": 0.4,
        "malicious_count": 11,
    },
    {
        "scenario": "labelflip_mu0p2",
        "attack": "label_flip",
        "malicious_fraction_nominal": 0.2,
        "malicious_count": 6,
    },
    {
        "scenario": "labelflip_mu0p4",
        "attack": "label_flip",
        "malicious_fraction_nominal": 0.4,
        "malicious_count": 11,
    },
)

TOTAL_ROUNDS = 100
ATTACK_START_ROUND = 20
CLIENTS_PER_OUTER_FOLD = 28
CLIENTS_PER_ROUND = 8

LOCAL_EPOCHS = 1
LOCAL_BATCH_SIZE = 64
LOCAL_OPTIMIZER = "AdamW"
LOCAL_LEARNING_RATE = 1e-3
LOCAL_WEIGHT_DECAY = 1e-4

EVALUATION_ROUNDS = tuple([0] + list(range(5, TOTAL_ROUNDS + 1, 5)))

MODEL_REPRESENTATION = "magnitude6_acc16_gyro"
MODEL_INPUT_CHANNELS = 6
MODEL_NORMALIZATION = "groupnorm"
MODEL_CHANNELS = (32, 64, 96)
MODEL_DROPOUT = 0.20
MODEL_GROUPS = 8
MODEL_PARAMETER_COUNT = 77004

SIGN_FLIP_SCALE = 5.0
LABEL_FLIP_RULE = "(y + 1) mod 12"

FEDPROX_MU = 0.01
TRIMMED_MEAN_K = 2

# Common normalized energy environment. These are synthetic normalized units,
# not physical joules.
ENERGY_INITIAL_LOW = 0.70
ENERGY_INITIAL_HIGH = 1.00
ENERGY_COMPUTE_FACTOR_LOW = 0.80
ENERGY_COMPUTE_FACTOR_HIGH = 1.20
ENERGY_STANDBY_COST = 0.0005
ENERGY_COMMUNICATION_COST = 0.004
ENERGY_COMPUTE_COEFFICIENT = 0.016
ENERGY_CRITICAL_THRESHOLD = 0.10
NETWORK_STOP_ACTIVE_FRACTION = 0.50

# TEA-FL published HAR-oriented hyperparameters, adapted to the common
# energy environment so all methods face identical client budgets/costs.
TEA_INITIAL_TRUST = 0.5
TEA_TRUST_BETA = 0.8
TEA_TRUST_FLOOR = 0.05
TEA_SELECTION_TRUST_WEIGHT = 0.8
TEA_EXPLORATION_RATIO = 0.1
TEA_AGG_TRUST_EXPONENT = 2.0
TEA_AGG_ENERGY_EXPONENT = 0.3

# FedLE adaptation to the present common energy environment.
FEDLE_PREFLIGHT_LOCAL_EPOCHS = 1
FEDLE_NUM_CLUSTERS = CLIENTS_PER_ROUND

# ARL-FL risk estimation.
ARL_RISK_BETA = 0.8
ARL_TEMPORAL_FIRST_SCORE = 0.0
ARL_GLOBAL_PRESSURE_QUANTILE = 0.75
ARL_STALENESS_CAP = 10

# ARL-FL adaptive robust aggregation.
ARL_CLIP_KAPPA_LOW_PRESSURE = 3.0
ARL_CLIP_KAPPA_HIGH_PRESSURE = 1.0
ARL_TRIM_K_LOW_PRESSURE = 1
ARL_TRIM_K_HIGH_PRESSURE = 2
ARL_PRESSURE_SWITCH = 0.5

BASE_SEED = 20260706


# =============================================================================
# Utilities
# =============================================================================

def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_subject(text: object) -> int:
    value = str(text).strip()
    if value.lower().startswith("subject"):
        value = value[7:]
    return int(value)


def deterministic_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int(value % (2**31 - 1))


def validate_partition_audit(
    partition_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    report_path = partition_root / "PARTITION_AUDIT_REPORT.txt"
    summary_path = partition_root / "alpha_partition_summary.csv"
    client_manifest_path = partition_root / "outer_fold_client_manifest.csv"

    for path in (report_path, summary_path, client_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    report_text = report_path.read_text(encoding="utf-8")
    if "STATUS\n------------------------------------------------------------------------------\nPASS" not in report_text:
        raise RuntimeError("Partition audit report does not contain PASS.")

    alpha_summary = pd.read_csv(summary_path)
    client_manifest = pd.read_csv(client_manifest_path)

    alpha1 = alpha_summary[
        np.isclose(alpha_summary["alpha"].astype(float), 1.0)
    ].iloc[0]
    alpha01 = alpha_summary[
        np.isclose(alpha_summary["alpha"].astype(float), 0.1)
    ].iloc[0]

    if not (
        float(alpha01["mean_normalized_label_entropy"])
        < float(alpha1["mean_normalized_label_entropy"])
    ):
        raise RuntimeError("alpha=0.1 does not have lower label entropy.")

    if not (
        float(alpha01["mean_js_divergence_from_fold"])
        > float(alpha1["mean_js_divergence_from_fold"])
    ):
        raise RuntimeError("alpha=0.1 does not have higher JSD.")

    if int(client_manifest["windows"].min()) < 128:
        raise RuntimeError("A federated client is below 128 windows.")

    expected_rows = 5 * 2 * 28
    if len(client_manifest) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} client manifest rows, "
            f"found {len(client_manifest)}."
        )

    return alpha_summary, client_manifest


def make_attack_identity_map(
    client_manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    # Client identities are paired across alpha and attack type:
    # same outer fold, malicious fraction, and FL seed -> same malicious IDs.
    unique_clients_by_fold: dict[int, list[str]] = {}

    for outer_fold in OUTER_FOLDS:
        fold_clients = sorted(
            client_manifest[
                client_manifest["outer_fold"].astype(int) == outer_fold
            ]["global_client_id"]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )

        if len(fold_clients) != CLIENTS_PER_OUTER_FOLD:
            raise RuntimeError(
                f"Outer fold {outer_fold}: expected 28 clients, "
                f"found {len(fold_clients)}."
            )

        unique_clients_by_fold[outer_fold] = fold_clients

    for outer_fold in OUTER_FOLDS:
        client_ids = unique_clients_by_fold[outer_fold]

        for fl_seed in FL_SEEDS:
            for malicious_count in (6, 11):
                rng = np.random.default_rng(
                    deterministic_seed(
                        BASE_SEED,
                        "malicious_ids",
                        outer_fold,
                        fl_seed,
                        malicious_count,
                    )
                )

                chosen = sorted(
                    rng.choice(
                        np.asarray(client_ids, dtype=object),
                        size=malicious_count,
                        replace=False,
                    ).tolist()
                )

                for client_id in chosen:
                    rows.append(
                        {
                            "outer_fold": outer_fold,
                            "fl_seed": fl_seed,
                            "malicious_count": malicious_count,
                            "realized_fraction": (
                                malicious_count / CLIENTS_PER_OUTER_FOLD
                            ),
                            "global_client_id": client_id,
                        }
                    )

    return pd.DataFrame(rows)


def make_energy_profiles(
    client_manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    # Initial energy and device compute factor are independent of alpha,
    # scenario, and method. The same client receives the same profile in all
    # matched conditions for a given outer fold and FL seed.
    for outer_fold in OUTER_FOLDS:
        client_ids = sorted(
            client_manifest[
                client_manifest["outer_fold"].astype(int) == outer_fold
            ]["global_client_id"]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )

        for fl_seed in FL_SEEDS:
            for client_id in client_ids:
                rng = np.random.default_rng(
                    deterministic_seed(
                        BASE_SEED,
                        "energy_profile",
                        outer_fold,
                        fl_seed,
                        client_id,
                    )
                )

                initial_energy = float(
                    rng.uniform(
                        ENERGY_INITIAL_LOW,
                        ENERGY_INITIAL_HIGH,
                    )
                )
                compute_factor = float(
                    rng.uniform(
                        ENERGY_COMPUTE_FACTOR_LOW,
                        ENERGY_COMPUTE_FACTOR_HIGH,
                    )
                )

                rows.append(
                    {
                        "outer_fold": outer_fold,
                        "fl_seed": fl_seed,
                        "global_client_id": client_id,
                        "initial_energy": initial_energy,
                        "compute_factor": compute_factor,
                    }
                )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the complete PAMAP2 federated experimental protocol "
            "before any federated model training."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--partition-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--central-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()

    partition_root = (
        args.partition_root.expanduser().resolve()
        if args.partition_root is not None
        else project_root
        / "outputs"
        / "protocols"
        / "pamap2_fl_partitions_v1"
    )

    central_root = (
        args.central_root.expanduser().resolve()
        if args.central_root is not None
        else project_root
        / "outputs"
        / "centralized"
        / "pamap2"
        / "outer_evaluation_v2"
    )

    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else project_root
        / "outputs"
        / "protocols"
        / "pamap2_fl_experiment_v1"
    )

    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(
            "ERROR: output directory already exists and is not empty:\n"
            f"  {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    partition_protocol_path = (
        partition_root / "FL_PARTITION_PROTOCOL.json"
    )
    central_summary_path = (
        central_root / "overall_outer_summary.json"
    )
    central_fold_path = (
        central_root / "outer_fold_summary.csv"
    )

    for path in (
        partition_protocol_path,
        central_summary_path,
        central_fold_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    alpha_summary, client_manifest = validate_partition_audit(
        partition_root
    )

    central_summary = json.loads(
        central_summary_path.read_text(encoding="utf-8")
    )
    central_fold = pd.read_csv(central_fold_path)

    if len(central_fold) != 5:
        raise RuntimeError("Expected five centralized outer folds.")

    print("=== Freeze PAMAP2 federated experimental protocol v1 ===")
    print(f"Project root:      {project_root}")
    print(f"Partition source:  {partition_root}")
    print(f"Central reference: {central_root}")
    print(f"Output:            {output_root}")
    print()

    # ------------------------------------------------------------------
    # Method manifest
    # ------------------------------------------------------------------
    method_rows = [
        {
            "method": "fedavg",
            "selection": "uniform random from eligible clients",
            "aggregation": "sample-size-weighted mean of model deltas",
            "energy_aware": False,
            "risk_or_trust_aware": False,
            "robust_aggregation": False,
        },
        {
            "method": "fedprox",
            "selection": "same matched random selection as FedAvg",
            "aggregation": "sample-size-weighted mean of model deltas",
            "energy_aware": False,
            "risk_or_trust_aware": False,
            "robust_aggregation": False,
        },
        {
            "method": "random_trimmed_mean",
            "selection": "same matched random selection as FedAvg",
            "aggregation": (
                f"coordinate-wise trimmed mean with fixed k={TRIMMED_MEAN_K}"
            ),
            "energy_aware": False,
            "risk_or_trust_aware": False,
            "robust_aggregation": True,
        },
        {
            "method": "fedle_adapted",
            "selection": (
                "one-time partial-model similarity clustering; inverse cluster "
                "size preference; higher residual energy preferred within cluster"
            ),
            "aggregation": "sample-size-weighted mean of model deltas",
            "energy_aware": True,
            "risk_or_trust_aware": False,
            "robust_aggregation": False,
        },
        {
            "method": "tea_fl",
            "selection": "published trust-energy exploit-explore orchestration",
            "aggregation": "published trust-energy-weighted averaging",
            "energy_aware": True,
            "risk_or_trust_aware": True,
            "robust_aggregation": False,
        },
        {
            "method": "arl_fl",
            "selection": (
                "Pareto risk-lifetime-staleness scheduling with adaptive exploration"
            ),
            "aggregation": (
                "independent adaptive norm clipping plus coordinate-wise trimmed mean"
            ),
            "energy_aware": True,
            "risk_or_trust_aware": True,
            "robust_aggregation": True,
        },
    ]

    method_df = pd.DataFrame(method_rows)
    method_df.to_csv(
        output_root / "method_manifest.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Scenario manifest
    # ------------------------------------------------------------------
    scenario_rows: list[dict[str, object]] = []

    for item in SCENARIOS:
        scenario_rows.append(
            {
                **item,
                "realized_fraction": (
                    item["malicious_count"] / CLIENTS_PER_OUTER_FOLD
                ),
                "attack_start_round": (
                    None
                    if item["attack"] == "none"
                    else ATTACK_START_ROUND
                ),
                "sign_flip_scale": (
                    SIGN_FLIP_SCALE
                    if item["attack"] == "sign_flip"
                    else None
                ),
                "label_flip_rule": (
                    LABEL_FLIP_RULE
                    if item["attack"] == "label_flip"
                    else None
                ),
            }
        )

    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(
        output_root / "scenario_manifest.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Matched-condition and 600-run campaign matrices
    # ------------------------------------------------------------------
    condition_rows: list[dict[str, object]] = []
    campaign_rows: list[dict[str, object]] = []

    condition_id = 0
    run_id = 0

    for outer_fold in OUTER_FOLDS:
        for alpha in ALPHAS:
            for scenario_item in SCENARIOS:
                for fl_seed in FL_SEEDS:
                    condition_id += 1

                    condition_seed = deterministic_seed(
                        BASE_SEED,
                        "condition",
                        outer_fold,
                        alpha,
                        scenario_item["scenario"],
                        fl_seed,
                    )
                    model_seed = deterministic_seed(
                        BASE_SEED,
                        "model",
                        outer_fold,
                        fl_seed,
                    )
                    random_schedule_seed = deterministic_seed(
                        BASE_SEED,
                        "random_schedule",
                        outer_fold,
                        fl_seed,
                    )

                    condition_rows.append(
                        {
                            "condition_id": condition_id,
                            "outer_fold": outer_fold,
                            "alpha": alpha,
                            "scenario": scenario_item["scenario"],
                            "attack": scenario_item["attack"],
                            "malicious_count": scenario_item["malicious_count"],
                            "fl_seed": fl_seed,
                            "condition_seed": condition_seed,
                            "model_seed": model_seed,
                            "random_schedule_seed": random_schedule_seed,
                        }
                    )

                    for method in METHODS:
                        run_id += 1
                        campaign_rows.append(
                            {
                                "run_id": run_id,
                                "condition_id": condition_id,
                                "outer_fold": outer_fold,
                                "alpha": alpha,
                                "scenario": scenario_item["scenario"],
                                "method": method,
                                "fl_seed": fl_seed,
                            }
                        )

    condition_df = pd.DataFrame(condition_rows)
    campaign_df = pd.DataFrame(campaign_rows)

    if len(condition_df) != 100:
        raise RuntimeError(
            f"Expected 100 matched conditions, found {len(condition_df)}."
        )
    if len(campaign_df) != 600:
        raise RuntimeError(
            f"Expected 600 campaign runs, found {len(campaign_df)}."
        )

    condition_df.to_csv(
        output_root / "matched_condition_manifest.csv",
        index=False,
    )
    campaign_df.to_csv(
        output_root / "campaign_matrix_600_runs.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Frozen malicious identities and client energy profiles
    # ------------------------------------------------------------------
    malicious_df = make_attack_identity_map(client_manifest)
    malicious_df.to_csv(
        output_root / "malicious_client_manifest.csv",
        index=False,
    )

    energy_df = make_energy_profiles(client_manifest)
    energy_df.to_csv(
        output_root / "client_energy_profile.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Protocol JSON
    # ------------------------------------------------------------------
    primary_central = central_summary["primary_subject_level_summary"]

    protocol = {
        "protocol_name": "PAMAP2 Federated Experimental Protocol v1",
        "status": "FROZEN_BEFORE_ANY_FL_TRAINING",
        "provenance": {
            "partition_protocol": str(partition_protocol_path),
            "partition_protocol_sha256": sha256_file(
                partition_protocol_path
            ),
            "central_summary": str(central_summary_path),
            "central_summary_sha256": sha256_file(
                central_summary_path
            ),
            "centralized_reference_macro_f1": float(
                primary_central[
                    "mean_across_outer_subjects_macro_f1"
                ]
            ),
        },
        "data": {
            "outer_folds": list(OUTER_FOLDS),
            "clients_per_fold": CLIENTS_PER_OUTER_FOLD,
            "subject_pure_clients": True,
            "heterogeneity_alphas": list(ALPHAS),
            "partition_manifests_frozen": True,
        },
        "model": {
            "representation": MODEL_REPRESENTATION,
            "input_channels": MODEL_INPUT_CHANNELS,
            "normalization": MODEL_NORMALIZATION,
            "conv_channels": list(MODEL_CHANNELS),
            "dropout": MODEL_DROPOUT,
            "groupnorm_groups": MODEL_GROUPS,
            "trainable_parameters": MODEL_PARAMETER_COUNT,
            "rationale": (
                "Magnitude6 was selected in 5/5 nested outer folds. "
                "GroupNorm is standardized across FL folds to avoid "
                "cross-client BatchNorm running-statistic aggregation."
            ),
        },
        "training": {
            "rounds": TOTAL_ROUNDS,
            "clients_per_round": CLIENTS_PER_ROUND,
            "participation_fraction": (
                CLIENTS_PER_ROUND / CLIENTS_PER_OUTER_FOLD
            ),
            "local_epochs": LOCAL_EPOCHS,
            "batch_size": LOCAL_BATCH_SIZE,
            "optimizer": LOCAL_OPTIMIZER,
            "learning_rate": LOCAL_LEARNING_RATE,
            "weight_decay": LOCAL_WEIGHT_DECAY,
            "fl_seeds": list(FL_SEEDS),
            "evaluation_rounds": list(EVALUATION_ROUNDS),
            "outer_test_metrics_never_used_for_training_or_selection": True,
        },
        "common_energy_environment": {
            "units": "normalized synthetic participation-energy units",
            "not_physical_joules": True,
            "initial_energy_distribution": [
                ENERGY_INITIAL_LOW,
                ENERGY_INITIAL_HIGH,
            ],
            "compute_factor_distribution": [
                ENERGY_COMPUTE_FACTOR_LOW,
                ENERGY_COMPUTE_FACTOR_HIGH,
            ],
            "standby_cost_per_round": ENERGY_STANDBY_COST,
            "participation_cost_formula": (
                "communication_cost + compute_coefficient * compute_factor "
                "* (client_windows / fold_median_client_windows)"
            ),
            "communication_cost": ENERGY_COMMUNICATION_COST,
            "compute_coefficient": ENERGY_COMPUTE_COEFFICIENT,
            "critical_energy_threshold": ENERGY_CRITICAL_THRESHOLD,
            "network_stop_active_fraction": NETWORK_STOP_ACTIVE_FRACTION,
            "early_stopped_model_carried_forward_to_round_100_for_reporting": True,
            "same_initial_profiles_across_methods_alpha_and_scenarios": True,
        },
        "threat_model": {
            "scenarios": scenario_rows,
            "malicious_id_pairing": (
                "same outer fold, malicious count, and FL seed use identical "
                "malicious client identities across alpha, attack type, and method"
            ),
            "attack_start_round": ATTACK_START_ROUND,
            "sign_flip": {
                "rule": "malicious delta = -scale * honest delta",
                "scale": SIGN_FLIP_SCALE,
            },
            "label_flip": {
                "rule": LABEL_FLIP_RULE,
            },
        },
        "methods": {
            "fedavg": {
                "selection": (
                    "matched deterministic random client permutation per round; "
                    "take first eligible clients"
                ),
                "aggregation": "sample-size-weighted mean",
            },
            "fedprox": {
                "selection": "same matched schedule as FedAvg",
                "proximal_mu": FEDPROX_MU,
                "aggregation": "sample-size-weighted mean",
            },
            "random_trimmed_mean": {
                "selection": "same matched schedule as FedAvg",
                "trim_count_each_tail": TRIMMED_MEAN_K,
                "aggregation": "coordinate-wise trimmed mean",
            },
            "fedle_adapted": {
                "preflight_local_epochs": FEDLE_PREFLIGHT_LOCAL_EPOCHS,
                "partial_layers": (
                    "first convolution layer plus final linear layer"
                ),
                "similarity": "cosine similarity",
                "clustering": "one-time K-means",
                "num_clusters": FEDLE_NUM_CLUSTERS,
                "selection_probability": (
                    "inverse cluster size preference multiplied by current "
                    "residual-energy fraction"
                ),
                "aggregation": "sample-size-weighted mean",
                "preflight_energy_and_communication_charged": True,
            },
            "tea_fl": {
                "initial_trust": TEA_INITIAL_TRUST,
                "trust_beta": TEA_TRUST_BETA,
                "trust_floor": TEA_TRUST_FLOOR,
                "selection_trust_weight": TEA_SELECTION_TRUST_WEIGHT,
                "exploration_ratio": TEA_EXPLORATION_RATIO,
                "aggregation_trust_exponent": TEA_AGG_TRUST_EXPONENT,
                "aggregation_energy_exponent": TEA_AGG_ENERGY_EXPONENT,
                "energy_environment": "common campaign energy environment",
            },
            "arl_fl": {
                "design_principle": (
                    "selection risk signals do not directly weight aggregation"
                ),
                "risk_signals": {
                    "direction_anomaly": (
                        "(1 - cosine(client_delta, coordinate_median_delta)) / 2"
                    ),
                    "norm_anomaly": (
                        "upper robust z-score of log update norm using median/MAD, "
                        "mapped to [0,1] by 1-exp(-z/3)"
                    ),
                    "temporal_inconsistency": (
                        "(1 - cosine(current_delta, previous_client_delta)) / 2; "
                        "first participation score = 0"
                    ),
                    "instantaneous_fusion": "median of the three signals",
                    "ema_beta": ARL_RISK_BETA,
                },
                "global_adversarial_pressure": {
                    "definition": (
                        f"quantile {ARL_GLOBAL_PRESSURE_QUANTILE} of selected-client "
                        "risk scores"
                    ),
                },
                "scheduler": {
                    "objectives": [
                        "minimize risk",
                        "minimize predicted participation cost / residual energy",
                        "maximize staleness",
                    ],
                    "algorithm": (
                        "non-dominated Pareto sorting with crowding-distance ordering"
                    ),
                    "staleness_cap_rounds": ARL_STALENESS_CAP,
                    "adaptive_exploration_slots": (
                        "3 if pressure <0.25; 2 if 0.25<=pressure<0.5; "
                        "1 if pressure>=0.5"
                    ),
                },
                "aggregation": {
                    "independent_from_energy_and_direct_risk_weights": True,
                    "norm_clipping": (
                        "median norm + kappa * 1.4826*MAD; "
                        "kappa = 3 - 2*pressure"
                    ),
                    "kappa_range": [
                        ARL_CLIP_KAPPA_HIGH_PRESSURE,
                        ARL_CLIP_KAPPA_LOW_PRESSURE,
                    ],
                    "trim_count": (
                        f"{ARL_TRIM_K_LOW_PRESSURE} if pressure < "
                        f"{ARL_PRESSURE_SWITCH}; "
                        f"{ARL_TRIM_K_HIGH_PRESSURE} otherwise"
                    ),
                    "rule": "coordinate-wise trimmed mean after adaptive clipping",
                },
            },
        },
        "matched_randomness": {
            "same_initial_model_per_outer_fold_and_fl_seed": True,
            "same_malicious_identities_across_methods": True,
            "same_energy_profiles_across_methods": True,
            "fedavg_fedprox_random_trimmed_mean_share_random_selection_schedule": True,
            "local_shuffle_seed_derivation": (
                "deterministic hash of base seed, outer fold, FL seed, round, client ID"
            ),
        },
        "metrics": {
            "primary_predictive": "final Macro-F1 at round 100",
            "predictive": [
                "Macro-F1",
                "balanced accuracy",
                "accuracy",
                "per-class F1",
                "learning-curve area over fixed evaluation rounds",
                "final-window variability",
            ],
            "security": [
                "attack degradation versus matched clean condition",
                "malicious participation exposure",
                "risk separation between malicious and benign clients",
                "risk detection delay after attack onset",
            ],
            "lifetime": [
                "first client dropout round",
                "75% active-client lifetime",
                "50% active-client lifetime",
                "active-client AUC",
                "final mean residual energy",
                "final minimum residual energy",
                "total normalized energy consumed",
                "Jain participation fairness",
                "energy-to-target",
            ],
            "systems": [
                "wall-clock time",
                "server orchestration overhead",
                "uploaded bytes",
                "downloaded bytes",
            ],
        },
        "statistical_reporting": {
            "primary_independent_unit": (
                "outer-fold mean after averaging the two FL seeds"
            ),
            "report": (
                "mean and sample standard deviation across five outer folds"
            ),
            "paired_comparisons": (
                "paired outer-fold differences against ARL-FL"
            ),
            "confidence_intervals": (
                "95% bootstrap confidence intervals over paired outer-fold differences"
            ),
            "exploratory_tests": (
                "Wilcoxon signed-rank with Holm correction across five baselines"
            ),
            "do_not_treat_10_fold_seed_runs_as_10_independent_subjects": True,
        },
        "campaign_size": {
            "matched_conditions": 100,
            "methods": len(METHODS),
            "total_runs": 600,
            "formula": "5 folds x 2 alphas x 5 scenarios x 2 FL seeds x 6 methods",
        },
    }

    protocol_path = output_root / "FL_EXPERIMENTAL_PROTOCOL_V1.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Human-readable report
    # ------------------------------------------------------------------
    alpha1 = alpha_summary[
        np.isclose(alpha_summary["alpha"].astype(float), 1.0)
    ].iloc[0]
    alpha01 = alpha_summary[
        np.isclose(alpha_summary["alpha"].astype(float), 0.1)
    ].iloc[0]

    report_lines = [
        "PAMAP2 FEDERATED EXPERIMENTAL PROTOCOL V1",
        "=" * 78,
        "",
        "STATUS",
        "-" * 78,
        "FROZEN BEFORE ANY FL TRAINING",
        "",
        "FEDERATED DATA",
        "-" * 78,
        "Outer folds: 5",
        "Training clients per fold: 28",
        "Client design: 4 subject-pure pseudo-clients per training subject",
        "Heterogeneity: alpha=1.0 and alpha=0.1",
        (
            f"alpha=1.0 audit: classes/client="
            f"{float(alpha1['mean_classes_present']):.2f}; "
            f"entropy={float(alpha1['mean_normalized_label_entropy']):.3f}; "
            f"JSD={float(alpha1['mean_js_divergence_from_fold']):.3f}"
        ),
        (
            f"alpha=0.1 audit: classes/client="
            f"{float(alpha01['mean_classes_present']):.2f}; "
            f"entropy={float(alpha01['mean_normalized_label_entropy']):.3f}; "
            f"JSD={float(alpha01['mean_js_divergence_from_fold']):.3f}"
        ),
        "",
        "GLOBAL MODEL",
        "-" * 78,
        "Representation: Magnitude6 (16g acceleration + gyroscope magnitudes)",
        "Input channels: 6",
        "Normalization layer: GroupNorm",
        "Architecture: 32 -> 64 -> 96 lightweight 1D-CNN",
        f"Trainable parameters: {MODEL_PARAMETER_COUNT:,}",
        "",
        "TRAINING",
        "-" * 78,
        f"Communication rounds: {TOTAL_ROUNDS}",
        f"Clients per round: {CLIENTS_PER_ROUND}/{CLIENTS_PER_OUTER_FOLD} "
        f"({CLIENTS_PER_ROUND / CLIENTS_PER_OUTER_FOLD:.1%})",
        f"Local epochs: {LOCAL_EPOCHS}",
        f"Local batch size: {LOCAL_BATCH_SIZE}",
        "Local optimizer: AdamW",
        "Learning rate: 0.001",
        "Weight decay: 0.0001",
        f"FL seeds per outer fold: {list(FL_SEEDS)}",
        "",
        "THREAT MODEL",
        "-" * 78,
        "Attack begins at round 20.",
        "Clean",
        "Sign-flip, nominal mu=0.2 -> 6/28 malicious clients",
        "Sign-flip, nominal mu=0.4 -> 11/28 malicious clients",
        "Label flip, nominal mu=0.2 -> 6/28 malicious clients",
        "Label flip, nominal mu=0.4 -> 11/28 malicious clients",
        "Sign-flip scale: -5 x honest delta",
        "Label flip: (y + 1) mod 12",
        "",
        "METHODS",
        "-" * 78,
        "1. FedAvg",
        "2. FedProx (mu=0.01)",
        "3. Random Selection + coordinate-wise Trimmed Mean (k=2)",
        "4. FedLE-adapted",
        "5. TEA-FL",
        "6. ARL-FL",
        "",
        "COMMON ENERGY ENVIRONMENT",
        "-" * 78,
        "Normalized simulation units; no claim of physical joules.",
        "Initial energy: Uniform(0.70, 1.00)",
        "Device compute factor: Uniform(0.80, 1.20)",
        "Standby cost per round: 0.0005",
        "Communication cost per participation: 0.004",
        (
            "Compute cost: 0.016 x compute_factor x "
            "(client_windows / fold_median_client_windows)"
        ),
        "Critical energy threshold: 0.10",
        "Network death threshold: fewer than 50% active clients",
        "",
        "ARL-FL SEPARATION PRINCIPLE",
        "-" * 78,
        "Scheduling uses risk, lifetime pressure, and staleness.",
        "Aggregation does not directly use residual energy or risk weights.",
        "Aggregation uses adaptive norm clipping and adaptive trimmed mean.",
        "",
        "PRIMARY REPORTING",
        "-" * 78,
        "Primary predictive metric: final Macro-F1 at round 100.",
        "Independent unit: outer-fold mean after averaging two FL seeds.",
        "Report mean +/- sample standard deviation across five outer folds.",
        "",
        "CAMPAIGN SIZE",
        "-" * 78,
        "5 outer folds x 2 alphas x 5 scenarios x 2 FL seeds = 100 matched conditions",
        "100 conditions x 6 methods = 600 PAMAP2 FL runs",
        "",
        "FILES",
        "-" * 78,
        "FL_EXPERIMENTAL_PROTOCOL_V1.json",
        "FL_PROTOCOL_REPORT.txt",
        "method_manifest.csv",
        "scenario_manifest.csv",
        "matched_condition_manifest.csv",
        "campaign_matrix_600_runs.csv",
        "malicious_client_manifest.csv",
        "client_energy_profile.csv",
    ]

    report_path = output_root / "FL_PROTOCOL_REPORT.txt"
    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("[OK] Partition audit revalidated.")
    print("[OK] 100 matched experimental conditions created.")
    print("[OK] 600-run campaign matrix created.")
    print("[OK] Malicious client identities frozen.")
    print("[OK] Client energy profiles frozen.")
    print()
    print("=== PAMAP2 FL experimental protocol frozen successfully ===")
    print(f"Protocol: {protocol_path}")
    print(f"Report:   {report_path}")
    print()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nProtocol freeze interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
