from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Frozen ARL-FL smoke condition
# =============================================================================

OUTER_FOLD = 1
ALPHA = 1.0
SCENARIO = "clean"
METHOD = "arl_fl"
FL_SEED = 123

SMOKE_ROUNDS = 3

CLIENTS_PER_FOLD = 28
CLIENTS_PER_ROUND = 8

RISK_BETA = 0.8
GLOBAL_PRESSURE_QUANTILE = 0.75
STALENESS_CAP = 10

CLIP_KAPPA_LOW_PRESSURE = 3.0
CLIP_KAPPA_HIGH_PRESSURE = 1.0

TRIM_K_LOW_PRESSURE = 1
TRIM_K_HIGH_PRESSURE = 2
PRESSURE_SWITCH = 0.5

CRITICAL_ENERGY = 0.10
STANDBY_COST = 0.0005
COMMUNICATION_COST = 0.004
COMPUTE_COEFFICIENT = 0.016

BASE_LOCAL_SEED = 20260706
BASE_ARL_SEED = 20260706

EPS = 1e-12


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


def deterministic_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int(value % (2**31 - 1))


def alpha_label(alpha: float) -> str:
    if math.isclose(alpha, 1.0):
        return "alpha1p0"
    if math.isclose(alpha, 0.1):
        return "alpha0p1"
    return str(alpha).replace(".", "p")


def parse_subject(value: object) -> int:
    text = str(value).strip()
    if text.lower().startswith("subject"):
        text = text[7:]
    return int(text)


def load_base_module(project_root: Path):
    base_path = project_root / "16_smoke_pamap2_federated_engine.py"

    if not base_path.is_file():
        raise FileNotFoundError(
            "Required validated base engine not found:\n"
            f"  {base_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "pamap2_federated_base",
        base_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not load the validated base engine module."
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module, base_path


def participation_cost(
    *,
    client_id: str,
    client_sizes: dict[str, int],
    compute_factor: dict[str, float],
    median_client_windows: float,
) -> float:
    return (
        COMMUNICATION_COST
        + COMPUTE_COEFFICIENT
        * compute_factor[client_id]
        * (
            client_sizes[client_id]
            / median_client_windows
        )
    )


def cosine_similarity(
    vector_a: torch.Tensor,
    vector_b: torch.Tensor,
) -> float:
    a = vector_a.to(dtype=torch.float64)
    b = vector_b.to(dtype=torch.float64)

    norm_a = float(torch.linalg.vector_norm(a).item())
    norm_b = float(torch.linalg.vector_norm(b).item())

    if norm_a <= EPS or norm_b <= EPS:
        return 0.0

    cosine = float(
        torch.dot(a, b).item()
        / (norm_a * norm_b)
    )

    return float(np.clip(cosine, -1.0, 1.0))


def flattened_delta_vector(
    *,
    global_state: dict[str, torch.Tensor],
    local_state: dict[str, torch.Tensor],
) -> torch.Tensor:
    parts: list[torch.Tensor] = []

    for key in sorted(global_state):
        global_tensor = global_state[key]
        local_tensor = local_state[key]

        if not torch.is_floating_point(global_tensor):
            continue

        delta = (
            local_tensor.detach().cpu().to(dtype=torch.float64)
            - global_tensor.detach().cpu().to(dtype=torch.float64)
        )

        parts.append(delta.reshape(-1))

    if not parts:
        raise RuntimeError("No floating-point model deltas found.")

    vector = torch.cat(parts)

    if not torch.isfinite(vector).all():
        raise RuntimeError("Non-finite flattened update delta.")

    return vector


# =============================================================================
# Pareto scheduling
# =============================================================================

def dominates(
    objective_a: np.ndarray,
    objective_b: np.ndarray,
) -> bool:
    return bool(
        np.all(objective_a <= objective_b)
        and np.any(objective_a < objective_b)
    )


def fast_non_dominated_sort(
    objective_matrix: np.ndarray,
) -> list[list[int]]:
    population_size = objective_matrix.shape[0]

    domination_sets: list[list[int]] = [
        [] for _ in range(population_size)
    ]
    dominated_counts = np.zeros(
        population_size,
        dtype=np.int64,
    )

    first_front: list[int] = []

    for p in range(population_size):
        for q in range(population_size):
            if p == q:
                continue

            if dominates(
                objective_matrix[p],
                objective_matrix[q],
            ):
                domination_sets[p].append(q)
            elif dominates(
                objective_matrix[q],
                objective_matrix[p],
            ):
                dominated_counts[p] += 1

        if dominated_counts[p] == 0:
            first_front.append(p)

    fronts: list[list[int]] = []

    if first_front:
        fronts.append(first_front)

    front_index = 0

    while front_index < len(fronts):
        next_front: list[int] = []

        for p in fronts[front_index]:
            for q in domination_sets[p]:
                dominated_counts[q] -= 1

                if dominated_counts[q] == 0:
                    next_front.append(q)

        if next_front:
            fronts.append(next_front)

        front_index += 1

    return fronts


def crowding_distance(
    objective_matrix: np.ndarray,
    front: list[int],
) -> dict[int, float]:
    if not front:
        return {}

    if len(front) <= 2:
        return {
            index: float("inf")
            for index in front
        }

    distances = {
        index: 0.0
        for index in front
    }

    num_objectives = objective_matrix.shape[1]

    for objective_index in range(num_objectives):
        ordered = sorted(
            front,
            key=lambda index: objective_matrix[
                index,
                objective_index,
            ],
        )

        minimum = float(
            objective_matrix[
                ordered[0],
                objective_index,
            ]
        )
        maximum = float(
            objective_matrix[
                ordered[-1],
                objective_index,
            ]
        )

        distances[ordered[0]] = float("inf")
        distances[ordered[-1]] = float("inf")

        if maximum - minimum <= EPS:
            continue

        for position in range(1, len(ordered) - 1):
            previous_value = float(
                objective_matrix[
                    ordered[position - 1],
                    objective_index,
                ]
            )
            next_value = float(
                objective_matrix[
                    ordered[position + 1],
                    objective_index,
                ]
            )

            if math.isinf(distances[ordered[position]]):
                continue

            distances[ordered[position]] += (
                next_value - previous_value
            ) / (maximum - minimum)

    return distances


def exploration_slots_from_pressure(
    pressure: float,
) -> int:
    if pressure < 0.25:
        return 3
    if pressure < 0.5:
        return 2
    return 1


def select_arl_clients(
    *,
    round_index: int,
    previous_global_pressure: float,
    client_ids: list[str],
    risk_state: dict[str, float],
    residual_energy: dict[str, float],
    predicted_costs: dict[str, float],
    last_selected_round: dict[str, int],
) -> tuple[list[str], pd.DataFrame]:
    eligible = [
        client_id
        for client_id in client_ids
        if residual_energy[client_id] >= CRITICAL_ENERGY
    ]

    if len(eligible) < CLIENTS_PER_ROUND:
        raise RuntimeError(
            f"Round {round_index}: only {len(eligible)} eligible clients."
        )

    exploration_slots = exploration_slots_from_pressure(
        previous_global_pressure
    )
    core_slots = CLIENTS_PER_ROUND - exploration_slots

    rows: list[dict[str, object]] = []
    objective_rows: list[list[float]] = []

    for client_id in eligible:
        lifetime_pressure = (
            predicted_costs[client_id]
            / max(residual_energy[client_id], EPS)
        )

        staleness = min(
            STALENESS_CAP,
            max(
                0,
                round_index - last_selected_round[client_id],
            ),
        )

        # All objectives are represented for minimization:
        # 1. minimize risk
        # 2. minimize depletion pressure
        # 3. maximize staleness -> minimize negative staleness
        objective_rows.append(
            [
                risk_state[client_id],
                lifetime_pressure,
                -float(staleness),
            ]
        )

        rows.append(
            {
                "round": round_index,
                "global_client_id": client_id,
                "risk_before_selection": risk_state[client_id],
                "residual_energy_before_selection": (
                    residual_energy[client_id]
                ),
                "predicted_participation_cost": (
                    predicted_costs[client_id]
                ),
                "lifetime_pressure": lifetime_pressure,
                "staleness": staleness,
                "scheduler_pressure_input": (
                    previous_global_pressure
                ),
                "exploration_slots": exploration_slots,
                "core_slots": core_slots,
            }
        )

    objective_matrix = np.asarray(
        objective_rows,
        dtype=np.float64,
    )

    if not np.isfinite(objective_matrix).all():
        raise RuntimeError("Non-finite Pareto objectives.")

    fronts = fast_non_dominated_sort(objective_matrix)

    if sum(len(front) for front in fronts) != len(eligible):
        raise RuntimeError("Pareto sorting did not cover all eligible clients.")

    front_rank: dict[int, int] = {}
    crowding: dict[int, float] = {}

    for rank, front in enumerate(fronts):
        distances = crowding_distance(
            objective_matrix,
            front,
        )

        for index in front:
            front_rank[index] = rank
            crowding[index] = distances[index]

    ordered_indices = sorted(
        range(len(eligible)),
        key=lambda index: (
            front_rank[index],
            -crowding[index],
            objective_matrix[index, 0],
            objective_matrix[index, 1],
            objective_matrix[index, 2],
            eligible[index],
        ),
    )

    core_indices = ordered_indices[:core_slots]
    core_selected = [
        eligible[index]
        for index in core_indices
    ]

    remaining = [
        client_id
        for client_id in eligible
        if client_id not in set(core_selected)
    ]

    if len(remaining) < exploration_slots:
        raise RuntimeError("ARL-FL exploration pool is too small.")

    rng = np.random.default_rng(
        deterministic_seed(
            BASE_ARL_SEED,
            "arl_exploration",
            OUTER_FOLD,
            FL_SEED,
            round_index,
        )
    )

    explore_selected = rng.choice(
        np.asarray(remaining, dtype=object),
        size=exploration_slots,
        replace=False,
    ).tolist()

    selected = core_selected + explore_selected

    if len(selected) != CLIENTS_PER_ROUND:
        raise RuntimeError("ARL-FL did not select exactly 8 clients.")
    if len(set(selected)) != CLIENTS_PER_ROUND:
        raise RuntimeError("ARL-FL selected duplicate clients.")

    core_set = set(core_selected)
    explore_set = set(explore_selected)

    audit_rows: list[dict[str, object]] = []

    for index, row in enumerate(rows):
        client_id = str(row["global_client_id"])

        if client_id in core_set:
            selection_mode = "pareto_core"
        elif client_id in explore_set:
            selection_mode = "explore"
        else:
            selection_mode = "not_selected"

        audit_rows.append(
            {
                **row,
                "pareto_front_rank": front_rank[index],
                "crowding_distance": crowding[index],
                "selection_mode": selection_mode,
                "selected": client_id in set(selected),
            }
        )

    return selected, pd.DataFrame(audit_rows)


# =============================================================================
# Risk estimation
# =============================================================================

def compute_round_risk(
    *,
    round_index: int,
    selected: list[str],
    global_state: dict[str, torch.Tensor],
    local_results,
    previous_update: dict[str, torch.Tensor],
    risk_state: dict[str, float],
) -> tuple[
    pd.DataFrame,
    dict[str, float],
    dict[str, torch.Tensor],
    float,
    dict[str, torch.Tensor],
]:
    result_by_client = {
        result.client_id: result
        for result in local_results
    }

    current_updates = {
        client_id: flattened_delta_vector(
            global_state=global_state,
            local_state=result_by_client[client_id].state_dict,
        )
        for client_id in selected
    }

    stacked = torch.stack(
        [current_updates[client_id] for client_id in selected],
        dim=0,
    )

    coordinate_median = torch.median(
        stacked,
        dim=0,
    ).values

    update_norms = {
        client_id: float(
            torch.linalg.vector_norm(
                current_updates[client_id]
            ).item()
        )
        for client_id in selected
    }

    log_norms = np.asarray(
        [
            math.log(update_norms[client_id] + EPS)
            for client_id in selected
        ],
        dtype=np.float64,
    )

    median_log_norm = float(np.median(log_norms))
    mad_log_norm = float(
        np.median(
            np.abs(log_norms - median_log_norm)
        )
    )

    updated_risk = dict(risk_state)
    updated_previous = dict(previous_update)
    rows: list[dict[str, object]] = []

    for client_id in selected:
        cosine_to_median = cosine_similarity(
            current_updates[client_id],
            coordinate_median,
        )

        direction_anomaly = float(
            np.clip(
                (1.0 - cosine_to_median) / 2.0,
                0.0,
                1.0,
            )
        )

        log_norm = math.log(
            update_norms[client_id] + EPS
        )

        robust_z = max(
            0.0,
            (
                log_norm - median_log_norm
            )
            / (
                1.4826 * mad_log_norm + EPS
            ),
        )

        norm_anomaly = float(
            np.clip(
                1.0 - math.exp(-robust_z / 3.0),
                0.0,
                1.0,
            )
        )

        if client_id in previous_update:
            temporal_cosine = cosine_similarity(
                current_updates[client_id],
                previous_update[client_id],
            )
            temporal_inconsistency = float(
                np.clip(
                    (1.0 - temporal_cosine) / 2.0,
                    0.0,
                    1.0,
                )
            )
            temporal_first_participation = False
        else:
            temporal_cosine = float("nan")
            temporal_inconsistency = 0.0
            temporal_first_participation = True

        instantaneous_risk = float(
            np.median(
                [
                    direction_anomaly,
                    norm_anomaly,
                    temporal_inconsistency,
                ]
            )
        )

        old_risk = risk_state[client_id]

        new_risk = float(
            np.clip(
                RISK_BETA * old_risk
                + (1.0 - RISK_BETA)
                * instantaneous_risk,
                0.0,
                1.0,
            )
        )

        updated_risk[client_id] = new_risk
        updated_previous[client_id] = (
            current_updates[client_id].clone()
        )

        rows.append(
            {
                "round": round_index,
                "global_client_id": client_id,
                "update_norm": update_norms[client_id],
                "cosine_to_coordinate_median": cosine_to_median,
                "direction_anomaly": direction_anomaly,
                "median_log_update_norm": median_log_norm,
                "mad_log_update_norm": mad_log_norm,
                "robust_upper_norm_z": robust_z,
                "norm_anomaly": norm_anomaly,
                "temporal_first_participation": (
                    temporal_first_participation
                ),
                "temporal_cosine": temporal_cosine,
                "temporal_inconsistency": temporal_inconsistency,
                "instantaneous_risk": instantaneous_risk,
                "risk_before": old_risk,
                "risk_after": new_risk,
            }
        )

    selected_risks = np.asarray(
        [
            updated_risk[client_id]
            for client_id in selected
        ],
        dtype=np.float64,
    )

    global_pressure = float(
        np.quantile(
            selected_risks,
            GLOBAL_PRESSURE_QUANTILE,
        )
    )

    if not np.isfinite(global_pressure):
        raise RuntimeError("Non-finite global adversarial pressure.")

    global_pressure = float(
        np.clip(global_pressure, 0.0, 1.0)
    )

    return (
        pd.DataFrame(rows),
        updated_risk,
        updated_previous,
        global_pressure,
        current_updates,
    )


# =============================================================================
# Adaptive robust aggregation
# =============================================================================

def adaptive_robust_aggregation(
    *,
    round_index: int,
    global_state: dict[str, torch.Tensor],
    local_results,
    current_updates: dict[str, torch.Tensor],
    global_pressure: float,
) -> tuple[
    dict[str, torch.Tensor],
    pd.DataFrame,
    pd.DataFrame,
]:
    selected = [
        result.client_id
        for result in local_results
    ]

    raw_norms = np.asarray(
        [
            float(
                torch.linalg.vector_norm(
                    current_updates[client_id]
                ).item()
            )
            for client_id in selected
        ],
        dtype=np.float64,
    )

    median_norm = float(np.median(raw_norms))
    mad_norm = float(
        np.median(
            np.abs(raw_norms - median_norm)
        )
    )

    kappa = float(
        CLIP_KAPPA_LOW_PRESSURE
        - (
            CLIP_KAPPA_LOW_PRESSURE
            - CLIP_KAPPA_HIGH_PRESSURE
        )
        * global_pressure
    )

    kappa = float(
        np.clip(
            kappa,
            CLIP_KAPPA_HIGH_PRESSURE,
            CLIP_KAPPA_LOW_PRESSURE,
        )
    )

    clip_threshold = float(
        median_norm
        + kappa * 1.4826 * mad_norm
    )

    if not np.isfinite(clip_threshold) or clip_threshold <= 0.0:
        clip_threshold = max(median_norm, EPS)

    trim_k = (
        TRIM_K_LOW_PRESSURE
        if global_pressure < PRESSURE_SWITCH
        else TRIM_K_HIGH_PRESSURE
    )

    if 2 * trim_k >= len(local_results):
        raise RuntimeError(
            "Adaptive trimmed mean would remove all updates."
        )

    clip_factors: dict[str, float] = {}

    for client_id, raw_norm in zip(selected, raw_norms):
        clip_factors[client_id] = float(
            min(
                1.0,
                clip_threshold / max(raw_norm, EPS),
            )
        )

    result_by_client = {
        result.client_id: result
        for result in local_results
    }

    new_state: dict[str, torch.Tensor] = {}

    for key, global_tensor in global_state.items():
        global64 = global_tensor.to(dtype=torch.float64)

        clipped_deltas: list[torch.Tensor] = []

        for client_id in selected:
            local64 = (
                result_by_client[client_id]
                .state_dict[key]
                .to(dtype=torch.float64)
            )

            delta = local64 - global64

            clipped_deltas.append(
                delta * clip_factors[client_id]
            )

        stacked = torch.stack(
            clipped_deltas,
            dim=0,
        )

        sorted_deltas, _ = torch.sort(
            stacked,
            dim=0,
        )

        retained = sorted_deltas[
            trim_k : len(local_results) - trim_k
        ]

        mean_delta = retained.mean(dim=0)

        new_state[key] = (
            global64 + mean_delta
        ).to(dtype=global_tensor.dtype)

    client_rows = [
        {
            "round": round_index,
            "global_client_id": client_id,
            "raw_update_norm": float(raw_norm),
            "clip_threshold": clip_threshold,
            "clip_factor": clip_factors[client_id],
            "clipped": clip_factors[client_id] < 1.0 - 1e-12,
        }
        for client_id, raw_norm in zip(selected, raw_norms)
    ]

    round_rows = [
        {
            "round": round_index,
            "global_pressure": global_pressure,
            "median_update_norm": median_norm,
            "mad_update_norm": mad_norm,
            "kappa": kappa,
            "clip_threshold": clip_threshold,
            "trim_k_each_tail": trim_k,
            "retained_updates_per_coordinate": (
                len(local_results) - 2 * trim_k
            ),
            "num_clipped_clients": sum(
                1
                for factor in clip_factors.values()
                if factor < 1.0 - 1e-12
            ),
            "direct_per_client_risk_weighting": False,
            "residual_energy_weighting": False,
        }
    ]

    return (
        new_state,
        pd.DataFrame(client_rows),
        pd.DataFrame(round_rows),
    )


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Three-round ARL-FL smoke test with multisignal risk, "
            "Pareto scheduling, adaptive exploration, clipping, "
            "and trimmed-mean aggregation."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()

    base, base_path = load_base_module(project_root)

    processed_root = (
        project_root
        / "data"
        / "processed"
        / "pamap2"
        / "protocol_v1_w256_s128"
    )

    fl_protocol_root = (
        project_root
        / "outputs"
        / "protocols"
        / "pamap2_fl_experiment_v1"
    )

    partition_root = (
        project_root
        / "outputs"
        / "protocols"
        / "pamap2_fl_partitions_v1"
    )

    evaluation_root = (
        project_root
        / "outputs"
        / "protocols"
        / "pamap2_evaluation_v2"
    )

    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else project_root
        / "outputs"
        / "federated"
        / "pamap2"
        / "smoke_arl_fl_v1"
    )

    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(
            "ERROR: output directory already exists and is not empty:\n"
            f"  {output_root}"
        )

    output_root.mkdir(parents=True, exist_ok=True)

    protocol_path = (
        fl_protocol_root / "FL_EXPERIMENTAL_PROTOCOL_V1.json"
    )

    condition_path = (
        fl_protocol_root / "matched_condition_manifest.csv"
    )

    energy_path = (
        fl_protocol_root / "client_energy_profile.csv"
    )

    client_manifest_path = (
        partition_root / "outer_fold_client_manifest.csv"
    )

    assignment_path = (
        partition_root
        / f"master_assignments_{alpha_label(ALPHA)}.csv"
    )

    outer_manifest_path = (
        evaluation_root / "outer_fold_manifest.csv"
    )

    for path in (
        protocol_path,
        condition_path,
        energy_path,
        client_manifest_path,
        assignment_path,
        outer_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    protocol = json.loads(
        protocol_path.read_text(encoding="utf-8")
    )

    if protocol.get("status") != "FROZEN_BEFORE_ANY_FL_TRAINING":
        raise RuntimeError("Unexpected FL protocol status.")

    conditions = pd.read_csv(condition_path)
    energy_profiles = pd.read_csv(energy_path)
    client_manifest = pd.read_csv(client_manifest_path)
    assignments = pd.read_csv(assignment_path)
    outer_manifest = pd.read_csv(outer_manifest_path)

    condition = conditions[
        (conditions["outer_fold"].astype(int) == OUTER_FOLD)
        & np.isclose(
            conditions["alpha"].astype(float),
            ALPHA,
        )
        & (
            conditions["scenario"].astype(str)
            == SCENARIO
        )
        & (
            conditions["fl_seed"].astype(int)
            == FL_SEED
        )
    ]

    if len(condition) != 1:
        raise RuntimeError(
            f"Expected one matched condition, found {len(condition)}."
        )

    condition_row = condition.iloc[0]
    model_seed = int(condition_row["model_seed"])

    outer_row = outer_manifest[
        outer_manifest["outer_fold"].astype(int)
        == OUTER_FOLD
    ]

    if len(outer_row) != 1:
        raise RuntimeError("Expected one outer-fold row.")

    outer_test_subject = parse_subject(
        outer_row.iloc[0]["outer_test_subject"]
    )

    fold_clients = client_manifest[
        (
            client_manifest["outer_fold"].astype(int)
            == OUTER_FOLD
        )
        & np.isclose(
            client_manifest["alpha"].astype(float),
            ALPHA,
        )
    ].copy()

    client_ids = sorted(
        fold_clients["global_client_id"]
        .astype(str)
        .unique()
        .tolist()
    )

    if len(client_ids) != CLIENTS_PER_FOLD:
        raise RuntimeError(
            f"Expected 28 clients, found {len(client_ids)}."
        )

    fold_assignments = assignments[
        assignments["global_client_id"]
        .astype(str)
        .isin(client_ids)
    ].copy()

    if outer_test_subject in set(
        fold_assignments["subject_id"].astype(int)
    ):
        raise RuntimeError("Outer test subject leaked into training.")

    retained_rows = np.sort(
        fold_assignments["row_index"]
        .astype(np.int64)
        .unique()
    )

    config = {
        "status": "FROZEN_ARL_FL_SMOKE_CONFIGURATION",
        "fl_protocol_path": str(protocol_path),
        "fl_protocol_sha256": sha256_file(protocol_path),
        "validated_base_engine_path": str(base_path),
        "validated_base_engine_sha256": sha256_file(base_path),
        "condition_id": int(condition_row["condition_id"]),
        "outer_fold": OUTER_FOLD,
        "outer_test_subject": outer_test_subject,
        "alpha": ALPHA,
        "scenario": SCENARIO,
        "method": METHOD,
        "fl_seed": FL_SEED,
        "model_seed": model_seed,
        "rounds": SMOKE_ROUNDS,
        "clients_per_round": CLIENTS_PER_ROUND,
        "risk": {
            "signals": [
                "direction anomaly",
                "robust upper norm anomaly",
                "temporal inconsistency",
            ],
            "instantaneous_fusion": "median",
            "ema_beta": RISK_BETA,
            "first_temporal_score": 0.0,
        },
        "global_pressure": {
            "selected_client_risk_quantile": (
                GLOBAL_PRESSURE_QUANTILE
            ),
        },
        "scheduler": {
            "objectives": [
                "minimize risk",
                "minimize predicted cost / residual energy",
                "maximize staleness",
            ],
            "algorithm": (
                "non-dominated Pareto sorting with crowding distance"
            ),
            "staleness_cap": STALENESS_CAP,
            "exploration_slots": (
                "3 if pressure <0.25; "
                "2 if 0.25<=pressure<0.5; "
                "1 otherwise"
            ),
            "pressure_timing": (
                "previous completed round pressure used for next selection; "
                "round 1 starts at pressure 0"
            ),
        },
        "aggregation": {
            "direct_per_client_risk_weighting": False,
            "residual_energy_weighting": False,
            "global_pressure_controls_robustness": True,
            "clip_rule": (
                "median norm + kappa * 1.4826 * MAD"
            ),
            "kappa": "3 - 2 * global pressure",
            "trim_k": (
                "1 if pressure <0.5, otherwise 2"
            ),
        },
        "outer_test_used_for_training_selection_or_normalization": False,
    }

    (output_root / "ARL_FL_SMOKE_CONFIG.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    device = base.select_device()

    print("=== PAMAP2 ARL-FL smoke test ===")
    print(
        f"Condition:         outer={OUTER_FOLD}, "
        f"alpha={ALPHA}, clean"
    )
    print(f"Method:            {METHOD}")
    print(f"FL seed:           {FL_SEED}")
    print(f"Smoke rounds:      {SMOKE_ROUNDS}")
    print(f"Clients:           {len(client_ids)}")
    print(f"Clients per round: {CLIENTS_PER_ROUND}")
    print(f"Outer test:        subject{outer_test_subject}")
    print(f"Output:            {output_root}")
    print(f"PyTorch:           {torch.__version__}")
    print(f"Device:            {device}")

    if device.type == "xpu":
        print(
            f"XPU device:         "
            f"{torch.xpu.get_device_name(0)}"
        )

    print()

    print("Loading data and building Magnitude6...")

    dataset = base.load_all_raw_scale_windows(
        processed_root
    )

    x_magnitude6 = base.build_magnitude6(
        dataset.x_raw_full36
    )

    mean, std = base.fit_retained_client_zscore(
        x_magnitude6,
        retained_rows,
    )

    np.save(
        output_root / "normalization_mean.npy",
        mean,
    )

    np.save(
        output_root / "normalization_std.npy",
        std,
    )

    client_tensors: dict[
        str,
        tuple[torch.Tensor, torch.Tensor],
    ] = {}

    for client_id in client_ids:
        rows = np.sort(
            fold_assignments[
                (
                    fold_assignments[
                        "global_client_id"
                    ].astype(str)
                    == client_id
                )
            ]["row_index"]
            .astype(np.int64)
            .to_numpy()
        )

        client_tensors[client_id] = (
            base.normalize_rows(
                x_magnitude6,
                rows,
                mean,
                std,
            ),
            torch.from_numpy(
                dataset.y[rows].astype(
                    np.int64,
                    copy=True,
                )
            ),
        )

    test_rows = np.where(
        dataset.subject == outer_test_subject
    )[0].astype(np.int64)

    x_test = base.normalize_rows(
        x_magnitude6,
        test_rows,
        mean,
        std,
    )

    y_test = torch.from_numpy(
        dataset.y[test_rows].astype(
            np.int64,
            copy=True,
        )
    )

    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    energy = energy_profiles[
        (
            energy_profiles["outer_fold"].astype(int)
            == OUTER_FOLD
        )
        & (
            energy_profiles["fl_seed"].astype(int)
            == FL_SEED
        )
        & (
            energy_profiles["global_client_id"]
            .astype(str)
            .isin(client_ids)
        )
    ].copy()

    if len(energy) != CLIENTS_PER_FOLD:
        raise RuntimeError(
            f"Expected 28 energy profiles, found {len(energy)}."
        )

    energy = (
        energy
        .set_index("global_client_id")
        .loc[client_ids]
        .reset_index()
    )

    initial_energy = {
        str(row.global_client_id): float(
            row.initial_energy
        )
        for row in energy.itertuples(index=False)
    }

    residual_energy = dict(initial_energy)

    compute_factor = {
        str(row.global_client_id): float(
            row.compute_factor
        )
        for row in energy.itertuples(index=False)
    }

    client_sizes = {
        client_id: int(
            client_tensors[client_id][1].shape[0]
        )
        for client_id in client_ids
    }

    median_client_windows = float(
        np.median(
            list(client_sizes.values())
        )
    )

    predicted_costs = {
        client_id: participation_cost(
            client_id=client_id,
            client_sizes=client_sizes,
            compute_factor=compute_factor,
            median_client_windows=(
                median_client_windows
            ),
        )
        for client_id in client_ids
    }

    risk_state = {
        client_id: 0.0
        for client_id in client_ids
    }

    previous_update: dict[
        str,
        torch.Tensor,
    ] = {}

    last_selected_round = {
        client_id: 0
        for client_id in client_ids
    }

    previous_global_pressure = 0.0

    print(
        f"[OK] Retained training windows: "
        f"{len(retained_rows)}"
    )
    print(
        f"[OK] Client size range: "
        f"{min(client_sizes.values())}-"
        f"{max(client_sizes.values())}"
    )
    print(
        f"[OK] Outer-test windows: "
        f"{len(y_test)}"
    )
    print()

    base.set_seed(model_seed)

    global_model = base.LightweightCNN1D(
        input_channels=6
    ).to(device)

    parameter_count = (
        base.count_trainable_parameters(
            global_model
        )
    )

    if parameter_count != 77004:
        raise RuntimeError(
            f"Unexpected model parameter count: "
            f"{parameter_count}"
        )

    global_state = base.cpu_state_dict(
        global_model
    )

    initial_metrics, _, _ = base.evaluate_model(
        global_model,
        test_loader,
        device,
    )

    learning_rows: list[
        dict[str, object]
    ] = [
        {
            "round": 0,
            "test_loss": initial_metrics.loss,
            "test_accuracy": initial_metrics.accuracy,
            "test_balanced_accuracy": (
                initial_metrics
                .balanced_accuracy
            ),
            "test_macro_f1": (
                initial_metrics.macro_f1
            ),
            "active_clients": CLIENTS_PER_FOLD,
            "mean_residual_energy": float(
                np.mean(
                    list(
                        residual_energy.values()
                    )
                )
            ),
            "min_residual_energy": float(
                np.min(
                    list(
                        residual_energy.values()
                    )
                )
            ),
            "mean_risk": 0.0,
            "max_risk": 0.0,
            "global_pressure": 0.0,
        }
    ]

    scheduler_frames: list[
        pd.DataFrame
    ] = []

    risk_frames: list[
        pd.DataFrame
    ] = []

    aggregation_client_frames: list[
        pd.DataFrame
    ] = []

    aggregation_round_frames: list[
        pd.DataFrame
    ] = []

    local_rows: list[
        dict[str, object]
    ] = []

    print(
        f"Round 0: "
        f"Macro-F1={initial_metrics.macro_f1:.4f}; "
        f"BalAcc="
        f"{initial_metrics.balanced_accuracy:.4f}; "
        f"Acc={initial_metrics.accuracy:.4f}; "
        f"pressure=0.0000"
    )
    print()

    for round_index in range(
        1,
        SMOKE_ROUNDS + 1,
    ):
        round_start = time.perf_counter()

        active_at_start = [
            client_id
            for client_id in client_ids
            if (
                residual_energy[client_id]
                >= CRITICAL_ENERGY
            )
        ]

        for client_id in active_at_start:
            residual_energy[client_id] = max(
                0.0,
                (
                    residual_energy[client_id]
                    - STANDBY_COST
                ),
            )

        selected, scheduler_df = (
            select_arl_clients(
                round_index=round_index,
                previous_global_pressure=(
                    previous_global_pressure
                ),
                client_ids=client_ids,
                risk_state=risk_state,
                residual_energy=(
                    residual_energy
                ),
                predicted_costs=(
                    predicted_costs
                ),
                last_selected_round=(
                    last_selected_round
                ),
            )
        )

        scheduler_frames.append(
            scheduler_df
        )

        mode_counts = (
            scheduler_df[
                scheduler_df["selected"]
            ]["selection_mode"]
            .value_counts()
            .to_dict()
        )

        exploration_slots = (
            exploration_slots_from_pressure(
                previous_global_pressure
            )
        )

        if (
            mode_counts.get(
                "pareto_core",
                0,
            )
            != (
                CLIENTS_PER_ROUND
                - exploration_slots
            )
        ):
            raise RuntimeError(
                f"Round {round_index}: "
                "unexpected Pareto-core count."
            )

        if (
            mode_counts.get(
                "explore",
                0,
            )
            != exploration_slots
        ):
            raise RuntimeError(
                f"Round {round_index}: "
                "unexpected exploration count."
            )

        print(
            f"Round {round_index}/"
            f"{SMOKE_ROUNDS}: "
            f"pressure_in="
            f"{previous_global_pressure:.4f}; "
            f"core="
            f"{CLIENTS_PER_ROUND - exploration_slots}; "
            f"explore={exploration_slots}"
        )

        local_results = []

        for (
            selection_rank,
            client_id,
        ) in enumerate(
            selected,
            start=1,
        ):
            x_client, y_client = (
                client_tensors[client_id]
            )

            local_seed = deterministic_seed(
                BASE_LOCAL_SEED,
                OUTER_FOLD,
                FL_SEED,
                round_index,
                client_id,
            )

            result = base.train_one_client(
                global_state=global_state,
                x_client=x_client,
                y_client=y_client,
                client_id=client_id,
                local_seed=local_seed,
                device=device,
            )

            local_results.append(result)

            local_rows.append(
                {
                    "round": round_index,
                    "selection_rank": (
                        selection_rank
                    ),
                    "global_client_id": (
                        client_id
                    ),
                    "selection_mode": (
                        scheduler_df[
                            (
                                scheduler_df[
                                    "global_client_id"
                                ]
                                == client_id
                            )
                        ][
                            "selection_mode"
                        ].iloc[0]
                    ),
                    "client_windows": (
                        client_sizes[client_id]
                    ),
                    "local_seed": local_seed,
                    "local_train_loss": (
                        result.train_loss
                    ),
                    "local_train_macro_f1": (
                        result.train_macro_f1
                    ),
                    "local_wall_seconds": (
                        result.wall_seconds
                    ),
                }
            )

        (
            risk_df,
            risk_state,
            previous_update,
            current_global_pressure,
            current_updates,
        ) = compute_round_risk(
            round_index=round_index,
            selected=selected,
            global_state=global_state,
            local_results=local_results,
            previous_update=(
                previous_update
            ),
            risk_state=risk_state,
        )

        risk_frames.append(risk_df)

        for client_id in selected:
            cost = predicted_costs[client_id]

            residual_energy[client_id] = max(
                0.0,
                (
                    residual_energy[client_id]
                    - cost
                ),
            )

            last_selected_round[
                client_id
            ] = round_index

        (
            global_state,
            aggregation_client_df,
            aggregation_round_df,
        ) = adaptive_robust_aggregation(
            round_index=round_index,
            global_state=global_state,
            local_results=local_results,
            current_updates=(
                current_updates
            ),
            global_pressure=(
                current_global_pressure
            ),
        )

        aggregation_client_frames.append(
            aggregation_client_df
        )

        aggregation_round_frames.append(
            aggregation_round_df
        )

        global_model.load_state_dict(
            global_state
        )
        global_model.to(device)

        metrics, _, _ = base.evaluate_model(
            global_model,
            test_loader,
            device,
        )

        active_after = sum(
            1
            for client_id in client_ids
            if (
                residual_energy[client_id]
                >= CRITICAL_ENERGY
            )
        )

        base.synchronize(device)

        wall_seconds = (
            time.perf_counter()
            - round_start
        )

        learning_rows.append(
            {
                "round": round_index,
                "test_loss": metrics.loss,
                "test_accuracy": (
                    metrics.accuracy
                ),
                "test_balanced_accuracy": (
                    metrics
                    .balanced_accuracy
                ),
                "test_macro_f1": (
                    metrics.macro_f1
                ),
                "active_clients": active_after,
                "mean_residual_energy": float(
                    np.mean(
                        list(
                            residual_energy
                            .values()
                        )
                    )
                ),
                "min_residual_energy": float(
                    np.min(
                        list(
                            residual_energy
                            .values()
                        )
                    )
                ),
                "mean_risk": float(
                    np.mean(
                        list(
                            risk_state
                            .values()
                        )
                    )
                ),
                "max_risk": float(
                    np.max(
                        list(
                            risk_state
                            .values()
                        )
                    )
                ),
                "global_pressure": (
                    current_global_pressure
                ),
            }
        )

        round_aggregation = (
            aggregation_round_df.iloc[0]
        )

        print(
            f"  [GLOBAL] "
            f"Macro-F1={metrics.macro_f1:.4f}; "
            f"BalAcc="
            f"{metrics.balanced_accuracy:.4f}; "
            f"Acc={metrics.accuracy:.4f}; "
            f"pressure_out="
            f"{current_global_pressure:.4f}; "
            f"kappa="
            f"{float(round_aggregation['kappa']):.4f}; "
            f"trim_k="
            f"{int(round_aggregation['trim_k_each_tail'])}; "
            f"clipped="
            f"{int(round_aggregation['num_clipped_clients'])}; "
            f"active={active_after}; "
            f"wall={wall_seconds:.1f}s"
        )
        print()

        previous_global_pressure = (
            current_global_pressure
        )

    learning_df = pd.DataFrame(
        learning_rows
    )

    scheduler_audit_df = pd.concat(
        scheduler_frames,
        ignore_index=True,
    )

    risk_signal_df = pd.concat(
        risk_frames,
        ignore_index=True,
    )

    aggregation_client_df = pd.concat(
        aggregation_client_frames,
        ignore_index=True,
    )

    aggregation_round_df = pd.concat(
        aggregation_round_frames,
        ignore_index=True,
    )

    selected_client_df = pd.DataFrame(
        local_rows
    )

    learning_df.to_csv(
        output_root / "learning_curve.csv",
        index=False,
    )

    scheduler_audit_df.to_csv(
        output_root / "scheduler_audit.csv",
        index=False,
    )

    risk_signal_df.to_csv(
        output_root / "risk_signal_log.csv",
        index=False,
    )

    aggregation_client_df.to_csv(
        output_root / "aggregation_client_log.csv",
        index=False,
    )

    aggregation_round_df.to_csv(
        output_root / "aggregation_round_log.csv",
        index=False,
    )

    selected_client_df.to_csv(
        output_root / "selected_client_log.csv",
        index=False,
    )

    final_state_df = pd.DataFrame(
        [
            {
                "global_client_id": client_id,
                "client_windows": (
                    client_sizes[client_id]
                ),
                "initial_energy": (
                    initial_energy[client_id]
                ),
                "residual_energy": (
                    residual_energy[client_id]
                ),
                "risk": (
                    risk_state[client_id]
                ),
                "last_selected_round": (
                    last_selected_round[
                        client_id
                    ]
                ),
                "active": (
                    residual_energy[client_id]
                    >= CRITICAL_ENERGY
                ),
            }
            for client_id in client_ids
        ]
    )

    final_state_df.to_csv(
        output_root / "final_client_state.csv",
        index=False,
    )

    torch.save(
        {
            "model_state_dict": global_state,
            "method": METHOD,
            "rounds_completed": (
                SMOKE_ROUNDS
            ),
            "risk_state": risk_state,
            "residual_energy": (
                residual_energy
            ),
            "last_selected_round": (
                last_selected_round
            ),
            "global_pressure": (
                previous_global_pressure
            ),
            "final_metrics": (
                learning_rows[-1]
            ),
        },
        output_root / "final_global_model.pt",
    )

    # ------------------------------------------------------------------
    # Validation assertions
    # ------------------------------------------------------------------
    if (
        len(selected_client_df)
        != SMOKE_ROUNDS
        * CLIENTS_PER_ROUND
    ):
        raise RuntimeError(
            "Unexpected number of ARL-FL "
            "local trainings."
        )

    if not np.all(
        selected_client_df
        .groupby("round")[
            "global_client_id"
        ]
        .nunique()
        .to_numpy()
        == CLIENTS_PER_ROUND
    ):
        raise RuntimeError(
            "Duplicate client selected "
            "within an ARL-FL round."
        )

    risk_columns = [
        "direction_anomaly",
        "norm_anomaly",
        "temporal_inconsistency",
        "instantaneous_risk",
        "risk_before",
        "risk_after",
    ]

    for column in risk_columns:
        if not (
            risk_signal_df[column]
            .between(
                0.0,
                1.0,
                inclusive="both",
            )
            .all()
        ):
            raise RuntimeError(
                f"Risk column escaped [0,1]: "
                f"{column}"
            )

    if not (
        aggregation_round_df[
            "kappa"
        ]
        .between(
            CLIP_KAPPA_HIGH_PRESSURE,
            CLIP_KAPPA_LOW_PRESSURE,
            inclusive="both",
        )
        .all()
    ):
        raise RuntimeError(
            "Adaptive kappa escaped frozen bounds."
        )

    if not set(
        aggregation_round_df[
            "trim_k_each_tail"
        ].astype(int)
    ).issubset(
        {
            TRIM_K_LOW_PRESSURE,
            TRIM_K_HIGH_PRESSURE,
        }
    ):
        raise RuntimeError(
            "Unexpected adaptive trim count."
        )

    if not (
        aggregation_round_df[
            "direct_per_client_risk_weighting"
        ]
        .eq(False)
        .all()
    ):
        raise RuntimeError(
            "Direct per-client risk weighting "
            "was unexpectedly enabled."
        )

    if not (
        aggregation_round_df[
            "residual_energy_weighting"
        ]
        .eq(False)
        .all()
    ):
        raise RuntimeError(
            "Residual-energy weighting "
            "was unexpectedly enabled."
        )

    if not np.isfinite(
        learning_df[
            [
                "test_loss",
                "test_accuracy",
                "test_balanced_accuracy",
                "test_macro_f1",
                "mean_residual_energy",
                "min_residual_energy",
                "mean_risk",
                "max_risk",
                "global_pressure",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Non-finite ARL-FL smoke metrics."
        )

    final_metrics = learning_df.iloc[-1]

    report_lines = [
        "PAMAP2 ARL-FL SMOKE TEST",
        "=" * 78,
        "",
        "STATUS",
        "-" * 78,
        "PASS",
        "",
        "CONDITION",
        "-" * 78,
        f"Outer fold: {OUTER_FOLD}",
        f"Outer test: subject{outer_test_subject}",
        f"Alpha: {ALPHA}",
        f"Scenario: {SCENARIO}",
        f"FL seed: {FL_SEED}",
        f"Rounds: {SMOKE_ROUNDS}",
        "",
        "RISK ESTIMATION",
        "-" * 78,
        "Signals: direction anomaly, robust upper norm anomaly, temporal inconsistency.",
        "Instantaneous fusion: median of the three signals.",
        f"Risk EMA beta: {RISK_BETA}",
        "First-participation temporal inconsistency: 0.",
        (
            "Global pressure: "
            f"{GLOBAL_PRESSURE_QUANTILE:.2f} quantile "
            "of selected-client risk after update."
        ),
        "",
        "PARETO SCHEDULING",
        "-" * 78,
        "Objective 1: minimize risk.",
        "Objective 2: minimize predicted participation cost / residual energy.",
        "Objective 3: maximize staleness.",
        "Algorithm: non-dominated sorting plus crowding-distance ordering.",
        f"Staleness cap: {STALENESS_CAP} rounds.",
        "Adaptive exploration: 3 / 2 / 1 slots for low / medium / high pressure.",
        "The previous completed round pressure controls the next round scheduler.",
        "",
        "ADAPTIVE ROBUST AGGREGATION",
        "-" * 78,
        "No residual-energy weighting.",
        "No direct per-client risk weighting.",
        "Global pressure adjusts robustness only.",
        "Clip threshold = median norm + kappa * 1.4826 * MAD.",
        "kappa = 3 - 2 * global pressure.",
        "trim k = 1 if pressure < 0.5, otherwise 2.",
        "Coordinate-wise trimmed mean is applied after norm clipping.",
        "",
        "DATA AND LEAKAGE",
        "-" * 78,
        f"Retained training windows: {len(retained_rows)}",
        f"Outer-test windows: {len(y_test)}",
        "Z-score fitted only on retained outer-training client windows.",
        (
            "Outer test did not contribute to risk, scheduling, energy, "
            "training, aggregation, or normalization."
        ),
        "",
        "RESULTS",
        "-" * 78,
        (
            f"Round 0 Macro-F1: "
            f"{learning_df.iloc[0]['test_macro_f1']:.4f}"
        ),
        (
            f"Round {SMOKE_ROUNDS} Macro-F1: "
            f"{float(final_metrics['test_macro_f1']):.4f}"
        ),
        (
            f"Round {SMOKE_ROUNDS} balanced accuracy: "
            f"{float(final_metrics['test_balanced_accuracy']):.4f}"
        ),
        (
            f"Round {SMOKE_ROUNDS} accuracy: "
            f"{float(final_metrics['test_accuracy']):.4f}"
        ),
        (
            f"Final active clients: "
            f"{int(final_metrics['active_clients'])}/"
            f"{CLIENTS_PER_FOLD}"
        ),
        (
            f"Final mean residual energy: "
            f"{float(final_metrics['mean_residual_energy']):.4f}"
        ),
        (
            f"Final minimum residual energy: "
            f"{float(final_metrics['min_residual_energy']):.4f}"
        ),
        (
            f"Final mean risk: "
            f"{float(final_metrics['mean_risk']):.4f}"
        ),
        (
            f"Final maximum risk: "
            f"{float(final_metrics['max_risk']):.4f}"
        ),
        (
            f"Final global pressure: "
            f"{float(final_metrics['global_pressure']):.4f}"
        ),
        "",
        "ROUND-LEVEL ROBUSTNESS",
        "-" * 78,
    ]

    for row in aggregation_round_df.itertuples(
        index=False
    ):
        report_lines.append(
            f"Round {int(row.round)}: "
            f"pressure={row.global_pressure:.4f}; "
            f"kappa={row.kappa:.4f}; "
            f"trim_k={int(row.trim_k_each_tail)}; "
            f"clipped={int(row.num_clipped_clients)}; "
            f"retained={int(row.retained_updates_per_coordinate)}"
        )

    report_lines.extend(
        [
            "",
            "FILES",
            "-" * 78,
            "ARL_FL_SMOKE_CONFIG.json",
            "ARL_FL_SMOKE_REPORT.txt",
            "scheduler_audit.csv",
            "risk_signal_log.csv",
            "aggregation_client_log.csv",
            "aggregation_round_log.csv",
            "selected_client_log.csv",
            "learning_curve.csv",
            "final_client_state.csv",
            "normalization_mean.npy",
            "normalization_std.npy",
            "final_global_model.pt",
        ]
    )

    report_path = (
        output_root
        / "ARL_FL_SMOKE_REPORT.txt"
    )

    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print(
        "=== ARL-FL smoke test "
        "completed successfully ==="
    )
    print(f"Report: {report_path}")
    print()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nARL-FL smoke test interrupted by user.",
            file=sys.stderr,
        )
        raise SystemExit(130)
