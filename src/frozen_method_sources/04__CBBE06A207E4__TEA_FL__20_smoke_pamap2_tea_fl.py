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
# Frozen TEA-FL smoke condition
# =============================================================================

OUTER_FOLD = 1
ALPHA = 1.0
SCENARIO = "clean"
METHOD = "tea_fl"
FL_SEED = 123

SMOKE_ROUNDS = 3
CLIENTS_PER_FOLD = 28
CLIENTS_PER_ROUND = 8

INITIAL_TRUST = 0.5
TRUST_BETA = 0.8
TRUST_FLOOR = 0.05
TRUST_CEILING = 1.0

SELECTION_TRUST_WEIGHT = 0.8
EXPLORATION_RATIO = 0.1

AGG_TRUST_EXPONENT = 2.0
AGG_ENERGY_EXPONENT = 0.3

CRITICAL_ENERGY = 0.10
STANDBY_COST = 0.0005
COMMUNICATION_COST = 0.004
COMPUTE_COEFFICIENT = 0.016

BASE_LOCAL_SEED = 20260706
BASE_TEA_SEED = 20260706
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
        raise RuntimeError("Could not load the validated base engine module.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, base_path


def minmax_normalize(
    values: dict[str, float],
    client_ids: list[str],
) -> dict[str, float]:
    array = np.asarray(
        [values[client_id] for client_id in client_ids],
        dtype=np.float64,
    )

    if not np.isfinite(array).all():
        raise RuntimeError("Non-finite values in min-max normalization.")

    minimum = float(array.min())
    maximum = float(array.max())

    if maximum - minimum <= EPS:
        return {
            client_id: 1.0
            for client_id in client_ids
        }

    return {
        client_id: float(
            (values[client_id] - minimum)
            / (maximum - minimum)
        )
        for client_id in client_ids
    }


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


# =============================================================================
# TEA-FL selection
# =============================================================================

def select_tea_clients(
    *,
    round_index: int,
    client_ids: list[str],
    trust: dict[str, float],
    residual_energy: dict[str, float],
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

    trust_norm = minmax_normalize(trust, eligible)
    energy_norm = minmax_normalize(residual_energy, eligible)

    scores = {
        client_id: (
            SELECTION_TRUST_WEIGHT * trust_norm[client_id]
            + (1.0 - SELECTION_TRUST_WEIGHT)
            * energy_norm[client_id]
        )
        for client_id in eligible
    }

    explore_count = max(
        1,
        int(math.ceil(CLIENTS_PER_ROUND * EXPLORATION_RATIO)),
    )
    exploit_count = CLIENTS_PER_ROUND - explore_count

    ranked = sorted(
        eligible,
        key=lambda client_id: (
            -scores[client_id],
            client_id,
        ),
    )

    exploit_selected = ranked[:exploit_count]
    exploration_pool = ranked[exploit_count:]

    if len(exploration_pool) < explore_count:
        raise RuntimeError("TEA-FL exploration pool is too small.")

    rng = np.random.default_rng(
        deterministic_seed(
            BASE_TEA_SEED,
            "tea_exploration",
            OUTER_FOLD,
            FL_SEED,
            round_index,
        )
    )

    explore_selected = rng.choice(
        np.asarray(exploration_pool, dtype=object),
        size=explore_count,
        replace=False,
    ).tolist()

    selected = exploit_selected + explore_selected

    if len(selected) != CLIENTS_PER_ROUND:
        raise RuntimeError("TEA-FL did not select exactly 8 clients.")
    if len(set(selected)) != CLIENTS_PER_ROUND:
        raise RuntimeError("TEA-FL selected duplicate clients.")

    exploit_set = set(exploit_selected)
    explore_set = set(explore_selected)

    audit_rows: list[dict[str, object]] = []

    for rank, client_id in enumerate(ranked, start=1):
        if client_id in exploit_set:
            selection_mode = "exploit"
        elif client_id in explore_set:
            selection_mode = "explore"
        else:
            selection_mode = "not_selected"

        audit_rows.append(
            {
                "round": round_index,
                "global_client_id": client_id,
                "eligible": True,
                "trust_before_round": trust[client_id],
                "residual_energy_before_selection": (
                    residual_energy[client_id]
                ),
                "normalized_trust": trust_norm[client_id],
                "normalized_energy": energy_norm[client_id],
                "joint_selection_score": scores[client_id],
                "score_rank": rank,
                "selection_mode": selection_mode,
                "selected": client_id in set(selected),
            }
        )

    return selected, pd.DataFrame(audit_rows)


# =============================================================================
# TEA-FL trust update
# =============================================================================

def flattened_state_vector(
    state_dict: dict[str, torch.Tensor],
) -> torch.Tensor:
    parts: list[torch.Tensor] = []

    for key in sorted(state_dict):
        tensor = state_dict[key]

        if not torch.is_floating_point(tensor):
            continue

        parts.append(
            tensor.detach()
            .cpu()
            .to(dtype=torch.float64)
            .reshape(-1)
        )

    if not parts:
        raise RuntimeError("No floating-point model tensors found.")

    return torch.cat(parts)


def update_trust_from_round_reference(
    *,
    round_index: int,
    selected: list[str],
    local_results,
    trust: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    by_client = {
        result.client_id: result
        for result in local_results
    }

    local_vectors = {
        client_id: flattened_state_vector(
            by_client[client_id].state_dict
        )
        for client_id in selected
    }

    reference = torch.stack(
        [local_vectors[client_id] for client_id in selected],
        dim=0,
    ).mean(dim=0)

    distances = {
        client_id: float(
            torch.linalg.vector_norm(
                local_vectors[client_id] - reference
            ).item()
        )
        for client_id in selected
    }

    distance_array = np.asarray(
        [distances[client_id] for client_id in selected],
        dtype=np.float64,
    )

    if not np.isfinite(distance_array).all():
        raise RuntimeError("Non-finite TEA-FL trust distances.")

    d_min = float(distance_array.min())
    d_max = float(distance_array.max())

    evidence: dict[str, float] = {}

    if d_max - d_min <= EPS:
        evidence = {
            client_id: 1.0
            for client_id in selected
        }
    else:
        for client_id in selected:
            normalized_distance = (
                distances[client_id] - d_min
            ) / (d_max - d_min)

            raw_evidence = 1.0 - normalized_distance
            evidence[client_id] = float(
                np.clip(
                    raw_evidence,
                    TRUST_FLOOR,
                    TRUST_CEILING,
                )
            )

    updated_trust = dict(trust)
    rows: list[dict[str, object]] = []

    for client_id in selected:
        old_trust = trust[client_id]

        new_trust = (
            TRUST_BETA * old_trust
            + (1.0 - TRUST_BETA) * evidence[client_id]
        )
        new_trust = float(
            np.clip(
                new_trust,
                TRUST_FLOOR,
                TRUST_CEILING,
            )
        )

        updated_trust[client_id] = new_trust

        rows.append(
            {
                "round": round_index,
                "global_client_id": client_id,
                "distance_to_equal_weight_reference": (
                    distances[client_id]
                ),
                "round_trust_evidence": evidence[client_id],
                "trust_before": old_trust,
                "trust_after": new_trust,
            }
        )

    return pd.DataFrame(rows), updated_trust


# =============================================================================
# TEA-FL aggregation
# =============================================================================

def tea_weighted_aggregation(
    *,
    global_state: dict[str, torch.Tensor],
    local_results,
    trust: dict[str, float],
    residual_energy: dict[str, float],
) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    unnormalized: dict[str, float] = {}

    for result in local_results:
        client_id = result.client_id

        weight = (
            float(result.windows)
            * (trust[client_id] ** AGG_TRUST_EXPONENT)
            * (
                residual_energy[client_id]
                ** AGG_ENERGY_EXPONENT
            )
        )

        if not np.isfinite(weight) or weight <= 0.0:
            raise RuntimeError(
                f"Invalid TEA-FL aggregation weight for {client_id}: {weight}"
            )

        unnormalized[client_id] = weight

    denominator = float(sum(unnormalized.values()))

    if not np.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError("Invalid TEA-FL aggregation denominator.")

    normalized = {
        client_id: value / denominator
        for client_id, value in unnormalized.items()
    }

    new_state: dict[str, torch.Tensor] = {}

    for key, global_tensor in global_state.items():
        global64 = global_tensor.to(dtype=torch.float64)
        accumulator = torch.zeros_like(global64)

        for result in local_results:
            client_id = result.client_id
            local64 = result.state_dict[key].to(dtype=torch.float64)
            delta = local64 - global64
            accumulator.add_(
                delta,
                alpha=normalized[client_id],
            )

        new_state[key] = (
            global64 + accumulator
        ).to(dtype=global_tensor.dtype)

    rows = [
        {
            "global_client_id": result.client_id,
            "client_windows": int(result.windows),
            "trust_used": trust[result.client_id],
            "residual_energy_used": (
                residual_energy[result.client_id]
            ),
            "trust_exponent": AGG_TRUST_EXPONENT,
            "energy_exponent": AGG_ENERGY_EXPONENT,
            "unnormalized_weight": unnormalized[result.client_id],
            "normalized_weight": normalized[result.client_id],
        }
        for result in local_results
    ]

    audit_df = pd.DataFrame(rows)

    if not np.isclose(
        audit_df["normalized_weight"].sum(),
        1.0,
        atol=1e-12,
    ):
        raise RuntimeError("TEA-FL normalized weights do not sum to 1.")

    return new_state, audit_df


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Three-round TEA-FL smoke test for the frozen PAMAP2 "
            "federated protocol."
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
        / "smoke_tea_fl_v1"
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
        & np.isclose(conditions["alpha"].astype(float), ALPHA)
        & (conditions["scenario"].astype(str) == SCENARIO)
        & (conditions["fl_seed"].astype(int) == FL_SEED)
    ]

    if len(condition) != 1:
        raise RuntimeError(
            f"Expected one matched condition, found {len(condition)}."
        )

    condition_row = condition.iloc[0]
    model_seed = int(condition_row["model_seed"])

    outer_row = outer_manifest[
        outer_manifest["outer_fold"].astype(int) == OUTER_FOLD
    ]
    if len(outer_row) != 1:
        raise RuntimeError("Expected one outer-fold row.")

    outer_test_subject = parse_subject(
        outer_row.iloc[0]["outer_test_subject"]
    )

    fold_clients = client_manifest[
        (client_manifest["outer_fold"].astype(int) == OUTER_FOLD)
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
        "status": "FROZEN_TEA_FL_SMOKE_CONFIGURATION",
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
        "initial_trust": INITIAL_TRUST,
        "trust_beta": TRUST_BETA,
        "trust_floor": TRUST_FLOOR,
        "selection_trust_weight": SELECTION_TRUST_WEIGHT,
        "exploration_ratio": EXPLORATION_RATIO,
        "explore_count_per_round": max(
            1,
            int(math.ceil(CLIENTS_PER_ROUND * EXPLORATION_RATIO)),
        ),
        "aggregation_trust_exponent": AGG_TRUST_EXPONENT,
        "aggregation_energy_exponent": AGG_ENERGY_EXPONENT,
        "round_reference": "equal-weight mean of selected local models",
        "trust_distance": "Euclidean distance over all floating-point model tensors",
        "trust_evidence_mapping": (
            "1 - minmax_normalized_distance, clipped to [0.05,1.0]"
        ),
        "aggregation_order": (
            "local training -> trust update -> energy update -> aggregation"
        ),
        "outer_test_used_for_training_selection_or_normalization": False,
    }

    (output_root / "TEA_FL_SMOKE_CONFIG.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    device = base.select_device()

    print("=== PAMAP2 TEA-FL smoke test ===")
    print(f"Condition:         outer={OUTER_FOLD}, alpha={ALPHA}, clean")
    print(f"Method:            {METHOD}")
    print(f"FL seed:           {FL_SEED}")
    print(f"Smoke rounds:      {SMOKE_ROUNDS}")
    print(f"Clients:           {len(client_ids)}")
    print(f"Clients per round: {CLIENTS_PER_ROUND}")
    print(f"Exploit/explore:   7/1")
    print(f"Outer test:        subject{outer_test_subject}")
    print(f"Output:            {output_root}")
    print(f"PyTorch:           {torch.__version__}")
    print(f"Device:            {device}")
    if device.type == "xpu":
        print(f"XPU device:         {torch.xpu.get_device_name(0)}")
    print()

    print("Loading data and building Magnitude6...")
    dataset = base.load_all_raw_scale_windows(processed_root)
    x_magnitude6 = base.build_magnitude6(dataset.x_raw_full36)

    mean, std = base.fit_retained_client_zscore(
        x_magnitude6,
        retained_rows,
    )

    np.save(output_root / "normalization_mean.npy", mean)
    np.save(output_root / "normalization_std.npy", std)

    client_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    for client_id in client_ids:
        rows = np.sort(
            fold_assignments[
                fold_assignments["global_client_id"].astype(str)
                == client_id
            ]["row_index"].astype(np.int64).to_numpy()
        )

        client_tensors[client_id] = (
            base.normalize_rows(
                x_magnitude6,
                rows,
                mean,
                std,
            ),
            torch.from_numpy(
                dataset.y[rows].astype(np.int64, copy=True)
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
        dataset.y[test_rows].astype(np.int64, copy=True)
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
        (energy_profiles["outer_fold"].astype(int) == OUTER_FOLD)
        & (energy_profiles["fl_seed"].astype(int) == FL_SEED)
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
        energy.set_index("global_client_id")
        .loc[client_ids]
        .reset_index()
    )

    initial_energy = {
        str(row.global_client_id): float(row.initial_energy)
        for row in energy.itertuples(index=False)
    }
    residual_energy = dict(initial_energy)

    compute_factor = {
        str(row.global_client_id): float(row.compute_factor)
        for row in energy.itertuples(index=False)
    }

    client_sizes = {
        client_id: int(client_tensors[client_id][1].shape[0])
        for client_id in client_ids
    }

    median_client_windows = float(
        np.median(list(client_sizes.values()))
    )

    trust = {
        client_id: INITIAL_TRUST
        for client_id in client_ids
    }

    print(f"[OK] Retained training windows: {len(retained_rows)}")
    print(
        f"[OK] Client size range: "
        f"{min(client_sizes.values())}–{max(client_sizes.values())}"
    )
    print(f"[OK] Outer-test windows: {len(y_test)}")
    print()

    base.set_seed(model_seed)
    global_model = base.LightweightCNN1D(input_channels=6).to(device)

    parameter_count = base.count_trainable_parameters(global_model)
    if parameter_count != 77004:
        raise RuntimeError(
            f"Unexpected model parameter count: {parameter_count}"
        )

    global_state = base.cpu_state_dict(global_model)

    initial_metrics, _, _ = base.evaluate_model(
        global_model,
        test_loader,
        device,
    )

    learning_rows: list[dict[str, object]] = [
        {
            "round": 0,
            "test_loss": initial_metrics.loss,
            "test_accuracy": initial_metrics.accuracy,
            "test_balanced_accuracy": (
                initial_metrics.balanced_accuracy
            ),
            "test_macro_f1": initial_metrics.macro_f1,
            "active_clients": CLIENTS_PER_FOLD,
            "mean_residual_energy": float(
                np.mean(list(residual_energy.values()))
            ),
            "min_residual_energy": float(
                np.min(list(residual_energy.values()))
            ),
            "mean_trust": float(np.mean(list(trust.values()))),
            "min_trust": float(np.min(list(trust.values()))),
            "max_trust": float(np.max(list(trust.values()))),
        }
    ]

    selection_frames: list[pd.DataFrame] = []
    trust_frames: list[pd.DataFrame] = []
    aggregation_frames: list[pd.DataFrame] = []
    local_rows: list[dict[str, object]] = []

    print(
        f"Round 0: Macro-F1={initial_metrics.macro_f1:.4f}; "
        f"BalAcc={initial_metrics.balanced_accuracy:.4f}; "
        f"Acc={initial_metrics.accuracy:.4f}; "
        f"Tmean={np.mean(list(trust.values())):.4f}"
    )
    print()

    for round_index in range(1, SMOKE_ROUNDS + 1):
        round_start = time.perf_counter()

        active_at_start = [
            client_id
            for client_id in client_ids
            if residual_energy[client_id] >= CRITICAL_ENERGY
        ]

        for client_id in active_at_start:
            residual_energy[client_id] = max(
                0.0,
                residual_energy[client_id] - STANDBY_COST,
            )

        selected, selection_df = select_tea_clients(
            round_index=round_index,
            client_ids=client_ids,
            trust=trust,
            residual_energy=residual_energy,
        )
        selection_frames.append(selection_df)

        selected_modes = (
            selection_df[selection_df["selected"]]
            ["selection_mode"]
            .value_counts()
            .to_dict()
        )

        if selected_modes.get("exploit", 0) != 7:
            raise RuntimeError(
                f"Round {round_index}: expected 7 exploit clients."
            )
        if selected_modes.get("explore", 0) != 1:
            raise RuntimeError(
                f"Round {round_index}: expected 1 explore client."
            )

        print(
            f"Round {round_index}/{SMOKE_ROUNDS}: "
            f"active={len(active_at_start)}; "
            "selected=8 (7 exploit + 1 explore)"
        )

        local_results = []

        for selection_rank, client_id in enumerate(selected, start=1):
            x_client, y_client = client_tensors[client_id]

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
                    "selection_rank": selection_rank,
                    "global_client_id": client_id,
                    "client_windows": client_sizes[client_id],
                    "local_seed": local_seed,
                    "local_train_loss": result.train_loss,
                    "local_train_macro_f1": (
                        result.train_macro_f1
                    ),
                    "local_wall_seconds": result.wall_seconds,
                }
            )

        trust_update_df, trust = (
            update_trust_from_round_reference(
                round_index=round_index,
                selected=selected,
                local_results=local_results,
                trust=trust,
            )
        )
        trust_frames.append(trust_update_df)

        # Frozen TEA-FL order: update energy after participation, then use
        # updated trust and residual energy in aggregation.
        for client_id in selected:
            cost = participation_cost(
                client_id=client_id,
                client_sizes=client_sizes,
                compute_factor=compute_factor,
                median_client_windows=median_client_windows,
            )

            residual_energy[client_id] = max(
                0.0,
                residual_energy[client_id] - cost,
            )

        global_state, aggregation_df = tea_weighted_aggregation(
            global_state=global_state,
            local_results=local_results,
            trust=trust,
            residual_energy=residual_energy,
        )

        aggregation_df.insert(0, "round", round_index)
        aggregation_frames.append(aggregation_df)

        global_model.load_state_dict(global_state)
        global_model.to(device)

        metrics, _, _ = base.evaluate_model(
            global_model,
            test_loader,
            device,
        )

        active_after = sum(
            1
            for client_id in client_ids
            if residual_energy[client_id] >= CRITICAL_ENERGY
        )

        base.synchronize(device)
        wall_seconds = time.perf_counter() - round_start

        learning_rows.append(
            {
                "round": round_index,
                "test_loss": metrics.loss,
                "test_accuracy": metrics.accuracy,
                "test_balanced_accuracy": (
                    metrics.balanced_accuracy
                ),
                "test_macro_f1": metrics.macro_f1,
                "active_clients": active_after,
                "mean_residual_energy": float(
                    np.mean(list(residual_energy.values()))
                ),
                "min_residual_energy": float(
                    np.min(list(residual_energy.values()))
                ),
                "mean_trust": float(
                    np.mean(list(trust.values()))
                ),
                "min_trust": float(
                    np.min(list(trust.values()))
                ),
                "max_trust": float(
                    np.max(list(trust.values()))
                ),
            }
        )

        print(
            f"  [GLOBAL] Macro-F1={metrics.macro_f1:.4f}; "
            f"BalAcc={metrics.balanced_accuracy:.4f}; "
            f"Acc={metrics.accuracy:.4f}; "
            f"Tmean={np.mean(list(trust.values())):.4f}; "
            f"Tmin={np.min(list(trust.values())):.4f}; "
            f"Tmax={np.max(list(trust.values())):.4f}; "
            f"active={active_after}; "
            f"wall={wall_seconds:.1f}s"
        )
        print()

    learning_df = pd.DataFrame(learning_rows)
    selection_audit_df = pd.concat(
        selection_frames,
        ignore_index=True,
    )
    trust_update_df = pd.concat(
        trust_frames,
        ignore_index=True,
    )
    aggregation_weight_df = pd.concat(
        aggregation_frames,
        ignore_index=True,
    )
    local_training_df = pd.DataFrame(local_rows)

    learning_df.to_csv(
        output_root / "learning_curve.csv",
        index=False,
    )
    selection_audit_df.to_csv(
        output_root / "selection_score_audit.csv",
        index=False,
    )
    trust_update_df.to_csv(
        output_root / "trust_update_log.csv",
        index=False,
    )
    aggregation_weight_df.to_csv(
        output_root / "aggregation_weight_log.csv",
        index=False,
    )
    local_training_df.to_csv(
        output_root / "selected_client_log.csv",
        index=False,
    )

    final_state_df = pd.DataFrame(
        [
            {
                "global_client_id": client_id,
                "client_windows": client_sizes[client_id],
                "initial_energy": initial_energy[client_id],
                "residual_energy": residual_energy[client_id],
                "trust": trust[client_id],
                "active": (
                    residual_energy[client_id] >= CRITICAL_ENERGY
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
            "rounds_completed": SMOKE_ROUNDS,
            "trust": trust,
            "residual_energy": residual_energy,
            "final_metrics": learning_rows[-1],
        },
        output_root / "final_global_model.pt",
    )

    # ------------------------------------------------------------------
    # Validation assertions
    # ------------------------------------------------------------------
    if len(local_training_df) != SMOKE_ROUNDS * CLIENTS_PER_ROUND:
        raise RuntimeError("Unexpected number of TEA-FL local trainings.")

    if not np.all(
        local_training_df.groupby("round")["global_client_id"]
        .nunique()
        .to_numpy()
        == CLIENTS_PER_ROUND
    ):
        raise RuntimeError("Duplicate client selected within a TEA-FL round.")

    if not (
        trust_update_df["trust_after"].between(
            TRUST_FLOOR,
            TRUST_CEILING,
        ).all()
    ):
        raise RuntimeError("TEA-FL trust escaped frozen bounds.")

    if not np.allclose(
        aggregation_weight_df.groupby("round")["normalized_weight"]
        .sum()
        .to_numpy(),
        1.0,
        atol=1e-12,
    ):
        raise RuntimeError("TEA-FL aggregation weights do not sum to 1.")

    if not np.isfinite(
        learning_df[
            [
                "test_loss",
                "test_accuracy",
                "test_balanced_accuracy",
                "test_macro_f1",
                "mean_residual_energy",
                "min_residual_energy",
                "mean_trust",
                "min_trust",
                "max_trust",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("Non-finite TEA-FL smoke metrics.")

    final_metrics = learning_df.iloc[-1]

    report_lines = [
        "PAMAP2 TEA-FL SMOKE TEST",
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
        "TEA-FL CLIENT STATE",
        "-" * 78,
        f"Initial trust: {INITIAL_TRUST}",
        f"Trust EMA beta: {TRUST_BETA}",
        f"Trust floor: {TRUST_FLOOR}",
        "Trust reference: equal-weight mean of selected local models.",
        "Trust distance: Euclidean distance over floating-point model tensors.",
        "Round evidence: inverse min-max distance mapped to [0.05, 1.0].",
        "",
        "SELECTION",
        "-" * 78,
        "Trust and residual energy min-max normalized over eligible clients.",
        (
            f"Joint score = {SELECTION_TRUST_WEIGHT:.1f} * normalized trust "
            f"+ {1.0 - SELECTION_TRUST_WEIGHT:.1f} * normalized energy."
        ),
        "Seven highest-ranked clients exploited per round.",
        "One client uniformly explored from the remaining eligible pool.",
        "",
        "AGGREGATION",
        "-" * 78,
        (
            "Unnormalized weight = client_windows * "
            f"trust^{AGG_TRUST_EXPONENT} * "
            f"residual_energy^{AGG_ENERGY_EXPONENT}."
        ),
        "Weights normalized across the eight selected clients.",
        "Updated trust and post-participation residual energy used.",
        "",
        "DATA AND LEAKAGE",
        "-" * 78,
        f"Retained training windows: {len(retained_rows)}",
        f"Outer-test windows: {len(y_test)}",
        "Z-score fitted only on retained outer-training client windows.",
        (
            "Outer test did not contribute to selection, trust, energy, "
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
            f"{int(final_metrics['active_clients'])}/{CLIENTS_PER_FOLD}"
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
            f"Final mean trust: "
            f"{float(final_metrics['mean_trust']):.4f}"
        ),
        (
            f"Final trust range: "
            f"{float(final_metrics['min_trust']):.4f}–"
            f"{float(final_metrics['max_trust']):.4f}"
        ),
        "",
        "FILES",
        "-" * 78,
        "TEA_FL_SMOKE_CONFIG.json",
        "TEA_FL_SMOKE_REPORT.txt",
        "selection_score_audit.csv",
        "trust_update_log.csv",
        "aggregation_weight_log.csv",
        "selected_client_log.csv",
        "learning_curve.csv",
        "final_client_state.csv",
        "normalization_mean.npy",
        "normalization_std.npy",
        "final_global_model.pt",
    ]

    report_path = output_root / "TEA_FL_SMOKE_REPORT.txt"
    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("=== TEA-FL smoke test completed successfully ===")
    print(f"Report: {report_path}")
    print()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nTEA-FL smoke test interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
