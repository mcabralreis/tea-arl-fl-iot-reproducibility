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
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


OUTER_FOLD = 1
ALPHA = 1.0
SCENARIO = "clean"
FL_SEED = 123
METHODS = ("fedprox", "random_trimmed_mean")
SMOKE_ROUNDS = 3
CLIENTS_PER_FOLD = 28
CLIENTS_PER_ROUND = 8
FEDPROX_MU = 0.01
TRIM_K = 2
BASE_LOCAL_SEED = 20260706


def load_base_module(project_root: Path):
    path = project_root / "16_smoke_pamap2_federated_engine.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"Required base engine not found: {path}\n"
            "Keep 16_smoke_pamap2_federated_engine.py in the project root."
        )

    spec = importlib.util.spec_from_file_location("pamap2_fl_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import base engine: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def train_one_client_method(
    *,
    base,
    method: str,
    global_state: dict[str, torch.Tensor],
    x_client: torch.Tensor,
    y_client: torch.Tensor,
    client_id: str,
    local_seed: int,
    device: torch.device,
):
    base.set_seed(local_seed)

    model = base.LightweightCNN1D(input_channels=6)
    model.load_state_dict(global_state)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base.LOCAL_LEARNING_RATE,
        weight_decay=base.LOCAL_WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    global_parameter_refs: dict[str, torch.Tensor] = {}
    if method == "fedprox":
        for name, _parameter in model.named_parameters():
            global_parameter_refs[name] = (
                global_state[name].to(device=device).detach()
            )

    generator = torch.Generator()
    generator.manual_seed(local_seed)

    loader = DataLoader(
        TensorDataset(x_client, y_client),
        batch_size=base.LOCAL_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        generator=generator,
    )

    total_task_loss = 0.0
    total_examples = 0
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    start = time.perf_counter()

    for _ in range(base.LOCAL_EPOCHS):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            task_loss = criterion(logits, y_batch)
            loss = task_loss

            if method == "fedprox":
                proximal_term = torch.zeros(
                    (), device=device, dtype=task_loss.dtype
                )
                for name, parameter in model.named_parameters():
                    proximal_term = proximal_term + torch.sum(
                        (parameter - global_parameter_refs[name]) ** 2
                    )
                loss = task_loss + 0.5 * FEDPROX_MU * proximal_term

            loss.backward()
            optimizer.step()

            batch_size = int(y_batch.shape[0])
            total_task_loss += float(task_loss.detach().cpu().item()) * batch_size
            total_examples += batch_size

            prediction = logits.detach().argmax(dim=1)
            all_true.append(y_batch.detach().cpu().numpy())
            all_pred.append(prediction.cpu().numpy())

    base.synchronize(device)
    wall_seconds = time.perf_counter() - start

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    result = base.LocalTrainResult(
        client_id=client_id,
        windows=int(y_client.shape[0]),
        train_loss=total_task_loss / max(total_examples, 1),
        train_macro_f1=float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        wall_seconds=wall_seconds,
        state_dict=base.cpu_state_dict(model),
    )

    del model
    del optimizer
    if device.type == "xpu":
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass

    return result


def trimmed_mean_state(
    *,
    global_state: dict[str, torch.Tensor],
    local_results,
) -> dict[str, torch.Tensor]:
    if len(local_results) != CLIENTS_PER_ROUND:
        raise RuntimeError(
            f"Trimmed mean expected {CLIENTS_PER_ROUND} updates."
        )
    if 2 * TRIM_K >= len(local_results):
        raise RuntimeError("Trimmed mean removes all updates.")

    new_state: dict[str, torch.Tensor] = {}

    for key, global_tensor in global_state.items():
        global64 = global_tensor.to(dtype=torch.float64)
        deltas = torch.stack(
            [
                result.state_dict[key].to(dtype=torch.float64) - global64
                for result in local_results
            ],
            dim=0,
        )

        sorted_deltas, _ = torch.sort(deltas, dim=0)
        retained = sorted_deltas[TRIM_K : len(local_results) - TRIM_K]
        mean_delta = retained.mean(dim=0)

        new_state[key] = (
            global64 + mean_delta
        ).to(dtype=global_tensor.dtype)

    return new_state


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Paired three-round smoke tests for FedProx and "
            "Random Selection + Trimmed Mean."
        )
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    base = load_base_module(project_root)

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
        / "smoke_fedprox_trimmed_v1"
    )

    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(
            "ERROR: output directory already exists and is not empty:\n"
            f"  {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    protocol_path = fl_protocol_root / "FL_EXPERIMENTAL_PROTOCOL_V1.json"
    condition_path = fl_protocol_root / "matched_condition_manifest.csv"
    energy_path = fl_protocol_root / "client_energy_profile.csv"
    client_manifest_path = partition_root / "outer_fold_client_manifest.csv"
    assignment_path = (
        partition_root
        / f"master_assignments_{base.alpha_label(ALPHA)}.csv"
    )
    outer_manifest_path = evaluation_root / "outer_fold_manifest.csv"

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

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
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
    random_schedule_seed = int(condition_row["random_schedule_seed"])

    outer_row = outer_manifest[
        outer_manifest["outer_fold"].astype(int) == OUTER_FOLD
    ]
    if len(outer_row) != 1:
        raise RuntimeError("Expected one outer-fold row.")

    outer_test_subject = base.parse_subject(
        outer_row.iloc[0]["outer_test_subject"]
    )

    fold_clients = client_manifest[
        (client_manifest["outer_fold"].astype(int) == OUTER_FOLD)
        & np.isclose(client_manifest["alpha"].astype(float), ALPHA)
    ].copy()

    client_ids = sorted(
        fold_clients["global_client_id"].astype(str).unique().tolist()
    )
    if len(client_ids) != CLIENTS_PER_FOLD:
        raise RuntimeError(
            f"Expected 28 clients, found {len(client_ids)}."
        )

    fold_assignments = assignments[
        assignments["global_client_id"].astype(str).isin(client_ids)
    ].copy()

    if outer_test_subject in set(
        fold_assignments["subject_id"].astype(int)
    ):
        raise RuntimeError("Outer test subject leaked into training.")

    retained_rows = np.sort(
        fold_assignments["row_index"].astype(np.int64).unique()
    )

    config = {
        "status": "FROZEN_PAIRED_SMOKE_CONFIGURATION",
        "fl_protocol_path": str(protocol_path),
        "fl_protocol_sha256": sha256_file(protocol_path),
        "condition_id": int(condition_row["condition_id"]),
        "outer_fold": OUTER_FOLD,
        "outer_test_subject": outer_test_subject,
        "alpha": ALPHA,
        "scenario": SCENARIO,
        "fl_seed": FL_SEED,
        "methods": list(METHODS),
        "rounds": SMOKE_ROUNDS,
        "clients_per_round": CLIENTS_PER_ROUND,
        "fedprox_mu": FEDPROX_MU,
        "trim_k_each_tail": TRIM_K,
        "shared_model_seed": model_seed,
        "shared_random_schedule_seed": random_schedule_seed,
        "outer_test_used_for_training_selection_or_normalization": False,
    }

    (output_root / "PAIRED_SMOKE_CONFIG.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    device = base.select_device()

    print("=== PAMAP2 paired method smoke tests ===")
    print(f"Condition:         outer={OUTER_FOLD}, alpha={ALPHA}, clean")
    print(f"Methods:           {', '.join(METHODS)}")
    print(f"FL seed:           {FL_SEED}")
    print(f"Rounds per method: {SMOKE_ROUNDS}")
    print(f"Clients:           {len(client_ids)}")
    print(f"Clients per round: {CLIENTS_PER_ROUND}")
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
            energy_profiles["global_client_id"].astype(str).isin(
                client_ids
            )
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

    print(f"[OK] Retained training windows: {len(retained_rows)}")
    print(
        f"[OK] Client size range: "
        f"{min(client_sizes.values())}–{max(client_sizes.values())}"
    )
    print(f"[OK] Outer-test windows: {len(y_test)}")
    print()

    method_learning: dict[str, pd.DataFrame] = {}
    method_selection: dict[str, pd.DataFrame] = {}
    method_final_energy: dict[str, dict[str, float]] = {}

    for method in METHODS:
        print("=" * 78)
        print(f"METHOD: {method}")
        print("=" * 78)

        method_root = output_root / method
        method_root.mkdir(parents=True, exist_ok=True)

        base.set_seed(model_seed)
        global_model = base.LightweightCNN1D(input_channels=6).to(device)

        if base.count_trainable_parameters(global_model) != 77004:
            raise RuntimeError("Unexpected model parameter count.")

        global_state = base.cpu_state_dict(global_model)
        residual_energy = dict(initial_energy)

        learning_rows: list[dict[str, object]] = []
        selection_rows: list[dict[str, object]] = []

        initial_metrics, _, _ = base.evaluate_model(
            global_model,
            test_loader,
            nn.CrossEntropyLoss(),
            device,
        )

        learning_rows.append(
            {
                "method": method,
                "round": 0,
                "test_loss": initial_metrics.loss,
                "test_accuracy": initial_metrics.accuracy,
                "test_balanced_accuracy": initial_metrics.balanced_accuracy,
                "test_macro_f1": initial_metrics.macro_f1,
                "active_clients": CLIENTS_PER_FOLD,
                "mean_residual_energy": float(
                    np.mean(list(residual_energy.values()))
                ),
                "min_residual_energy": float(
                    np.min(list(residual_energy.values()))
                ),
            }
        )

        print(
            f"Round 0: Macro-F1={initial_metrics.macro_f1:.4f}; "
            f"BalAcc={initial_metrics.balanced_accuracy:.4f}; "
            f"Acc={initial_metrics.accuracy:.4f}"
        )

        for round_index in range(1, SMOKE_ROUNDS + 1):
            round_start = time.perf_counter()

            active_at_start = [
                client_id
                for client_id in client_ids
                if residual_energy[client_id] >= base.CRITICAL_ENERGY
            ]
            for client_id in active_at_start:
                residual_energy[client_id] = max(
                    0.0,
                    residual_energy[client_id] - base.STANDBY_COST,
                )

            eligible = [
                client_id
                for client_id in client_ids
                if residual_energy[client_id] >= base.CRITICAL_ENERGY
            ]

            permutation_rng = np.random.default_rng(
                base.deterministic_seed(
                    random_schedule_seed,
                    "round_permutation",
                    round_index,
                )
            )
            full_permutation = permutation_rng.permutation(
                np.asarray(client_ids, dtype=object)
            ).tolist()

            eligible_set = set(eligible)
            selected = [
                client_id
                for client_id in full_permutation
                if client_id in eligible_set
            ][:CLIENTS_PER_ROUND]

            if len(selected) != CLIENTS_PER_ROUND:
                raise RuntimeError("Could not select 8 eligible clients.")

            local_results = []

            for selection_rank, client_id in enumerate(selected, start=1):
                x_client, y_client = client_tensors[client_id]

                local_seed = base.deterministic_seed(
                    BASE_LOCAL_SEED,
                    OUTER_FOLD,
                    FL_SEED,
                    round_index,
                    client_id,
                )

                result = train_one_client_method(
                    base=base,
                    method=method,
                    global_state=global_state,
                    x_client=x_client,
                    y_client=y_client,
                    client_id=client_id,
                    local_seed=local_seed,
                    device=device,
                )
                local_results.append(result)

                participation_cost = (
                    base.COMMUNICATION_COST
                    + base.COMPUTE_COEFFICIENT
                    * compute_factor[client_id]
                    * (
                        client_sizes[client_id]
                        / median_client_windows
                    )
                )

                residual_energy[client_id] = max(
                    0.0,
                    residual_energy[client_id] - participation_cost,
                )

                selection_rows.append(
                    {
                        "method": method,
                        "round": round_index,
                        "selection_rank": selection_rank,
                        "global_client_id": client_id,
                        "client_windows": client_sizes[client_id],
                        "local_seed": local_seed,
                        "local_train_loss": result.train_loss,
                        "local_train_macro_f1": result.train_macro_f1,
                        "local_wall_seconds": result.wall_seconds,
                        "participation_cost": participation_cost,
                        "residual_energy_after_round": residual_energy[client_id],
                    }
                )

            if method == "fedprox":
                global_state = base.fedavg_weighted_state(
                    global_state=global_state,
                    local_results=local_results,
                )
            elif method == "random_trimmed_mean":
                global_state = trimmed_mean_state(
                    global_state=global_state,
                    local_results=local_results,
                )
            else:
                raise RuntimeError(f"Unexpected method: {method}")

            global_model.load_state_dict(global_state)
            global_model.to(device)

            metrics, _, _ = base.evaluate_model(
                global_model,
                test_loader,
                nn.CrossEntropyLoss(),
                device,
            )

            active_after = sum(
                1
                for client_id in client_ids
                if residual_energy[client_id] >= base.CRITICAL_ENERGY
            )

            base.synchronize(device)
            round_wall_seconds = time.perf_counter() - round_start

            learning_rows.append(
                {
                    "method": method,
                    "round": round_index,
                    "test_loss": metrics.loss,
                    "test_accuracy": metrics.accuracy,
                    "test_balanced_accuracy": metrics.balanced_accuracy,
                    "test_macro_f1": metrics.macro_f1,
                    "active_clients": active_after,
                    "mean_residual_energy": float(
                        np.mean(list(residual_energy.values()))
                    ),
                    "min_residual_energy": float(
                        np.min(list(residual_energy.values()))
                    ),
                }
            )

            print(
                f"Round {round_index}/{SMOKE_ROUNDS}: "
                f"Macro-F1={metrics.macro_f1:.4f}; "
                f"BalAcc={metrics.balanced_accuracy:.4f}; "
                f"Acc={metrics.accuracy:.4f}; "
                f"active={active_after}; "
                f"wall={round_wall_seconds:.1f}s"
            )

        learning_df = pd.DataFrame(learning_rows)
        selection_df = pd.DataFrame(selection_rows)

        learning_df.to_csv(method_root / "learning_curve.csv", index=False)
        selection_df.to_csv(
            method_root / "selected_client_log.csv",
            index=False,
        )

        torch.save(
            {
                "model_state_dict": global_state,
                "method": method,
                "rounds_completed": SMOKE_ROUNDS,
                "final_metrics": learning_rows[-1],
            },
            method_root / "final_global_model.pt",
        )

        method_learning[method] = learning_df
        method_selection[method] = selection_df
        method_final_energy[method] = dict(residual_energy)
        print()

    schedule_rows: list[dict[str, object]] = []
    left = method_selection[METHODS[0]]
    right = method_selection[METHODS[1]]

    for round_index in range(1, SMOKE_ROUNDS + 1):
        left_ids = left[
            left["round"].astype(int) == round_index
        ].sort_values("selection_rank")["global_client_id"].astype(str).tolist()

        right_ids = right[
            right["round"].astype(int) == round_index
        ].sort_values("selection_rank")["global_client_id"].astype(str).tolist()

        identical = left_ids == right_ids
        schedule_rows.append(
            {
                "round": round_index,
                "fedprox_selected": "|".join(left_ids),
                "trimmed_selected": "|".join(right_ids),
                "identical": identical,
            }
        )

        if not identical:
            raise RuntimeError(
                f"Matched schedule failed at round {round_index}."
            )

    schedule_df = pd.DataFrame(schedule_rows)
    schedule_df.to_csv(
        output_root / "matched_schedule_check.csv",
        index=False,
    )

    max_energy_difference = max(
        abs(
            method_final_energy[METHODS[0]][client_id]
            - method_final_energy[METHODS[1]][client_id]
        )
        for client_id in client_ids
    )

    if max_energy_difference > 1e-12:
        raise RuntimeError(
            "Matched energy evolution failed: "
            f"max diff={max_energy_difference}"
        )

    combined_learning = pd.concat(
        [method_learning[method] for method in METHODS],
        ignore_index=True,
    )
    combined_learning.to_csv(
        output_root / "combined_learning_curve.csv",
        index=False,
    )

    final_rows = (
        combined_learning[
            combined_learning["round"].astype(int) == SMOKE_ROUNDS
        ]
        .copy()
        .sort_values("method")
    )
    final_rows.to_csv(
        output_root / "final_method_results.csv",
        index=False,
    )

    if not np.isfinite(
        combined_learning[
            [
                "test_loss",
                "test_accuracy",
                "test_balanced_accuracy",
                "test_macro_f1",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("Non-finite smoke metrics.")

    report_lines = [
        "PAMAP2 FEDPROX + TRIMMED-MEAN PAIRED SMOKE TEST",
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
        f"Rounds per method: {SMOKE_ROUNDS}",
        "",
        "METHODS",
        "-" * 78,
        f"FedProx: proximal mu={FEDPROX_MU}",
        (
            "Random + Trimmed Mean: "
            f"coordinate-wise k={TRIM_K} removed from each tail"
        ),
        "",
        "MATCHED CONTROLS",
        "-" * 78,
        "Same initial global model seed.",
        "Same client permutation schedule.",
        "Same 8 selected clients in every paired round.",
        "Same local shuffle seeds for the same client-round pair.",
        "Same initial energy profiles and participation costs.",
        (
            "Maximum final per-client energy difference: "
            f"{max_energy_difference:.3e}"
        ),
        "",
        "DATA AND LEAKAGE",
        "-" * 78,
        f"Retained training windows: {len(retained_rows)}",
        f"Outer-test windows: {len(y_test)}",
        "Z-score fitted only on retained outer-training client windows.",
        "Outer test did not contribute to training, selection, or normalization.",
        "",
        "RESULTS",
        "-" * 78,
    ]

    for row in final_rows.itertuples(index=False):
        report_lines.append(
            f"{row.method}: "
            f"Round {SMOKE_ROUNDS} Macro-F1={row.test_macro_f1:.4f}; "
            f"BalAcc={row.test_balanced_accuracy:.4f}; "
            f"Acc={row.test_accuracy:.4f}; "
            f"active={int(row.active_clients)}"
        )

    report_lines.extend(
        [
            "",
            "FILES",
            "-" * 78,
            "PAIRED_SMOKE_CONFIG.json",
            "PAIRED_SMOKE_REPORT.txt",
            "combined_learning_curve.csv",
            "final_method_results.csv",
            "matched_schedule_check.csv",
            "normalization_mean.npy",
            "normalization_std.npy",
            "fedprox/learning_curve.csv",
            "fedprox/selected_client_log.csv",
            "fedprox/final_global_model.pt",
            "random_trimmed_mean/learning_curve.csv",
            "random_trimmed_mean/selected_client_log.csv",
            "random_trimmed_mean/final_global_model.pt",
        ]
    )

    report_path = output_root / "PAIRED_SMOKE_REPORT.txt"
    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("=== Paired method smoke tests completed successfully ===")
    print(f"Report: {report_path}")
    print()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nPaired smoke test interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
