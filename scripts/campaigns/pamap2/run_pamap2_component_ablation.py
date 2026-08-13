from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


RUNNER_ID = "PAMAP2_FROZEN_COMPONENT_ABLATION_RUNNER_V1"
TOTAL_ROUNDS = 100
CLIENTS_PER_FOLD = 28
CLIENTS_PER_ROUND = 8
NUM_CLASSES = 12
LOCAL_BATCH_SIZE = 64
ATTACK_START_ROUND = 20
SIGN_FLIP_SCALE = 5.0
CRITICAL_ENERGY = 0.10
STANDBY_COST = 0.0005
COMMUNICATION_COST = 0.004
COMPUTE_COEFFICIENT = 0.016
NETWORK_STOP_ACTIVE_FRACTION = 0.5
BASE_LOCAL_SEED = 20260706
EVALUATION_ROUNDS = tuple(range(0, 101, 5))
EXPECTED_METHOD_FAMILIES = (
    "tea_fl",
    "arl_fl",
)
EXPECTED_VARIANTS = (
    "tea_trust_only",
    "tea_energy_only",
    "arl_no_energy",
    "arl_no_pressure",
)
ABLATION_RUNS = 240
EXPECTED_SCENARIOS = (
    "clean",
    "signflip_mu0p2",
    "signflip_mu0p4",
    "labelflip_mu0p2",
    "labelflip_mu0p4",
)
EPS = 1e-12


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


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


def load_module(path: Path, module_name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def runner_paths(project_root: Path) -> dict[str, Path]:
    freeze_root = (
        project_root
        / "outputs"
        / "protocols"
        / "pamap2_source_interface_freeze_v1r2"
    )
    source_root = freeze_root / "frozen_source_provenance"
    input_root = freeze_root / "frozen_input_provenance"
    partition_root = (
        project_root
        / "outputs"
        / "protocols"
        / "pamap2_fl_partitions_v1"
    )
    processed_root = (
        project_root
        / "data"
        / "processed"
        / "pamap2"
        / "protocol_v1_w256_s128"
    )
    campaign_root = (
        project_root
        / "outputs"
        / "federated"
        / "pamap2"
        / "ablation_campaign_v1"
    )
    ablation_authorization_root = (
        project_root
        / "outputs"
        / "federated"
        / "pamap2"
        / "ablation_campaign_authorization_v1"
    )
    return {
        "freeze_root": freeze_root,
        "source_root": source_root,
        "input_root": input_root,
        "partition_root": partition_root,
        "processed_root": processed_root,
        "campaign_root": campaign_root,
        "gate117_binding": freeze_root / "GATE117R2_FINAL_BINDING.json",
        "source_manifest": freeze_root / "SOURCE_MANIFEST_SHA256.csv",
        "input_manifest": freeze_root / "FROZEN_INPUT_MANIFEST_SHA256.csv",
        "source_audit": freeze_root / "PAMAP2_SOURCE_INTERFACE_AUDIT.json",
        "campaign_matrix": (
            ablation_authorization_root
            / "PAMAP2_ABLATION_CAMPAIGN_MATRIX_240.csv"
        ),
        "conditions": input_root / "matched_condition_manifest.csv",
        "energy": input_root / "client_energy_profile.csv",
        "malicious": input_root / "malicious_client_manifest.csv",
        "client_manifest": input_root / "outer_fold_client_manifest.csv",
        "outer_manifest": input_root / "outer_fold_manifest.csv",
        "protocol": input_root / "FL_EXPERIMENTAL_PROTOCOL_V1.json",
    }


def verify_csv_manifest(
    manifest_path: Path,
    content_root: Path,
) -> tuple[int, list[dict[str, str]]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        path = content_root / row["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        expected = str(row["sha256"]).upper()
        if observed != expected:
            raise RuntimeError(
                f"Manifest mismatch for {path}: {observed} != {expected}"
            )
    return len(rows), rows


def verify_runner_binding(
    runner_file: Path,
    binding_path: Path,
) -> dict[str, Any]:
    if not binding_path.is_file():
        raise FileNotFoundError(binding_path)
    binding = json.loads(binding_path.read_text(encoding="utf-8-sig"))
    if binding.get("runner_id") != RUNNER_ID:
        raise RuntimeError("Unexpected runner ID in binding.")
    observed_runner_sha = sha256_file(runner_file)
    if binding.get("runner_sha256") != observed_runner_sha:
        raise RuntimeError("Runner SHA256 does not match binding.")
    final_field = "runner_build_binding_sha256"
    expected = binding.get(final_field)
    if not isinstance(expected, str):
        raise RuntimeError("Runner binding has no final binding SHA256.")
    core = dict(binding)
    core.pop(final_field)
    if canonical_sha256(core) != expected:
        raise RuntimeError("Runner build binding canonical replay failed.")
    return binding


def configure_frozen_modules(
    modules: dict[str, Any],
    *,
    outer_fold: int,
    fl_seed: int,
) -> None:
    base = modules["base"]
    prox = modules["prox"]
    fedle = modules["fedle"]
    tea = modules["tea"]
    arl = modules["arl"]

    for module in (base, prox, fedle, tea, arl):
        if hasattr(module, "OUTER_FOLD"):
            module.OUTER_FOLD = int(outer_fold)
        if hasattr(module, "FL_SEED"):
            module.FL_SEED = int(fl_seed)
        if hasattr(module, "CLIENTS_PER_FOLD"):
            module.CLIENTS_PER_FOLD = CLIENTS_PER_FOLD
        if hasattr(module, "CLIENTS_PER_ROUND"):
            module.CLIENTS_PER_ROUND = CLIENTS_PER_ROUND
        if hasattr(module, "CRITICAL_ENERGY"):
            module.CRITICAL_ENERGY = CRITICAL_ENERGY
        if hasattr(module, "COMMUNICATION_COST"):
            module.COMMUNICATION_COST = COMMUNICATION_COST
        if hasattr(module, "BASE_LOCAL_SEED"):
            module.BASE_LOCAL_SEED = BASE_LOCAL_SEED

    if hasattr(fedle, "BASE_FEDLE_SEED"):
        fedle.BASE_FEDLE_SEED = BASE_LOCAL_SEED
    if hasattr(tea, "BASE_TEA_SEED"):
        tea.BASE_TEA_SEED = BASE_LOCAL_SEED
    if hasattr(arl, "BASE_ARL_SEED"):
        arl.BASE_ARL_SEED = BASE_LOCAL_SEED


def load_frozen_modules(project_root: Path) -> dict[str, Any]:
    paths = runner_paths(project_root)
    source_root = paths["source_root"]
    modules = {
        "base": load_module(
            source_root / "16_smoke_pamap2_federated_engine.py",
            "pamap2_frozen_base_v1",
        ),
        "prox": load_module(
            source_root / "18_smoke_pamap2_fedprox_trimmed_mean_fixed.py",
            "pamap2_frozen_prox_trim_v1",
        ),
        "fedle": load_module(
            source_root / "19b_smoke_pamap2_fedle_adapted_delta.py",
            "pamap2_frozen_fedle_v1",
        ),
        "tea": load_module(
            source_root / "20_smoke_pamap2_tea_fl.py",
            "pamap2_frozen_tea_v1",
        ),
        "arl": load_module(
            source_root / "21_smoke_pamap2_arl_fl.py",
            "pamap2_frozen_arl_v1",
        ),
    }
    configure_frozen_modules(modules, outer_fold=1, fl_seed=123)
    return modules


def make_local_result(
    base,
    *,
    client_id: str,
    windows: int,
    state_dict: dict[str, torch.Tensor],
    train_loss: float = 0.0,
    train_macro_f1: float = 0.0,
    wall_seconds: float = 0.0,
):
    return base.LocalTrainResult(
        client_id=client_id,
        windows=int(windows),
        train_loss=float(train_loss),
        train_macro_f1=float(train_macro_f1),
        wall_seconds=float(wall_seconds),
        state_dict={
            key: value.detach().cpu().clone()
            for key, value in state_dict.items()
        },
    )


def sign_flip_result(
    base,
    *,
    global_state: dict[str, torch.Tensor],
    honest_result,
    scale: float = SIGN_FLIP_SCALE,
):
    attacked_state: dict[str, torch.Tensor] = {}
    for key, global_tensor in global_state.items():
        local_tensor = honest_result.state_dict[key]
        if torch.is_floating_point(global_tensor):
            global64 = global_tensor.to(dtype=torch.float64)
            local64 = local_tensor.to(dtype=torch.float64)
            attacked = global64 - float(scale) * (local64 - global64)
            attacked_state[key] = attacked.to(dtype=global_tensor.dtype)
        else:
            attacked_state[key] = local_tensor.detach().cpu().clone()
    return make_local_result(
        base,
        client_id=str(honest_result.client_id),
        windows=int(honest_result.windows),
        train_loss=float(honest_result.train_loss),
        train_macro_f1=float(honest_result.train_macro_f1),
        wall_seconds=float(honest_result.wall_seconds),
        state_dict=attacked_state,
    )


def label_flip(y: torch.Tensor) -> torch.Tensor:
    return torch.remainder(y.to(dtype=torch.int64) + 1, NUM_CLASSES)


def contract_check(
    project_root: Path,
    runner_file: Path,
    binding_path: Path,
) -> dict[str, Any]:
    paths = runner_paths(project_root)
    required = [
        paths["gate117_binding"],
        paths["source_manifest"],
        paths["input_manifest"],
        paths["source_audit"],
        paths["campaign_matrix"],
        paths["conditions"],
        paths["energy"],
        paths["malicious"],
        paths["client_manifest"],
        paths["outer_manifest"],
        paths["protocol"],
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    gate117 = json.loads(
        paths["gate117_binding"].read_text(encoding="utf-8-sig")
    )
    if gate117.get("status") != "PASS":
        raise RuntimeError("Gate-117R2 is not PASS.")
    if gate117.get("gate117r2_final_binding_sha256") != (
        "9E6F793BBA851B46B36BF5063224551EE02AC0D6BFFAE7ED51E5A98DEFA1CBE3"
    ):
        raise RuntimeError("Unexpected Gate-117R2 binding.")

    binding = verify_runner_binding(runner_file, binding_path)
    source_count, _ = verify_csv_manifest(
        paths["source_manifest"],
        paths["source_root"],
    )
    input_count, _ = verify_csv_manifest(
        paths["input_manifest"],
        paths["input_root"],
    )
    if source_count != 10:
        raise RuntimeError(f"Expected 10 frozen sources, found {source_count}.")
    if input_count != 12:
        raise RuntimeError(f"Expected 12 frozen inputs, found {input_count}.")

    campaign = pd.read_csv(paths["campaign_matrix"])
    if len(campaign) != ABLATION_RUNS:
        raise RuntimeError(
            f"Ablation campaign matrix does not contain {ABLATION_RUNS} rows."
        )
    if sorted(campaign["ablation_run_id"].astype(int).tolist()) != list(
        range(1, ABLATION_RUNS + 1)
    ):
        raise RuntimeError(
            "Ablation campaign run IDs are not exactly 1..240."
        )
    if set(campaign["method_family"].astype(str)) != set(
        EXPECTED_METHOD_FAMILIES
    ):
        raise RuntimeError("Ablation method families are unexpected.")
    if set(campaign["variant"].astype(str)) != set(EXPECTED_VARIANTS):
        raise RuntimeError("Ablation variants do not match the frozen design.")
    if not set(campaign["scenario"].astype(str)).issubset(
        set(EXPECTED_SCENARIOS)
    ):
        raise RuntimeError("Ablation scenarios are not frozen scenarios.")

    modules = load_frozen_modules(project_root)
    base = modules["base"]
    prox = modules["prox"]
    fedle = modules["fedle"]
    tea = modules["tea"]
    arl = modules["arl"]

    required_callables = {
        "base": (
            "load_all_raw_scale_windows",
            "build_magnitude6",
            "fit_retained_client_zscore",
            "normalize_rows",
            "LightweightCNN1D",
            "train_one_client",
            "fedavg_weighted_state",
            "evaluate_model",
        ),
        "prox": ("train_one_client_method", "trimmed_mean_state"),
        "fedle": (
            "train_one_client",
            "weighted_fedavg_state",
            "run_preflight",
            "cluster_preflight_vectors",
            "select_fedle_clients",
        ),
        "tea": (
            "select_tea_clients",
            "update_trust_from_round_reference",
            "tea_weighted_aggregation",
        ),
        "arl": (
            "select_arl_clients",
            "compute_round_risk",
            "adaptive_robust_aggregation",
        ),
    }
    for module_name, names in required_callables.items():
        module = modules[module_name]
        missing = [name for name in names if not callable(getattr(module, name, None))]
        if missing:
            raise RuntimeError(
                f"{module_name} missing required callables: {missing}"
            )

    base.set_seed(123)
    model = base.LightweightCNN1D(input_channels=6)
    if base.count_trainable_parameters(model) != 77004:
        raise RuntimeError("Unexpected model parameter count.")
    global_state = base.cpu_state_dict(model)
    client_ids = [f"client_{index:02d}" for index in range(28)]
    local_results = []
    for index, client_id in enumerate(client_ids[:8]):
        state: dict[str, torch.Tensor] = {}
        for key, tensor in global_state.items():
            if torch.is_floating_point(tensor):
                state[key] = tensor + (index + 1) * 1e-4
            else:
                state[key] = tensor.clone()
        local_results.append(
            make_local_result(
                base,
                client_id=client_id,
                windows=128 + index,
                state_dict=state,
            )
        )

    fedavg_state = base.fedavg_weighted_state(
        global_state=global_state,
        local_results=local_results,
    )
    trimmed_state = prox.trimmed_mean_state(
        global_state=global_state,
        local_results=local_results,
    )
    if set(fedavg_state) != set(global_state):
        raise RuntimeError("FedAvg synthetic state keys mismatch.")
    if set(trimmed_state) != set(global_state):
        raise RuntimeError("Trimmed-mean synthetic state keys mismatch.")

    trust = {client_id: 0.5 for client_id in client_ids}
    trust_df, trust = tea.update_trust_from_round_reference(
        round_index=1,
        selected=client_ids[:8],
        local_results=local_results,
        trust=trust,
    )
    residual = {client_id: 0.9 for client_id in client_ids}
    tea_state, tea_aggregation_df = tea.tea_weighted_aggregation(
        global_state=global_state,
        local_results=local_results,
        trust=trust,
        residual_energy=residual,
    )
    if (
        len(trust_df) != 8
        or set(tea_state) != set(global_state)
        or len(tea_aggregation_df) != 8
    ):
        raise RuntimeError("TEA-FL synthetic contract failed.")

    risk_state = {client_id: 0.0 for client_id in client_ids}
    (
        risk_df,
        updated_risk,
        previous_update,
        pressure,
        current_updates,
    ) = arl.compute_round_risk(
        round_index=1,
        selected=client_ids[:8],
        global_state=global_state,
        local_results=local_results,
        previous_update={},
        risk_state=risk_state,
    )
    arl_state, aggregation_client_df, aggregation_round_df = (
        arl.adaptive_robust_aggregation(
            round_index=1,
            global_state=global_state,
            local_results=local_results,
            current_updates=current_updates,
            global_pressure=pressure,
        )
    )
    if (
        len(risk_df) != 8
        or len(updated_risk) != 28
        or len(previous_update) != 8
        or set(arl_state) != set(global_state)
        or len(aggregation_client_df) != 8
        or len(aggregation_round_df) != 1
    ):
        raise RuntimeError("ARL-FL synthetic contract failed.")

    initial_energy = {client_id: 0.9 for client_id in client_ids}
    cluster_assignment = {
        client_id: index % 8
        for index, client_id in enumerate(client_ids)
    }
    cluster_sizes = dict(
        pd.Series(list(cluster_assignment.values())).value_counts().sort_index()
    )
    selected_fedle, fedle_df = fedle.select_fedle_clients(
        round_index=1,
        client_ids=client_ids,
        residual_energy=residual,
        initial_energy=initial_energy,
        cluster_assignment=cluster_assignment,
        cluster_sizes={int(k): int(v) for k, v in cluster_sizes.items()},
    )
    selected_tea, tea_select_df = tea.select_tea_clients(
        round_index=1,
        client_ids=client_ids,
        trust=trust,
        residual_energy=residual,
    )
    predicted_costs = {client_id: 0.02 for client_id in client_ids}
    last_selected = {client_id: 0 for client_id in client_ids}
    selected_arl, arl_select_df = arl.select_arl_clients(
        round_index=1,
        previous_global_pressure=0.0,
        client_ids=client_ids,
        risk_state=risk_state,
        residual_energy=residual,
        predicted_costs=predicted_costs,
        last_selected_round=last_selected,
    )
    if not all(
        len(selected) == 8 and len(set(selected)) == 8
        for selected in (selected_fedle, selected_tea, selected_arl)
    ):
        raise RuntimeError("Synthetic selector cardinality failed.")
    if len(fedle_df) < 8 or len(tea_select_df) < 8 or len(arl_select_df) < 8:
        raise RuntimeError("Synthetic selector audits are incomplete.")

    attacked = sign_flip_result(
        base,
        global_state=global_state,
        honest_result=local_results[0],
    )
    first_float_key = next(
        key for key, value in global_state.items() if torch.is_floating_point(value)
    )
    honest_delta = (
        local_results[0].state_dict[first_float_key].to(torch.float64)
        - global_state[first_float_key].to(torch.float64)
    )
    attacked_delta = (
        attacked.state_dict[first_float_key].to(torch.float64)
        - global_state[first_float_key].to(torch.float64)
    )
    if not torch.allclose(
        attacked_delta,
        -SIGN_FLIP_SCALE * honest_delta,
        atol=1e-6,
        rtol=1e-5,
    ):
        raise RuntimeError("Synthetic sign-flip contract failed.")
    labels = torch.tensor([0, 1, 11], dtype=torch.int64)
    if label_flip(labels).tolist() != [1, 2, 0]:
        raise RuntimeError("Synthetic label-flip contract failed.")

    del model
    report = {
        "status": "PAMAP2_SCIENTIFIC_RUNNER_CONTRACT_CHECK_PASS",
        "runner_id": RUNNER_ID,
        "runner_sha256": binding["runner_sha256"],
        "runner_build_binding_sha256": binding[
            "runner_build_binding_sha256"
        ],
        "campaign_rows_verified": ABLATION_RUNS,
        "source_files_verified": source_count,
        "input_files_verified": input_count,
        "method_families": list(EXPECTED_METHOD_FAMILIES),
        "variants": list(EXPECTED_VARIANTS),
        "scenarios": list(EXPECTED_SCENARIOS),
        "model_parameter_count": 77004,
        "synthetic_aggregators_verified": [
            "fedavg",
            "random_trimmed_mean",
            "tea_fl",
            "arl_fl",
        ],
        "synthetic_selectors_verified": [
            "fedle_adapted",
            "tea_fl",
            "arl_fl",
        ],
        "synthetic_attacks_verified": [
            "sign_flip",
            "label_flip",
        ],
        "fedprox_training_signature_verified": True,
        "scientific_training_started": False,
        "scientific_optimizer_steps_executed": 0,
        "pamap2_600_run_campaign_started": False,
    }
    return report


def read_single_row(frame: pd.DataFrame, mask, label: str) -> pd.Series:
    selected = frame[mask]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one {label} row, found {len(selected)}.")
    return selected.iloc[0]


def predicted_participation_cost(
    client_id: str,
    *,
    client_sizes: dict[str, int],
    compute_factor: dict[str, float],
    median_client_windows: float,
) -> float:
    return (
        COMMUNICATION_COST
        + COMPUTE_COEFFICIENT
        * compute_factor[client_id]
        * (client_sizes[client_id] / median_client_windows)
    )


def random_matched_selection(
    *,
    round_index: int,
    random_schedule_seed: int,
    client_ids: list[str],
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
    rng = np.random.default_rng(
        deterministic_seed(
            random_schedule_seed,
            "round_permutation",
            round_index,
        )
    )
    permutation = rng.permutation(
        np.asarray(client_ids, dtype=object)
    ).tolist()
    eligible_set = set(eligible)
    selected = [
        str(client_id)
        for client_id in permutation
        if str(client_id) in eligible_set
    ][:CLIENTS_PER_ROUND]
    selected_set = set(selected)
    audit = pd.DataFrame(
        [
            {
                "round": round_index,
                "global_client_id": client_id,
                "eligible": client_id in eligible_set,
                "selected": client_id in selected_set,
                "selection_mode": "matched_random",
            }
            for client_id in client_ids
        ]
    )
    return selected, audit


def optimizer_steps_for_client(windows: int, local_epochs: int = 1) -> int:
    return int(local_epochs) * int(math.ceil(int(windows) / LOCAL_BATCH_SIZE))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def run_scientific(
    project_root: Path,
    runner_file: Path,
    binding_path: Path,
    run_id: int,
    authorization_path: Path,
) -> dict[str, Any]:
    paths = runner_paths(project_root)
    binding = verify_runner_binding(runner_file, binding_path)

    if not authorization_path.is_file():
        raise FileNotFoundError(authorization_path)
    authorization = json.loads(
        authorization_path.read_text(encoding="utf-8-sig")
    )
    if authorization.get("status") != "AUTHORIZED":
        raise RuntimeError("Authorization status is not AUTHORIZED.")
    if int(authorization.get("run_id", -1)) != int(run_id):
        raise RuntimeError("Authorization run ID mismatch.")
    if authorization.get("runner_sha256") != binding["runner_sha256"]:
        raise RuntimeError("Authorization runner SHA mismatch.")
    if authorization.get("runner_build_binding_sha256") != binding[
        "runner_build_binding_sha256"
    ]:
        raise RuntimeError("Authorization build-binding mismatch.")
    auth_field = "authorization_binding_sha256"
    auth_expected = authorization.get(auth_field)
    if not isinstance(auth_expected, str):
        raise RuntimeError("Authorization has no canonical binding.")
    auth_core = dict(authorization)
    auth_core.pop(auth_field)
    if canonical_sha256(auth_core) != auth_expected:
        raise RuntimeError("Authorization canonical replay failed.")

    campaign = pd.read_csv(paths["campaign_matrix"])
    row = read_single_row(
        campaign,
        campaign["ablation_run_id"].astype(int) == int(run_id),
        "ablation campaign",
    )
    outer_fold = int(row["outer_fold"])
    alpha = float(row["alpha"])
    scenario = str(row["scenario"])
    method = str(row["method_family"])
    variant = str(row["variant"])
    fl_seed = int(row["fl_seed"])
    condition_id = int(row["source_condition_id"])
    reference_run_id = int(row["reference_run_id"])
    if method not in EXPECTED_METHOD_FAMILIES:
        raise RuntimeError(f"Unexpected ablation method family: {method}")
    if variant not in EXPECTED_VARIANTS:
        raise RuntimeError(f"Unexpected ablation variant: {variant}")
    if variant.startswith("tea_") and method != "tea_fl":
        raise RuntimeError("TEA ablation variant has non-TEA method family.")
    if variant.startswith("arl_") and method != "arl_fl":
        raise RuntimeError("ARL ablation variant has non-ARL method family.")

    run_name = (
        f"ablation_run_{run_id:03d}"
        f"__fold{outer_fold}"
        f"__{alpha_label(alpha)}"
        f"__{scenario}"
        f"__{variant}"
        f"__seed{fl_seed}"
    )
    output_root = paths["campaign_root"] / run_name
    if output_root.exists():
        complete_path = output_root / "RUN_COMPLETE.json"
        if complete_path.is_file():
            existing = json.loads(
                complete_path.read_text(encoding="utf-8-sig")
            )
            if int(existing.get("run_contract", {}).get("run_id", -1)) == run_id:
                return {
                    "status": "SCIENTIFIC_RUN_ALREADY_COMPLETE",
                    "run_id": run_id,
                    "output_root": str(output_root),
                }
        raise RuntimeError(
            f"Refusing existing partial or mismatched run directory: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=False)

    conditions = pd.read_csv(paths["conditions"])
    condition = read_single_row(
        conditions,
        (
            (conditions["condition_id"].astype(int) == condition_id)
            & (conditions["outer_fold"].astype(int) == outer_fold)
            & np.isclose(conditions["alpha"].astype(float), alpha)
            & (conditions["scenario"].astype(str) == scenario)
            & (conditions["fl_seed"].astype(int) == fl_seed)
        ),
        "matched condition",
    )
    model_seed = int(condition["model_seed"])
    random_schedule_seed = int(condition["random_schedule_seed"])
    attack = str(condition["attack"])
    malicious_count = int(condition["malicious_count"])

    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8-sig"))
    if protocol.get("status") != "FROZEN_BEFORE_ANY_FL_TRAINING":
        raise RuntimeError("Unexpected frozen protocol status.")
    if int(protocol["training"]["rounds"]) != TOTAL_ROUNDS:
        raise RuntimeError("Unexpected round count.")
    if int(protocol["training"]["clients_per_round"]) != CLIENTS_PER_ROUND:
        raise RuntimeError("Unexpected clients-per-round.")

    modules = load_frozen_modules(project_root)
    configure_frozen_modules(
        modules,
        outer_fold=outer_fold,
        fl_seed=fl_seed,
    )
    base = modules["base"]
    prox = modules["prox"]
    fedle = modules["fedle"]
    tea = modules["tea"]
    arl = modules["arl"]

    client_manifest = pd.read_csv(paths["client_manifest"])
    fold_clients = client_manifest[
        (client_manifest["outer_fold"].astype(int) == outer_fold)
        & np.isclose(client_manifest["alpha"].astype(float), alpha)
    ].copy()
    client_ids = sorted(
        fold_clients["global_client_id"].astype(str).unique().tolist()
    )
    if len(client_ids) != CLIENTS_PER_FOLD:
        raise RuntimeError(
            f"Expected 28 clients, found {len(client_ids)}."
        )

    assignment_path = (
        paths["partition_root"]
        / f"master_assignments_{alpha_label(alpha)}.csv"
    )
    if not assignment_path.is_file():
        raise FileNotFoundError(assignment_path)
    assignments = pd.read_csv(assignment_path)
    fold_assignments = assignments[
        assignments["global_client_id"].astype(str).isin(client_ids)
    ].copy()

    outer_manifest = pd.read_csv(paths["outer_manifest"])
    outer_row = read_single_row(
        outer_manifest,
        outer_manifest["outer_fold"].astype(int) == outer_fold,
        "outer-fold",
    )
    outer_test_subject = parse_subject(outer_row["outer_test_subject"])
    if outer_test_subject in set(
        fold_assignments["subject_id"].astype(int)
    ):
        raise RuntimeError("Outer test subject leaked into training clients.")

    retained_rows = np.sort(
        fold_assignments["row_index"].astype(np.int64).unique()
    )
    dataset = base.load_all_raw_scale_windows(paths["processed_root"])
    x_magnitude6 = base.build_magnitude6(dataset.x_raw_full36)
    mean, std = base.fit_retained_client_zscore(
        x_magnitude6,
        retained_rows,
    )

    client_tensors: dict[
        str,
        tuple[torch.Tensor, torch.Tensor],
    ] = {}
    for client_id in client_ids:
        client_rows = np.sort(
            fold_assignments[
                fold_assignments["global_client_id"].astype(str)
                == client_id
            ]["row_index"].astype(np.int64).to_numpy()
        )
        x_client = base.normalize_rows(
            x_magnitude6,
            client_rows,
            mean,
            std,
        )
        y_client = torch.from_numpy(
            dataset.y[client_rows].astype(np.int64, copy=True)
        )
        client_tensors[client_id] = (x_client, y_client)

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

    energy_profiles = pd.read_csv(paths["energy"])
    energy = energy_profiles[
        (energy_profiles["outer_fold"].astype(int) == outer_fold)
        & (energy_profiles["fl_seed"].astype(int) == fl_seed)
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
        str(item.global_client_id): float(item.initial_energy)
        for item in energy.itertuples(index=False)
    }
    compute_factor = {
        str(item.global_client_id): float(item.compute_factor)
        for item in energy.itertuples(index=False)
    }
    residual_energy = dict(initial_energy)
    client_sizes = {
        client_id: int(client_tensors[client_id][1].shape[0])
        for client_id in client_ids
    }
    median_client_windows = float(
        np.median(list(client_sizes.values()))
    )
    predicted_costs = {
        client_id: predicted_participation_cost(
            client_id,
            client_sizes=client_sizes,
            compute_factor=compute_factor,
            median_client_windows=median_client_windows,
        )
        for client_id in client_ids
    }

    if malicious_count > 0:
        malicious_frame = pd.read_csv(paths["malicious"])
        malicious_rows = malicious_frame[
            (malicious_frame["outer_fold"].astype(int) == outer_fold)
            & (malicious_frame["fl_seed"].astype(int) == fl_seed)
            & (
                malicious_frame["malicious_count"].astype(int)
                == malicious_count
            )
        ]
        malicious_clients = set(
            malicious_rows["global_client_id"].astype(str).tolist()
        )
        if len(malicious_clients) != malicious_count:
            raise RuntimeError(
                f"Expected {malicious_count} malicious clients, "
                f"found {len(malicious_clients)}."
            )
    else:
        malicious_clients = set()

    device = base.select_device()
    base.set_seed(model_seed)
    global_model = base.LightweightCNN1D(input_channels=6).to(device)
    if base.count_trainable_parameters(global_model) != 77004:
        raise RuntimeError("Unexpected model parameter count.")
    global_state = base.cpu_state_dict(global_model)

    config = {
        "runner_id": RUNNER_ID,
        "runner_sha256": binding["runner_sha256"],
        "runner_build_binding_sha256": binding[
            "runner_build_binding_sha256"
        ],
        "authorization_binding_sha256": auth_expected,
        "run_id": run_id,
        "condition_id": condition_id,
        "outer_fold": outer_fold,
        "outer_test_subject": outer_test_subject,
        "alpha": alpha,
        "scenario": scenario,
        "attack": attack,
        "malicious_count": malicious_count,
        "method_family": method,
        "variant": variant,
        "reference_full_method_run_id": reference_run_id,
        "fl_seed": fl_seed,
        "model_seed": model_seed,
        "random_schedule_seed": random_schedule_seed,
        "total_rounds": TOTAL_ROUNDS,
        "clients_per_fold": CLIENTS_PER_FOLD,
        "clients_per_round": CLIENTS_PER_ROUND,
        "evaluation_rounds": list(EVALUATION_ROUNDS),
        "scientific_run": True,
    }
    write_json(output_root / "RUN_CONFIG.json", config)
    write_json(
        output_root / "RUN_STATE.json",
        {
            "status": "IN_PROGRESS",
            "completed_round": 0,
            "scientific_optimizer_steps_accounted": 0,
        },
    )
    np.save(output_root / "normalization_mean.npy", mean)
    np.save(output_root / "normalization_std.npy", std)

    evaluation_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    attack_rows: list[dict[str, Any]] = []
    method_audit_frames: list[pd.DataFrame] = []
    aggregation_frames: list[pd.DataFrame] = []

    trust = {client_id: 0.5 for client_id in client_ids}
    risk_state = {client_id: 0.0 for client_id in client_ids}
    previous_update: dict[str, torch.Tensor] = {}
    last_selected_round = {client_id: 0 for client_id in client_ids}
    previous_global_pressure = 0.0
    cluster_assignment: dict[str, int] = {}
    cluster_sizes: dict[int, int] = {}
    scientific_optimizer_steps = 0
    early_stop_round: int | None = None

    if method == "fedle_adapted":
        preflight_df, feature_matrix = fedle.run_preflight(
            client_ids=client_ids,
            client_tensors=client_tensors,
            global_state=global_state,
            residual_energy=residual_energy,
            client_sizes=client_sizes,
            compute_factor=compute_factor,
            median_client_windows=median_client_windows,
            device=device,
        )
        scientific_optimizer_steps += sum(
            optimizer_steps_for_client(client_sizes[client_id])
            for client_id in client_ids
        )
        cluster_df, similarity_matrix, silhouette = (
            fedle.cluster_preflight_vectors(
                client_ids=client_ids,
                feature_matrix=feature_matrix,
            )
        )
        cluster_assignment = {
            str(item.global_client_id): int(item.cluster_id)
            for item in cluster_df.itertuples(index=False)
        }
        cluster_sizes = {
            int(cluster_id): int(count)
            for cluster_id, count in (
                cluster_df.groupby("cluster_id")
                .size()
                .to_dict()
                .items()
            )
        }
        preflight_df.to_csv(
            output_root / "FEDLE_PREFLIGHT.csv",
            index=False,
        )
        cluster_df.to_csv(
            output_root / "FEDLE_CLUSTERS.csv",
            index=False,
        )
        np.save(
            output_root / "FEDLE_SIMILARITY_MATRIX.npy",
            similarity_matrix,
        )
        write_json(
            output_root / "FEDLE_PREFLIGHT_SUMMARY.json",
            {
                "silhouette_cosine": silhouette,
                "optimizer_steps_accounted": sum(
                    optimizer_steps_for_client(client_sizes[client_id])
                    for client_id in client_ids
                ),
            },
        )

    def append_evaluation(round_index: int) -> None:
        metrics, y_true_eval, y_pred_eval = base.evaluate_model(
            global_model,
            test_loader,
            device,
        )
        per_class = []
        from sklearn.metrics import f1_score
        class_f1 = f1_score(
            y_true_eval,
            y_pred_eval,
            labels=list(range(NUM_CLASSES)),
            average=None,
            zero_division=0,
        )
        for class_id, score in enumerate(class_f1):
            per_class.append(float(score))
        evaluation_rows.append(
            {
                "round": round_index,
                "test_loss": float(metrics.loss),
                "test_accuracy": float(metrics.accuracy),
                "test_balanced_accuracy": float(
                    metrics.balanced_accuracy
                ),
                "test_macro_f1": float(metrics.macro_f1),
                "per_class_f1_json": json.dumps(per_class),
                "active_clients": sum(
                    value >= CRITICAL_ENERGY
                    for value in residual_energy.values()
                ),
                "mean_residual_energy": float(
                    np.mean(list(residual_energy.values()))
                ),
                "min_residual_energy": float(
                    np.min(list(residual_energy.values()))
                ),
                "global_pressure": (
                    float(previous_global_pressure)
                    if method == "arl_fl"
                    else None
                ),
                "scientific_optimizer_steps_accounted": (
                    scientific_optimizer_steps
                ),
            }
        )

    append_evaluation(0)

    for round_index in range(1, TOTAL_ROUNDS + 1):
        round_start = time.perf_counter()
        active_at_start = [
            client_id
            for client_id in client_ids
            if residual_energy[client_id] >= CRITICAL_ENERGY
        ]
        if len(active_at_start) < math.ceil(
            CLIENTS_PER_FOLD * NETWORK_STOP_ACTIVE_FRACTION
        ):
            early_stop_round = round_index
            for evaluation_round in EVALUATION_ROUNDS:
                if (
                    evaluation_round >= round_index
                    and evaluation_round
                    not in {int(item["round"]) for item in evaluation_rows}
                ):
                    append_evaluation(evaluation_round)
            break

        for client_id in active_at_start:
            residual_energy[client_id] = max(
                0.0,
                residual_energy[client_id] - STANDBY_COST,
            )

        neutral_active_energy = {
            client_id: (
                1.0
                if residual_energy[client_id] >= CRITICAL_ENERGY
                else 0.0
            )
            for client_id in client_ids
        }
        neutral_trust = {client_id: 0.5 for client_id in client_ids}
        neutral_predicted_cost = float(
            np.median(
                [
                    predicted_costs[client_id]
                    for client_id in active_at_start
                ]
            )
        )
        neutral_predicted_costs = {
            client_id: neutral_predicted_cost
            for client_id in client_ids
        }

        if variant == "tea_trust_only":
            selected, selection_df = tea.select_tea_clients(
                round_index=round_index,
                client_ids=client_ids,
                trust=trust,
                residual_energy=neutral_active_energy,
            )
        elif variant == "tea_energy_only":
            selected, selection_df = tea.select_tea_clients(
                round_index=round_index,
                client_ids=client_ids,
                trust=neutral_trust,
                residual_energy=residual_energy,
            )
        elif variant == "arl_no_energy":
            selected, selection_df = arl.select_arl_clients(
                round_index=round_index,
                previous_global_pressure=previous_global_pressure,
                client_ids=client_ids,
                risk_state=risk_state,
                residual_energy=neutral_active_energy,
                predicted_costs=neutral_predicted_costs,
                last_selected_round=last_selected_round,
            )
        elif variant == "arl_no_pressure":
            selected, selection_df = arl.select_arl_clients(
                round_index=round_index,
                previous_global_pressure=0.0,
                client_ids=client_ids,
                risk_state=risk_state,
                residual_energy=residual_energy,
                predicted_costs=predicted_costs,
                last_selected_round=last_selected_round,
            )
        else:
            raise RuntimeError(f"Unexpected ablation variant: {variant}")

        if len(selected) != CLIENTS_PER_ROUND or len(set(selected)) != CLIENTS_PER_ROUND:
            raise RuntimeError(
                f"Round {round_index}: invalid selected-client cardinality."
            )
        selection_df = selection_df.copy()
        selection_df["method_family"] = method
        selection_df["variant"] = variant
        selection_df["scenario"] = scenario
        selection_df["is_malicious"] = (
            selection_df["global_client_id"].astype(str).isin(
                malicious_clients
            )
        )
        method_audit_frames.append(selection_df)

        local_results = []
        round_malicious_selected = 0
        round_attacks_applied = 0
        round_optimizer_steps = 0

        for rank, client_id in enumerate(selected, start=1):
            x_client, y_client_honest = client_tensors[client_id]
            is_malicious = client_id in malicious_clients
            attack_active = (
                round_index >= ATTACK_START_ROUND
                and is_malicious
                and attack != "none"
            )
            y_train = y_client_honest
            if attack_active and attack == "label_flip":
                y_train = label_flip(y_client_honest)

            local_seed = deterministic_seed(
                BASE_LOCAL_SEED,
                outer_fold,
                fl_seed,
                round_index,
                client_id,
            )
            honest_result = base.train_one_client(
                global_state=global_state,
                x_client=x_client,
                y_client=y_train,
                client_id=client_id,
                local_seed=local_seed,
                device=device,
            )

            transmitted_result = honest_result
            if attack_active and attack == "sign_flip":
                transmitted_result = sign_flip_result(
                    base,
                    global_state=global_state,
                    honest_result=honest_result,
                )
            if is_malicious:
                round_malicious_selected += 1
            if attack_active:
                round_attacks_applied += 1

            steps = optimizer_steps_for_client(
                int(honest_result.windows)
            )
            round_optimizer_steps += steps
            scientific_optimizer_steps += steps
            local_results.append(transmitted_result)
            selection_rows.append(
                {
                    "round": round_index,
                    "selection_rank": rank,
                    "global_client_id": client_id,
                    "is_malicious": is_malicious,
                    "attack_active": attack_active,
                    "attack_type": attack if attack_active else "none",
                    "client_windows": int(honest_result.windows),
                    "local_seed": local_seed,
                    "local_train_loss": float(
                        honest_result.train_loss
                    ),
                    "local_train_macro_f1": float(
                        honest_result.train_macro_f1
                    ),
                    "local_wall_seconds": float(
                        honest_result.wall_seconds
                    ),
                    "optimizer_steps_accounted": steps,
                }
            )
            attack_rows.append(
                {
                    "round": round_index,
                    "global_client_id": client_id,
                    "is_malicious": is_malicious,
                    "attack_active": attack_active,
                    "attack_type": attack if attack_active else "none",
                }
            )

        raw_current_global_pressure = previous_global_pressure
        current_global_pressure = previous_global_pressure
        if method == "tea_fl":
            trust_df, trust = tea.update_trust_from_round_reference(
                round_index=round_index,
                selected=selected,
                local_results=local_results,
                trust=trust,
            )
            trust_df = trust_df.copy()
            trust_df["audit_type"] = "trust_update"
            trust_df["variant"] = variant
            trust_df["trust_applied_to_decision"] = (
                variant == "tea_trust_only"
            )
            trust_df["is_malicious"] = (
                trust_df["global_client_id"].astype(str).isin(
                    malicious_clients
                )
            )
            method_audit_frames.append(trust_df)
        elif method == "arl_fl":
            (
                risk_df,
                risk_state,
                previous_update,
                raw_current_global_pressure,
                current_updates,
            ) = arl.compute_round_risk(
                round_index=round_index,
                selected=selected,
                global_state=global_state,
                local_results=local_results,
                previous_update=previous_update,
                risk_state=risk_state,
            )
            current_global_pressure = (
                0.0
                if variant == "arl_no_pressure"
                else float(raw_current_global_pressure)
            )
            risk_df = risk_df.copy()
            risk_df["audit_type"] = "risk_update"
            risk_df["variant"] = variant
            risk_df["raw_global_pressure"] = float(
                raw_current_global_pressure
            )
            risk_df["applied_global_pressure"] = float(
                current_global_pressure
            )
            risk_df["is_malicious"] = (
                risk_df["global_client_id"].astype(str).isin(
                    malicious_clients
                )
            )
            method_audit_frames.append(risk_df)

        for client_id in selected:
            residual_energy[client_id] = max(
                0.0,
                residual_energy[client_id] - predicted_costs[client_id],
            )
            if method == "arl_fl":
                last_selected_round[client_id] = round_index

        if variant == "tea_trust_only":
            global_state, tea_aggregation_df = tea.tea_weighted_aggregation(
                global_state=global_state,
                local_results=local_results,
                trust=trust,
                residual_energy=neutral_active_energy,
            )
            tea_aggregation_df = tea_aggregation_df.copy()
            tea_aggregation_df["effective_component_mode"] = (
                "trust_only_energy_neutral"
            )
            tea_aggregation_df["variant"] = variant
            tea_aggregation_df["audit_type"] = "tea_aggregation"
            tea_aggregation_df["is_malicious"] = (
                tea_aggregation_df["global_client_id"].astype(str).isin(
                    malicious_clients
                )
            )
            aggregation_frames.append(tea_aggregation_df)
        elif variant == "tea_energy_only":
            global_state, tea_aggregation_df = tea.tea_weighted_aggregation(
                global_state=global_state,
                local_results=local_results,
                trust=neutral_trust,
                residual_energy=residual_energy,
            )
            tea_aggregation_df = tea_aggregation_df.copy()
            tea_aggregation_df["effective_component_mode"] = (
                "energy_only_trust_neutral"
            )
            tea_aggregation_df["variant"] = variant
            tea_aggregation_df["audit_type"] = "tea_aggregation"
            tea_aggregation_df["is_malicious"] = (
                tea_aggregation_df["global_client_id"].astype(str).isin(
                    malicious_clients
                )
            )
            aggregation_frames.append(tea_aggregation_df)
        elif variant in {"arl_no_energy", "arl_no_pressure"}:
            (
                global_state,
                aggregation_client_df,
                aggregation_round_df,
            ) = arl.adaptive_robust_aggregation(
                round_index=round_index,
                global_state=global_state,
                local_results=local_results,
                current_updates=current_updates,
                global_pressure=current_global_pressure,
            )
            aggregation_client_df = aggregation_client_df.copy()
            aggregation_client_df["variant"] = variant
            aggregation_client_df["is_malicious"] = (
                aggregation_client_df[
                    "global_client_id"
                ].astype(str).isin(malicious_clients)
            )
            aggregation_round_df = aggregation_round_df.copy()
            aggregation_round_df["variant"] = variant
            aggregation_round_df["raw_global_pressure"] = float(
                raw_current_global_pressure
            )
            aggregation_round_df["applied_global_pressure"] = float(
                current_global_pressure
            )
            aggregation_frames.append(aggregation_client_df)
            aggregation_frames.append(aggregation_round_df)
        else:
            raise RuntimeError(f"Unexpected ablation variant: {variant}")

        previous_global_pressure = (
            0.0
            if variant == "arl_no_pressure"
            else float(current_global_pressure)
        )
        global_model.load_state_dict(global_state)
        global_model.to(device)

        if round_index in EVALUATION_ROUNDS:
            append_evaluation(round_index)

        progress_rows.append(
            {
                "round": round_index,
                "selected_clients": len(selected),
                "malicious_selected": round_malicious_selected,
                "attacks_applied": round_attacks_applied,
                "active_clients_after_round": sum(
                    value >= CRITICAL_ENERGY
                    for value in residual_energy.values()
                ),
                "mean_residual_energy": float(
                    np.mean(list(residual_energy.values()))
                ),
                "min_residual_energy": float(
                    np.min(list(residual_energy.values()))
                ),
                "round_optimizer_steps_accounted": round_optimizer_steps,
                "cumulative_optimizer_steps_accounted": (
                    scientific_optimizer_steps
                ),
                "global_pressure": (
                    float(current_global_pressure)
                    if method == "arl_fl"
                    else None
                ),
                "round_wall_seconds": (
                    time.perf_counter() - round_start
                ),
            }
        )
        write_json(
            output_root / "RUN_STATE.json",
            {
                "status": "IN_PROGRESS",
                "completed_round": round_index,
                "scientific_optimizer_steps_accounted": (
                    scientific_optimizer_steps
                ),
            },
        )

    if not evaluation_rows or int(evaluation_rows[-1]["round"]) != TOTAL_ROUNDS:
        append_evaluation(TOTAL_ROUNDS)

    write_csv(output_root / "EVALUATION_METRICS.csv", evaluation_rows)
    write_csv(output_root / "ROUND_PROGRESS.csv", progress_rows)
    write_csv(output_root / "CLIENT_SELECTION.csv", selection_rows)
    write_csv(output_root / "ATTACK_AUDIT.csv", attack_rows)
    if method_audit_frames:
        pd.concat(
            method_audit_frames,
            ignore_index=True,
            sort=False,
        ).to_csv(
            output_root / "METHOD_AUDIT.csv",
            index=False,
        )
    else:
        pd.DataFrame().to_csv(
            output_root / "METHOD_AUDIT.csv",
            index=False,
        )
    if aggregation_frames:
        pd.concat(
            aggregation_frames,
            ignore_index=True,
            sort=False,
        ).to_csv(
            output_root / "AGGREGATION_AUDIT.csv",
            index=False,
        )
    else:
        pd.DataFrame().to_csv(
            output_root / "AGGREGATION_AUDIT.csv",
            index=False,
        )

    final_metrics = evaluation_rows[-1]
    active_counts = [
        int(item["active_clients"])
        for item in evaluation_rows
    ]
    first_dropout_round = next(
        (
            int(item["round"])
            for item in evaluation_rows
            if int(item["active_clients"]) < CLIENTS_PER_FOLD
        ),
        None,
    )
    threshold75 = math.ceil(0.75 * CLIENTS_PER_FOLD)
    threshold50 = math.ceil(0.50 * CLIENTS_PER_FOLD)
    lifetime75 = next(
        (
            int(item["round"])
            for item in evaluation_rows
            if int(item["active_clients"]) < threshold75
        ),
        TOTAL_ROUNDS,
    )
    lifetime50 = next(
        (
            int(item["round"])
            for item in evaluation_rows
            if int(item["active_clients"]) < threshold50
        ),
        TOTAL_ROUNDS,
    )
    participation_counts = pd.Series(
        [row["global_client_id"] for row in selection_rows]
    ).value_counts()
    counts = np.asarray(
        [int(participation_counts.get(client_id, 0)) for client_id in client_ids],
        dtype=np.float64,
    )
    jain = float(
        (counts.sum() ** 2)
        / max(len(counts) * float(np.square(counts).sum()), EPS)
    )
    total_initial_energy = float(sum(initial_energy.values()))
    total_residual_energy = float(sum(residual_energy.values()))

    run_contract = {
        "runner_id": RUNNER_ID,
        "runner_sha256": binding["runner_sha256"],
        "runner_build_binding_sha256": binding[
            "runner_build_binding_sha256"
        ],
        "run_id": run_id,
        "condition_id": condition_id,
        "outer_fold": outer_fold,
        "alpha": alpha,
        "scenario": scenario,
        "method_family": method,
        "variant": variant,
        "reference_full_method_run_id": reference_run_id,
        "fl_seed": fl_seed,
        "scientific_run": True,
    }
    complete_core = {
        "status": "SCIENTIFIC_RUN_COMPLETE",
        "run_contract": run_contract,
        "final_metrics_round_100": {
            "loss": float(final_metrics["test_loss"]),
            "accuracy": float(final_metrics["test_accuracy"]),
            "balanced_accuracy": float(
                final_metrics["test_balanced_accuracy"]
            ),
            "macro_f1": float(final_metrics["test_macro_f1"]),
            "per_class_f1": json.loads(
                final_metrics["per_class_f1_json"]
            ),
        },
        "lifetime_metrics": {
            "early_stop_round": early_stop_round,
            "first_client_dropout_round": first_dropout_round,
            "active_client_lifetime_75pct": lifetime75,
            "active_client_lifetime_50pct": lifetime50,
            "final_active_clients": int(active_counts[-1]),
            "final_mean_residual_energy": float(
                np.mean(list(residual_energy.values()))
            ),
            "final_min_residual_energy": float(
                np.min(list(residual_energy.values()))
            ),
            "total_normalized_energy_consumed": (
                total_initial_energy - total_residual_energy
            ),
            "jain_participation_fairness": jain,
        },
        "scientific_optimizer_steps_accounted": (
            scientific_optimizer_steps
        ),
        "scientific_training_started": True,
        "scientific_metrics_computed": True,
    }
    result_binding = canonical_sha256(complete_core)
    complete = dict(complete_core)
    complete["run_result_binding_sha256"] = result_binding
    write_json(output_root / "RUN_COMPLETE.json", complete)
    write_json(
        output_root / "RUN_STATE.json",
        {
            "status": "COMPLETE",
            "completed_round": TOTAL_ROUNDS,
            "scientific_optimizer_steps_accounted": (
                scientific_optimizer_steps
            ),
            "run_result_binding_sha256": result_binding,
        },
    )

    file_rows = []
    for path in sorted(output_root.iterdir()):
        if path.is_file() and path.name != "RUN_FILE_SHA256.csv":
            file_rows.append(
                {
                    "filename": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    pd.DataFrame(file_rows).to_csv(
        output_root / "RUN_FILE_SHA256.csv",
        index=False,
    )

    print(json.dumps(complete, indent=2))
    return complete


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--binding-json", type=Path, required=True)
    parser.add_argument("--contract-check", action="store_true")
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    runner_file = Path(__file__).resolve()
    binding_path = args.binding_json.expanduser().resolve()

    if args.contract_check:
        report = contract_check(
            project_root,
            runner_file,
            binding_path,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.run_id is None or args.authorization is None:
        raise SystemExit(
            "A scientific execution requires --run-id and --authorization."
        )
    if not 1 <= int(args.run_id) <= ABLATION_RUNS:
        raise SystemExit(
            f"--run-id must be between 1 and {ABLATION_RUNS}."
        )

    run_scientific(
        project_root,
        runner_file,
        binding_path,
        int(args.run_id),
        args.authorization.expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
