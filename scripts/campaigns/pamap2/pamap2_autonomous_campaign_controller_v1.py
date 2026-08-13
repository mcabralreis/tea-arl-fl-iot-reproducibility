from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

RUNNER_ID = "PAMAP2_FROZEN_600_RUN_SCIENTIFIC_CAMPAIGN_RUNNER_V1"
EXPECTED_RUNNER_SHA256 = "7B43B8023DFC7D80C1BE409FA0EEF21E7BBA3024996FDB7A342638E5EE9B0108"
EXPECTED_RUNNER_BUILD_BINDING = "F26B90B821B08C5CF031A6A205FB3DDAD313BF769DF390E2995BE8AD6B1BDBFA"
EXPECTED_RUNNER_BINDING_FILE_SHA256 = "AA2E7771736E10EFDD97A681B9886EE8A38E6090D9632DF9CDC25770CB3837ED"
METHODS = ("fedavg", "fedprox", "random_trimmed_mean", "fedle_adapted", "tea_fl", "arl_fl")
START_RUN = 169
END_RUN = 600
GROUP_SIZE = 6
TOTAL_ROUNDS = 100
EXPECTED_CLASSES = 12
REQUIRED_FILES = {
    "RUN_COMPLETE.json",
    "RUN_STATE.json",
    "RUN_CONFIG.json",
    "EVALUATION_METRICS.csv",
    "ROUND_PROGRESS.csv",
    "CLIENT_SELECTION.csv",
    "ATTACK_AUDIT.csv",
    "METHOD_AUDIT.csv",
    "AGGREGATION_AUDIT.csv",
    "normalization_mean.npy",
    "normalization_std.npy",
    "RUN_FILE_SHA256.csv",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8", newline="\n")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise RuntimeError(f"Invalid boolean value: {value!r}")


def finite_number(value: Any, label: str) -> float:
    number = float(value)
    require(math.isfinite(number), f"{label} is not finite.")
    return number


def finite_probability(value: Any, label: str) -> float:
    number = finite_number(value, label)
    require(0.0 <= number <= 1.0, f"{label} outside [0,1].")
    return number


def alpha_token(alpha: float) -> str:
    if math.isclose(alpha, 1.0):
        return "alpha1p0"
    if math.isclose(alpha, 0.1):
        return "alpha0p1"
    raise RuntimeError(f"Unsupported frozen alpha: {alpha}")


def run_name_from_row(row: dict[str, str]) -> str:
    return (
        f"run_{int(row['run_id']):03d}"
        f"__fold{int(row['outer_fold'])}"
        f"__{alpha_token(float(row['alpha']))}"
        f"__{row['scenario']}"
        f"__{row['method']}"
        f"__seed{int(row['fl_seed'])}"
    )


def replay_binding(path: Path, field: str, expected: str | None = None) -> dict[str, Any]:
    value = read_json(path)
    observed = value.get(field)
    require(isinstance(observed, str), f"Missing binding field {field}: {path}")
    if expected is not None:
        require(observed == expected, f"Unexpected binding value in {path}")
    core = dict(value)
    core.pop(field)
    require(canonical_sha256(core) == observed, f"Canonical binding replay failed: {path}")
    return value


def verify_manifest(run_root: Path) -> int:
    rows = read_csv(run_root / "RUN_FILE_SHA256.csv")
    require(rows, f"Empty run manifest: {run_root}")
    filenames = [str(row["filename"]) for row in rows]
    require(len(filenames) == len(set(filenames)), f"Duplicate manifest filenames: {run_root}")
    for row in rows:
        path = run_root / str(row["filename"])
        require(path.is_file(), f"Manifest file missing: {path}")
        require(int(row["size_bytes"]) == path.stat().st_size, f"Manifest size mismatch: {path}")
        require(str(row["sha256"]).upper() == sha256_file(path), f"Manifest SHA mismatch: {path}")
    actual = {
        path.name
        for path in run_root.iterdir()
        if path.is_file() and path.name != "RUN_FILE_SHA256.csv"
    }
    require(actual == set(filenames), f"Run inventory differs from manifest: {run_root}")
    return len(rows)


def contract_from_campaign(row: dict[str, str]) -> dict[str, Any]:
    return {
        "condition_id": int(row["condition_id"]),
        "outer_fold": int(row["outer_fold"]),
        "alpha": float(row["alpha"]),
        "scenario": str(row["scenario"]),
        "method": str(row["method"]),
        "fl_seed": int(row["fl_seed"]),
    }


def find_condition(
    conditions: list[dict[str, str]],
    campaign_row: dict[str, str],
) -> dict[str, str]:
    matches = [
        row
        for row in conditions
        if int(row["condition_id"]) == int(campaign_row["condition_id"])
        and int(row["outer_fold"]) == int(campaign_row["outer_fold"])
        and math.isclose(float(row["alpha"]), float(campaign_row["alpha"]))
        and str(row["scenario"]) == str(campaign_row["scenario"])
        and int(row["fl_seed"]) == int(campaign_row["fl_seed"])
    ]
    require(len(matches) == 1, f"Expected one matched condition for run {campaign_row['run_id']}; found {len(matches)}")
    return matches[0]


def malicious_clients_for_condition(
    malicious_rows: list[dict[str, str]],
    condition: dict[str, str],
) -> set[str]:
    malicious_count = int(condition["malicious_count"])
    if malicious_count == 0:
        return set()
    clients = {
        str(row["global_client_id"])
        for row in malicious_rows
        if int(row["outer_fold"]) == int(condition["outer_fold"])
        and int(row["fl_seed"]) == int(condition["fl_seed"])
        and int(row["malicious_count"]) == malicious_count
    }
    require(
        len(clients) == malicious_count,
        f"Malicious-client manifest mismatch for condition {condition['condition_id']}: expected {malicious_count}, found {len(clients)}",
    )
    return clients


def expected_attack_type(attack: str, active: bool) -> str:
    if not active or attack == "none":
        return "none"
    if attack in {"sign_flip", "label_flip"}:
        return attack
    raise RuntimeError(f"Unsupported frozen attack type: {attack}")


def mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def audit_robust_method(
    run_root: Path,
    method: str,
    rounds: int,
    attack_start_round: int,
) -> dict[str, Any] | None:
    if method == "tea_fl":
        trust_rows = [
            row for row in read_csv(run_root / "METHOD_AUDIT.csv")
            if row.get("audit_type", "") == "trust_update"
        ]
        aggregation_rows = [
            row for row in read_csv(run_root / "AGGREGATION_AUDIT.csv")
            if row.get("audit_type", "") == "tea_aggregation"
        ]
        require(len(trust_rows) == rounds * 8, f"TEA trust-row count mismatch: {run_root}")
        require(len(aggregation_rows) == rounds * 8, f"TEA aggregation-row count mismatch: {run_root}")
        weight_sums: defaultdict[int, float] = defaultdict(float)
        malicious_trust: list[float] = []
        honest_trust: list[float] = []
        malicious_weight: list[float] = []
        honest_weight: list[float] = []
        for row in trust_rows:
            round_id = int(row["round"])
            trust_after = finite_probability(row["trust_after"], "TEA trust_after")
            if round_id >= attack_start_round:
                (malicious_trust if parse_bool(row["is_malicious"]) else honest_trust).append(trust_after)
        for index, row in enumerate(aggregation_rows):
            round_id = int(row.get("round") or (index // 8) + 1)
            weight = finite_probability(row["normalized_weight"], "TEA normalized_weight")
            weight_sums[round_id] += weight
            if round_id >= attack_start_round:
                (malicious_weight if parse_bool(row["is_malicious"]) else honest_weight).append(weight)
        require(
            all(math.isclose(weight_sums[round_id], 1.0, abs_tol=1e-10) for round_id in range(1, rounds + 1)),
            f"TEA normalized weights do not sum to one: {run_root}",
        )
        return {
            "audit_type": "tea_fl",
            "trust_update_rows": len(trust_rows),
            "aggregation_rows": len(aggregation_rows),
            "post_attack_mean_trust_malicious": mean(malicious_trust),
            "post_attack_mean_trust_honest": mean(honest_trust),
            "post_attack_mean_normalized_weight_malicious": mean(malicious_weight),
            "post_attack_mean_normalized_weight_honest": mean(honest_weight),
        }
    if method == "arl_fl":
        risk_rows = [
            row for row in read_csv(run_root / "METHOD_AUDIT.csv")
            if row.get("audit_type", "") == "risk_update"
        ]
        aggregation_rows = read_csv(run_root / "AGGREGATION_AUDIT.csv")
        client_rows = [
            row for row in aggregation_rows
            if str(row.get("global_client_id", "")).strip()
            and str(row.get("clip_factor", "")).strip()
        ]
        round_rows = [
            row for row in aggregation_rows
            if str(row.get("global_pressure", "")).strip()
            and not str(row.get("global_client_id", "")).strip()
        ]
        require(len(risk_rows) == rounds * 8, f"ARL risk-row count mismatch: {run_root}")
        require(len(client_rows) == rounds * 8, f"ARL client aggregation-row count mismatch: {run_root}")
        require(len(round_rows) == rounds, f"ARL round aggregation-row count mismatch: {run_root}")
        malicious_risk: list[float] = []
        honest_risk: list[float] = []
        malicious_clip: list[float] = []
        honest_clip: list[float] = []
        for row in risk_rows:
            round_id = int(row["round"])
            risk_after = finite_probability(row["risk_after"], "ARL risk_after")
            if round_id >= attack_start_round:
                (malicious_risk if parse_bool(row["is_malicious"]) else honest_risk).append(risk_after)
        for row in client_rows:
            round_id = int(row["round"])
            clip_factor = finite_probability(row["clip_factor"], "ARL clip_factor")
            if round_id >= attack_start_round:
                (malicious_clip if parse_bool(row["is_malicious"]) else honest_clip).append(clip_factor)
        for row in round_rows:
            finite_number(row["global_pressure"], "ARL global_pressure")
        return {
            "audit_type": "arl_fl",
            "risk_update_rows": len(risk_rows),
            "aggregation_client_rows": len(client_rows),
            "aggregation_round_rows": len(round_rows),
            "post_attack_mean_risk_malicious": mean(malicious_risk),
            "post_attack_mean_risk_honest": mean(honest_risk),
            "post_attack_mean_clip_factor_malicious": mean(malicious_clip),
            "post_attack_mean_clip_factor_honest": mean(honest_clip),
        }
    return None


def audit_run(
    *,
    run_id: int,
    campaign_row: dict[str, str],
    condition: dict[str, str],
    run_root: Path,
    authorization_path: Path,
    malicious_clients: set[str],
    attack_start_round: int,
) -> dict[str, Any]:
    require(run_root.is_dir(), f"Missing run directory: {run_root}")
    files = {path.name for path in run_root.iterdir() if path.is_file()}
    require(REQUIRED_FILES.issubset(files), f"Run {run_id} is missing required files")
    manifest_entries = verify_manifest(run_root)

    authorization = replay_binding(authorization_path, "authorization_binding_sha256")
    expected_contract = contract_from_campaign(campaign_row)
    require(int(authorization.get("run_id", -1)) == run_id, f"Authorization run ID mismatch: {run_id}")
    require(authorization.get("runner_sha256") == EXPECTED_RUNNER_SHA256, f"Authorization runner mismatch: {run_id}")
    require(
        authorization.get("runner_build_binding_sha256") == EXPECTED_RUNNER_BUILD_BINDING,
        f"Authorization build binding mismatch: {run_id}",
    )
    for key, expected in expected_contract.items():
        observed = authorization.get(key)
        if key == "alpha":
            require(math.isclose(float(observed), float(expected)), f"Authorization {key} mismatch: {run_id}")
        else:
            require(observed == expected, f"Authorization {key} mismatch: {run_id}")
    require(authorization.get("scientific_run") is True, f"Authorization is not scientific: {run_id}")
    require(authorization.get("resume_allowed") is False, f"Authorization permits resume: {run_id}")

    complete = read_json(run_root / "RUN_COMPLETE.json")
    require(complete.get("status") == "SCIENTIFIC_RUN_COMPLETE", f"Run status mismatch: {run_id}")
    result_binding = complete.get("run_result_binding_sha256")
    require(isinstance(result_binding, str), f"Run has no result binding: {run_id}")
    core = dict(complete)
    core.pop("run_result_binding_sha256")
    require(canonical_sha256(core) == result_binding, f"Run result-binding replay failed: {run_id}")

    contract = complete.get("run_contract", {})
    require(contract.get("runner_id") == RUNNER_ID, f"Runner ID mismatch: {run_id}")
    require(contract.get("runner_sha256") == EXPECTED_RUNNER_SHA256, f"Runner SHA mismatch: {run_id}")
    require(
        contract.get("runner_build_binding_sha256") == EXPECTED_RUNNER_BUILD_BINDING,
        f"Runner build binding mismatch: {run_id}",
    )
    require(int(contract.get("run_id", -1)) == run_id, f"Run contract ID mismatch: {run_id}")
    require(contract.get("scientific_run") is True, f"Run contract is not scientific: {run_id}")
    for key, expected in expected_contract.items():
        observed = contract.get(key)
        if key == "alpha":
            require(math.isclose(float(observed), float(expected)), f"Run contract {key} mismatch: {run_id}")
        else:
            require(observed == expected, f"Run contract {key} mismatch: {run_id}")

    config = read_json(run_root / "RUN_CONFIG.json")
    require(config.get("runner_id") == RUNNER_ID, f"RUN_CONFIG runner ID mismatch: {run_id}")
    require(config.get("runner_sha256") == EXPECTED_RUNNER_SHA256, f"RUN_CONFIG runner SHA mismatch: {run_id}")
    require(
        config.get("runner_build_binding_sha256") == EXPECTED_RUNNER_BUILD_BINDING,
        f"RUN_CONFIG build binding mismatch: {run_id}",
    )
    require(
        config.get("authorization_binding_sha256") == authorization["authorization_binding_sha256"],
        f"RUN_CONFIG authorization binding mismatch: {run_id}",
    )
    require(int(config.get("total_rounds", -1)) == TOTAL_ROUNDS, f"RUN_CONFIG round count mismatch: {run_id}")
    clients_per_round = int(config.get("clients_per_round", -1))
    require(clients_per_round == 8, f"RUN_CONFIG clients-per-round mismatch: {run_id}")
    evaluation_rounds = [int(value) for value in config.get("evaluation_rounds", [])]
    require(evaluation_rounds == list(range(0, 101, 5)), f"RUN_CONFIG evaluation rounds mismatch: {run_id}")
    require(str(config.get("attack")) == str(condition["attack"]), f"RUN_CONFIG attack mismatch: {run_id}")
    require(int(config.get("malicious_count", -1)) == int(condition["malicious_count"]), f"RUN_CONFIG malicious count mismatch: {run_id}")

    metrics = complete.get("final_metrics_round_100", {})
    finite_number(metrics.get("loss"), f"Run {run_id} loss")
    finite_probability(metrics.get("accuracy"), f"Run {run_id} accuracy")
    finite_probability(metrics.get("balanced_accuracy"), f"Run {run_id} balanced accuracy")
    finite_probability(metrics.get("macro_f1"), f"Run {run_id} macro-F1")
    per_class = metrics.get("per_class_f1", [])
    require(len(per_class) == EXPECTED_CLASSES, f"Run {run_id} per-class F1 count mismatch")
    for index, value in enumerate(per_class):
        finite_probability(value, f"Run {run_id} class {index} F1")

    lifetime = complete.get("lifetime_metrics", {})
    early_stop_round = lifetime.get("early_stop_round")
    rounds = TOTAL_ROUNDS if early_stop_round is None else int(early_stop_round) - 1
    require(0 <= rounds <= TOTAL_ROUNDS, f"Run {run_id} completed-round count is invalid")
    require(int(lifetime.get("final_active_clients", -1)) >= 0, f"Run {run_id} final active-client count invalid")
    finite_number(lifetime.get("final_mean_residual_energy"), f"Run {run_id} final mean residual energy")
    finite_number(lifetime.get("final_min_residual_energy"), f"Run {run_id} final minimum residual energy")
    finite_number(lifetime.get("total_normalized_energy_consumed"), f"Run {run_id} energy")
    finite_probability(lifetime.get("jain_participation_fairness"), f"Run {run_id} Jain fairness")

    optimizer_steps = int(complete.get("scientific_optimizer_steps_accounted", -1))
    require(optimizer_steps > 0, f"Run {run_id} optimizer steps are not positive")
    require(complete.get("scientific_training_started") is True, f"Run {run_id} training marker missing")
    require(complete.get("scientific_metrics_computed") is True, f"Run {run_id} metrics marker missing")

    state = read_json(run_root / "RUN_STATE.json")
    require(state.get("status") == "COMPLETE", f"Run {run_id} state is not COMPLETE")
    require(int(state.get("completed_round", -1)) == TOTAL_ROUNDS, f"Run {run_id} final completed_round mismatch")
    require(
        int(state.get("scientific_optimizer_steps_accounted", -1)) == optimizer_steps,
        f"Run {run_id} state optimizer ledger mismatch",
    )
    require(state.get("run_result_binding_sha256") == result_binding, f"Run {run_id} state binding mismatch")

    evaluation = read_csv(run_root / "EVALUATION_METRICS.csv")
    require([int(row["round"]) for row in evaluation] == evaluation_rounds, f"Run {run_id} evaluation chain mismatch")
    progress = read_csv(run_root / "ROUND_PROGRESS.csv")
    require([int(row["round"]) for row in progress] == list(range(1, rounds + 1)), f"Run {run_id} progress chain mismatch")
    selection = read_csv(run_root / "CLIENT_SELECTION.csv")
    attack_audit = read_csv(run_root / "ATTACK_AUDIT.csv")
    expected_selection_rows = rounds * clients_per_round
    require(len(selection) == expected_selection_rows, f"Run {run_id} selection-row count mismatch")
    require(len(attack_audit) == expected_selection_rows, f"Run {run_id} attack-row count mismatch")

    preflight_steps = 0
    if str(campaign_row["method"]) == "fedle_adapted":
        for name in (
            "FEDLE_PREFLIGHT.csv",
            "FEDLE_CLUSTERS.csv",
            "FEDLE_SIMILARITY_MATRIX.npy",
            "FEDLE_PREFLIGHT_SUMMARY.json",
        ):
            require((run_root / name).is_file(), f"Run {run_id} missing {name}")
        preflight = read_json(run_root / "FEDLE_PREFLIGHT_SUMMARY.json")
        preflight_steps = int(preflight.get("optimizer_steps_accounted", -1))
        require(preflight_steps > 0, f"Run {run_id} FedLE preflight optimizer ledger invalid")
        require(len(read_csv(run_root / "FEDLE_PREFLIGHT.csv")) == 28, f"Run {run_id} FedLE preflight row count mismatch")
        require(len(read_csv(run_root / "FEDLE_CLUSTERS.csv")) == 28, f"Run {run_id} FedLE cluster row count mismatch")

    malicious_by_round: defaultdict[int, int] = defaultdict(int)
    attacks_by_round: defaultdict[int, int] = defaultdict(int)
    steps_by_round: defaultdict[int, int] = defaultdict(int)
    malicious_rows = 0
    active_attack_rows = 0
    selected_client_steps = 0
    attack = str(condition["attack"])
    malicious_count = int(condition["malicious_count"])

    for selection_row, attack_row in zip(selection, attack_audit):
        round_id = int(selection_row["round"])
        require(int(attack_row["round"]) == round_id, f"Run {run_id} attack/selection round alignment mismatch")
        require(
            str(attack_row["global_client_id"]) == str(selection_row["global_client_id"]),
            f"Run {run_id} attack/selection client alignment mismatch",
        )
        client_id = str(selection_row["global_client_id"])
        is_malicious = client_id in malicious_clients
        active = bool(is_malicious and malicious_count > 0 and attack != "none" and round_id >= attack_start_round)
        expected_type = expected_attack_type(attack, active)
        for row in (selection_row, attack_row):
            require(parse_bool(row["is_malicious"]) == is_malicious, f"Run {run_id} malicious marker mismatch")
            require(parse_bool(row["attack_active"]) == active, f"Run {run_id} attack-active marker mismatch")
            require(str(row["attack_type"]) == expected_type, f"Run {run_id} attack-type mismatch")
        local_steps = int(selection_row["optimizer_steps_accounted"])
        require(local_steps > 0, f"Run {run_id} selected-client optimizer steps invalid")
        selected_client_steps += local_steps
        malicious_rows += int(is_malicious)
        active_attack_rows += int(active)
        malicious_by_round[round_id] += int(is_malicious)
        attacks_by_round[round_id] += int(active)
        steps_by_round[round_id] += local_steps

    require(selected_client_steps + preflight_steps == optimizer_steps, f"Run {run_id} optimizer-step decomposition mismatch")
    cumulative = preflight_steps
    for row in progress:
        round_id = int(row["round"])
        cumulative += steps_by_round[round_id]
        require(int(row["malicious_selected"]) == malicious_by_round[round_id], f"Run {run_id} malicious-selection progress mismatch")
        require(int(row["attacks_applied"]) == attacks_by_round[round_id], f"Run {run_id} attack progress mismatch")
        require(int(row["round_optimizer_steps_accounted"]) == steps_by_round[round_id], f"Run {run_id} round optimizer ledger mismatch")
        require(int(row["cumulative_optimizer_steps_accounted"]) == cumulative, f"Run {run_id} cumulative optimizer ledger mismatch")
    require(cumulative == optimizer_steps, f"Run {run_id} final optimizer ledger mismatch")

    if malicious_count == 0 or attack == "none":
        require(active_attack_rows == 0, f"Run {run_id} unexpectedly applied attacks")
        first_attack_round: int | None = None
        last_attack_round: int | None = None
    else:
        active_rows = [row for row in attack_audit if parse_bool(row["attack_active"])]
        require(active_rows, f"Run {run_id} has no active attack rows")
        first_attack_round = min(int(row["round"]) for row in active_rows)
        last_attack_round = max(int(row["round"]) for row in active_rows)
        require(first_attack_round >= attack_start_round, f"Run {run_id} attack began before frozen attack start")

    robust_diagnostics = audit_robust_method(
        run_root,
        str(campaign_row["method"]),
        rounds,
        attack_start_round,
    )

    return {
        "run_id": run_id,
        "run_name": run_root.name,
        "condition_id": int(campaign_row["condition_id"]),
        "outer_fold": int(campaign_row["outer_fold"]),
        "alpha": float(campaign_row["alpha"]),
        "scenario": str(campaign_row["scenario"]),
        "method": str(campaign_row["method"]),
        "fl_seed": int(campaign_row["fl_seed"]),
        "attack": attack,
        "malicious_count": malicious_count,
        "completed_rounds": rounds,
        "run_state_completed_round": int(state["completed_round"]),
        "manifest_entries": manifest_entries,
        "evaluation_rows": len(evaluation),
        "progress_rows": len(progress),
        "selection_rows": len(selection),
        "malicious_selection_rows": malicious_rows,
        "attacks_applied_rows": active_attack_rows,
        "first_attack_round": first_attack_round,
        "last_attack_round": last_attack_round,
        "preflight_optimizer_steps": preflight_steps,
        "selected_client_optimizer_steps": selected_client_steps,
        "scientific_optimizer_steps": optimizer_steps,
        "run_result_binding_sha256": result_binding,
        "authorization_file_sha256": sha256_file(authorization_path),
        "authorization_binding_sha256": authorization["authorization_binding_sha256"],
        "macro_f1": float(metrics["macro_f1"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "accuracy": float(metrics["accuracy"]),
        "robust_method_diagnostics": robust_diagnostics,
    }


def authorization_for_run(
    *,
    run_id: int,
    campaign_row: dict[str, str],
    condition: dict[str, str],
    master_binding: str,
    gate148_binding: str,
    controller_sha256: str,
) -> dict[str, Any]:
    core = {
        "status": "AUTHORIZED",
        "authorization_scope": f"PAMAP2_AUTONOMOUS_PROGRESSIVE_RUN_{run_id:03d}_ONLY",
        "authorization_mode": "PROGRESSIVE_CONDITIONAL_AFTER_PRIOR_ACCEPTED_GROUP",
        "master_authorization_binding_sha256": master_binding,
        "gate148_final_binding_sha256": gate148_binding,
        "controller_sha256": controller_sha256,
        "runner_id": RUNNER_ID,
        "runner_sha256": EXPECTED_RUNNER_SHA256,
        "runner_build_binding_sha256": EXPECTED_RUNNER_BUILD_BINDING,
        "run_id": run_id,
        "condition_id": int(campaign_row["condition_id"]),
        "outer_fold": int(campaign_row["outer_fold"]),
        "alpha": float(campaign_row["alpha"]),
        "scenario": str(campaign_row["scenario"]),
        "method": str(campaign_row["method"]),
        "fl_seed": int(campaign_row["fl_seed"]),
        "attack": str(condition["attack"]),
        "malicious_count": int(condition["malicious_count"]),
        "scientific_run": True,
        "resume_allowed": False,
        "campaign_authorized": False,
        "unconditional_future_runs_authorized": False,
        "next_run_requires_current_run_audit_pass": True,
        "next_group_requires_current_group_audit_pass": True,
    }
    value = dict(core)
    value["authorization_binding_sha256"] = canonical_sha256(core)
    return value


def validate_campaign_structure(campaign: dict[int, dict[str, str]]) -> None:
    require(sorted(campaign) == list(range(1, 601)), "Campaign matrix must contain run IDs 1-600 exactly")
    for group_start in range(1, 601, GROUP_SIZE):
        rows = [campaign[run_id] for run_id in range(group_start, group_start + GROUP_SIZE)]
        require(
            tuple(str(row["method"]) for row in rows) == METHODS,
            f"Method order mismatch in campaign group {group_start}-{group_start + 5}",
        )
        keys = ("condition_id", "outer_fold", "alpha", "scenario", "fl_seed")
        for key in keys:
            values = {str(row[key]) for row in rows}
            require(len(values) == 1, f"Campaign group {group_start}-{group_start + 5} varies in {key}")


def execute_runner(
    *,
    python_executable: Path,
    runner_path: Path,
    project_root: Path,
    binding_path: Path,
    run_id: int,
    authorization_path: Path,
    log_handle,
) -> None:
    command = [
        str(python_executable),
        str(runner_path),
        "--project-root",
        str(project_root),
        "--binding-json",
        str(binding_path),
        "--run-id",
        str(run_id),
        "--authorization",
        str(authorization_path),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log_handle.write(line)
        log_handle.flush()
    return_code = process.wait()
    require(return_code == 0, f"PAMAP2 run {run_id} failed with exit code {return_code}")


@contextmanager
def exclusive_controller_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        handle.seek(0)
        handle.write(b"0")
        handle.flush()
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another autonomous PAMAP2 controller appears to be active.") from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Another autonomous PAMAP2 controller appears to be active.") from exc
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--gate148-binding", type=Path, required=True)
    parser.add_argument("--master-authorization", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    gate148_binding_path = args.gate148_binding.expanduser().resolve()
    master_path = args.master_authorization.expanduser().resolve()
    authorization_root = gate148_binding_path.parent
    controller_path = Path(__file__).resolve()
    controller_output_root = (
        project_root
        / "outputs"
        / "federated"
        / "pamap2"
        / "autonomous_campaign_runs169_600_v1"
    )
    controller_output_root.mkdir(parents=True, exist_ok=True)
    authorizations_root = controller_output_root / "authorizations"
    groups_root = controller_output_root / "group_audits"
    authorizations_root.mkdir(parents=True, exist_ok=True)
    groups_root.mkdir(parents=True, exist_ok=True)
    log_path = controller_output_root / "AUTONOMOUS_CAMPAIGN_EXECUTION_LOG.txt"
    state_path = controller_output_root / "AUTONOMOUS_CONTROLLER_STATE.json"
    failure_path = controller_output_root / "AUTONOMOUS_CONTROLLER_FAILURE.json"
    lock_path = controller_output_root / "AUTONOMOUS_CONTROLLER.lock"

    runner_root = project_root / "outputs" / "federated" / "pamap2" / "scientific_campaign_runner_v1"
    runner_path = runner_root / "run_pamap2_scientific_campaign.py"
    runner_binding_path = runner_root / "PAMAP2_SCIENTIFIC_RUNNER_BINDING.json"
    campaign_root = project_root / "outputs" / "federated" / "pamap2" / "scientific_campaign_600runs_v1"
    frozen_root = (
        project_root
        / "outputs"
        / "protocols"
        / "pamap2_source_interface_freeze_v1r2"
        / "frozen_input_provenance"
    )
    campaign_matrix_path = frozen_root / "campaign_matrix_600_runs.csv"
    conditions_path = frozen_root / "matched_condition_manifest.csv"
    malicious_manifest_path = frozen_root / "malicious_client_manifest.csv"
    protocol_path = frozen_root / "FL_EXPERIMENTAL_PROTOCOL_V1.json"

    with exclusive_controller_lock(lock_path):
        current_run: int | None = None
        try:
            gate148 = replay_binding(gate148_binding_path, "gate148_final_binding_sha256")
            require(gate148.get("status") == "PASS", "Gate-148 is not PASS")
            require(gate148.get("authorized_start_run") == START_RUN, "Gate-148 start run mismatch")
            require(gate148.get("authorized_end_run") == END_RUN, "Gate-148 end run mismatch")
            require(gate148.get("group_size") == GROUP_SIZE, "Gate-148 group size mismatch")
            gate148_binding = str(gate148["gate148_final_binding_sha256"])

            master = replay_binding(master_path, "master_authorization_binding_sha256")
            require(master.get("status") == "AUTHORIZED", "Master authorization is not AUTHORIZED")
            require(sha256_file(master_path) == gate148.get("master_authorization_file_sha256"), "Master authorization file SHA mismatch")
            require(master["master_authorization_binding_sha256"] == gate148.get("master_authorization_binding_sha256"), "Master authorization binding mismatch")
            require(int(master.get("start_run", -1)) == START_RUN, "Master authorization start mismatch")
            require(int(master.get("end_run", -1)) == END_RUN, "Master authorization end mismatch")
            require(int(master.get("group_size", -1)) == GROUP_SIZE, "Master authorization group size mismatch")
            require(master.get("sequential_execution_required") is True, "Master authorization does not require sequential execution")
            require(master.get("parallel_execution_authorized") is False, "Master authorization permits parallel execution")
            require(master.get("partial_run_policy") == "REFUSE_AND_STOP", "Master authorization partial-run policy mismatch")
            controller_sha256 = sha256_file(controller_path)
            require(controller_sha256 == master.get("controller_sha256"), "Controller SHA mismatch")
            require(controller_sha256 == gate148.get("controller_sha256"), "Gate-148 controller SHA mismatch")

            require(runner_path.is_file(), f"Runner missing: {runner_path}")
            require(sha256_file(runner_path) == EXPECTED_RUNNER_SHA256, "Runner SHA mismatch")
            require(runner_binding_path.is_file(), f"Runner binding missing: {runner_binding_path}")
            require(sha256_file(runner_binding_path) == EXPECTED_RUNNER_BINDING_FILE_SHA256, "Runner binding file SHA mismatch")
            runner_binding = replay_binding(
                runner_binding_path,
                "runner_build_binding_sha256",
                EXPECTED_RUNNER_BUILD_BINDING,
            )
            require(runner_binding.get("runner_id") == RUNNER_ID, "Runner binding ID mismatch")
            require(runner_binding.get("runner_sha256") == EXPECTED_RUNNER_SHA256, "Runner binding SHA mismatch")

            campaign_rows = read_csv(campaign_matrix_path)
            require(len(campaign_rows) == 600, "Campaign matrix row count mismatch")
            campaign = {int(row["run_id"]): row for row in campaign_rows}
            require(len(campaign) == 600, "Campaign run IDs are not unique")
            validate_campaign_structure(campaign)
            conditions = read_csv(conditions_path)
            malicious_rows = read_csv(malicious_manifest_path)
            protocol = read_json(protocol_path)
            attack_start_round = int(protocol["threat_model"]["attack_start_round"])
            require(attack_start_round == 20, "Frozen attack start round mismatch")

            expected_prior_names = {run_name_from_row(campaign[run_id]) for run_id in range(1, START_RUN)}
            unknown_dirs = {
                path.name
                for path in campaign_root.iterdir()
                if path.is_dir() and path.name not in {run_name_from_row(campaign[run_id]) for run_id in range(1, 601)}
            }
            require(not unknown_dirs, f"Unknown campaign run directories: {sorted(unknown_dirs)}")
            missing_prior = [name for name in expected_prior_names if not (campaign_root / name / "RUN_COMPLETE.json").is_file()]
            require(not missing_prior, f"Missing accepted prior run(s), first: {missing_prior[:3]}")

            master_binding = str(master["master_authorization_binding_sha256"])
            accepted_steps = int(master["scientific_optimizer_steps_before_autonomous_execution"])
            accepted_runs = int(master["scientific_runs_complete_before_autonomous_execution"])
            require(accepted_runs == 168, "Master authorization prior run ledger mismatch")
            require(accepted_steps == 958565, "Master authorization prior optimizer ledger mismatch")
            previous_group_binding = gate148_binding

            with log_path.open("a", encoding="utf-8", newline="\n") as log_handle:
                header = (
                    f"\n[{utc_now()}] PAMAP2 autonomous controller start/restart\n"
                    f"Scope: runs {START_RUN}-{END_RUN}; sequential; groups of {GROUP_SIZE}; fail closed\n"
                )
                print(header, end="", flush=True)
                log_handle.write(header)
                log_handle.flush()

                for group_start in range(START_RUN, END_RUN + 1, GROUP_SIZE):
                    group_end = group_start + GROUP_SIZE - 1
                    group_results: list[dict[str, Any]] = []
                    print(f"\nAUTONOMOUS GROUP {group_start}-{group_end} START", flush=True)
                    log_handle.write(f"\nAUTONOMOUS GROUP {group_start}-{group_end} START\n")
                    log_handle.flush()

                    for run_id in range(group_start, group_end + 1):
                        current_run = run_id
                        row = campaign[run_id]
                        condition = find_condition(conditions, row)
                        malicious_clients = malicious_clients_for_condition(malicious_rows, condition)
                        run_name = run_name_from_row(row)
                        run_root = campaign_root / run_name
                        authorization_path = authorizations_root / f"RUN{run_id:03d}_AUTHORIZATION.json"
                        expected_authorization = authorization_for_run(
                            run_id=run_id,
                            campaign_row=row,
                            condition=condition,
                            master_binding=master_binding,
                            gate148_binding=gate148_binding,
                            controller_sha256=controller_sha256,
                        )
                        if authorization_path.exists():
                            existing_authorization = read_json(authorization_path)
                            require(existing_authorization == expected_authorization, f"Existing authorization differs for run {run_id}")
                        else:
                            write_json(authorization_path, expected_authorization)

                        if run_root.exists():
                            complete_path = run_root / "RUN_COMPLETE.json"
                            require(
                                complete_path.is_file(),
                                f"Refusing partial run directory: {run_root}. No automatic deletion or resume is permitted.",
                            )
                            print(f"AUDIT EXISTING COMPLETE run {run_id} / {run_name}", flush=True)
                            log_handle.write(f"AUDIT EXISTING COMPLETE run {run_id} / {run_name}\n")
                            log_handle.flush()
                        else:
                            print(f"START NEW run {run_id} / {run_name}", flush=True)
                            log_handle.write(f"START NEW run {run_id} / {run_name}\n")
                            log_handle.flush()
                            execute_runner(
                                python_executable=Path(sys.executable),
                                runner_path=runner_path,
                                project_root=project_root,
                                binding_path=runner_binding_path,
                                run_id=run_id,
                                authorization_path=authorization_path,
                                log_handle=log_handle,
                            )
                            require((run_root / "RUN_COMPLETE.json").is_file(), f"Run {run_id} returned success without RUN_COMPLETE.json")

                        result = audit_run(
                            run_id=run_id,
                            campaign_row=row,
                            condition=condition,
                            run_root=run_root,
                            authorization_path=authorization_path,
                            malicious_clients=malicious_clients,
                            attack_start_round=attack_start_round,
                        )
                        group_results.append(result)
                        print(
                            f"AUDIT PASS run {run_id} / steps={result['scientific_optimizer_steps']} / "
                            f"macro_f1={result['macro_f1']:.12f}",
                            flush=True,
                        )
                        log_handle.write(
                            f"AUDIT PASS run {run_id} / steps={result['scientific_optimizer_steps']} / "
                            f"binding={result['run_result_binding_sha256']}\n"
                        )
                        log_handle.flush()

                    require(tuple(result["method"] for result in group_results) == METHODS, f"Group {group_start}-{group_end} method sequence mismatch")
                    require(len({result["condition_id"] for result in group_results}) == 1, f"Group {group_start}-{group_end} condition mismatch")
                    group_steps = sum(int(result["scientific_optimizer_steps"]) for result in group_results)
                    accepted_steps += group_steps
                    accepted_runs += GROUP_SIZE
                    group_audit = {
                        "status": "PASS",
                        "scope": f"PAMAP2_AUTONOMOUS_GROUP_{group_start}_{group_end}_AUDIT",
                        "group_start_run": group_start,
                        "group_end_run": group_end,
                        "condition_id": group_results[0]["condition_id"],
                        "outer_fold": group_results[0]["outer_fold"],
                        "alpha": group_results[0]["alpha"],
                        "scenario": group_results[0]["scenario"],
                        "fl_seed": group_results[0]["fl_seed"],
                        "methods": list(METHODS),
                        "run_audits": group_results,
                        "group_scientific_optimizer_steps": group_steps,
                        "scientific_runs_complete": accepted_runs,
                        "accepted_scientific_optimizer_steps": accepted_steps,
                        "previous_accepted_binding_sha256": previous_group_binding,
                        "scientific_training_executed_by_auditor": False,
                        "scientific_optimizer_steps_executed_by_auditor": 0,
                    }
                    group_audit_path = groups_root / f"GROUP_{group_start:03d}_{group_end:03d}_AUDIT.json"
                    write_json(group_audit_path, group_audit)
                    group_report_lines = [
                        f"PAMAP2 AUTONOMOUS GROUP {group_start}-{group_end} AUDIT",
                        "=" * 78,
                        "",
                        "STATUS",
                        "-" * 78,
                        "PASS",
                        "",
                        f"Condition ID: {group_results[0]['condition_id']}",
                        f"Outer fold: {group_results[0]['outer_fold']}",
                        f"Alpha: {group_results[0]['alpha']}",
                        f"Scenario: {group_results[0]['scenario']}",
                        f"FL seed: {group_results[0]['fl_seed']}",
                        f"Runs: {group_start}-{group_end}",
                        f"Group optimizer steps: {group_steps}",
                        f"Scientific runs complete: {accepted_runs}/600",
                        f"Accepted scientific optimizer steps: {accepted_steps}",
                        "",
                        "RUNS",
                        "-" * 78,
                    ]
                    for result in group_results:
                        group_report_lines.append(
                            f"run {result['run_id']}: {result['method']}; "
                            f"steps={result['scientific_optimizer_steps']}; "
                            f"macro_f1={result['macro_f1']:.12f}; "
                            f"binding={result['run_result_binding_sha256']}"
                        )
                    group_report_path = groups_root / f"GROUP_{group_start:03d}_{group_end:03d}_REPORT.txt"
                    group_report_path.write_text("\n".join(group_report_lines) + "\n", encoding="utf-8", newline="\n")
                    binding_core = {
                        "status": "PASS",
                        "scope": group_audit["scope"],
                        "group_start_run": group_start,
                        "group_end_run": group_end,
                        "previous_accepted_binding_sha256": previous_group_binding,
                        "gate148_final_binding_sha256": gate148_binding,
                        "master_authorization_binding_sha256": master_binding,
                        "controller_sha256": controller_sha256,
                        "audit_sha256": sha256_file(group_audit_path),
                        "report_sha256": sha256_file(group_report_path),
                        "run_result_bindings": {
                            str(result["run_id"]): result["run_result_binding_sha256"]
                            for result in group_results
                        },
                        "group_scientific_optimizer_steps": group_steps,
                        "scientific_runs_complete": accepted_runs,
                        "accepted_scientific_optimizer_steps": accepted_steps,
                    }
                    group_binding = canonical_sha256(binding_core)
                    binding_value = dict(binding_core)
                    binding_value["group_final_binding_sha256"] = group_binding
                    group_binding_path = groups_root / f"GROUP_{group_start:03d}_{group_end:03d}_FINAL_BINDING.json"
                    write_json(group_binding_path, binding_value)
                    previous_group_binding = group_binding

                    state = {
                        "status": "GROUP_ACCEPTED",
                        "updated_utc": utc_now(),
                        "last_accepted_group_start": group_start,
                        "last_accepted_group_end": group_end,
                        "last_accepted_group_binding_sha256": group_binding,
                        "scientific_runs_complete": accepted_runs,
                        "accepted_scientific_optimizer_steps": accepted_steps,
                        "next_run": group_end + 1 if group_end < END_RUN else None,
                    }
                    write_json(state_path, state)
                    if failure_path.exists():
                        failure_path.unlink()
                    print(
                        f"AUTONOMOUS GROUP {group_start}-{group_end} PASS / "
                        f"runs={accepted_runs}/600 / accepted_steps={accepted_steps}",
                        flush=True,
                    )
                    log_handle.write(
                        f"AUTONOMOUS GROUP {group_start}-{group_end} PASS / "
                        f"binding={group_binding} / runs={accepted_runs}/600 / accepted_steps={accepted_steps}\n"
                    )
                    log_handle.flush()

                require(accepted_runs == 600, f"Final accepted run ledger mismatch: {accepted_runs}")
                final_report = controller_output_root / "PAMAP2_AUTONOMOUS_CAMPAIGN_FINAL_REPORT.txt"
                final_report.write_text(
                    "\n".join(
                        [
                            "PAMAP2 AUTONOMOUS CAMPAIGN COMPLETION REPORT",
                            "=" * 78,
                            "",
                            "STATUS",
                            "-" * 78,
                            "PASS",
                            "",
                            "Scientific runs complete: 600/600",
                            f"Accepted scientific optimizer steps: {accepted_steps}",
                            f"Final accepted group binding SHA256: {previous_group_binding}",
                            "Execution mode: sequential / no runner resume / fail closed",
                            "Conditional group audits: 72/72 PASS",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                final_core = {
                    "status": "PASS",
                    "scope": "PAMAP2_AUTONOMOUS_RUNS169_600_COMPLETE",
                    "gate148_final_binding_sha256": gate148_binding,
                    "master_authorization_binding_sha256": master_binding,
                    "controller_sha256": controller_sha256,
                    "final_group_binding_sha256": previous_group_binding,
                    "final_report_sha256": sha256_file(final_report),
                    "scientific_runs_complete": accepted_runs,
                    "accepted_scientific_optimizer_steps": accepted_steps,
                    "groups_accepted": 72,
                }
                final_binding = canonical_sha256(final_core)
                final_value = dict(final_core)
                final_value["autonomous_campaign_final_binding_sha256"] = final_binding
                write_json(
                    controller_output_root / "PAMAP2_AUTONOMOUS_CAMPAIGN_FINAL_BINDING.json",
                    final_value,
                )
                write_json(
                    state_path,
                    {
                        "status": "COMPLETE",
                        "updated_utc": utc_now(),
                        "scientific_runs_complete": accepted_runs,
                        "accepted_scientific_optimizer_steps": accepted_steps,
                        "final_group_binding_sha256": previous_group_binding,
                        "autonomous_campaign_final_binding_sha256": final_binding,
                        "next_run": None,
                    },
                )
                print("=" * 78)
                print("PAMAP2_AUTONOMOUS_RUNS169_600_COMPLETE")
                print("=" * 78)
                print("Scientific runs complete: 600/600")
                print(f"Accepted scientific optimizer steps: {accepted_steps}")
                print(f"Autonomous campaign final binding SHA256: {final_binding}")
                print(f"Final report: {final_report}")
                return 0
        except Exception as exc:
            failure = {
                "status": "STOPPED_FAIL_CLOSED",
                "failed_utc": utc_now(),
                "current_run": current_run,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "automatic_partial_run_deletion_performed": False,
                "automatic_runner_resume_performed": False,
                "future_runs_started_after_failure": False,
            }
            write_json(failure_path, failure)
            write_json(
                state_path,
                {
                    "status": "STOPPED_FAIL_CLOSED",
                    "updated_utc": utc_now(),
                    "current_run": current_run,
                    "error": str(exc),
                    "next_run": current_run,
                },
            )
            print("=" * 78, file=sys.stderr)
            print("PAMAP2_AUTONOMOUS_CONTROLLER_STOPPED_FAIL_CLOSED", file=sys.stderr)
            print("=" * 78, file=sys.stderr)
            print(f"Current run: {current_run}", file=sys.stderr)
            print(f"Error: {exc}", file=sys.stderr)
            print(f"Failure report: {failure_path}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
