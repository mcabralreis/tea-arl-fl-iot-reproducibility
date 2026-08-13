from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GATE_ID = "GATE-149R"
SCOPE = "PAMAP2_REPAIRED_FREEZE_AND_DESCRIPTIVE_AGGREGATION_OF_ACCEPTED_600_RUN_CAMPAIGN"
RUNNER_ID = "PAMAP2_FROZEN_600_RUN_SCIENTIFIC_CAMPAIGN_RUNNER_V1"
EXPECTED_RUNNER_SHA256 = "7B43B8023DFC7D80C1BE409FA0EEF21E7BBA3024996FDB7A342638E5EE9B0108"
EXPECTED_RUNNER_BUILD_BINDING = "F26B90B821B08C5CF031A6A205FB3DDAD313BF769DF390E2995BE8AD6B1BDBFA"
EXPECTED_GATE148_FINAL_BINDING = "ED865AD6C73D223221D040106C4307747C9C47411CAB05FEA36AEF70CA9C5050"
EXPECTED_MASTER_AUTHORIZATION_BINDING = "551359371C40693C2493834C8E72CF959BFB185AF8937187D248B08BA08873FE"
EXPECTED_CONTROLLER_SHA256 = "D72A1A0B4E6E77A7E5D6E86FA69363E3EE0B133787FFFB794BC13A042A759281"
EXPECTED_FINAL_GROUP_BINDING = "AF67EC72689B28D286BF3481C14340B2BF39603BF2140742BED73E466CF1E5A7"
EXPECTED_AUTONOMOUS_FINAL_BINDING = "0AE27C139DCBB1B5F5FEEADBEADBBA89C3D3F67C875987B04CA4F8BC2F44DCD9"
EXPECTED_ACCEPTED_OPTIMIZER_STEPS = 3401459
EXPECTED_RUNS = 600
EXPECTED_EVALUATION_ROWS = 600 * 21
EXPECTED_ARL_PRESSURE_ROWS = 100 * 21
EXPECTED_NON_ARL_BLANK_PRESSURE_ROWS = 500 * 21
METHODS = ("fedavg", "fedprox", "random_trimmed_mean", "fedle_adapted", "tea_fl", "arl_fl")
SCENARIOS = ("clean", "signflip_mu0p2", "signflip_mu0p4", "labelflip_mu0p2", "labelflip_mu0p4")
ALPHAS = (1.0, 0.1)
SEEDS = (123, 456)
FOLDS = (1, 2, 3, 4, 5)
TOTAL_ROUNDS = 100
EXPECTED_CLASSES = 12
REPAIRS_GATE149_PACKAGE_SHA256 = "3D93ED9CBB702053DB40FB5005ED751084C5D23369FE321FFAC0209412FE4457"
REPAIR_REASON = "EVALUATION_METRICS.global_pressure is numeric only for arl_fl and blank for all other methods"

REQUIRED_RUN_FILES = {
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Missing CSV file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        require(bool(rows), f"Cannot infer CSV fields for empty table: {path}")
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def replay_binding(path: Path, binding_field: str, expected: str | None = None) -> dict[str, Any]:
    value = read_json(path)
    observed = value.get(binding_field)
    require(isinstance(observed, str), f"Missing binding field {binding_field}: {path}")
    if expected is not None:
        require(observed == expected, f"Unexpected binding value in {path}: {observed}")
    core = dict(value)
    core.pop(binding_field)
    require(canonical_sha256(core) == observed, f"Canonical binding replay failed: {path}")
    return value


def finite_number(value: Any, label: str) -> float:
    number = float(value)
    require(math.isfinite(number), f"Non-finite value for {label}: {value}")
    return number


def finite_probability(value: Any, label: str) -> float:
    number = finite_number(value, label)
    require(0.0 <= number <= 1.0, f"Probability outside [0,1] for {label}: {number}")
    return number


def optional_finite_number(value: Any, label: str) -> float | None:
    text = "" if value is None else str(value).strip()
    if text == "":
        return None
    return finite_number(text, label)


def alpha_token(alpha: float) -> str:
    require(any(math.isclose(alpha, allowed) for allowed in ALPHAS), f"Unexpected alpha: {alpha}")
    return "alpha1p0" if math.isclose(alpha, 1.0) else "alpha0p1"


def run_name_from_row(row: dict[str, str]) -> str:
    return (
        f"run_{int(row['run_id']):03d}"
        f"__fold{int(row['outer_fold'])}"
        f"__{alpha_token(float(row['alpha']))}"
        f"__{row['scenario']}"
        f"__{row['method']}"
        f"__seed{int(row['fl_seed'])}"
    )


def verify_run_manifest(run_root: Path) -> int:
    rows = read_csv(run_root / "RUN_FILE_SHA256.csv")
    require(rows, f"Empty run manifest: {run_root}")
    filenames = [str(row["filename"]) for row in rows]
    require(len(filenames) == len(set(filenames)), f"Duplicate manifest filenames: {run_root}")

    for row in rows:
        file_path = run_root / str(row["filename"])
        require(file_path.is_file(), f"Manifest subject missing: {file_path}")
        require(int(row["size_bytes"]) == file_path.stat().st_size, f"Manifest size mismatch: {file_path}")
        require(str(row["sha256"]).upper() == sha256_file(file_path), f"Manifest SHA mismatch: {file_path}")

    actual = {
        path.name
        for path in run_root.iterdir()
        if path.is_file() and path.name != "RUN_FILE_SHA256.csv"
    }
    require(actual == set(filenames), f"Run file inventory differs from manifest: {run_root}")
    require(REQUIRED_RUN_FILES.issubset(actual | {"RUN_FILE_SHA256.csv"}), f"Missing required run files: {run_root}")
    return len(rows)


def preserve_existing_output(output_root: Path) -> str | None:
    if not output_root.exists():
        return None
    if not any(output_root.iterdir()):
        output_root.rmdir()
        return None
    index = 1
    while True:
        candidate = output_root.with_name(f"{output_root.name}_previous_output_v{index}")
        if not candidate.exists():
            shutil.move(str(output_root), str(candidate))
            return str(candidate)
        index += 1


def same_float(a: Any, b: Any, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def summary_stats(values: list[float]) -> dict[str, float]:
    require(values, "Cannot summarize an empty list")
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def verify_autonomous_chain(
    *,
    gate148_root: Path,
    autonomous_root: Path,
) -> dict[str, Any]:
    gate148_binding_path = gate148_root / "GATE148_FINAL_BINDING.json"
    master_path = gate148_root / "PAMAP2_AUTONOMOUS_MASTER_AUTHORIZATION.json"
    controller_path = gate148_root / "pamap2_autonomous_campaign_controller_v1.py"

    gate148 = replay_binding(
        gate148_binding_path,
        "gate148_final_binding_sha256",
        EXPECTED_GATE148_FINAL_BINDING,
    )
    require(gate148.get("status") == "PASS", "Gate-148 status is not PASS")
    require(gate148.get("runner_sha256") == EXPECTED_RUNNER_SHA256, "Gate-148 runner SHA mismatch")
    require(
        gate148.get("runner_build_binding_sha256") == EXPECTED_RUNNER_BUILD_BINDING,
        "Gate-148 runner build binding mismatch",
    )

    master = replay_binding(
        master_path,
        "master_authorization_binding_sha256",
        EXPECTED_MASTER_AUTHORIZATION_BINDING,
    )
    require(master.get("status") == "AUTHORIZED", "Master authorization status mismatch")
    require(master.get("start_run") == 169 and master.get("end_run") == 600, "Master authorization range mismatch")
    require(master.get("group_count") == 72 and master.get("group_size") == 6, "Master group structure mismatch")
    require(master.get("controller_sha256") == EXPECTED_CONTROLLER_SHA256, "Master controller SHA mismatch")
    require(sha256_file(controller_path) == EXPECTED_CONTROLLER_SHA256, "Controller file SHA mismatch")

    group_root = autonomous_root / "group_audits"
    require(group_root.is_dir(), f"Group audit directory missing: {group_root}")

    previous_binding = EXPECTED_GATE148_FINAL_BINDING
    expected_runs_complete = 168
    expected_steps = 958565
    group_count = 0

    for start_run in range(169, 601, 6):
        end_run = start_run + 5
        audit_path = group_root / f"GROUP_{start_run}_{end_run}_AUDIT.json"
        report_path = group_root / f"GROUP_{start_run}_{end_run}_REPORT.txt"
        binding_path = group_root / f"GROUP_{start_run}_{end_run}_FINAL_BINDING.json"

        audit = read_json(audit_path)
        binding = replay_binding(binding_path, "group_final_binding_sha256")
        report_text = report_path.read_text(encoding="utf-8-sig")

        require(audit.get("status") == "PASS", f"Group {start_run}-{end_run} audit is not PASS")
        require(binding.get("status") == "PASS", f"Group {start_run}-{end_run} binding is not PASS")
        require("\nPASS\n" in report_text, f"Group {start_run}-{end_run} report has no PASS marker")
        require(audit.get("group_start_run") == start_run and audit.get("group_end_run") == end_run, "Group audit range mismatch")
        require(binding.get("group_start_run") == start_run and binding.get("group_end_run") == end_run, "Group binding range mismatch")
        require(audit.get("previous_accepted_binding_sha256") == previous_binding, "Group audit chain mismatch")
        require(binding.get("previous_accepted_binding_sha256") == previous_binding, "Group binding chain mismatch")
        require(binding.get("audit_sha256") == sha256_file(audit_path), "Group audit file SHA mismatch")
        require(binding.get("report_sha256") == sha256_file(report_path), "Group report file SHA mismatch")
        require(binding.get("gate148_final_binding_sha256") == EXPECTED_GATE148_FINAL_BINDING, "Group Gate-148 link mismatch")
        require(
            binding.get("master_authorization_binding_sha256") == EXPECTED_MASTER_AUTHORIZATION_BINDING,
            "Group master authorization link mismatch",
        )
        require(binding.get("controller_sha256") == EXPECTED_CONTROLLER_SHA256, "Group controller link mismatch")
        require(audit.get("scientific_training_executed_by_auditor") is False, "Group auditor reports training")
        require(int(audit.get("scientific_optimizer_steps_executed_by_auditor", -1)) == 0, "Group auditor optimizer ledger is not zero")

        run_audits = audit.get("run_audits")
        require(isinstance(run_audits, list) and len(run_audits) == 6, "Group run-audit count mismatch")
        require([int(item["run_id"]) for item in run_audits] == list(range(start_run, end_run + 1)), "Group run IDs mismatch")
        require([str(item["method"]) for item in run_audits] == list(METHODS), "Group method order mismatch")

        group_steps = sum(int(item["scientific_optimizer_steps"]) for item in run_audits)
        require(group_steps == int(audit["group_scientific_optimizer_steps"]), "Group audit optimizer sum mismatch")
        require(group_steps == int(binding["group_scientific_optimizer_steps"]), "Group binding optimizer sum mismatch")
        require(
            binding.get("run_result_bindings")
            == {str(item["run_id"]): item["run_result_binding_sha256"] for item in run_audits},
            "Group run-result binding map mismatch",
        )

        expected_runs_complete += 6
        expected_steps += group_steps
        require(int(audit["scientific_runs_complete"]) == expected_runs_complete, "Group audit run ledger mismatch")
        require(int(binding["scientific_runs_complete"]) == expected_runs_complete, "Group binding run ledger mismatch")
        require(int(audit["accepted_scientific_optimizer_steps"]) == expected_steps, "Group audit optimizer ledger mismatch")
        require(int(binding["accepted_scientific_optimizer_steps"]) == expected_steps, "Group binding optimizer ledger mismatch")

        previous_binding = str(binding["group_final_binding_sha256"])
        group_count += 1

    require(group_count == 72, "Autonomous group count mismatch")
    require(previous_binding == EXPECTED_FINAL_GROUP_BINDING, "Final group binding mismatch")
    require(expected_runs_complete == EXPECTED_RUNS, "Final group run ledger mismatch")
    require(expected_steps == EXPECTED_ACCEPTED_OPTIMIZER_STEPS, "Final group optimizer ledger mismatch")

    final_report_path = autonomous_root / "PAMAP2_AUTONOMOUS_CAMPAIGN_FINAL_REPORT.txt"
    final_binding_path = autonomous_root / "PAMAP2_AUTONOMOUS_CAMPAIGN_FINAL_BINDING.json"
    state_path = autonomous_root / "AUTONOMOUS_CONTROLLER_STATE.json"

    final_binding = replay_binding(
        final_binding_path,
        "autonomous_campaign_final_binding_sha256",
        EXPECTED_AUTONOMOUS_FINAL_BINDING,
    )
    state = read_json(state_path)

    require(final_binding.get("status") == "PASS", "Autonomous final binding is not PASS")
    require(final_binding.get("final_group_binding_sha256") == EXPECTED_FINAL_GROUP_BINDING, "Final binding group link mismatch")
    require(final_binding.get("final_report_sha256") == sha256_file(final_report_path), "Autonomous final report SHA mismatch")
    require(int(final_binding.get("scientific_runs_complete", -1)) == EXPECTED_RUNS, "Autonomous final run ledger mismatch")
    require(
        int(final_binding.get("accepted_scientific_optimizer_steps", -1)) == EXPECTED_ACCEPTED_OPTIMIZER_STEPS,
        "Autonomous final optimizer ledger mismatch",
    )
    require(int(final_binding.get("groups_accepted", -1)) == 72, "Autonomous final group count mismatch")

    require(state.get("status") == "COMPLETE", "Autonomous controller state is not COMPLETE")
    require(state.get("next_run") is None, "Autonomous controller still has a next run")
    require(int(state.get("scientific_runs_complete", -1)) == EXPECTED_RUNS, "Controller final run ledger mismatch")
    require(
        int(state.get("accepted_scientific_optimizer_steps", -1)) == EXPECTED_ACCEPTED_OPTIMIZER_STEPS,
        "Controller final optimizer ledger mismatch",
    )
    require(state.get("final_group_binding_sha256") == EXPECTED_FINAL_GROUP_BINDING, "Controller final group link mismatch")
    require(
        state.get("autonomous_campaign_final_binding_sha256") == EXPECTED_AUTONOMOUS_FINAL_BINDING,
        "Controller autonomous final binding link mismatch",
    )

    failure_files = list(autonomous_root.glob("*FAIL*")) + list(autonomous_root.glob("*ERROR*"))
    require(not failure_files, f"Unexpected autonomous failure files: {failure_files}")

    return {
        "gate148_final_binding_sha256": EXPECTED_GATE148_FINAL_BINDING,
        "master_authorization_binding_sha256": EXPECTED_MASTER_AUTHORIZATION_BINDING,
        "controller_sha256": EXPECTED_CONTROLLER_SHA256,
        "final_group_binding_sha256": EXPECTED_FINAL_GROUP_BINDING,
        "autonomous_campaign_final_binding_sha256": EXPECTED_AUTONOMOUS_FINAL_BINDING,
        "autonomous_groups_verified": group_count,
        "scientific_runs_complete": expected_runs_complete,
        "accepted_scientific_optimizer_steps": expected_steps,
        "autonomous_final_report_sha256": sha256_file(final_report_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    pamap_root = project_root / "outputs" / "federated" / "pamap2"
    campaign_root = pamap_root / "scientific_campaign_600runs_v1"
    gate148_root = pamap_root / "autonomous_campaign_authorization_v1"
    autonomous_root = pamap_root / "autonomous_campaign_runs169_600_v1"
    frozen_root = (
        project_root
        / "outputs"
        / "protocols"
        / "pamap2_source_interface_freeze_v1r2"
        / "frozen_input_provenance"
    )
    campaign_matrix_path = frozen_root / "campaign_matrix_600_runs.csv"
    conditions_path = frozen_root / "matched_condition_manifest.csv"
    output_root = pamap_root / "postcampaign_freeze_v1"

    require(campaign_root.is_dir(), f"Campaign root missing: {campaign_root}")
    require(gate148_root.is_dir(), f"Gate-148 root missing: {gate148_root}")
    require(autonomous_root.is_dir(), f"Autonomous root missing: {autonomous_root}")
    require(campaign_matrix_path.is_file(), f"Campaign matrix missing: {campaign_matrix_path}")
    require(conditions_path.is_file(), f"Condition manifest missing: {conditions_path}")

    preserved_output = preserve_existing_output(output_root)
    output_root.mkdir(parents=True, exist_ok=False)

    chain = verify_autonomous_chain(gate148_root=gate148_root, autonomous_root=autonomous_root)

    campaign_rows = read_csv(campaign_matrix_path)
    require(len(campaign_rows) == EXPECTED_RUNS, "Campaign matrix row count mismatch")
    campaign = {int(row["run_id"]): row for row in campaign_rows}
    require(len(campaign) == EXPECTED_RUNS, "Campaign run IDs are not unique")
    require(set(campaign) == set(range(1, EXPECTED_RUNS + 1)), "Campaign run ID range mismatch")

    factorial_keys = {
        (
            int(row["outer_fold"]),
            float(row["alpha"]),
            str(row["scenario"]),
            str(row["method"]),
            int(row["fl_seed"]),
        )
        for row in campaign_rows
    }
    expected_factorial = {
        (fold, alpha, scenario, method, seed)
        for fold in FOLDS
        for alpha in ALPHAS
        for scenario in SCENARIOS
        for method in METHODS
        for seed in SEEDS
    }
    require(factorial_keys == expected_factorial, "Campaign factorial structure mismatch")

    conditions = read_csv(conditions_path)
    conditions_by_id = {int(row["condition_id"]): row for row in conditions}
    require(len(conditions_by_id) == 100, "Matched condition manifest should contain 100 unique conditions")

    expected_run_names = {run_name_from_row(row) for row in campaign_rows}
    actual_run_directories = {path.name for path in campaign_root.iterdir() if path.is_dir()}
    require(actual_run_directories == expected_run_names, "Campaign run-directory inventory differs from frozen matrix")

    run_level_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    binding_ledger_rows: list[dict[str, Any]] = []
    early_stop_rows: list[dict[str, Any]] = []

    arl_pressure_rows_verified = 0
    non_arl_blank_pressure_rows_verified = 0
    total_optimizer_steps = 0
    total_manifest_entries = 0
    run_bindings: dict[str, str] = {}

    for run_id in range(1, EXPECTED_RUNS + 1):
        campaign_row = campaign[run_id]
        run_root = campaign_root / run_name_from_row(campaign_row)
        require(run_root.is_dir(), f"Run directory missing: {run_root}")

        manifest_entries = verify_run_manifest(run_root)
        total_manifest_entries += manifest_entries

        complete = read_json(run_root / "RUN_COMPLETE.json")
        require(complete.get("status") == "SCIENTIFIC_RUN_COMPLETE", f"Run {run_id} completion status mismatch")
        result_binding = complete.get("run_result_binding_sha256")
        require(isinstance(result_binding, str), f"Run {run_id} has no result binding")
        complete_core = dict(complete)
        complete_core.pop("run_result_binding_sha256")
        require(canonical_sha256(complete_core) == result_binding, f"Run {run_id} result-binding replay failed")

        contract = complete.get("run_contract")
        require(isinstance(contract, dict), f"Run {run_id} contract is missing")
        require(contract.get("runner_id") == RUNNER_ID, f"Run {run_id} runner ID mismatch")
        require(contract.get("runner_sha256") == EXPECTED_RUNNER_SHA256, f"Run {run_id} runner SHA mismatch")
        require(
            contract.get("runner_build_binding_sha256") == EXPECTED_RUNNER_BUILD_BINDING,
            f"Run {run_id} runner build binding mismatch",
        )
        require(contract.get("scientific_run") is True, f"Run {run_id} is not marked scientific")

        for key, caster in (
            ("run_id", int),
            ("condition_id", int),
            ("outer_fold", int),
            ("scenario", str),
            ("method", str),
            ("fl_seed", int),
        ):
            require(caster(contract[key]) == caster(campaign_row[key]), f"Run {run_id} contract {key} mismatch")
        require(same_float(contract["alpha"], campaign_row["alpha"]), f"Run {run_id} contract alpha mismatch")

        config = read_json(run_root / "RUN_CONFIG.json")
        state = read_json(run_root / "RUN_STATE.json")
        require(config.get("scientific_run") is True, f"Run {run_id} config is not scientific")
        require(int(config.get("run_id", -1)) == run_id, f"Run {run_id} config ID mismatch")
        require(config.get("runner_sha256") == EXPECTED_RUNNER_SHA256, f"Run {run_id} config runner SHA mismatch")
        require(
            config.get("runner_build_binding_sha256") == EXPECTED_RUNNER_BUILD_BINDING,
            f"Run {run_id} config build binding mismatch",
        )
        require(int(config.get("total_rounds", -1)) == TOTAL_ROUNDS, f"Run {run_id} configured round count mismatch")
        require(int(config.get("clients_per_round", -1)) == 8, f"Run {run_id} clients-per-round mismatch")
        require(config.get("evaluation_rounds") == list(range(0, 101, 5)), f"Run {run_id} evaluation schedule mismatch")

        condition = conditions_by_id[int(campaign_row["condition_id"])]
        for key, caster in (
            ("outer_fold", int),
            ("scenario", str),
            ("fl_seed", int),
        ):
            require(caster(condition[key]) == caster(campaign_row[key]), f"Run {run_id} condition {key} mismatch")
        require(same_float(condition["alpha"], campaign_row["alpha"]), f"Run {run_id} condition alpha mismatch")
        require(str(config.get("attack")) == str(condition["attack"]), f"Run {run_id} attack mismatch")
        require(int(config.get("malicious_count", -1)) == int(condition["malicious_count"]), f"Run {run_id} malicious count mismatch")

        optimizer_steps = int(complete.get("scientific_optimizer_steps_accounted", -1))
        require(optimizer_steps > 0, f"Run {run_id} optimizer ledger is not positive")
        require(complete.get("scientific_training_started") is True, f"Run {run_id} training marker missing")
        require(complete.get("scientific_metrics_computed") is True, f"Run {run_id} metrics marker missing")
        require(state.get("status") == "COMPLETE", f"Run {run_id} state is not COMPLETE")
        require(int(state.get("completed_round", -1)) == TOTAL_ROUNDS, f"Run {run_id} state completed_round mismatch")
        require(
            int(state.get("scientific_optimizer_steps_accounted", -1)) == optimizer_steps,
            f"Run {run_id} optimizer ledger differs between completion and state",
        )
        require(state.get("run_result_binding_sha256") == result_binding, f"Run {run_id} state binding mismatch")

        metrics = complete.get("final_metrics_round_100")
        lifetime = complete.get("lifetime_metrics")
        require(isinstance(metrics, dict), f"Run {run_id} final metrics missing")
        require(isinstance(lifetime, dict), f"Run {run_id} lifetime metrics missing")

        loss = finite_number(metrics["loss"], f"run {run_id} loss")
        accuracy = finite_probability(metrics["accuracy"], f"run {run_id} accuracy")
        balanced_accuracy = finite_probability(metrics["balanced_accuracy"], f"run {run_id} balanced accuracy")
        macro_f1 = finite_probability(metrics["macro_f1"], f"run {run_id} macro-F1")
        per_class = metrics.get("per_class_f1")
        require(isinstance(per_class, list) and len(per_class) == EXPECTED_CLASSES, f"Run {run_id} per-class F1 count mismatch")
        per_class_values = [
            finite_probability(value, f"run {run_id} class {index} F1")
            for index, value in enumerate(per_class)
        ]

        early_stop_round = lifetime.get("early_stop_round")
        completed_training_rounds = TOTAL_ROUNDS if early_stop_round is None else int(early_stop_round) - 1
        require(0 <= completed_training_rounds <= TOTAL_ROUNDS, f"Run {run_id} completed training rounds invalid")

        evaluation = read_csv(run_root / "EVALUATION_METRICS.csv")
        require(len(evaluation) == 21, f"Run {run_id} evaluation row count mismatch")
        require([int(row["round"]) for row in evaluation] == list(range(0, 101, 5)), f"Run {run_id} evaluation chain mismatch")
        final_evaluation = evaluation[-1]
        require(same_float(final_evaluation["test_loss"], loss), f"Run {run_id} final loss differs from evaluation row 100")
        require(same_float(final_evaluation["test_accuracy"], accuracy), f"Run {run_id} final accuracy differs from evaluation row 100")
        require(
            same_float(final_evaluation["test_balanced_accuracy"], balanced_accuracy),
            f"Run {run_id} final balanced accuracy differs from evaluation row 100",
        )
        require(same_float(final_evaluation["test_macro_f1"], macro_f1), f"Run {run_id} final macro-F1 differs from evaluation row 100")
        require(int(final_evaluation["scientific_optimizer_steps_accounted"]) == optimizer_steps, f"Run {run_id} final evaluation optimizer ledger mismatch")

        progress = read_csv(run_root / "ROUND_PROGRESS.csv")
        selection = read_csv(run_root / "CLIENT_SELECTION.csv")
        attack_audit = read_csv(run_root / "ATTACK_AUDIT.csv")
        require(len(progress) == completed_training_rounds, f"Run {run_id} progress row count mismatch")
        require([int(row["round"]) for row in progress] == list(range(1, completed_training_rounds + 1)), f"Run {run_id} progress round chain mismatch")
        require(len(selection) == completed_training_rounds * 8, f"Run {run_id} client-selection row count mismatch")
        require(len(attack_audit) == completed_training_rounds * 8, f"Run {run_id} attack-audit row count mismatch")

        final_active_clients = int(lifetime["final_active_clients"])
        final_mean_energy = finite_number(lifetime["final_mean_residual_energy"], f"run {run_id} final mean energy")
        final_min_energy = finite_number(lifetime["final_min_residual_energy"], f"run {run_id} final minimum energy")
        total_energy = finite_number(lifetime["total_normalized_energy_consumed"], f"run {run_id} total energy")
        fairness = finite_probability(lifetime["jain_participation_fairness"], f"run {run_id} Jain fairness")

        run_row: dict[str, Any] = {
            "run_id": run_id,
            "condition_id": int(campaign_row["condition_id"]),
            "outer_fold": int(campaign_row["outer_fold"]),
            "outer_test_subject": int(config["outer_test_subject"]),
            "alpha": float(campaign_row["alpha"]),
            "scenario": str(campaign_row["scenario"]),
            "attack": str(config["attack"]),
            "malicious_count": int(config["malicious_count"]),
            "method": str(campaign_row["method"]),
            "fl_seed": int(campaign_row["fl_seed"]),
            "model_seed": int(config["model_seed"]),
            "random_schedule_seed": int(config["random_schedule_seed"]),
            "early_stop_round": "" if early_stop_round is None else int(early_stop_round),
            "completed_training_rounds": completed_training_rounds,
            "loss": loss,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
            "first_client_dropout_round": "" if lifetime.get("first_client_dropout_round") is None else int(lifetime["first_client_dropout_round"]),
            "active_client_lifetime_75pct": "" if lifetime.get("active_client_lifetime_75pct") is None else int(lifetime["active_client_lifetime_75pct"]),
            "active_client_lifetime_50pct": "" if lifetime.get("active_client_lifetime_50pct") is None else int(lifetime["active_client_lifetime_50pct"]),
            "final_active_clients": final_active_clients,
            "final_mean_residual_energy": final_mean_energy,
            "final_min_residual_energy": final_min_energy,
            "total_normalized_energy_consumed": total_energy,
            "jain_participation_fairness": fairness,
            "scientific_optimizer_steps_accounted": optimizer_steps,
            "run_result_binding_sha256": result_binding,
            "run_file_manifest_sha256": sha256_file(run_root / "RUN_FILE_SHA256.csv"),
            "run_manifest_entries": manifest_entries,
        }
        for class_index, class_f1 in enumerate(per_class_values):
            run_row[f"class_{class_index:02d}_f1"] = class_f1

        run_level_rows.append(run_row)
        run_bindings[str(run_id)] = result_binding
        total_optimizer_steps += optimizer_steps

        for class_index, class_f1 in enumerate(per_class_values):
            per_class_rows.append(
                {
                    "run_id": run_id,
                    "condition_id": int(campaign_row["condition_id"]),
                    "outer_fold": int(campaign_row["outer_fold"]),
                    "alpha": float(campaign_row["alpha"]),
                    "scenario": str(campaign_row["scenario"]),
                    "method": str(campaign_row["method"]),
                    "fl_seed": int(campaign_row["fl_seed"]),
                    "class_index": class_index,
                    "class_f1": class_f1,
                }
            )

        for row in evaluation:
            method = str(campaign_row["method"])
            pressure = optional_finite_number(
                row.get("global_pressure", ""),
                f"run {run_id} trajectory pressure",
            )
            if method == "arl_fl":
                require(
                    pressure is not None,
                    f"Run {run_id} arl_fl trajectory row {row.get('round')} has blank global_pressure",
                )
                require(
                    0.0 <= pressure <= 1.0,
                    f"Run {run_id} arl_fl trajectory pressure outside [0,1]: {pressure}",
                )
                arl_pressure_rows_verified += 1
            else:
                require(
                    pressure is None,
                    f"Run {run_id} non-arl trajectory row {row.get('round')} unexpectedly has global_pressure",
                )
                non_arl_blank_pressure_rows_verified += 1

            trajectory_rows.append(
                {
                    "run_id": run_id,
                    "condition_id": int(campaign_row["condition_id"]),
                    "outer_fold": int(campaign_row["outer_fold"]),
                    "alpha": float(campaign_row["alpha"]),
                    "scenario": str(campaign_row["scenario"]),
                    "method": method,
                    "fl_seed": int(campaign_row["fl_seed"]),
                    "round": int(row["round"]),
                    "test_loss": finite_number(row["test_loss"], f"run {run_id} trajectory loss"),
                    "test_accuracy": finite_probability(row["test_accuracy"], f"run {run_id} trajectory accuracy"),
                    "test_balanced_accuracy": finite_probability(
                        row["test_balanced_accuracy"], f"run {run_id} trajectory balanced accuracy"
                    ),
                    "test_macro_f1": finite_probability(row["test_macro_f1"], f"run {run_id} trajectory macro-F1"),
                    "active_clients": int(row["active_clients"]),
                    "mean_residual_energy": finite_number(row["mean_residual_energy"], f"run {run_id} trajectory mean energy"),
                    "min_residual_energy": finite_number(row["min_residual_energy"], f"run {run_id} trajectory minimum energy"),
                    "global_pressure": "" if pressure is None else pressure,
                    "scientific_optimizer_steps_accounted": int(row["scientific_optimizer_steps_accounted"]),
                }
            )

        binding_ledger_rows.append(
            {
                "run_id": run_id,
                "run_name": run_root.name,
                "scientific_optimizer_steps_accounted": optimizer_steps,
                "run_result_binding_sha256": result_binding,
                "run_file_manifest_sha256": sha256_file(run_root / "RUN_FILE_SHA256.csv"),
            }
        )

        if early_stop_round is not None:
            early_stop_rows.append(
                {
                    "run_id": run_id,
                    "outer_fold": int(campaign_row["outer_fold"]),
                    "alpha": float(campaign_row["alpha"]),
                    "scenario": str(campaign_row["scenario"]),
                    "method": str(campaign_row["method"]),
                    "fl_seed": int(campaign_row["fl_seed"]),
                    "early_stop_round": int(early_stop_round),
                    "completed_training_rounds": completed_training_rounds,
                    "run_result_binding_sha256": result_binding,
                }
            )

    require(len(run_level_rows) == EXPECTED_RUNS, "Run-level result row count mismatch")
    require(len(per_class_rows) == EXPECTED_RUNS * EXPECTED_CLASSES, "Per-class table row count mismatch")
    require(len(trajectory_rows) == EXPECTED_EVALUATION_ROWS, "Evaluation trajectory row count mismatch")
    require(arl_pressure_rows_verified == EXPECTED_ARL_PRESSURE_ROWS, "ARL global_pressure row count mismatch")
    require(non_arl_blank_pressure_rows_verified == EXPECTED_NON_ARL_BLANK_PRESSURE_ROWS, "Non-ARL blank global_pressure row count mismatch")
    require(len(run_bindings) == EXPECTED_RUNS, "Run-result binding ledger count mismatch")
    require(total_optimizer_steps == EXPECTED_ACCEPTED_OPTIMIZER_STEPS, "Full campaign optimizer ledger mismatch")

    grouped: defaultdict[tuple[float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_level_rows:
        grouped[(float(row["alpha"]), str(row["scenario"]), str(row["method"]))].append(row)

    descriptive_rows: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        for scenario in SCENARIOS:
            for method in METHODS:
                rows = grouped[(alpha, scenario, method)]
                require(len(rows) == 10, f"Descriptive aggregation expected n=10 for {alpha}/{scenario}/{method}")

                result: dict[str, Any] = {
                    "alpha": alpha,
                    "scenario": scenario,
                    "method": method,
                    "n_runs": len(rows),
                    "n_outer_folds": len({int(row["outer_fold"]) for row in rows}),
                    "n_fl_seeds": len({int(row["fl_seed"]) for row in rows}),
                    "early_stop_count": sum(1 for row in rows if row["early_stop_round"] != ""),
                }

                metric_names = (
                    "loss",
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "final_active_clients",
                    "final_mean_residual_energy",
                    "final_min_residual_energy",
                    "total_normalized_energy_consumed",
                    "jain_participation_fairness",
                    "scientific_optimizer_steps_accounted",
                )
                for metric_name in metric_names:
                    stats = summary_stats([float(row[metric_name]) for row in rows])
                    for stat_name, stat_value in stats.items():
                        result[f"{metric_name}_{stat_name}"] = stat_value

                descriptive_rows.append(result)

    require(len(descriptive_rows) == 60, "Descriptive summary row count mismatch")

    run_level_path = output_root / "PAMAP2_RUN_LEVEL_RESULTS_600.csv"
    per_class_path = output_root / "PAMAP2_PER_CLASS_F1_7200_ROWS.csv"
    trajectory_path = output_root / "PAMAP2_EVALUATION_TRAJECTORIES_12600_ROWS.csv"
    binding_ledger_path = output_root / "PAMAP2_RUN_RESULT_BINDING_LEDGER_600.csv"
    early_stop_path = output_root / "PAMAP2_EARLY_STOP_RUNS.csv"
    descriptive_path = output_root / "PAMAP2_DESCRIPTIVE_SUMMARY_60_CONDITIONS.csv"

    write_csv(run_level_path, run_level_rows)
    write_csv(per_class_path, per_class_rows)
    write_csv(trajectory_path, trajectory_rows)
    write_csv(binding_ledger_path, binding_ledger_rows)
    write_csv(
        early_stop_path,
        early_stop_rows,
        fieldnames=[
            "run_id",
            "outer_fold",
            "alpha",
            "scenario",
            "method",
            "fl_seed",
            "early_stop_round",
            "completed_training_rounds",
            "run_result_binding_sha256",
        ],
    )
    write_csv(descriptive_path, descriptive_rows)

    dictionary_path = output_root / "PAMAP2_POSTCAMPAIGN_DATA_DICTIONARY.txt"
    dictionary_lines = [
        "PAMAP2 POST-CAMPAIGN FREEZE DATA DICTIONARY",
        "=" * 78,
        "",
        "PAMAP2_RUN_LEVEL_RESULTS_600.csv",
        "  One row per accepted scientific run. Final predictive, lifetime, energy,",
        "  fairness, optimizer-ledger, per-class F1, and binding fields.",
        "",
        "PAMAP2_PER_CLASS_F1_7200_ROWS.csv",
        "  Long-format table: 600 runs x 12 activity classes.",
        "",
        "PAMAP2_EVALUATION_TRAJECTORIES_12600_ROWS.csv",
        "  Long-format evaluation table: 600 runs x 21 frozen evaluation rounds.",
        "  global_pressure is numeric and required only for arl_fl; it is intentionally",
        "  blank for fedavg, fedprox, random_trimmed_mean, fedle_adapted and tea_fl.",
        "",
        "PAMAP2_RUN_RESULT_BINDING_LEDGER_600.csv",
        "  Compact scientific result-binding and optimizer-step ledger.",
        "",
        "PAMAP2_EARLY_STOP_RUNS.csv",
        "  Protocol-defined early stopping events. These runs remain COMPLETE and",
        "  scientifically accepted; completed_training_rounds = early_stop_round - 1.",
        "",
        "PAMAP2_DESCRIPTIVE_SUMMARY_60_CONDITIONS.csv",
        "  Descriptive aggregation by alpha x scenario x method. Each row has n=10",
        "  paired observations from 5 outer folds x 2 FL seeds. No inferential",
        "  significance test is performed by Gate-149R.",
    ]
    dictionary_path.write_text("\n".join(dictionary_lines) + "\n", encoding="utf-8", newline="\n")

    audit = {
        "gate_id": GATE_ID,
        "status": "PASS",
        "scope": SCOPE,
        "generated_utc": utc_now(),
        "preserved_previous_output": preserved_output,
        "repairs_gate149_package_sha256": REPAIRS_GATE149_PACKAGE_SHA256,
        "repair_reason": REPAIR_REASON,
        "source_chain": chain,
        "runner_id": RUNNER_ID,
        "runner_sha256": EXPECTED_RUNNER_SHA256,
        "runner_build_binding_sha256": EXPECTED_RUNNER_BUILD_BINDING,
        "campaign_matrix_sha256": sha256_file(campaign_matrix_path),
        "matched_condition_manifest_sha256": sha256_file(conditions_path),
        "scientific_runs_verified": len(run_level_rows),
        "run_manifest_entries_verified": total_manifest_entries,
        "evaluation_rows_verified": len(trajectory_rows),
        "arl_global_pressure_rows_verified": arl_pressure_rows_verified,
        "non_arl_blank_global_pressure_rows_verified": non_arl_blank_pressure_rows_verified,
        "per_class_rows_verified": len(per_class_rows),
        "early_stop_runs": len(early_stop_rows),
        "accepted_scientific_optimizer_steps": total_optimizer_steps,
        "run_result_bindings": run_bindings,
        "descriptive_summary_rows": len(descriptive_rows),
        "inferential_statistical_testing_executed": False,
        "scientific_training_executed_by_gate149r": False,
        "scientific_optimizer_steps_executed_by_gate149r": 0,
        "new_scientific_runs_started_by_gate149r": 0,
    }
    audit_path = output_root / "GATE149R_AUDIT.json"
    write_json(audit_path, audit)

    report_lines = [
        "PAMAP2 GATE-149R REPAIRED POST-CAMPAIGN FREEZE AND DESCRIPTIVE AGGREGATION",
        "=" * 78,
        "",
        "STATUS",
        "-" * 78,
        "PASS",
        "",
        "ACCEPTED SCIENTIFIC CAMPAIGN",
        "-" * 78,
        f"Scientific runs verified: {len(run_level_rows)}/600",
        f"Run manifest entries verified: {total_manifest_entries}",
        f"Evaluation rows verified: {len(trajectory_rows)}",
        f"ARL numeric global_pressure rows verified: {arl_pressure_rows_verified}",
        f"Non-ARL blank global_pressure rows verified: {non_arl_blank_pressure_rows_verified}",
        f"Per-class F1 rows verified: {len(per_class_rows)}",
        f"Accepted scientific optimizer steps: {total_optimizer_steps}",
        f"Protocol-defined early-stop runs: {len(early_stop_rows)}",
        "Missing run IDs: NONE",
        "Duplicate run IDs: NONE",
        "",
        "BINDING CHAIN",
        "-" * 78,
        f"Gate-148 final binding: {EXPECTED_GATE148_FINAL_BINDING}",
        f"Master authorization binding: {EXPECTED_MASTER_AUTHORIZATION_BINDING}",
        f"Final accepted group binding: {EXPECTED_FINAL_GROUP_BINDING}",
        f"Autonomous campaign final binding: {EXPECTED_AUTONOMOUS_FINAL_BINDING}",
        "Autonomous conditional group audits: 72/72 PASS",
        "All 600 run-result bindings replayed: PASS",
        "All 600 run-file manifests replayed: PASS",
        "",
        "OUTPUT TABLES",
        "-" * 78,
        "Run-level result rows: 600",
        "Per-class F1 rows: 7200",
        "Evaluation trajectory rows: 12600",
        "Descriptive summary rows: 60",
        "",
        "SCIENTIFIC BOUNDARY",
        "-" * 78,
        "Scientific training executed by Gate-149R: NO",
        "Scientific optimizer steps executed by Gate-149R: 0",
        "New scientific runs started by Gate-149R: 0",
        "Inferential statistical testing executed by Gate-149R: NO",
    ]
    report_path = output_root / "GATE149R_REPORT.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8", newline="\n")

    output_hashes = {
        "run_level_results_sha256": sha256_file(run_level_path),
        "per_class_f1_sha256": sha256_file(per_class_path),
        "evaluation_trajectories_sha256": sha256_file(trajectory_path),
        "run_result_binding_ledger_sha256": sha256_file(binding_ledger_path),
        "early_stop_runs_sha256": sha256_file(early_stop_path),
        "descriptive_summary_sha256": sha256_file(descriptive_path),
        "data_dictionary_sha256": sha256_file(dictionary_path),
        "audit_sha256": sha256_file(audit_path),
        "report_sha256": sha256_file(report_path),
    }

    binding_core = {
        "gate_id": GATE_ID,
        "status": "PASS",
        "scope": SCOPE,
        "repairs_gate149_package_sha256": REPAIRS_GATE149_PACKAGE_SHA256,
        "repair_reason": REPAIR_REASON,
        "gate148_final_binding_sha256": EXPECTED_GATE148_FINAL_BINDING,
        "master_authorization_binding_sha256": EXPECTED_MASTER_AUTHORIZATION_BINDING,
        "controller_sha256": EXPECTED_CONTROLLER_SHA256,
        "final_group_binding_sha256": EXPECTED_FINAL_GROUP_BINDING,
        "autonomous_campaign_final_binding_sha256": EXPECTED_AUTONOMOUS_FINAL_BINDING,
        "runner_sha256": EXPECTED_RUNNER_SHA256,
        "runner_build_binding_sha256": EXPECTED_RUNNER_BUILD_BINDING,
        "scientific_runs_verified": len(run_level_rows),
        "accepted_scientific_optimizer_steps": total_optimizer_steps,
        "early_stop_runs": len(early_stop_rows),
        "evaluation_rows_verified": len(trajectory_rows),
        "arl_global_pressure_rows_verified": arl_pressure_rows_verified,
        "non_arl_blank_global_pressure_rows_verified": non_arl_blank_pressure_rows_verified,
        "descriptive_summary_rows": len(descriptive_rows),
        **output_hashes,
        "scientific_training_executed_by_gate149r": False,
        "scientific_optimizer_steps_executed_by_gate149r": 0,
        "new_scientific_runs_started_by_gate149r": 0,
        "inferential_statistical_testing_executed_by_gate149r": False,
    }
    final_binding = dict(binding_core)
    final_binding["gate149r_final_binding_sha256"] = canonical_sha256(binding_core)
    final_binding_path = output_root / "GATE149R_FINAL_BINDING.json"
    write_json(final_binding_path, final_binding)

    manifest_path = output_root / "MANIFEST_SHA256.csv"
    manifest_subjects = sorted(
        [path for path in output_root.iterdir() if path.is_file() and path.name != manifest_path.name],
        key=lambda path: path.name,
    )
    manifest_rows = [
        {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in manifest_subjects
    ]
    write_csv(manifest_path, manifest_rows)

    print("=" * 78)
    print("GATE149R_PASS")
    print("=" * 78)
    print("Scientific runs verified: 600/600")
    print(f"Accepted scientific optimizer steps: {total_optimizer_steps}")
    print(f"Run manifest entries verified: {total_manifest_entries}")
    print(f"Evaluation rows verified: {len(trajectory_rows)}")
    print(f"ARL numeric global_pressure rows verified: {arl_pressure_rows_verified}")
    print(f"Non-ARL blank global_pressure rows verified: {non_arl_blank_pressure_rows_verified}")
    print(f"Protocol-defined early-stop runs: {len(early_stop_rows)}")
    print("Autonomous conditional group audits: 72/72 PASS")
    print("Scientific training executed by Gate-149R: NO")
    print("Scientific optimizer steps executed by Gate-149R: 0")
    print("New scientific runs started by Gate-149R: 0")
    print("Inferential statistical testing executed by Gate-149R: NO")
    print(f"Gate-149R final binding SHA256: {final_binding['gate149r_final_binding_sha256']}")
    print(f"Report: {report_path}")
    print(f"Run-level results: {run_level_path}")
    print(f"Descriptive summary: {descriptive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
