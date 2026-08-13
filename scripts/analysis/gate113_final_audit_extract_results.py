from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

GATE_ID = "GATE113"
EXPECTED_RUN_COUNT = 180
EXPECTED_ROUNDS = 20
EXPECTED_STEPS_PER_RUN = 36000
EXPECTED_FAILED_PARTIAL_STEPS = 1801
EXPECTED_TECHNICAL_STEPS = 825
EXPECTED_ACCEPTED_STEPS = EXPECTED_RUN_COUNT * EXPECTED_STEPS_PER_RUN
EXPECTED_TOTAL_SCIENTIFIC_STEPS = EXPECTED_ACCEPTED_STEPS + EXPECTED_FAILED_PARTIAL_STEPS

METHODS = [
    "FEDAVG",
    "FEDPROX",
    "RANDOM_TRIMMED_MEAN",
    "FEDLE_ADAPTED",
    "TEA_FL",
    "ARL_FL",
]
SCENARIOS = ["CLEAN", "SIGNFLIP_MU0P2", "SIGNFLIP_MU0P4"]
CONFIGS = [
    ("alpha_0p1_seed_123", 0.1, 123),
    ("alpha_0p1_seed_42", 0.1, 42),
    ("alpha_0p1_seed_456", 0.1, 456),
    ("alpha_0p1_seed_789", 0.1, 789),
    ("alpha_0p1_seed_999", 0.1, 999),
    ("alpha_1p0_seed_123", 1.0, 123),
    ("alpha_1p0_seed_42", 1.0, 42),
    ("alpha_1p0_seed_456", 1.0, 456),
    ("alpha_1p0_seed_789", 1.0, 789),
    ("alpha_1p0_seed_999", 1.0, 999),
]

RUNNER_V4_SHA = "F0D3CE89AA9A9B14FF3084E913A5301732CD8DE0854A95E1F3E865B9FD2BC688"
RUNNER_V4_BUILD = "333EDF80A9B1272CC745B76F70F8DCB47D005332028E2581BD24994810EEFF52"
RUNNER_V5V6_SHA = "38F2047D95802F625FE9C597EAC46407F26A3A2874902D6BC3515F798D850CA1"
RUNNER_V5_BUILD = "4093D90C4BE87B4008237EB7CA16B0EBC9F0DDD08EF42EC443C8A5E494187274"
RUNNER_V6R1_BUILD = "D9BE0D1FF33073102A32B2092E8A5ACD0FDD3A36FEBB9142D88DEAF81E64D611"
CONTEXT_BRIDGE_SHA = "A745D35AD96CA98BF251963ECFB18A53D3CAFE00C3441E49A46B63097B9DF7D6"
GATE110R_FINAL_BINDING = "0C28C742B71130899F30BF10685A9DE58C4CCDC0D1A6AD2A166E4BAABB8771A8"
GATE111_FINAL_BINDING = "E0541D9009D08BCBE591F5E99BC76733FDE3D68024A80C81DCCD77B7741942AF"
GATE112_FINAL_BINDING = "EBA8926B48B6A744AEE0BC7C23197C8EE14899362165CA0D79AD49B2697F78D9"
GATE112_AUTH_BINDING = "8BBE6B5E1F53E60AC0649676C732EE5CEF9D31DC691165D1CF04C3475350D1A1"

REQUIRED_FILES = [
    "BEST_STATE.pt",
    "ROUND_CHECKPOINT.pt",
    "ROUND_VALIDATION_METRICS.csv",
    "RUN_COMPLETE.json",
    "RUN_CONTRACT.json",
    "RUN_STATE.json",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def same_number(a: Any, b: Any) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-12)


def compare_metric_dict(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key in ["loss", "accuracy", "macro_f1", "balanced_accuracy"]:
        require(key in actual and key in expected, f"{label}: missing metric {key}")
        require(same_number(actual[key], expected[key]), f"{label}: mismatch for {key}")
    require(int(actual["round"]) == int(expected["round"]), f"{label}: round mismatch")


def expected_contract_for_index(run_index: int) -> tuple[str, float, int, str, str]:
    require(1 <= run_index <= EXPECTED_RUN_COUNT, f"Invalid run index: {run_index}")
    zero = run_index - 1
    config_pos = zero // (len(METHODS) * len(SCENARIOS))
    inside = zero % (len(METHODS) * len(SCENARIOS))
    method_pos = inside // len(SCENARIOS)
    scenario_pos = inside % len(SCENARIOS)
    config_id, alpha, seed = CONFIGS[config_pos]
    return config_id, alpha, seed, METHODS[method_pos], SCENARIOS[scenario_pos]


def expected_runner_pair(run_index: int) -> tuple[str, str, str]:
    if run_index in {1, 2, 3, 19}:
        return RUNNER_V4_SHA, RUNNER_V4_BUILD, "RUNNER_V4_INITIAL_ACCEPTED"
    if run_index in {4, 5, 6}:
        return RUNNER_V5V6_SHA, RUNNER_V5_BUILD, "RUNNER_V5_FEDPROX_REPAIR_ACCEPTED"
    return RUNNER_V5V6_SHA, RUNNER_V6R1_BUILD, "RUNNER_V6R1_CONTEXT_BRIDGE_ACCEPTED"


def mean_std(values: list[float]) -> tuple[float, float]:
    require(values, "Cannot summarize an empty metric group")
    mean_value = statistics.fmean(values)
    std_value = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean_value, std_value


def format_float(value: float) -> str:
    return f"{value:.12f}"


def build_summary_rows(master_rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in master_rows:
        groups[tuple(row[key] for key in keys)].append(row)

    output: list[dict[str, Any]] = []
    for group_key in sorted(groups, key=lambda item: tuple(str(x) for x in item)):
        rows = groups[group_key]
        record = {key: value for key, value in zip(keys, group_key)}
        record["n"] = len(rows)
        for metric in ["test_accuracy", "test_macro_f1", "test_balanced_accuracy", "test_loss"]:
            values = [float(row[metric]) for row in rows]
            mean_value, std_value = mean_std(values)
            record[f"{metric}_mean"] = format_float(mean_value)
            record[f"{metric}_std"] = format_float(std_value)
            record[f"{metric}_min"] = format_float(min(values))
            record[f"{metric}_max"] = format_float(max(values))
        record["best_round_mean"] = format_float(statistics.fmean(float(r["best_round"]) for r in rows))
        output.append(record)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--gate-script", required=True)
    parser.add_argument("--analysis-script", required=True)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    gate_script = Path(args.gate_script).resolve()
    analysis_script = Path(args.analysis_script).resolve()
    cic_root = project_root / "outputs" / "ciciot2023"
    campaign_root = cic_root / "scientific_campaign_180runs_v1"
    output_root = cic_root / "scientific_campaign_final_audit_v1"

    require(campaign_root.is_dir(), f"Campaign root not found: {campaign_root}")
    require(gate_script.is_file(), f"Gate script not found: {gate_script}")
    require(analysis_script.is_file(), f"Analysis script not found: {analysis_script}")
    require(not output_root.exists(), f"Output already exists and was not preserved: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    runner_path = cic_root / "scientific_campaign_runner_v6r1" / "run_ciciot2023_scientific_campaign.py"
    bridge_path = cic_root / "scientific_campaign_runner_v6r1" / "ciciot2023_scientific_context_bridge_v1r1.py"
    require(runner_path.is_file(), f"Frozen runner missing: {runner_path}")
    require(bridge_path.is_file(), f"Context bridge missing: {bridge_path}")
    require(sha256_file(runner_path) == RUNNER_V5V6_SHA, "Frozen runner SHA256 mismatch")
    require(sha256_file(bridge_path) == CONTEXT_BRIDGE_SHA, "Context bridge SHA256 mismatch")

    completion_path = cic_root / "scientific_campaign_remaining_169_authorization_v1" / "SCIENTIFIC_CAMPAIGN_180RUNS_COMPLETE.json"
    gate112_path = cic_root / "scientific_campaign_remaining_169_authorization_v1" / "GATE112_FINAL_BINDING.json"
    gate110r_path = cic_root / "scientific_campaign_context_bridge_repair_v1r1" / "GATE110R_FINAL_BINDING.json"
    require(completion_path.is_file(), f"Final completion record missing: {completion_path}")
    require(gate112_path.is_file(), f"Gate-112 binding missing: {gate112_path}")
    require(gate110r_path.is_file(), f"Gate-110R binding missing: {gate110r_path}")

    completion = read_json(completion_path)
    require(completion.get("status") == "CICIOT2023_180_RUN_SCIENTIFIC_CAMPAIGN_COMPLETE", "Unexpected final completion status")
    require(int(completion.get("complete_scientific_runs", -1)) == EXPECTED_RUN_COUNT, "Final completion run count mismatch")
    require(int(completion.get("accepted_complete_run_optimizer_steps", -1)) == EXPECTED_ACCEPTED_STEPS, "Accepted step ledger mismatch")
    require(int(completion.get("preserved_failed_partial_optimizer_steps", -1)) == EXPECTED_FAILED_PARTIAL_STEPS, "Failed-partial step ledger mismatch")
    require(int(completion.get("total_scientific_optimizer_steps_including_failed_partials", -1)) == EXPECTED_TOTAL_SCIENTIFIC_STEPS, "Total scientific step ledger mismatch")
    require(int(completion.get("technical_optimizer_steps", -1)) == EXPECTED_TECHNICAL_STEPS, "Technical step ledger mismatch")
    require(completion.get("pamap2_600_run_campaign_started") is False, "PAMAP2 campaign boundary violated")

    gate110r = read_json(gate110r_path)
    require(gate110r.get("gate110r_final_binding_sha256") == GATE110R_FINAL_BINDING, "Gate-110R final binding mismatch")
    require(int(gate110r.get("failed_run4_partial_steps", -1)) == 1, "Run-4 partial ledger mismatch")
    require(int(gate110r.get("failed_run7_partial_steps", -1)) == 1800, "Run-7 partial ledger mismatch")

    gate112 = read_json(gate112_path)
    require(gate112.get("gate110r_final_binding_sha256") == GATE110R_FINAL_BINDING, "Gate-112 to Gate-110R binding mismatch")
    require(gate112.get("gate111_final_binding_sha256") == GATE111_FINAL_BINDING, "Gate-112 to Gate-111 binding mismatch")
    require(gate112.get("gate112_final_binding_sha256") == GATE112_FINAL_BINDING, "Gate-112 final binding mismatch")
    require(gate112.get("remaining_campaign_authorization_binding_sha256") == GATE112_AUTH_BINDING, "Gate-112 authorization binding mismatch")

    run_dirs = sorted(
        [path for path in campaign_root.iterdir() if path.is_dir() and path.name.startswith("run_")],
        key=lambda path: path.name,
    )
    require(len(run_dirs) == EXPECTED_RUN_COUNT, f"Expected 180 run directories, found {len(run_dirs)}")

    master_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    seen_bindings: set[str] = set()
    matrix_counter: Counter[tuple[str, str, str]] = Counter()
    runner_counter: Counter[str] = Counter()
    run_state_counter: Counter[str] = Counter()

    for run_dir in run_dirs:
        for filename in REQUIRED_FILES:
            require((run_dir / filename).is_file(), f"Missing required file: {run_dir.name}/{filename}")

        complete_path = run_dir / "RUN_COMPLETE.json"
        contract_path = run_dir / "RUN_CONTRACT.json"
        state_path = run_dir / "RUN_STATE.json"
        metrics_path = run_dir / "ROUND_VALIDATION_METRICS.csv"

        complete = read_json(complete_path)
        contract = read_json(contract_path)
        state = read_json(state_path)
        metric_rows = read_csv(metrics_path)

        require(complete.get("status") == "SCIENTIFIC_RUN_COMPLETE", f"Incomplete run marker: {run_dir.name}")
        require(complete.get("scientific_training_started") is True, f"Training-start flag mismatch: {run_dir.name}")
        require(complete.get("scientific_metrics_computed") is True, f"Metrics flag mismatch: {run_dir.name}")
        require(int(complete.get("scientific_optimizer_steps_expected", -1)) == EXPECTED_STEPS_PER_RUN, f"Step count mismatch: {run_dir.name}")
        require(complete.get("run_contract") == contract, f"RUN_CONTRACT mismatch: {run_dir.name}")

        binding = str(complete.get("run_result_binding_sha256", ""))
        require(len(binding) == 64 and all(ch in "0123456789ABCDEF" for ch in binding), f"Invalid result binding: {run_dir.name}")
        replay_payload = dict(complete)
        replay_payload.pop("run_result_binding_sha256", None)
        require(canonical_sha(replay_payload) == binding, f"Result-binding replay failed: {run_dir.name}")
        require(binding not in seen_bindings, f"Duplicate result binding: {binding}")
        seen_bindings.add(binding)

        run_index = int(contract["run_index"])
        require(run_index not in seen_indices, f"Duplicate run index: {run_index}")
        seen_indices.add(run_index)
        expected_config, alpha, seed, expected_method, expected_scenario = expected_contract_for_index(run_index)
        require(contract["config_id"] == expected_config, f"Config mismatch at run {run_index}")
        require(int(contract["experimental_seed"]) == seed, f"Seed mismatch at run {run_index}")
        require(contract["method_id"] == expected_method, f"Method mismatch at run {run_index}")
        require(contract["scenario_id"] == expected_scenario, f"Scenario mismatch at run {run_index}")
        require(contract.get("scientific_run") is True, f"Scientific-run flag mismatch at run {run_index}")

        expected_name = f"run_{run_index:03d}__{expected_config}__{expected_method}__{expected_scenario}"
        require(run_dir.name == expected_name, f"Run-directory name mismatch: expected {expected_name}, found {run_dir.name}")

        expected_runner, expected_build, runner_generation = expected_runner_pair(run_index)
        require(contract["runner_sha256"] == expected_runner, f"Runner SHA mismatch at run {run_index}")
        require(contract["build_binding_sha256"] == expected_build, f"Build binding mismatch at run {run_index}")
        runner_counter[runner_generation] += 1

        require(len(metric_rows) == EXPECTED_ROUNDS, f"Expected 20 metric rows at run {run_index}")
        rounds = [int(row["Round"]) for row in metric_rows]
        require(rounds == list(range(1, EXPECTED_ROUNDS + 1)), f"Round chain mismatch at run {run_index}")
        for row in metric_rows:
            for metric in ["loss", "accuracy", "macro_f1", "balanced_accuracy"]:
                value = float(row[metric])
                require(math.isfinite(value), f"Non-finite {metric} at run {run_index}, round {row['Round']}")

        best_row = max(metric_rows, key=lambda row: float(row["macro_f1"]))
        expected_best = {
            "loss": float(best_row["loss"]),
            "accuracy": float(best_row["accuracy"]),
            "macro_f1": float(best_row["macro_f1"]),
            "balanced_accuracy": float(best_row["balanced_accuracy"]),
            "round": int(best_row["Round"]),
        }
        compare_metric_dict(complete["best_validation"], expected_best, f"Best validation run {run_index}")
        require(int(state.get("completed_round", -1)) == EXPECTED_ROUNDS, f"RUN_STATE completed round mismatch at run {run_index}")
        require(state.get("status") in {"IN_PROGRESS", "SCIENTIFIC_RUN_COMPLETE"}, f"Unexpected RUN_STATE status at run {run_index}")
        run_state_counter[str(state.get("status"))] += 1
        compare_metric_dict(state["best"], complete["best_validation"], f"RUN_STATE best run {run_index}")

        for filename in REQUIRED_FILES:
            file_path = run_dir / filename
            file_rows.append({
                "run_index": run_index,
                "run_directory": run_dir.name,
                "filename": filename,
                "size_bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            })

        test = complete["test_metrics"]
        validation = complete["best_validation"]
        for metric_name, value in test.items():
            require(math.isfinite(float(value)), f"Non-finite test metric {metric_name} at run {run_index}")

        matrix_counter[(expected_config, expected_method, expected_scenario)] += 1
        master_rows.append({
            "run_index": run_index,
            "run_directory": run_dir.name,
            "config_id": expected_config,
            "alpha": alpha,
            "experimental_seed": seed,
            "method_id": expected_method,
            "scenario_id": expected_scenario,
            "runner_generation": runner_generation,
            "runner_sha256": contract["runner_sha256"],
            "build_binding_sha256": contract["build_binding_sha256"],
            "best_round": int(validation["round"]),
            "validation_loss": format_float(float(validation["loss"])),
            "validation_accuracy": format_float(float(validation["accuracy"])),
            "validation_macro_f1": format_float(float(validation["macro_f1"])),
            "validation_balanced_accuracy": format_float(float(validation["balanced_accuracy"])),
            "test_loss": format_float(float(test["loss"])),
            "test_accuracy": format_float(float(test["accuracy"])),
            "test_macro_f1": format_float(float(test["macro_f1"])),
            "test_balanced_accuracy": format_float(float(test["balanced_accuracy"])),
            "scientific_optimizer_steps": EXPECTED_STEPS_PER_RUN,
            "run_result_binding_sha256": binding,
            "run_state_status": state.get("status"),
        })

    require(seen_indices == set(range(1, EXPECTED_RUN_COUNT + 1)), "Run-index coverage is not exactly 1..180")
    expected_matrix = {(config, method, scenario) for config, _, _ in CONFIGS for method in METHODS for scenario in SCENARIOS}
    require(set(matrix_counter) == expected_matrix, "Experimental matrix coverage mismatch")
    require(all(count == 1 for count in matrix_counter.values()), "Experimental matrix contains duplicate cells")

    master_rows.sort(key=lambda row: int(row["run_index"]))
    master_fields = list(master_rows[0].keys())
    master_path = output_root / "MASTER_RUN_RESULTS.csv"
    write_csv(master_path, master_rows, master_fields)

    file_rows.sort(key=lambda row: (int(row["run_index"]), row["filename"]))
    file_hash_path = output_root / "RUN_FILE_SHA256.csv"
    write_csv(file_hash_path, file_rows, ["run_index", "run_directory", "filename", "size_bytes", "sha256"])

    alpha_summary = build_summary_rows(master_rows, ["alpha", "method_id", "scenario_id"])
    alpha_summary_path = output_root / "DESCRIPTIVE_SUMMARY_BY_ALPHA_METHOD_SCENARIO.csv"
    write_csv(alpha_summary_path, alpha_summary, list(alpha_summary[0].keys()))

    overall_summary = build_summary_rows(master_rows, ["method_id", "scenario_id"])
    overall_summary_path = output_root / "DESCRIPTIVE_SUMMARY_BY_METHOD_SCENARIO.csv"
    write_csv(overall_summary_path, overall_summary, list(overall_summary[0].keys()))

    audit_details = {
        "status": "PASS",
        "gate_id": GATE_ID,
        "scope": "FINAL_180_RUN_POST_AUDIT_AND_RESULTS_EXTRACTION_ONLY",
        "scientific_training_executed_by_gate113": False,
        "optimizer_steps_executed_by_gate113": 0,
        "pamap2_600_run_campaign_started": False,
        "campaign_root": str(campaign_root),
        "complete_scientific_runs": EXPECTED_RUN_COUNT,
        "experimental_matrix_cells": len(matrix_counter),
        "unique_run_result_bindings": len(seen_bindings),
        "six_file_inventories_verified": EXPECTED_RUN_COUNT,
        "twenty_round_metric_chains_verified": EXPECTED_RUN_COUNT,
        "result_binding_replays_verified": EXPECTED_RUN_COUNT,
        "accepted_complete_run_optimizer_steps": EXPECTED_ACCEPTED_STEPS,
        "preserved_failed_partial_optimizer_steps": EXPECTED_FAILED_PARTIAL_STEPS,
        "total_scientific_optimizer_steps_including_failed_partials": EXPECTED_TOTAL_SCIENTIFIC_STEPS,
        "technical_optimizer_steps": EXPECTED_TECHNICAL_STEPS,
        "runner_generation_counts": dict(sorted(runner_counter.items())),
        "run_state_status_counts": dict(sorted(run_state_counter.items())),
        "note_run_state": "RUN_STATE.json remains an in-run checkpoint record. IN_PROGRESS with completed_round=20 is accepted only when RUN_COMPLETE.json and its canonical result binding pass.",
        "frozen_runner_sha256": sha256_file(runner_path),
        "context_bridge_sha256": sha256_file(bridge_path),
        "gate110r_final_binding_sha256": GATE110R_FINAL_BINDING,
        "gate111_final_binding_sha256": GATE111_FINAL_BINDING,
        "gate112_final_binding_sha256": GATE112_FINAL_BINDING,
        "gate112_remaining_authorization_binding_sha256": GATE112_AUTH_BINDING,
        "final_completion_record_sha256": sha256_file(completion_path),
        "gate_script_sha256": sha256_file(gate_script),
        "analysis_script_sha256": sha256_file(analysis_script),
    }
    audit_json_path = output_root / "GATE113_AUDIT.json"
    write_json(audit_json_path, audit_details)

    output_hashes = {
        "MASTER_RUN_RESULTS.csv": sha256_file(master_path),
        "RUN_FILE_SHA256.csv": sha256_file(file_hash_path),
        "DESCRIPTIVE_SUMMARY_BY_ALPHA_METHOD_SCENARIO.csv": sha256_file(alpha_summary_path),
        "DESCRIPTIVE_SUMMARY_BY_METHOD_SCENARIO.csv": sha256_file(overall_summary_path),
        "GATE113_AUDIT.json": sha256_file(audit_json_path),
    }
    final_binding = {
        **audit_details,
        "output_file_sha256": output_hashes,
    }
    final_binding["gate113_final_binding_sha256"] = canonical_sha(final_binding)
    final_binding_path = output_root / "GATE113_FINAL_BINDING.json"
    write_json(final_binding_path, final_binding)

    report_lines = [
        "CICIoT2023 GATE-113 FINAL 180-RUN POST-AUDIT",
        "=" * 78,
        "",
        "STATUS",
        "-" * 78,
        "PASS",
        "",
        "SCOPE",
        "-" * 78,
        "FINAL_180_RUN_POST_AUDIT_AND_RESULTS_EXTRACTION_ONLY",
        "Scientific training executed by Gate-113: NO",
        "Optimizer steps executed by Gate-113: 0",
        "PAMAP2 600-run campaign started: NO",
        "",
        "FINAL CAMPAIGN AUDIT",
        "-" * 78,
        f"Complete scientific runs: {EXPECTED_RUN_COUNT}/180",
        f"Experimental matrix cells verified: {len(matrix_counter)}/180",
        f"Six-file inventories verified: {EXPECTED_RUN_COUNT}/180",
        f"Twenty-round metric chains verified: {EXPECTED_RUN_COUNT}/180",
        f"Canonical result-binding replays verified: {EXPECTED_RUN_COUNT}/180",
        f"Unique result bindings: {len(seen_bindings)}/180",
        "",
        "RUNNER GENERATIONS",
        "-" * 78,
    ]
    for key, value in sorted(runner_counter.items()):
        report_lines.append(f"{key}: {value} runs")
    report_lines.extend([
        "",
        "LEDGER",
        "-" * 78,
        f"Accepted complete-run scientific optimizer steps: {EXPECTED_ACCEPTED_STEPS}",
        f"Preserved failed-partial scientific optimizer steps: {EXPECTED_FAILED_PARTIAL_STEPS}",
        f"Total scientific optimizer steps including failed partials: {EXPECTED_TOTAL_SCIENTIFIC_STEPS}",
        f"Technical optimizer steps: {EXPECTED_TECHNICAL_STEPS}",
        "",
        "RUN_STATE NOTE",
        "-" * 78,
        f"Status counts: {dict(sorted(run_state_counter.items()))}",
        "RUN_STATE.json is an in-run checkpoint record. Its IN_PROGRESS value is not",
        "treated as final status; acceptance is based on RUN_COMPLETE.json, completed_round=20,",
        "the 20-round validation chain, six-file inventory, and canonical result-binding replay.",
        "No scientific output was modified.",
        "",
        "EXTRACTED OUTPUTS",
        "-" * 78,
        "MASTER_RUN_RESULTS.csv: 180 rows",
        "RUN_FILE_SHA256.csv: 1080 rows",
        "DESCRIPTIVE_SUMMARY_BY_ALPHA_METHOD_SCENARIO.csv: 36 rows",
        "DESCRIPTIVE_SUMMARY_BY_METHOD_SCENARIO.csv: 18 rows",
        "GATE113_AUDIT.json",
        "GATE113_FINAL_BINDING.json",
        "",
        "FINAL BINDING",
        "-" * 78,
        final_binding["gate113_final_binding_sha256"],
        "",
    ])
    report_path = output_root / "GATE113_REPORT.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8", newline="\n")

    manifest_rows: list[dict[str, Any]] = []
    for path in sorted(output_root.iterdir(), key=lambda p: p.name):
        if path.is_file() and path.name != "MANIFEST_SHA256.csv":
            manifest_rows.append({
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    manifest_path = output_root / "MANIFEST_SHA256.csv"
    write_csv(manifest_path, manifest_rows, ["filename", "size_bytes", "sha256"])

    print("=" * 78)
    print("GATE-113 PASS")
    print("=" * 78)
    print(f"Complete scientific runs audited: {EXPECTED_RUN_COUNT}/180")
    print(f"Experimental matrix cells verified: {len(matrix_counter)}/180")
    print(f"Six-file inventories verified: {EXPECTED_RUN_COUNT}/180")
    print(f"Twenty-round metric chains verified: {EXPECTED_RUN_COUNT}/180")
    print(f"Canonical result-binding replays verified: {EXPECTED_RUN_COUNT}/180")
    print(f"Unique result bindings: {len(seen_bindings)}/180")
    print(f"Accepted complete-run scientific optimizer steps: {EXPECTED_ACCEPTED_STEPS}")
    print(f"Preserved failed-partial scientific optimizer steps: {EXPECTED_FAILED_PARTIAL_STEPS}")
    print(f"Total scientific optimizer steps including failed partials: {EXPECTED_TOTAL_SCIENTIFIC_STEPS}")
    print(f"Technical optimizer steps: {EXPECTED_TECHNICAL_STEPS}")
    print(f"Runner generation counts: {dict(sorted(runner_counter.items()))}")
    print(f"RUN_STATE status counts: {dict(sorted(run_state_counter.items()))}")
    print(f"Gate-113 final binding SHA256: {final_binding['gate113_final_binding_sha256']}")
    print("Scientific training executed by Gate-113: NO")
    print("Optimizer steps executed by Gate-113: 0")
    print("PAMAP2 600-run campaign started: NO")
    print("")
    print("Gate-113 report:")
    print(f"  {report_path}")
    print("Master results:")
    print(f"  {master_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"GATE-113 ERROR: {exc}", file=sys.stderr)
        raise
