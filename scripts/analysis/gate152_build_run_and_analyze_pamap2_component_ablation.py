from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import traceback
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata

GATE_ID = "GATE-152"
SCOPE = "PAMAP2_PRESPECIFIED_COMPONENT_ABLATION_CAMPAIGN_AND_PAIRED_ANALYSIS"
EXPECTED_GATE151_FINAL_BINDING = "3ECB12E02890E852C0C02BA50F3F6F29976853B1E5B101B5F09095EE377A7DBB"
EXPECTED_GATE151_BINDING_FILE_SHA256 = "CA6C79D0184DD73E3DB472862942E4BBABCDE2883EBC298003CCE1FA38769264"
EXPECTED_GATE149R_RUN_LEVEL_SHA256 = "25B41F60D9EFAB10BA92ACEBE9FA36432DB29D70C0220F87206FAB0BE5309931"
EXPECTED_ORIGINAL_RUNNER_ID = "PAMAP2_FROZEN_600_RUN_SCIENTIFIC_CAMPAIGN_RUNNER_V1"
EXPECTED_ORIGINAL_RUNNER_SHA256 = "7B43B8023DFC7D80C1BE409FA0EEF21E7BBA3024996FDB7A342638E5EE9B0108"
EXPECTED_ORIGINAL_RUNNER_BUILD_BINDING = "F26B90B821B08C5CF031A6A205FB3DDAD313BF769DF390E2995BE8AD6B1BDBFA"
EXPECTED_ORIGINAL_RUNNER_BINDING_FILE_SHA256 = "AA2E7771736E10EFDD97A681B9886EE8A38E6090D9632DF9CDC25770CB3837ED"
ABLATION_RUNNER_ID = "PAMAP2_FROZEN_COMPONENT_ABLATION_RUNNER_V1"
TOTAL_ABLATION_RUNS = 240
TOTAL_ROUNDS = 100
EXPECTED_EVALUATION_ROUNDS = list(range(0, 101, 5))
EXPECTED_CLASSES = 12
CLIENTS_PER_ROUND = 8
FOLDS = (1, 2, 3, 4, 5)
SEEDS = (123, 456)
BOOTSTRAP_REPLICATES = 5000
ALPHA_LEVEL = 0.05

SELECTED_CONDITIONS: tuple[tuple[float, str, str], ...] = (
    (0.1, "clean", "No attack under strong non-IID heterogeneity"),
    (1.0, "labelflip_mu0p4", "Strong label-flip condition where proposed methods remain competitive"),
    (0.1, "signflip_mu0p2", "Moderate sign-flip under strong non-IID heterogeneity"),
    (1.0, "signflip_mu0p2", "Moderate sign-flip under milder heterogeneity"),
    (0.1, "signflip_mu0p4", "Strong sign-flip under strong non-IID heterogeneity"),
    (1.0, "signflip_mu0p4", "Strong sign-flip under milder heterogeneity"),
)

VARIANTS: tuple[tuple[str, str, str, str], ...] = (
    (
        "tea_trust_only",
        "tea_fl",
        "TEA-FL trust-only",
        "Full TEA-FL minus this variant estimates the energy-awareness contribution",
    ),
    (
        "tea_energy_only",
        "tea_fl",
        "TEA-FL energy-only",
        "Full TEA-FL minus this variant estimates the trust contribution",
    ),
    (
        "arl_no_energy",
        "arl_fl",
        "ARL-FL without energy-aware selection",
        "Full ARL-FL minus this variant estimates the energy-aware selection contribution",
    ),
    (
        "arl_no_pressure",
        "arl_fl",
        "ARL-FL without adaptive global pressure",
        "Full ARL-FL minus this variant estimates the adaptive-pressure contribution",
    ),
)

VARIANT_LABELS = {item[0]: item[2] for item in VARIANTS}
VARIANT_COMPONENT = {
    "tea_trust_only": "TEA energy awareness",
    "tea_energy_only": "TEA trust mechanism",
    "arl_no_energy": "ARL energy-aware selection",
    "arl_no_pressure": "ARL adaptive global pressure",
}
METHOD_LABELS = {"tea_fl": "TEA-FL", "arl_fl": "ARL-FL"}
METRICS: dict[str, dict[str, Any]] = {
    "macro_f1": {
        "label": "Macro-F1",
        "direction": 1,
        "endpoint_role": "primary_effectiveness",
    },
    "total_normalized_energy_consumed": {
        "label": "Total normalized energy consumed",
        "direction": -1,
        "endpoint_role": "primary_sustainability",
    },
    "final_active_clients": {
        "label": "Final active clients",
        "direction": 1,
        "endpoint_role": "secondary_retention",
    },
    "jain_participation_fairness": {
        "label": "Jain participation fairness",
        "direction": 1,
        "endpoint_role": "secondary_fairness",
    },
}
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
    require(path.is_file(), f"Missing JSON: {path}")
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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8", newline="\n")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def replay_binding(
    path: Path,
    final_field: str,
    expected_value: str | None = None,
) -> dict[str, Any]:
    value = read_json(path)
    observed = value.get(final_field)
    require(isinstance(observed, str), f"Missing binding field {final_field}: {path}")
    if expected_value is not None:
        require(observed == expected_value, f"Unexpected binding value: {path}")
    core = dict(value)
    core.pop(final_field)
    require(canonical_sha256(core) == observed, f"Canonical binding replay failed: {path}")
    return value


def verify_csv_manifest(manifest_path: Path, content_root: Path) -> int:
    rows = read_csv_rows(manifest_path)
    require(rows, f"Empty manifest: {manifest_path}")
    for row in rows:
        path = content_root / str(row["filename"])
        require(path.is_file(), f"Manifest file missing: {path}")
        require(int(row["size_bytes"]) == path.stat().st_size, f"Manifest size mismatch: {path}")
        require(
            str(row["sha256"]).upper() == sha256_file(path),
            f"Manifest SHA mismatch: {path}",
        )
    return len(rows)


def finite_number(value: Any, label: str) -> float:
    number = float(value)
    require(math.isfinite(number), f"{label} is not finite")
    return number


def finite_probability(value: Any, label: str) -> float:
    number = finite_number(value, label)
    require(0.0 <= number <= 1.0, f"{label} outside [0,1]")
    return number


def alpha_token(alpha: float) -> str:
    if math.isclose(alpha, 0.1):
        return "alpha0p1"
    if math.isclose(alpha, 1.0):
        return "alpha1p0"
    raise RuntimeError(f"Unsupported alpha: {alpha}")


def run_name(row: dict[str, Any]) -> str:
    return (
        f"ablation_run_{int(row['ablation_run_id']):03d}"
        f"__fold{int(row['outer_fold'])}"
        f"__{alpha_token(float(row['alpha']))}"
        f"__{row['scenario']}"
        f"__{row['variant']}"
        f"__seed{int(row['fl_seed'])}"
    )


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    n = len(values)
    order = np.argsort(values)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank_index, original_index in enumerate(order):
        candidate = (n - rank_index) * values[original_index]
        running = max(running, candidate)
        adjusted[original_index] = min(1.0, running)
    return adjusted


def exact_signed_rank(differences: np.ndarray) -> dict[str, float]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    nonzero = values[np.abs(values) > 1e-15]
    n = len(nonzero)
    if n == 0:
        return {
            "n_nonzero": 0,
            "w_plus": 0.0,
            "w_minus": 0.0,
            "statistic_min": 0.0,
            "p_exact_two_sided": 1.0,
            "rank_biserial": 0.0,
        }
    require(n <= 20, f"Exact signed-rank enumeration exceeds supported n=20: {n}")
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = nonzero > 0
    w_plus = float(ranks[positive].sum())
    w_minus = float(ranks[~positive].sum())
    total_rank = float(ranks.sum())

    subset_sums = np.empty(1 << n, dtype=float)
    for mask in range(1 << n):
        value = 0.0
        for bit in range(n):
            if mask & (1 << bit):
                value += float(ranks[bit])
        subset_sums[mask] = value

    lower = float(np.mean(subset_sums <= w_plus + 1e-12))
    upper = float(np.mean(subset_sums >= w_plus - 1e-12))
    p_value = min(1.0, 2.0 * min(lower, upper))
    rank_biserial = (w_plus - w_minus) / total_rank if total_rank else 0.0
    return {
        "n_nonzero": int(n),
        "w_plus": w_plus,
        "w_minus": w_minus,
        "statistic_min": min(w_plus, w_minus),
        "p_exact_two_sided": p_value,
        "rank_biserial": rank_biserial,
    }

def hodges_lehmann_paired(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    walsh = []
    for index, left in enumerate(values):
        for right in values[index:]:
            walsh.append((left + right) / 2.0)
    return float(np.median(np.asarray(walsh, dtype=float)))


def deterministic_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int(int.from_bytes(digest[:8], "big") % (2**31 - 1))


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    require(len(array) > 0, "Bootstrap input is empty")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = None
    try:
        handle = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(handle, f"pid={os.getpid()}\nstarted_utc={utc_now()}\n".encode("utf-8"))
        os.close(handle)
        handle = None
        yield
    finally:
        if handle is not None:
            os.close(handle)
        if path.exists():
            path.unlink()


def verify_gate151(project_root: Path) -> dict[str, Any]:
    gate151_root = project_root / "outputs" / "federated" / "pamap2" / "article_analysis_v1"
    binding_path = gate151_root / "GATE151_FINAL_BINDING.json"
    require(binding_path.is_file(), f"Gate-151 binding missing: {binding_path}")
    require(
        sha256_file(binding_path) == EXPECTED_GATE151_BINDING_FILE_SHA256,
        "Gate-151 binding file SHA mismatch",
    )
    binding = replay_binding(
        binding_path,
        "gate151_final_binding_sha256",
        EXPECTED_GATE151_FINAL_BINDING,
    )
    require(binding.get("status") == "PASS", "Gate-151 is not PASS")
    require(int(binding.get("scientific_runs_verified", -1)) == 600, "Gate-151 run count mismatch")
    require(int(binding.get("accepted_scientific_optimizer_steps", -1)) == 3401459, "Gate-151 optimizer ledger mismatch")
    return binding


def verify_original_runner(project_root: Path) -> tuple[Path, Path, dict[str, Any]]:
    runner_root = project_root / "outputs" / "federated" / "pamap2" / "scientific_campaign_runner_v1"
    runner_path = runner_root / "run_pamap2_scientific_campaign.py"
    binding_path = runner_root / "PAMAP2_SCIENTIFIC_RUNNER_BINDING.json"
    require(runner_path.is_file(), f"Original runner missing: {runner_path}")
    require(binding_path.is_file(), f"Original runner binding missing: {binding_path}")
    require(sha256_file(runner_path) == EXPECTED_ORIGINAL_RUNNER_SHA256, "Original runner SHA mismatch")
    require(
        sha256_file(binding_path) == EXPECTED_ORIGINAL_RUNNER_BINDING_FILE_SHA256,
        "Original runner binding file SHA mismatch",
    )
    binding = replay_binding(
        binding_path,
        "runner_build_binding_sha256",
        EXPECTED_ORIGINAL_RUNNER_BUILD_BINDING,
    )
    require(binding.get("runner_id") == EXPECTED_ORIGINAL_RUNNER_ID, "Original runner ID mismatch")
    require(binding.get("runner_sha256") == EXPECTED_ORIGINAL_RUNNER_SHA256, "Original runner binding SHA mismatch")
    return runner_path, binding_path, binding


def frozen_paths(project_root: Path) -> dict[str, Path]:
    freeze_root = project_root / "outputs" / "protocols" / "pamap2_source_interface_freeze_v1r2"
    source_root = freeze_root / "frozen_source_provenance"
    input_root = freeze_root / "frozen_input_provenance"
    return {
        "freeze_root": freeze_root,
        "source_root": source_root,
        "input_root": input_root,
        "source_manifest": freeze_root / "SOURCE_MANIFEST_SHA256.csv",
        "input_manifest": freeze_root / "FROZEN_INPUT_MANIFEST_SHA256.csv",
        "campaign_matrix": input_root / "campaign_matrix_600_runs.csv",
        "conditions": input_root / "matched_condition_manifest.csv",
    }


def build_design_matrix(project_root: Path) -> pd.DataFrame:
    paths = frozen_paths(project_root)
    source_count = verify_csv_manifest(paths["source_manifest"], paths["source_root"])
    input_count = verify_csv_manifest(paths["input_manifest"], paths["input_root"])
    require(source_count == 10, f"Expected 10 frozen source files, found {source_count}")
    require(input_count == 12, f"Expected 12 frozen input files, found {input_count}")

    campaign = pd.read_csv(paths["campaign_matrix"])
    conditions = pd.read_csv(paths["conditions"])
    require(len(campaign) == 600, "Original campaign matrix does not have 600 rows")
    require(sorted(campaign["run_id"].astype(int).tolist()) == list(range(1, 601)), "Original run IDs mismatch")

    rows: list[dict[str, Any]] = []
    design_condition_id = 0
    ablation_run_id = 0
    for alpha, scenario, rationale in SELECTED_CONDITIONS:
        design_condition_id += 1
        for fold in FOLDS:
            for seed in SEEDS:
                condition_candidates = conditions[
                    (conditions["outer_fold"].astype(int) == fold)
                    & np.isclose(conditions["alpha"].astype(float), alpha)
                    & (conditions["scenario"].astype(str) == scenario)
                    & (conditions["fl_seed"].astype(int) == seed)
                ]
                require(len(condition_candidates) == 1, f"Expected one frozen condition for {alpha}/{scenario}/fold{fold}/seed{seed}")
                condition = condition_candidates.iloc[0]
                source_condition_id = int(condition["condition_id"])
                for variant, method_family, variant_label, component_rationale in VARIANTS:
                    reference_candidates = campaign[
                        (campaign["outer_fold"].astype(int) == fold)
                        & np.isclose(campaign["alpha"].astype(float), alpha)
                        & (campaign["scenario"].astype(str) == scenario)
                        & (campaign["fl_seed"].astype(int) == seed)
                        & (campaign["method"].astype(str) == method_family)
                    ]
                    require(len(reference_candidates) == 1, "Expected one matched full-method reference run")
                    reference = reference_candidates.iloc[0]
                    require(int(reference["condition_id"]) == source_condition_id, "Reference condition ID mismatch")
                    ablation_run_id += 1
                    rows.append(
                        {
                            "ablation_run_id": ablation_run_id,
                            "design_condition_id": design_condition_id,
                            "reference_run_id": int(reference["run_id"]),
                            "source_condition_id": source_condition_id,
                            "outer_fold": fold,
                            "alpha": float(alpha),
                            "scenario": scenario,
                            "attack": str(condition["attack"]),
                            "malicious_count": int(condition["malicious_count"]),
                            "method_family": method_family,
                            "variant": variant,
                            "variant_label": variant_label,
                            "component_isolated": VARIANT_COMPONENT[variant],
                            "fl_seed": seed,
                            "model_seed": int(condition["model_seed"]),
                            "random_schedule_seed": int(condition["random_schedule_seed"]),
                            "condition_rationale": rationale,
                            "component_rationale": component_rationale,
                        }
                    )
    frame = pd.DataFrame(rows)
    require(len(frame) == TOTAL_ABLATION_RUNS, f"Expected {TOTAL_ABLATION_RUNS} ablation rows")
    require(frame["ablation_run_id"].tolist() == list(range(1, TOTAL_ABLATION_RUNS + 1)), "Ablation run IDs mismatch")
    require(frame.groupby(["alpha", "scenario", "variant"]).size().eq(10).all(), "Each condition/variant must contain 10 paired blocks")
    require(frame["reference_run_id"].nunique() == 120, "Expected 120 unique full-method reference runs")
    return frame


def build_authorization(
    project_root: Path,
    runner_source: Path,
    python_exe: Path,
) -> dict[str, Any]:
    pamap_root = project_root / "outputs" / "federated" / "pamap2"
    authorization_root = pamap_root / "ablation_campaign_authorization_v1"
    runner_root = pamap_root / "ablation_campaign_runner_v1"
    matrix_path = authorization_root / "PAMAP2_ABLATION_CAMPAIGN_MATRIX_240.csv"
    build_binding_path = authorization_root / "GATE152_BUILD_FINAL_BINDING.json"
    runner_path = runner_root / "run_pamap2_component_ablation.py"
    runner_binding_path = runner_root / "PAMAP2_COMPONENT_ABLATION_RUNNER_BINDING.json"

    verify_gate151(project_root)
    verify_original_runner(project_root)

    freeze_result = (
        pamap_root
        / "postcampaign_freeze_v1"
        / "PAMAP2_RUN_LEVEL_RESULTS_600.csv"
    )
    require(freeze_result.is_file(), f"Gate-149R run-level freeze missing: {freeze_result}")
    require(sha256_file(freeze_result) == EXPECTED_GATE149R_RUN_LEVEL_SHA256, "Gate-149R run-level results SHA mismatch")

    expected_runner_source_sha = sha256_file(runner_source)

    if authorization_root.exists() or runner_root.exists():
        require(authorization_root.is_dir(), "Ablation authorization root is not a directory")
        require(runner_root.is_dir(), "Ablation runner root is not a directory")
        build_binding = replay_binding(
            build_binding_path,
            "gate152_build_final_binding_sha256",
        )
        require(build_binding.get("status") == "PASS", "Existing Gate-152 build is not PASS")
        require(build_binding.get("gate151_final_binding_sha256") == EXPECTED_GATE151_FINAL_BINDING, "Existing build Gate-151 binding mismatch")
        require(build_binding.get("ablation_runner_source_sha256") == expected_runner_source_sha, "Existing build runner-source SHA mismatch")
        require(runner_path.is_file(), "Existing ablation runner missing")
        require(runner_binding_path.is_file(), "Existing ablation runner binding missing")
        require(sha256_file(runner_path) == build_binding["ablation_runner_sha256"], "Existing ablation runner SHA mismatch")
        replay_binding(
            runner_binding_path,
            "runner_build_binding_sha256",
            build_binding["ablation_runner_build_binding_sha256"],
        )
        require(matrix_path.is_file(), "Existing ablation matrix missing")
        require(sha256_file(matrix_path) == build_binding["ablation_matrix_sha256"], "Existing ablation matrix SHA mismatch")
        return build_binding

    authorization_root.mkdir(parents=True, exist_ok=False)
    runner_root.mkdir(parents=True, exist_ok=False)

    design = build_design_matrix(project_root)
    design.to_csv(matrix_path, index=False)
    matrix_sha = sha256_file(matrix_path)

    shutil.copy2(runner_source, runner_path)
    runner_sha = sha256_file(runner_path)
    require(runner_sha == expected_runner_source_sha, "Runner copy SHA mismatch")

    original_runner_path, original_binding_path, original_binding = verify_original_runner(project_root)
    paths = frozen_paths(project_root)

    runner_binding_core = {
        "runner_id": ABLATION_RUNNER_ID,
        "status": "FROZEN_BEFORE_COMPONENT_ABLATION_TRAINING",
        "scope": "PAMAP2_COMPONENT_ABLATION_240_RUNS_ONLY",
        "runner_sha256": runner_sha,
        "gate151_final_binding_sha256": EXPECTED_GATE151_FINAL_BINDING,
        "original_scientific_runner_sha256": EXPECTED_ORIGINAL_RUNNER_SHA256,
        "original_scientific_runner_build_binding_sha256": EXPECTED_ORIGINAL_RUNNER_BUILD_BINDING,
        "original_runner_binding_file_sha256": sha256_file(original_binding_path),
        "original_campaign_matrix_sha256": sha256_file(paths["campaign_matrix"]),
        "original_condition_manifest_sha256": sha256_file(paths["conditions"]),
        "frozen_source_manifest_sha256": sha256_file(paths["source_manifest"]),
        "frozen_input_manifest_sha256": sha256_file(paths["input_manifest"]),
        "ablation_matrix_sha256": matrix_sha,
        "total_ablation_runs": TOTAL_ABLATION_RUNS,
        "selected_conditions": [
            {"alpha": alpha, "scenario": scenario, "rationale": rationale}
            for alpha, scenario, rationale in SELECTED_CONDITIONS
        ],
        "variants": [
            {
                "variant": variant,
                "method_family": family,
                "label": label,
                "rationale": rationale,
            }
            for variant, family, label, rationale in VARIANTS
        ],
        "scientific_training_executed_during_build": False,
        "scientific_optimizer_steps_executed_during_build": 0,
    }
    runner_binding = dict(runner_binding_core)
    runner_binding["runner_build_binding_sha256"] = canonical_sha256(runner_binding_core)
    write_json(runner_binding_path, runner_binding)

    contract = subprocess.run(
        [
            str(python_exe),
            str(runner_path),
            "--project-root",
            str(project_root),
            "--binding-json",
            str(runner_binding_path),
            "--contract-check",
        ],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (authorization_root / "ABLATION_RUNNER_CONTRACT_CHECK_STDOUT.txt").write_text(
        contract.stdout,
        encoding="utf-8",
        newline="\n",
    )
    (authorization_root / "ABLATION_RUNNER_CONTRACT_CHECK_STDERR.txt").write_text(
        contract.stderr,
        encoding="utf-8",
        newline="\n",
    )
    require(contract.returncode == 0, "Ablation runner contract check failed")

    authorization_manifest_rows: list[dict[str, Any]] = []
    for row in design.to_dict(orient="records"):
        run_id = int(row["ablation_run_id"])
        authorization_core = {
            "status": "AUTHORIZED",
            "authorization_scope": "ONE_PAMAP2_COMPONENT_ABLATION_RUN_ONLY",
            "gate151_final_binding_sha256": EXPECTED_GATE151_FINAL_BINDING,
            "runner_id": ABLATION_RUNNER_ID,
            "runner_sha256": runner_sha,
            "runner_build_binding_sha256": runner_binding["runner_build_binding_sha256"],
            "ablation_matrix_sha256": matrix_sha,
            "run_id": run_id,
            "variant": str(row["variant"]),
            "method_family": str(row["method_family"]),
            "reference_full_method_run_id": int(row["reference_run_id"]),
            "outer_fold": int(row["outer_fold"]),
            "alpha": float(row["alpha"]),
            "scenario": str(row["scenario"]),
            "fl_seed": int(row["fl_seed"]),
            "full_campaign_authorized": False,
        }
        authorization = dict(authorization_core)
        authorization["authorization_binding_sha256"] = canonical_sha256(authorization_core)
        auth_path = authorization_root / f"ABLATION_RUN_{run_id:03d}_AUTHORIZATION.json"
        write_json(auth_path, authorization)
        authorization_manifest_rows.append(
            {
                "ablation_run_id": run_id,
                "filename": auth_path.name,
                "size_bytes": auth_path.stat().st_size,
                "sha256": sha256_file(auth_path),
                "authorization_binding_sha256": authorization["authorization_binding_sha256"],
            }
        )
    auth_manifest_path = authorization_root / "ABLATION_AUTHORIZATION_MANIFEST_240.csv"
    write_csv_rows(auth_manifest_path, authorization_manifest_rows)

    report_lines = [
        "PAMAP2 GATE-152 COMPONENT ABLATION DESIGN AND RUNNER BUILD",
        "=" * 78,
        "",
        "STATUS",
        "-" * 78,
        "PASS",
        "",
        "SCIENTIFIC DESIGN",
        "-" * 78,
        f"Prespecified conditions: {len(SELECTED_CONDITIONS)}",
        f"Component-ablation variants: {len(VARIANTS)}",
        f"Matched fold x seed blocks per condition/variant: 10",
        f"Authorized ablation runs: {TOTAL_ABLATION_RUNS}",
        "Full TEA-FL and ARL-FL references are reused from the frozen 600-run campaign.",
        "",
        "SCIENTIFIC BOUNDARY",
        "-" * 78,
        "Scientific training executed during Gate-152 build: NO",
        "Scientific optimizer steps executed during Gate-152 build: 0",
        "CICIoT2023 scientific training started: NO",
        "",
    ]
    build_report_path = authorization_root / "GATE152_BUILD_REPORT.txt"
    build_report_path.write_text("\n".join(report_lines), encoding="utf-8", newline="\n")

    build_audit = {
        "gate_id": GATE_ID,
        "status": "PASS",
        "scope": SCOPE,
        "generated_utc": utc_now(),
        "gate151_final_binding_sha256": EXPECTED_GATE151_FINAL_BINDING,
        "original_runner_sha256": sha256_file(original_runner_path),
        "original_runner_build_binding_sha256": original_binding["runner_build_binding_sha256"],
        "ablation_runner_source_sha256": expected_runner_source_sha,
        "ablation_runner_sha256": runner_sha,
        "ablation_runner_build_binding_sha256": runner_binding["runner_build_binding_sha256"],
        "ablation_matrix_sha256": matrix_sha,
        "authorization_manifest_sha256": sha256_file(auth_manifest_path),
        "authorized_ablation_runs": TOTAL_ABLATION_RUNS,
        "prespecified_conditions": len(SELECTED_CONDITIONS),
        "variants": len(VARIANTS),
        "scientific_training_executed_during_build": False,
        "scientific_optimizer_steps_executed_during_build": 0,
        "ciciot2023_scientific_training_started": False,
    }
    build_audit_path = authorization_root / "GATE152_BUILD_AUDIT.json"
    write_json(build_audit_path, build_audit)

    build_binding_core = {
        "gate_id": GATE_ID,
        "status": "PASS",
        "scope": "PAMAP2_COMPONENT_ABLATION_BUILD_AND_PROGRESSIVE_AUTHORIZATION",
        "gate151_final_binding_sha256": EXPECTED_GATE151_FINAL_BINDING,
        "ablation_runner_source_sha256": expected_runner_source_sha,
        "ablation_runner_sha256": runner_sha,
        "ablation_runner_build_binding_sha256": runner_binding["runner_build_binding_sha256"],
        "ablation_runner_binding_file_sha256": sha256_file(runner_binding_path),
        "ablation_matrix_sha256": matrix_sha,
        "authorization_manifest_sha256": sha256_file(auth_manifest_path),
        "build_report_sha256": sha256_file(build_report_path),
        "build_audit_sha256": sha256_file(build_audit_path),
        "authorized_ablation_runs": TOTAL_ABLATION_RUNS,
        "full_600_run_campaign_results_modified": False,
        "scientific_training_executed_during_build": False,
        "scientific_optimizer_steps_executed_during_build": 0,
    }
    build_binding = dict(build_binding_core)
    build_binding["gate152_build_final_binding_sha256"] = canonical_sha256(build_binding_core)
    write_json(build_binding_path, build_binding)
    return build_binding


def audit_run(
    row: dict[str, Any],
    *,
    run_root: Path,
    runner_sha: str,
    runner_build_binding: str,
) -> dict[str, Any]:
    for name in REQUIRED_RUN_FILES:
        require((run_root / name).is_file(), f"Missing run file {name}: {run_root}")

    complete = read_json(run_root / "RUN_COMPLETE.json")
    require(complete.get("status") == "SCIENTIFIC_RUN_COMPLETE", f"Run not complete: {run_root}")
    result_binding = complete.get("run_result_binding_sha256")
    require(isinstance(result_binding, str), "Run result binding missing")
    complete_core = dict(complete)
    complete_core.pop("run_result_binding_sha256")
    require(canonical_sha256(complete_core) == result_binding, "Run result binding replay failed")

    contract = complete.get("run_contract", {})
    require(contract.get("runner_id") == ABLATION_RUNNER_ID, "Ablation runner ID mismatch")
    require(contract.get("runner_sha256") == runner_sha, "Ablation runner SHA mismatch")
    require(contract.get("runner_build_binding_sha256") == runner_build_binding, "Ablation runner build binding mismatch")
    require(int(contract.get("run_id", -1)) == int(row["ablation_run_id"]), "Ablation run ID mismatch")
    require(contract.get("variant") == str(row["variant"]), "Ablation variant mismatch")
    require(contract.get("method_family") == str(row["method_family"]), "Ablation method family mismatch")
    require(int(contract.get("reference_full_method_run_id", -1)) == int(row["reference_run_id"]), "Reference run mismatch")
    require(int(contract.get("outer_fold", -1)) == int(row["outer_fold"]), "Fold mismatch")
    require(math.isclose(float(contract.get("alpha")), float(row["alpha"])), "Alpha mismatch")
    require(contract.get("scenario") == str(row["scenario"]), "Scenario mismatch")
    require(int(contract.get("fl_seed", -1)) == int(row["fl_seed"]), "Seed mismatch")

    state = read_json(run_root / "RUN_STATE.json")
    require(state.get("status") == "COMPLETE", "Run state not COMPLETE")
    require(int(state.get("completed_round", -1)) == TOTAL_ROUNDS, "Run state completed round mismatch")
    require(state.get("run_result_binding_sha256") == result_binding, "Run-state result binding mismatch")

    manifest_rows = read_csv_rows(run_root / "RUN_FILE_SHA256.csv")
    require(manifest_rows, "Empty run manifest")
    manifest_names = {str(item["filename"]) for item in manifest_rows}
    require("RUN_FILE_SHA256.csv" not in manifest_names, "Run manifest must not self-reference")
    for item in manifest_rows:
        path = run_root / str(item["filename"])
        require(path.is_file(), f"Manifest member missing: {path}")
        require(path.stat().st_size == int(item["size_bytes"]), f"Manifest size mismatch: {path}")
        require(sha256_file(path) == str(item["sha256"]).upper(), f"Manifest SHA mismatch: {path}")

    evaluations = pd.read_csv(run_root / "EVALUATION_METRICS.csv")
    require(len(evaluations) == len(EXPECTED_EVALUATION_ROUNDS), "Evaluation row count mismatch")
    require(evaluations["round"].astype(int).tolist() == EXPECTED_EVALUATION_ROUNDS, "Evaluation rounds mismatch")
    for column in ["test_accuracy", "test_balanced_accuracy", "test_macro_f1"]:
        values = evaluations[column].astype(float).to_numpy()
        require(np.isfinite(values).all(), f"Non-finite evaluation metric: {column}")
        require(((values >= 0.0) & (values <= 1.0)).all(), f"Evaluation metric outside [0,1]: {column}")
    final_eval = evaluations.iloc[-1]
    final_metrics = complete["final_metrics_round_100"]
    require(math.isclose(float(final_eval["test_macro_f1"]), float(final_metrics["macro_f1"]), rel_tol=0, abs_tol=1e-12), "Final macro-F1 mismatch")
    require(math.isclose(float(final_eval["test_balanced_accuracy"]), float(final_metrics["balanced_accuracy"]), rel_tol=0, abs_tol=1e-12), "Final balanced accuracy mismatch")
    per_class = final_metrics["per_class_f1"]
    require(isinstance(per_class, list) and len(per_class) == EXPECTED_CLASSES, "Per-class F1 length mismatch")
    for index, value in enumerate(per_class):
        finite_probability(value, f"class {index} F1")

    progress = pd.read_csv(run_root / "ROUND_PROGRESS.csv")
    early_stop_round = complete["lifetime_metrics"].get("early_stop_round")
    expected_progress_rows = TOTAL_ROUNDS if early_stop_round is None else int(early_stop_round) - 1
    require(len(progress) == expected_progress_rows, "Progress-row count inconsistent with early stop")
    if len(progress):
        require(progress["round"].astype(int).tolist() == list(range(1, len(progress) + 1)), "Progress rounds mismatch")
        final_cumulative = int(progress.iloc[-1]["cumulative_optimizer_steps_accounted"])
    else:
        final_cumulative = 0

    selections = pd.read_csv(run_root / "CLIENT_SELECTION.csv")
    attacks = pd.read_csv(run_root / "ATTACK_AUDIT.csv")
    require(len(selections) == CLIENTS_PER_ROUND * len(progress), "Selection-row count mismatch")
    require(len(attacks) == len(selections), "Attack audit row count mismatch")
    optimizer_steps = int(complete["scientific_optimizer_steps_accounted"])
    selection_optimizer_steps = int(selections["optimizer_steps_accounted"].astype(int).sum()) if len(selections) else 0
    require(selection_optimizer_steps == optimizer_steps, "Optimizer-step selection ledger mismatch")
    require(final_cumulative == optimizer_steps, "Optimizer-step progress ledger mismatch")
    require(int(state["scientific_optimizer_steps_accounted"]) == optimizer_steps, "Optimizer-step state ledger mismatch")

    lifetime = complete["lifetime_metrics"]
    run_manifest_binding = canonical_sha256(
        [
            {
                "filename": str(item["filename"]),
                "size_bytes": int(item["size_bytes"]),
                "sha256": str(item["sha256"]).upper(),
            }
            for item in manifest_rows
        ]
    )
    return {
        "ablation_run_id": int(row["ablation_run_id"]),
        "reference_run_id": int(row["reference_run_id"]),
        "source_condition_id": int(row["source_condition_id"]),
        "outer_fold": int(row["outer_fold"]),
        "alpha": float(row["alpha"]),
        "scenario": str(row["scenario"]),
        "method_family": str(row["method_family"]),
        "variant": str(row["variant"]),
        "variant_label": VARIANT_LABELS[str(row["variant"])],
        "component_isolated": VARIANT_COMPONENT[str(row["variant"])],
        "fl_seed": int(row["fl_seed"]),
        "early_stop_round": early_stop_round,
        "loss": float(final_metrics["loss"]),
        "accuracy": float(final_metrics["accuracy"]),
        "balanced_accuracy": float(final_metrics["balanced_accuracy"]),
        "macro_f1": float(final_metrics["macro_f1"]),
        "final_active_clients": int(lifetime["final_active_clients"]),
        "final_mean_residual_energy": float(lifetime["final_mean_residual_energy"]),
        "final_min_residual_energy": float(lifetime["final_min_residual_energy"]),
        "total_normalized_energy_consumed": float(lifetime["total_normalized_energy_consumed"]),
        "jain_participation_fairness": float(lifetime["jain_participation_fairness"]),
        "scientific_optimizer_steps_accounted": optimizer_steps,
        "run_result_binding_sha256": result_binding,
        "run_manifest_binding_sha256": run_manifest_binding,
        "run_manifest_entries": len(manifest_rows),
    }


def execute_campaign(
    project_root: Path,
    build_binding: dict[str, Any],
    python_exe: Path,
) -> list[dict[str, Any]]:
    pamap_root = project_root / "outputs" / "federated" / "pamap2"
    authorization_root = pamap_root / "ablation_campaign_authorization_v1"
    runner_root = pamap_root / "ablation_campaign_runner_v1"
    campaign_root = pamap_root / "ablation_campaign_v1"
    control_root = pamap_root / "ablation_campaign_controller_v1"
    matrix_path = authorization_root / "PAMAP2_ABLATION_CAMPAIGN_MATRIX_240.csv"
    runner_path = runner_root / "run_pamap2_component_ablation.py"
    runner_binding_path = runner_root / "PAMAP2_COMPONENT_ABLATION_RUNNER_BINDING.json"
    execution_log = control_root / "ABLATION_CAMPAIGN_EXECUTION_LOG.txt"
    state_path = control_root / "ABLATION_CONTROLLER_STATE.json"
    failure_path = control_root / "ABLATION_CONTROLLER_FAILURE.json"
    lock_path = control_root / "ABLATION_CAMPAIGN.lock"

    control_root.mkdir(parents=True, exist_ok=True)
    campaign_root.mkdir(parents=True, exist_ok=True)

    runner_binding = replay_binding(
        runner_binding_path,
        "runner_build_binding_sha256",
        build_binding["ablation_runner_build_binding_sha256"],
    )
    require(sha256_file(runner_path) == build_binding["ablation_runner_sha256"], "Ablation runner changed after build")
    design = pd.read_csv(matrix_path)
    require(len(design) == TOTAL_ABLATION_RUNS, "Ablation matrix row count changed")

    def log(message: str) -> None:
        stamp = f"[{utc_now()}] {message}"
        print(stamp, flush=True)
        with execution_log.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(stamp + "\n")

    with exclusive_lock(lock_path):
        try:
            if failure_path.exists():
                failure_path.unlink()
            completed_rows: list[dict[str, Any]] = []
            for item in design.to_dict(orient="records"):
                run_id = int(item["ablation_run_id"])
                current_name = run_name(item)
                current_root = campaign_root / current_name
                auth_path = authorization_root / f"ABLATION_RUN_{run_id:03d}_AUTHORIZATION.json"

                if current_root.exists():
                    complete_path = current_root / "RUN_COMPLETE.json"
                    require(complete_path.is_file(), f"Refusing partial ablation run directory: {current_root}")
                    audited = audit_run(
                        item,
                        run_root=current_root,
                        runner_sha=runner_binding["runner_sha256"],
                        runner_build_binding=runner_binding["runner_build_binding_sha256"],
                    )
                    completed_rows.append(audited)
                    log(f"SKIP VERIFIED COMPLETE {run_id:03d}/{TOTAL_ABLATION_RUNS}: {current_name}")
                else:
                    log(f"START {run_id:03d}/{TOTAL_ABLATION_RUNS}: {current_name}")
                    process = subprocess.run(
                        [
                            str(python_exe),
                            str(runner_path),
                            "--project-root",
                            str(project_root),
                            "--binding-json",
                            str(runner_binding_path),
                            "--run-id",
                            str(run_id),
                            "--authorization",
                            str(auth_path),
                        ],
                        cwd=str(project_root),
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                    with execution_log.open("a", encoding="utf-8", newline="\n") as handle:
                        handle.write(process.stdout)
                        if process.stdout and not process.stdout.endswith("\n"):
                            handle.write("\n")
                        if process.stderr:
                            handle.write("--- STDERR ---\n")
                            handle.write(process.stderr)
                            if not process.stderr.endswith("\n"):
                                handle.write("\n")
                    require(process.returncode == 0, f"Ablation run {run_id} exited with code {process.returncode}")
                    audited = audit_run(
                        item,
                        run_root=current_root,
                        runner_sha=runner_binding["runner_sha256"],
                        runner_build_binding=runner_binding["runner_build_binding_sha256"],
                    )
                    completed_rows.append(audited)
                    log(
                        f"COMPLETE {run_id:03d}/{TOTAL_ABLATION_RUNS}: "
                        f"macro_f1={audited['macro_f1']:.6f}; "
                        f"energy={audited['total_normalized_energy_consumed']:.6f}"
                    )

                write_json(
                    state_path,
                    {
                        "status": "RUNNING" if run_id < TOTAL_ABLATION_RUNS else "RUNS_COMPLETE",
                        "completed_ablation_runs": run_id,
                        "total_ablation_runs": TOTAL_ABLATION_RUNS,
                        "last_completed_ablation_run_id": run_id,
                        "next_ablation_run_id": run_id + 1 if run_id < TOTAL_ABLATION_RUNS else None,
                        "accepted_ablation_optimizer_steps": int(
                            sum(row["scientific_optimizer_steps_accounted"] for row in completed_rows)
                        ),
                        "gate152_build_final_binding_sha256": build_binding["gate152_build_final_binding_sha256"],
                        "updated_utc": utc_now(),
                    },
                )
            require(len(completed_rows) == TOTAL_ABLATION_RUNS, "Not all ablation runs were audited")
            require([row["ablation_run_id"] for row in completed_rows] == list(range(1, TOTAL_ABLATION_RUNS + 1)), "Ablation run sequence mismatch")
            return completed_rows
        except Exception as exc:
            write_json(
                failure_path,
                {
                    "status": "FAIL_CLOSED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "generated_utc": utc_now(),
                    "scientific_training_may_have_started_for_current_run": True,
                    "no_subsequent_ablation_run_started_after_failure": True,
                },
            )
            raise


def preserve_existing_output(path: Path) -> str | None:
    if not path.exists():
        return None
    if not any(path.iterdir()):
        path.rmdir()
        return None
    binding_path = path / "GATE152_FINAL_BINDING.json"
    if binding_path.is_file():
        existing = replay_binding(binding_path, "gate152_final_binding_sha256")
        if existing.get("status") == "PASS":
            return "ALREADY_COMPLETE"
    index = 1
    while True:
        candidate = path.with_name(f"{path.name}_previous_output_v{index}")
        if not candidate.exists():
            shutil.move(str(path), str(candidate))
            return str(candidate)
        index += 1


def copy_frozen_source_snapshot(project_root: Path, analysis_root: Path) -> tuple[int, str]:
    paths = frozen_paths(project_root)
    manifest_rows = read_csv_rows(paths["source_manifest"])
    snapshot_root = analysis_root / "frozen_source_snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    copied_rows: list[dict[str, Any]] = []
    for row in manifest_rows:
        source = paths["source_root"] / str(row["filename"])
        destination = snapshot_root / str(row["filename"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        observed = sha256_file(destination)
        require(observed == str(row["sha256"]).upper(), "Frozen source snapshot SHA mismatch")
        copied_rows.append(
            {
                "filename": str(row["filename"]),
                "size_bytes": destination.stat().st_size,
                "sha256": observed,
            }
        )
    snapshot_manifest = analysis_root / "PAMAP2_FROZEN_SOURCE_SNAPSHOT_MANIFEST.csv"
    write_csv_rows(snapshot_manifest, copied_rows)
    return len(copied_rows), sha256_file(snapshot_manifest)


def analyze_campaign(
    project_root: Path,
    build_binding: dict[str, Any],
    audited_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    pamap_root = project_root / "outputs" / "federated" / "pamap2"
    analysis_root = pamap_root / "ablation_analysis_v1"
    previous = preserve_existing_output(analysis_root)
    if previous == "ALREADY_COMPLETE":
        final_binding = replay_binding(
            analysis_root / "GATE152_FINAL_BINDING.json",
            "gate152_final_binding_sha256",
        )
        handoff_root = project_root / "outputs" / "handoff"
        handoff_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = handoff_root / f"pamap2_gate152_component_ablation_{timestamp}.zip"
        with zipfile.ZipFile(
            zip_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for path in sorted(analysis_root.rglob("*")):
                if path.is_file():
                    archive.write(
                        path,
                        arcname=str(
                            Path("ablation_analysis_v1")
                            / path.relative_to(analysis_root)
                        ).replace("\\", "/"),
                    )
        final_binding["_evidence_zip_path"] = str(zip_path)
        final_binding["_evidence_zip_sha256"] = sha256_file(zip_path)
        return final_binding
    analysis_root.mkdir(parents=True, exist_ok=False)

    authorization_root = pamap_root / "ablation_campaign_authorization_v1"
    runner_root = pamap_root / "ablation_campaign_runner_v1"
    control_root = pamap_root / "ablation_campaign_controller_v1"
    freeze_root = pamap_root / "postcampaign_freeze_v1"
    design_path = authorization_root / "PAMAP2_ABLATION_CAMPAIGN_MATRIX_240.csv"
    reference_path = freeze_root / "PAMAP2_RUN_LEVEL_RESULTS_600.csv"
    require(sha256_file(reference_path) == EXPECTED_GATE149R_RUN_LEVEL_SHA256, "Reference freeze changed")

    ablation = pd.DataFrame(audited_rows).sort_values("ablation_run_id").reset_index(drop=True)
    reference = pd.read_csv(reference_path)
    merged = ablation.merge(
        reference,
        left_on="reference_run_id",
        right_on="run_id",
        how="left",
        suffixes=("_ablation", "_full"),
        validate="many_to_one",
    )
    require(len(merged) == TOTAL_ABLATION_RUNS, "Ablation/reference merge row count mismatch")
    require(merged["run_id"].notna().all(), "Missing full-method reference")
    require(
        (merged["outer_fold_ablation"].astype(int) == merged["outer_fold_full"].astype(int)).all(),
        "Reference fold mismatch",
    )
    require(
        np.isclose(merged["alpha_ablation"].astype(float), merged["alpha_full"].astype(float)).all(),
        "Reference alpha mismatch",
    )
    require(
        (merged["scenario_ablation"].astype(str) == merged["scenario_full"].astype(str)).all(),
        "Reference scenario mismatch",
    )
    require(
        (merged["fl_seed_ablation"].astype(int) == merged["fl_seed_full"].astype(int)).all(),
        "Reference seed mismatch",
    )
    require(
        (merged["method_family"].astype(str) == merged["method"].astype(str)).all(),
        "Reference method mismatch",
    )

    ablation_path = analysis_root / "PAMAP2_ABLATION_RUN_RESULTS_240.csv"
    ablation.to_csv(ablation_path, index=False)

    design = pd.read_csv(design_path)
    descriptive_rows: list[dict[str, Any]] = []
    for (alpha, scenario, variant), group in ablation.groupby(["alpha", "scenario", "variant"], sort=False):
        record: dict[str, Any] = {
            "alpha": float(alpha),
            "scenario": str(scenario),
            "variant": str(variant),
            "variant_label": VARIANT_LABELS[str(variant)],
            "component_isolated": VARIANT_COMPONENT[str(variant)],
            "method_family": str(group["method_family"].iloc[0]),
            "n_runs": len(group),
        }
        for metric in METRICS:
            values = group[metric].astype(float).to_numpy()
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_std"] = float(np.std(values, ddof=1))
            record[f"{metric}_median"] = float(np.median(values))
        descriptive_rows.append(record)
    descriptive = pd.DataFrame(descriptive_rows)
    require(len(descriptive) == len(SELECTED_CONDITIONS) * len(VARIANTS), "Descriptive row count mismatch")
    descriptive_path = analysis_root / "PAMAP2_ABLATION_DESCRIPTIVE_24_ROWS.csv"
    descriptive.to_csv(descriptive_path, index=False)

    comparison_rows: list[dict[str, Any]] = []
    for (alpha, scenario, variant), group in merged.groupby(
        ["alpha_ablation", "scenario_ablation", "variant"],
        sort=False,
    ):
        require(len(group) == 10, "Each ablation comparison requires 10 paired blocks")
        method_family = str(group["method_family"].iloc[0])
        for metric, meta in METRICS.items():
            ablation_values = group[f"{metric}_ablation"].astype(float).to_numpy()
            full_values = group[f"{metric}_full"].astype(float).to_numpy()
            direction = int(meta["direction"])
            benefit = direction * (full_values - ablation_values)
            signed = exact_signed_rank(benefit)
            ci_low, ci_high = bootstrap_mean_ci(
                benefit,
                replicates=BOOTSTRAP_REPLICATES,
                seed=deterministic_seed(GATE_ID, alpha, scenario, variant, metric),
            )
            comparison_rows.append(
                {
                    "alpha": float(alpha),
                    "scenario": str(scenario),
                    "method_family": method_family,
                    "full_method_label": METHOD_LABELS[method_family],
                    "variant": str(variant),
                    "variant_label": VARIANT_LABELS[str(variant)],
                    "component_isolated": VARIANT_COMPONENT[str(variant)],
                    "metric": metric,
                    "metric_label": str(meta["label"]),
                    "endpoint_role": str(meta["endpoint_role"]),
                    "higher_is_better": direction == 1,
                    "n_paired_blocks": len(group),
                    "mean_full_method": float(np.mean(full_values)),
                    "mean_ablation_variant": float(np.mean(ablation_values)),
                    "mean_component_benefit_full_minus_ablation": float(np.mean(benefit)),
                    "bootstrap_95ci_component_benefit_low": ci_low,
                    "bootstrap_95ci_component_benefit_high": ci_high,
                    "hodges_lehmann_component_benefit": hodges_lehmann_paired(benefit),
                    "rank_biserial_positive_favors_full_method": signed["rank_biserial"],
                    "p_exact_two_sided": signed["p_exact_two_sided"],
                    "n_nonzero_differences": signed["n_nonzero"],
                    "component_benefit_direction": (
                        "full_method_better"
                        if float(np.mean(benefit)) > 0
                        else "ablation_variant_better"
                        if float(np.mean(benefit)) < 0
                        else "tie"
                    ),
                }
            )
    comparisons = pd.DataFrame(comparison_rows)
    require(len(comparisons) == len(SELECTED_CONDITIONS) * len(VARIANTS) * len(METRICS), "Comparison row count mismatch")

    comparisons["p_holm_within_condition_metric_method"] = np.nan
    for _, indices in comparisons.groupby(
        ["alpha", "scenario", "metric", "method_family"],
        sort=False,
    ).groups.items():
        index_list = list(indices)
        comparisons.loc[index_list, "p_holm_within_condition_metric_method"] = holm_adjust(
            comparisons.loc[index_list, "p_exact_two_sided"].astype(float).to_numpy()
        )
    comparisons["significant_holm_0p05"] = (
        comparisons["p_holm_within_condition_metric_method"].astype(float) < ALPHA_LEVEL
    )
    comparisons["bootstrap_ci_excludes_zero"] = (
        (comparisons["bootstrap_95ci_component_benefit_low"] > 0)
        | (comparisons["bootstrap_95ci_component_benefit_high"] < 0)
    )
    comparisons_path = analysis_root / "PAMAP2_ABLATION_PAIRED_COMPONENT_TESTS_96_ROWS.csv"
    comparisons.to_csv(comparisons_path, index=False)

    component_rows: list[dict[str, Any]] = []
    for (variant, metric), group in comparisons.groupby(["variant", "metric"], sort=False):
        component_rows.append(
            {
                "variant": str(variant),
                "variant_label": VARIANT_LABELS[str(variant)],
                "component_isolated": VARIANT_COMPONENT[str(variant)],
                "method_family": str(group["method_family"].iloc[0]),
                "metric": str(metric),
                "metric_label": str(group["metric_label"].iloc[0]),
                "conditions_analyzed": len(group),
                "conditions_mean_benefit_positive": int(
                    (group["mean_component_benefit_full_minus_ablation"].astype(float) > 0).sum()
                ),
                "conditions_holm_significant_positive": int(
                    (
                        group["significant_holm_0p05"].astype(bool)
                        & (group["mean_component_benefit_full_minus_ablation"].astype(float) > 0)
                    ).sum()
                ),
                "mean_of_condition_mean_benefits": float(
                    group["mean_component_benefit_full_minus_ablation"].astype(float).mean()
                ),
                "minimum_condition_mean_benefit": float(
                    group["mean_component_benefit_full_minus_ablation"].astype(float).min()
                ),
                "maximum_condition_mean_benefit": float(
                    group["mean_component_benefit_full_minus_ablation"].astype(float).max()
                ),
            }
        )
    component_summary = pd.DataFrame(component_rows)
    component_summary_path = analysis_root / "PAMAP2_ABLATION_COMPONENT_SUMMARY_16_ROWS.csv"
    component_summary.to_csv(component_summary_path, index=False)

    ledger = ablation[
        [
            "ablation_run_id",
            "reference_run_id",
            "outer_fold",
            "alpha",
            "scenario",
            "method_family",
            "variant",
            "fl_seed",
            "scientific_optimizer_steps_accounted",
            "run_result_binding_sha256",
            "run_manifest_binding_sha256",
        ]
    ].copy()
    ledger_path = analysis_root / "PAMAP2_ABLATION_BINDING_LEDGER_240.csv"
    ledger.to_csv(ledger_path, index=False)

    source_count, source_snapshot_manifest_sha = copy_frozen_source_snapshot(
        project_root,
        analysis_root,
    )

    shutil.copy2(design_path, analysis_root / design_path.name)
    shutil.copy2(
        authorization_root / "GATE152_BUILD_FINAL_BINDING.json",
        analysis_root / "GATE152_BUILD_FINAL_BINDING.json",
    )
    shutil.copy2(
        runner_root / "PAMAP2_COMPONENT_ABLATION_RUNNER_BINDING.json",
        analysis_root / "PAMAP2_COMPONENT_ABLATION_RUNNER_BINDING.json",
    )
    shutil.copy2(
        runner_root / "run_pamap2_component_ablation.py",
        analysis_root / "run_pamap2_component_ablation.py",
    )
    shutil.copy2(
        control_root / "ABLATION_CAMPAIGN_EXECUTION_LOG.txt",
        analysis_root / "ABLATION_CAMPAIGN_EXECUTION_LOG.txt",
    )
    shutil.copy2(
        control_root / "ABLATION_CONTROLLER_STATE.json",
        analysis_root / "ABLATION_CONTROLLER_STATE.json",
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    condition_labels = [
        f"a={alpha:g} | {scenario.replace('_', ' ')}"
        for alpha, scenario, _ in SELECTED_CONDITIONS
    ]
    variant_order = [item[0] for item in VARIANTS]

    macro = comparisons[comparisons["metric"] == "macro_f1"].copy()
    macro["condition_label"] = macro.apply(
        lambda row: f"a={float(row['alpha']):g} | {str(row['scenario']).replace('_', ' ')}",
        axis=1,
    )
    macro_pivot = macro.pivot(
        index="variant",
        columns="condition_label",
        values="mean_component_benefit_full_minus_ablation",
    ).reindex(index=variant_order, columns=condition_labels)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    image = ax.imshow(macro_pivot.to_numpy(), aspect="auto")
    ax.set_xticks(range(len(condition_labels)), labels=condition_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(variant_order)), labels=[VARIANT_LABELS[item] for item in variant_order])
    ax.set_title("Macro-F1 component contribution: full method minus ablation")
    ax.set_xlabel("Prespecified condition")
    ax.set_ylabel("Ablation variant")
    fig.colorbar(image, ax=ax, label="Macro-F1 benefit of full component")
    fig.tight_layout()
    fig1 = analysis_root / "FIG152_01_MACRO_F1_COMPONENT_CONTRIBUTION_HEATMAP.png"
    fig.savefig(fig1, dpi=300)
    plt.close(fig)

    energy = comparisons[comparisons["metric"] == "total_normalized_energy_consumed"].copy()
    energy["condition_label"] = energy.apply(
        lambda row: f"a={float(row['alpha']):g} | {str(row['scenario']).replace('_', ' ')}",
        axis=1,
    )
    energy_pivot = energy.pivot(
        index="variant",
        columns="condition_label",
        values="mean_component_benefit_full_minus_ablation",
    ).reindex(index=variant_order, columns=condition_labels)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    image = ax.imshow(energy_pivot.to_numpy(), aspect="auto")
    ax.set_xticks(range(len(condition_labels)), labels=condition_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(variant_order)), labels=[VARIANT_LABELS[item] for item in variant_order])
    ax.set_title("Energy component contribution: positive values favor the full method")
    ax.set_xlabel("Prespecified condition")
    ax.set_ylabel("Ablation variant")
    fig.colorbar(image, ax=ax, label="Normalized energy benefit of full component")
    fig.tight_layout()
    fig2 = analysis_root / "FIG152_02_ENERGY_COMPONENT_CONTRIBUTION_HEATMAP.png"
    fig.savefig(fig2, dpi=300)
    plt.close(fig)

    overall_points = (
        merged.groupby(["variant", "method_family"], as_index=False)
        .agg(
            macro_f1_ablation=("macro_f1_ablation", "mean"),
            macro_f1_full=("macro_f1_full", "mean"),
            energy_ablation=("total_normalized_energy_consumed_ablation", "mean"),
            energy_full=("total_normalized_energy_consumed_full", "mean"),
        )
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    for item in overall_points.itertuples(index=False):
        ax.plot(
            [item.energy_ablation, item.energy_full],
            [item.macro_f1_ablation, item.macro_f1_full],
            marker="o",
        )
        ax.annotate(
            VARIANT_LABELS[str(item.variant)],
            (item.energy_ablation, item.macro_f1_ablation),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
        ax.annotate(
            METHOD_LABELS[str(item.method_family)],
            (item.energy_full, item.macro_f1_full),
            xytext=(4, -10),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Total normalized energy consumed (lower is better)")
    ax.set_ylabel("Macro-F1 (higher is better)")
    ax.set_title("Full methods and matched component ablations")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig3 = analysis_root / "FIG152_03_FULL_VS_ABLATION_EFFECTIVENESS_ENERGY.png"
    fig.savefig(fig3, dpi=300)
    plt.close(fig)

    primary = comparisons[comparisons["metric"].isin(["macro_f1", "total_normalized_energy_consumed"])].copy()
    primary["condition_label"] = primary.apply(
        lambda row: f"a={float(row['alpha']):g}\n{str(row['scenario'])}",
        axis=1,
    )
    figure_rows = []
    for index, item in enumerate(primary.itertuples(index=False)):
        figure_rows.append((index, item))
    fig, ax = plt.subplots(figsize=(12, max(7, len(figure_rows) * 0.16)))
    y = np.arange(len(figure_rows))
    means = np.asarray([float(item.mean_component_benefit_full_minus_ablation) for _, item in figure_rows])
    lows = np.asarray([float(item.bootstrap_95ci_component_benefit_low) for _, item in figure_rows])
    highs = np.asarray([float(item.bootstrap_95ci_component_benefit_high) for _, item in figure_rows])
    ax.errorbar(means, y, xerr=[means - lows, highs - means], fmt="o", capsize=2)
    labels = [
        f"{VARIANT_LABELS[str(item.variant)]} | {item.metric_label} | {item.condition_label}"
        for _, item in figure_rows
    ]
    ax.set_yticks(y, labels=labels, fontsize=7)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Component benefit: full method minus ablation (direction-normalized)")
    ax.set_title("Prespecified primary component effects with bootstrap 95% intervals")
    ax.invert_yaxis()
    fig.tight_layout()
    fig4 = analysis_root / "FIG152_04_PRIMARY_COMPONENT_EFFECTS_FOREST.png"
    fig.savefig(fig4, dpi=300)
    plt.close(fig)

    optimizer_steps = int(ablation["scientific_optimizer_steps_accounted"].astype(int).sum())
    early_stops = int(ablation["early_stop_round"].notna().sum())
    significant_primary_positive = int(
        (
            comparisons["endpoint_role"].astype(str).str.startswith("primary")
            & comparisons["significant_holm_0p05"].astype(bool)
            & (comparisons["mean_component_benefit_full_minus_ablation"].astype(float) > 0)
        ).sum()
    )

    report_lines = [
        "PAMAP2 GATE-152 COMPONENT ABLATION CAMPAIGN AND PAIRED ANALYSIS",
        "=" * 78,
        "",
        "STATUS",
        "-" * 78,
        "PASS",
        "",
        "VERIFIED EXECUTION",
        "-" * 78,
        f"Component-ablation runs verified: {len(ablation)}/{TOTAL_ABLATION_RUNS}",
        f"Accepted component-ablation optimizer steps: {optimizer_steps}",
        f"Protocol-defined early-stop ablation runs: {early_stops}",
        f"Frozen full-method reference runs used: {ablation['reference_run_id'].nunique()}",
        f"Prespecified conditions: {len(SELECTED_CONDITIONS)}",
        f"Component-ablation variants: {len(VARIANTS)}",
        "",
        "PAIRED COMPONENT ANALYSIS",
        "-" * 78,
        f"Paired comparison rows: {len(comparisons)}",
        f"Bootstrap replicates per comparison: {BOOTSTRAP_REPLICATES}",
        f"Positive Holm-significant primary component effects: {significant_primary_positive}",
        "Positive direction means that the complete TEA-FL or ARL-FL method outperformed the matched ablation.",
        "",
        "SCIENTIFIC BOUNDARY",
        "-" * 78,
        "Original PAMAP2 600-run results modified: NO",
        "New PAMAP2 component-ablation runs executed: YES",
        f"New PAMAP2 component-ablation runs: {TOTAL_ABLATION_RUNS}",
        f"Scientific optimizer steps executed by Gate-152 ablation campaign: {optimizer_steps}",
        "CICIoT2023 scientific training started: NO",
        "",
    ]
    report_path = analysis_root / "GATE152_REPORT.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8", newline="\n")

    output_paths = [
        ablation_path,
        descriptive_path,
        comparisons_path,
        component_summary_path,
        ledger_path,
        analysis_root / design_path.name,
        analysis_root / "GATE152_BUILD_FINAL_BINDING.json",
        analysis_root / "PAMAP2_COMPONENT_ABLATION_RUNNER_BINDING.json",
        analysis_root / "run_pamap2_component_ablation.py",
        analysis_root / "ABLATION_CAMPAIGN_EXECUTION_LOG.txt",
        analysis_root / "ABLATION_CONTROLLER_STATE.json",
        analysis_root / "PAMAP2_FROZEN_SOURCE_SNAPSHOT_MANIFEST.csv",
        fig1,
        fig2,
        fig3,
        fig4,
        report_path,
    ]
    output_hashes = {path.name: sha256_file(path) for path in output_paths}

    audit = {
        "gate_id": GATE_ID,
        "status": "PASS",
        "scope": SCOPE,
        "generated_utc": utc_now(),
        "gate151_final_binding_sha256": EXPECTED_GATE151_FINAL_BINDING,
        "gate152_build_final_binding_sha256": build_binding["gate152_build_final_binding_sha256"],
        "component_ablation_runs_verified": len(ablation),
        "accepted_component_ablation_optimizer_steps": optimizer_steps,
        "protocol_defined_early_stop_ablation_runs": early_stops,
        "full_method_reference_runs_used": int(ablation["reference_run_id"].nunique()),
        "prespecified_conditions": len(SELECTED_CONDITIONS),
        "component_ablation_variants": len(VARIANTS),
        "paired_comparison_rows": len(comparisons),
        "bootstrap_replicates_per_comparison": BOOTSTRAP_REPLICATES,
        "frozen_source_files_snapshotted": source_count,
        "frozen_source_snapshot_manifest_sha256": source_snapshot_manifest_sha,
        "article_figures_created": 4,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "output_hashes_before_audit_binding_manifest": output_hashes,
        "original_600_run_results_modified": False,
        "new_pamap2_component_ablation_runs_started": TOTAL_ABLATION_RUNS,
        "scientific_optimizer_steps_executed_by_gate152": optimizer_steps,
        "ciciot2023_scientific_training_started": False,
    }
    audit_path = analysis_root / "GATE152_AUDIT.json"
    write_json(audit_path, audit)

    binding_core = {
        "gate_id": GATE_ID,
        "status": "PASS",
        "scope": SCOPE,
        "gate151_final_binding_sha256": EXPECTED_GATE151_FINAL_BINDING,
        "gate152_build_final_binding_sha256": build_binding["gate152_build_final_binding_sha256"],
        "component_ablation_runs_verified": len(ablation),
        "accepted_component_ablation_optimizer_steps": optimizer_steps,
        "ablation_result_ledger_sha256": sha256_file(ledger_path),
        "ablation_run_results_sha256": sha256_file(ablation_path),
        "ablation_descriptive_sha256": sha256_file(descriptive_path),
        "ablation_paired_tests_sha256": sha256_file(comparisons_path),
        "ablation_component_summary_sha256": sha256_file(component_summary_path),
        "frozen_source_snapshot_manifest_sha256": source_snapshot_manifest_sha,
        "report_sha256": sha256_file(report_path),
        "audit_sha256": sha256_file(audit_path),
        "original_600_run_results_modified": False,
        "new_pamap2_component_ablation_runs_started": TOTAL_ABLATION_RUNS,
        "scientific_optimizer_steps_executed_by_gate152": optimizer_steps,
        "ciciot2023_scientific_training_started": False,
    }
    final_binding = dict(binding_core)
    final_binding["gate152_final_binding_sha256"] = canonical_sha256(binding_core)
    final_binding_path = analysis_root / "GATE152_FINAL_BINDING.json"
    write_json(final_binding_path, final_binding)

    manifest_rows: list[dict[str, Any]] = []
    for path in sorted(analysis_root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST_SHA256.csv":
            manifest_rows.append(
                {
                    "filename": str(path.relative_to(analysis_root)).replace("\\", "/"),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest_path = analysis_root / "MANIFEST_SHA256.csv"
    write_csv_rows(manifest_path, manifest_rows)

    handoff_root = project_root / "outputs" / "handoff"
    handoff_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = handoff_root / f"pamap2_gate152_component_ablation_{timestamp}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(analysis_root.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    arcname=str(Path("ablation_analysis_v1") / path.relative_to(analysis_root)).replace("\\", "/"),
                )
    zip_sha = sha256_file(zip_path)

    final_state = read_json(control_root / "ABLATION_CONTROLLER_STATE.json")
    final_state.update(
        {
            "status": "COMPLETE",
            "analysis_status": "PASS",
            "gate152_final_binding_sha256": final_binding["gate152_final_binding_sha256"],
            "evidence_zip": str(zip_path),
            "evidence_zip_sha256": zip_sha,
            "updated_utc": utc_now(),
        }
    )
    write_json(control_root / "ABLATION_CONTROLLER_STATE.json", final_state)

    final_binding["_evidence_zip_path"] = str(zip_path)
    final_binding["_evidence_zip_sha256"] = zip_sha
    return final_binding


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--runner-source", type=Path, required=True)
    parser.add_argument("--start-campaign", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    runner_source = args.runner_source.expanduser().resolve()
    python_exe = project_root / ".venv" / "Scripts" / "python.exe"
    require(project_root.is_dir(), f"Project root not found: {project_root}")
    require(runner_source.is_file(), f"Ablation runner source not found: {runner_source}")
    require(python_exe.is_file(), f"Project Python executable not found: {python_exe}")

    print("Running Gate-152 PAMAP2 component-ablation build and progressive controller...", flush=True)
    print("Prespecified ablation runs: 240", flush=True)
    print("Mode: sequential / resumable only across fully audited runs / fail closed", flush=True)
    print("Original PAMAP2 600-run results modified: NO", flush=True)
    print("CICIoT2023 scientific training started: NO", flush=True)

    build_binding = build_authorization(
        project_root,
        runner_source,
        python_exe,
    )
    print(
        "Gate-152 build binding SHA256: "
        + build_binding["gate152_build_final_binding_sha256"],
        flush=True,
    )

    if not args.start_campaign:
        print("GATE152_BUILD_PASS", flush=True)
        print("Scientific training executed in build-only mode: NO", flush=True)
        return 0

    audited_rows = execute_campaign(
        project_root,
        build_binding,
        python_exe,
    )
    final_binding = analyze_campaign(
        project_root,
        build_binding,
        audited_rows,
    )

    print("=" * 78)
    print("GATE152_PASS")
    print("=" * 78)
    print(f"Component-ablation runs verified: {final_binding['component_ablation_runs_verified']}/240")
    print(
        "Accepted component-ablation optimizer steps: "
        + str(final_binding["accepted_component_ablation_optimizer_steps"])
    )
    print("Original PAMAP2 600-run results modified: NO")
    print("New PAMAP2 component-ablation runs started: 240")
    print(
        "Scientific optimizer steps executed by Gate-152: "
        + str(final_binding["scientific_optimizer_steps_executed_by_gate152"])
    )
    print("CICIoT2023 scientific training started by Gate-152: NO")
    print(
        "Gate-152 final binding SHA256: "
        + final_binding["gate152_final_binding_sha256"]
    )
    print(
        "Report: "
        + str(
            project_root
            / "outputs"
            / "federated"
            / "pamap2"
            / "ablation_analysis_v1"
            / "GATE152_REPORT.txt"
        )
    )
    print("Evidence ZIP: " + final_binding["_evidence_zip_path"])
    print("Evidence ZIP SHA256: " + final_binding["_evidence_zip_sha256"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("GATE152_FAIL_CLOSED", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
