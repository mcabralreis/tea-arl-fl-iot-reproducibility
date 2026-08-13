from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import t

GATE_ID = "GATE-151"
SCOPE = "PAMAP2_MULTIOBJECTIVE_TEMPORAL_AND_PER_CLASS_ARTICLE_ANALYSIS"
EXPECTED_GATE149R_FINAL_BINDING = "B73E05C64EAF424BF6014E076B1848DD2409F54F07307B083EBD5F9CB42AE2AC"
EXPECTED_GATE150_FINAL_BINDING = "3EBF4CFB46EDFF04BC54D12CFBA90CE176AAA90C3CD23DC234EA0605682EBB51"
EXPECTED_RUN_LEVEL_SHA256 = "25B41F60D9EFAB10BA92ACEBE9FA36432DB29D70C0220F87206FAB0BE5309931"
EXPECTED_TRAJECTORY_SHA256 = "0DF29A6EFBDF8918AA38D3C627B30454D76815947857807E72701477A5DA4E0C"
EXPECTED_PER_CLASS_SHA256 = "798DA4B5C55B6C0A3EABCB22B7C18069101E449472B80DFC79E6663C7D6CF6FE"
EXPECTED_GATE150_ARTICLE_FOCUSED_SHA256 = "0B69E8240A2E324B9B66FDCBDA7053B0EE4F8742CA17688C6A62D42FC81D0E05"
EXPECTED_GATE150_OMNIBUS_SHA256 = "DCAA203DD1BEEE25BEF82BD3EA9ABE9F411B7D374E2092B5631129378DE524D7"
EXPECTED_RUNS = 600
EXPECTED_TRAJECTORY_ROWS = 12600
EXPECTED_PER_CLASS_ROWS = 7200
EXPECTED_OPTIMIZER_STEPS = 3401459
ATTACK_START_ROUND = 20
METHODS = (
    "fedavg",
    "fedprox",
    "random_trimmed_mean",
    "fedle_adapted",
    "tea_fl",
    "arl_fl",
)
METHOD_LABELS = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "random_trimmed_mean": "Random Trimmed Mean",
    "fedle_adapted": "FedLE-adapted",
    "tea_fl": "TEA-FL",
    "arl_fl": "ARL-FL",
}
ALPHAS = (0.1, 1.0)
SCENARIOS = (
    "clean",
    "labelflip_mu0p2",
    "labelflip_mu0p4",
    "signflip_mu0p2",
    "signflip_mu0p4",
)
SCENARIO_LABELS = {
    "clean": "Clean",
    "labelflip_mu0p2": "Label flip, mu=0.2",
    "labelflip_mu0p4": "Label flip, mu=0.4",
    "signflip_mu0p2": "Sign flip, mu=0.2",
    "signflip_mu0p4": "Sign flip, mu=0.4",
}
ACTIVITY_IDS = (1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24)
ACTIVITY_NAMES = {
    1: "Lying",
    2: "Sitting",
    3: "Standing",
    4: "Walking",
    5: "Running",
    6: "Cycling",
    7: "Nordic walking",
    12: "Ascending stairs",
    13: "Descending stairs",
    16: "Vacuum cleaning",
    17: "Ironing",
    24: "Rope jumping",
}
EVAL_ROUNDS = tuple(range(0, 101, 5))


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
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_canonical_binding(path: Path, final_field: str, expected: str) -> dict[str, Any]:
    require(path.is_file(), f"Binding file missing: {path}")
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    observed_declared = str(obj.get(final_field, "")).upper()
    require(observed_declared == expected, f"Declared binding mismatch in {path.name}: {observed_declared}")
    payload = dict(obj)
    payload.pop(final_field, None)
    reproduced = canonical_sha256(payload)
    require(reproduced == expected, f"Canonical binding replay mismatch in {path.name}: {reproduced}")
    return obj


def minmax_benefit(values: pd.Series, higher_is_better: bool) -> pd.Series:
    values = values.astype(float)
    lo = float(values.min())
    hi = float(values.max())
    if math.isclose(lo, hi, rel_tol=0.0, abs_tol=1e-15):
        return pd.Series(np.ones(len(values)), index=values.index, dtype=float)
    norm = (values - lo) / (hi - lo)
    return norm if higher_is_better else 1.0 - norm


def pareto_mask(values: np.ndarray) -> np.ndarray:
    """All columns are benefit-oriented: higher is better."""
    n = values.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(values[j] >= values[i] - 1e-12) and np.any(values[j] > values[i] + 1e-12):
                mask[i] = False
                break
    return mask


def auc_normalized(rounds: np.ndarray, values: np.ndarray, start: int, end: int) -> float:
    keep = (rounds >= start) & (rounds <= end)
    x = rounds[keep].astype(float)
    y = values[keep].astype(float)
    require(len(x) >= 2, f"Insufficient points for AUC {start}-{end}")
    return float(np.trapezoid(y, x) / (end - start))


def mean_ci(values: pd.Series) -> tuple[float, float, float, float]:
    arr = values.dropna().astype(float).to_numpy()
    n = len(arr)
    if n == 0:
        return math.nan, math.nan, math.nan, math.nan
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if n <= 1 or math.isclose(std, 0.0, abs_tol=1e-15):
        return mean, std, mean, mean
    half = float(t.ppf(0.975, df=n - 1) * std / math.sqrt(n))
    return mean, std, mean - half, mean + half


def rank_desc(values: pd.Series) -> pd.Series:
    return values.rank(method="average", ascending=False)


def plot_scatter(overall: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.scatter(overall["total_normalized_energy_consumed_mean"], overall["macro_f1_mean"], s=70)
    for row in overall.itertuples(index=False):
        ax.annotate(row.method_label, (row.total_normalized_energy_consumed_mean, row.macro_f1_mean), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Total normalized energy consumed (lower is better)")
    ax.set_ylabel("Macro-F1 (higher is better)")
    ax.set_title("PAMAP2 overall effectiveness-energy trade-off")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(summary: pd.DataFrame, alpha: float, scenario: str, output: Path) -> None:
    subset = summary[(np.isclose(summary["alpha"], alpha)) & (summary["scenario"] == scenario)]
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    for method in METHODS:
        part = subset[subset["method"] == method].sort_values("round")
        ax.plot(part["round"], part["test_macro_f1_mean"], marker="o", markersize=3, linewidth=1.5, label=METHOD_LABELS[method])
        ax.fill_between(part["round"], part["test_macro_f1_ci95_low"], part["test_macro_f1_ci95_high"], alpha=0.12)
    ax.axvline(ATTACK_START_ROUND, linestyle="--", linewidth=1.2, label="Attack starts")
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Macro-F1")
    ax.set_title(f"Macro-F1 trajectory: {SCENARIO_LABELS[scenario]}, alpha={alpha:g}")
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_active_clients(summary: pd.DataFrame, output: Path) -> None:
    grouped = summary.groupby(["method", "round"], as_index=False)["active_clients_mean"].mean()
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    for method in METHODS:
        part = grouped[grouped["method"] == method].sort_values("round")
        ax.plot(part["round"], part["active_clients_mean"], marker="o", markersize=3, linewidth=1.5, label=METHOD_LABELS[method])
    ax.axvline(ATTACK_START_ROUND, linestyle="--", linewidth=1.2, label="Attack starts")
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Mean active clients")
    ax.set_title("PAMAP2 active-client trajectories across all conditions")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_arl_pressure(summary: pd.DataFrame, output: Path) -> None:
    subset = summary[(summary["method"] == "arl_fl") & (np.isclose(summary["alpha"], 1.0))]
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    for scenario in SCENARIOS:
        part = subset[subset["scenario"] == scenario].sort_values("round")
        ax.plot(part["round"], part["global_pressure_mean"], marker="o", markersize=3, linewidth=1.5, label=SCENARIO_LABELS[scenario])
    ax.axvline(ATTACK_START_ROUND, linestyle="--", linewidth=1.2, label="Attack starts")
    ax.set_xlabel("Federated round")
    ax.set_ylabel("ARL-FL global pressure")
    ax.set_title("ARL-FL global pressure by scenario, alpha=1.0")
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_class_heatmap(class_overall: pd.DataFrame, output: Path) -> None:
    matrix = class_overall.pivot(index="method_label", columns="activity_name", values="class_f1_mean").reindex([METHOD_LABELS[m] for m in METHODS])
    fig, ax = plt.subplots(figsize=(13.0, 5.5))
    image = ax.imshow(matrix.to_numpy(), aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=9)
    ax.set_title("Mean per-activity F1 across all PAMAP2 conditions")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean F1")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_condition_heatmap(condition_methods: pd.DataFrame, output: Path) -> None:
    work = condition_methods.copy()
    work["condition_label"] = work.apply(lambda r: f"a={r.alpha:g} | {SCENARIO_LABELS[r.scenario]}", axis=1)
    matrix = work.pivot(index="method_label", columns="condition_label", values="macro_f1_mean").reindex([METHOD_LABELS[m] for m in METHODS])
    ordered_cols = [f"a={a:g} | {SCENARIO_LABELS[s]}" for a in ALPHAS for s in SCENARIOS]
    matrix = matrix.reindex(columns=ordered_cols)
    fig, ax = plt.subplots(figsize=(13.0, 5.5))
    image = ax.imshow(matrix.to_numpy(), aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=9)
    ax.set_title("Macro-F1 across PAMAP2 heterogeneity and attack conditions")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean Macro-F1")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate-151 PAMAP2 multidimensional article analysis")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    freeze_root = project_root / "outputs" / "federated" / "pamap2" / "postcampaign_freeze_v1"
    gate150_root = project_root / "outputs" / "federated" / "pamap2" / "inferential_statistics_v1"
    output_root = project_root / "outputs" / "federated" / "pamap2" / "article_analysis_v1"
    handoff_root = project_root / "outputs" / "handoff"

    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"Gate-151 output directory already exists and is not empty: {output_root}")
    if output_root.exists():
        output_root.rmdir()
    output_root.mkdir(parents=True, exist_ok=False)
    handoff_root.mkdir(parents=True, exist_ok=True)

    run_path = freeze_root / "PAMAP2_RUN_LEVEL_RESULTS_600.csv"
    trajectory_path = freeze_root / "PAMAP2_EVALUATION_TRAJECTORIES_12600_ROWS.csv"
    class_path = freeze_root / "PAMAP2_PER_CLASS_F1_7200_ROWS.csv"
    gate149r_binding_path = freeze_root / "GATE149R_FINAL_BINDING.json"
    gate150_binding_path = gate150_root / "GATE150_FINAL_BINDING.json"
    gate150_article_path = gate150_root / "PAMAP2_ARTICLE_FOCUSED_COMPARISONS.csv"
    gate150_omnibus_path = gate150_root / "PAMAP2_OMNIBUS_FRIEDMAN_TESTS.csv"

    verify_canonical_binding(gate149r_binding_path, "gate149r_final_binding_sha256", EXPECTED_GATE149R_FINAL_BINDING)
    gate150_binding = verify_canonical_binding(gate150_binding_path, "gate150_final_binding_sha256", EXPECTED_GATE150_FINAL_BINDING)

    expected_hashes = {
        run_path: EXPECTED_RUN_LEVEL_SHA256,
        trajectory_path: EXPECTED_TRAJECTORY_SHA256,
        class_path: EXPECTED_PER_CLASS_SHA256,
        gate150_article_path: EXPECTED_GATE150_ARTICLE_FOCUSED_SHA256,
        gate150_omnibus_path: EXPECTED_GATE150_OMNIBUS_SHA256,
    }
    for path, expected in expected_hashes.items():
        require(path.is_file(), f"Required input missing: {path}")
        observed = sha256_file(path)
        require(observed == expected, f"Input SHA256 mismatch for {path.name}: {observed}")

    require(str(gate150_binding["gate149r_final_binding_sha256"]).upper() == EXPECTED_GATE149R_FINAL_BINDING, "Gate-150 provenance does not bind Gate-149R")

    runs = pd.read_csv(run_path)
    trajectories = pd.read_csv(trajectory_path)
    per_class = pd.read_csv(class_path)
    gate150_article = pd.read_csv(gate150_article_path)
    gate150_omnibus = pd.read_csv(gate150_omnibus_path)

    require(len(runs) == EXPECTED_RUNS, f"Expected {EXPECTED_RUNS} runs, found {len(runs)}")
    require(runs["run_id"].nunique() == EXPECTED_RUNS, "Run IDs are not unique")
    require(set(runs["method"]) == set(METHODS), "Method set mismatch")
    require(int(runs["scientific_optimizer_steps_accounted"].sum()) == EXPECTED_OPTIMIZER_STEPS, "Optimizer-step ledger mismatch")
    require(len(trajectories) == EXPECTED_TRAJECTORY_ROWS, "Trajectory row-count mismatch")
    require(len(per_class) == EXPECTED_PER_CLASS_ROWS, "Per-class row-count mismatch")
    require(trajectories.groupby("run_id")["round"].nunique().eq(21).all(), "Every run must contain 21 evaluation rounds")
    require(set(trajectories["round"].astype(int).unique()) == set(EVAL_ROUNDS), "Evaluation-round set mismatch")
    require(per_class.groupby("run_id")["class_index"].nunique().eq(12).all(), "Every run must contain 12 class F1 rows")

    # Reconcile final trajectory values with run-level data.
    final_traj = trajectories[trajectories["round"] == 100][["run_id", "test_macro_f1", "test_balanced_accuracy", "active_clients", "mean_residual_energy"]]
    reconcile = runs[["run_id", "macro_f1", "balanced_accuracy", "final_active_clients", "final_mean_residual_energy"]].merge(final_traj, on="run_id", validate="one_to_one")
    require(np.allclose(reconcile["macro_f1"], reconcile["test_macro_f1"], rtol=0.0, atol=1e-12), "Final Macro-F1 reconciliation failed")
    require(np.allclose(reconcile["balanced_accuracy"], reconcile["test_balanced_accuracy"], rtol=0.0, atol=1e-12), "Final balanced-accuracy reconciliation failed")
    require(np.allclose(reconcile["final_active_clients"], reconcile["active_clients"], rtol=0.0, atol=1e-12), "Final active-client reconciliation failed")
    require(np.allclose(reconcile["final_mean_residual_energy"], reconcile["mean_residual_energy"], rtol=0.0, atol=1e-12), "Final residual-energy reconciliation failed")

    # ------------------------------------------------------------------
    # Overall and condition-level multiobjective summaries.
    # ------------------------------------------------------------------
    metric_specs = {
        "macro_f1": True,
        "balanced_accuracy": True,
        "total_normalized_energy_consumed": False,
        "final_mean_residual_energy": True,
        "final_active_clients": True,
        "jain_participation_fairness": True,
    }
    agg_map = {metric: ["mean", "std", "median"] for metric in metric_specs}
    overall = runs.groupby("method", as_index=False).agg(agg_map)
    overall.columns = ["method"] + [f"{a}_{b}" for a, b in overall.columns.tolist()[1:]]
    overall["method_label"] = overall["method"].map(METHOD_LABELS)

    for metric, higher in metric_specs.items():
        overall[f"{metric}_benefit_norm"] = minmax_benefit(overall[f"{metric}_mean"], higher)
        overall[f"{metric}_rank"] = rank_desc(overall[f"{metric}_benefit_norm"])

    overall["score_effectiveness_energy"] = 0.5 * overall["macro_f1_benefit_norm"] + 0.5 * overall["total_normalized_energy_consumed_benefit_norm"]
    overall["score_balanced_four_objective"] = (
        overall["macro_f1_benefit_norm"]
        + overall["total_normalized_energy_consumed_benefit_norm"]
        + overall["final_active_clients_benefit_norm"]
        + overall["jain_participation_fairness_benefit_norm"]
    ) / 4.0
    overall["score_retention_priority"] = (
        0.35 * overall["macro_f1_benefit_norm"]
        + 0.20 * overall["total_normalized_energy_consumed_benefit_norm"]
        + 0.30 * overall["final_active_clients_benefit_norm"]
        + 0.15 * overall["jain_participation_fairness_benefit_norm"]
    )
    overall["rank_effectiveness_energy"] = rank_desc(overall["score_effectiveness_energy"])
    overall["rank_balanced_four_objective"] = rank_desc(overall["score_balanced_four_objective"])
    overall["rank_retention_priority"] = rank_desc(overall["score_retention_priority"])

    p2 = overall[["macro_f1_benefit_norm", "total_normalized_energy_consumed_benefit_norm"]].to_numpy(float)
    p4 = overall[["macro_f1_benefit_norm", "total_normalized_energy_consumed_benefit_norm", "final_active_clients_benefit_norm", "jain_participation_fairness_benefit_norm"]].to_numpy(float)
    overall["pareto_macro_f1_energy"] = pareto_mask(p2)
    overall["pareto_four_objective"] = pareto_mask(p4)
    overall = overall.sort_values("rank_balanced_four_objective")
    overall.to_csv(output_root / "PAMAP2_OVERALL_MULTIOBJECTIVE_METHODS.csv", index=False)

    condition_rows: list[dict[str, Any]] = []
    for (alpha, scenario), group in runs.groupby(["alpha", "scenario"], sort=False):
        summary = group.groupby("method", as_index=False).agg({metric: "mean" for metric in metric_specs})
        for metric, higher in metric_specs.items():
            summary[f"{metric}_benefit_norm"] = minmax_benefit(summary[metric], higher)
        summary["pareto_macro_f1_energy"] = pareto_mask(summary[["macro_f1_benefit_norm", "total_normalized_energy_consumed_benefit_norm"]].to_numpy(float))
        summary["pareto_four_objective"] = pareto_mask(summary[["macro_f1_benefit_norm", "total_normalized_energy_consumed_benefit_norm", "final_active_clients_benefit_norm", "jain_participation_fairness_benefit_norm"]].to_numpy(float))
        summary["score_effectiveness_energy"] = 0.5 * summary["macro_f1_benefit_norm"] + 0.5 * summary["total_normalized_energy_consumed_benefit_norm"]
        summary["score_balanced_four_objective"] = (
            summary["macro_f1_benefit_norm"] + summary["total_normalized_energy_consumed_benefit_norm"] + summary["final_active_clients_benefit_norm"] + summary["jain_participation_fairness_benefit_norm"]
        ) / 4.0
        for row in summary.to_dict("records"):
            row.update({"alpha": float(alpha), "scenario": scenario, "method_label": METHOD_LABELS[row["method"]]})
            condition_rows.append(row)
    condition_pareto = pd.DataFrame(condition_rows)
    condition_pareto.to_csv(output_root / "PAMAP2_CONDITION_MULTIOBJECTIVE_PARETO.csv", index=False)

    # ------------------------------------------------------------------
    # Per-run temporal features and trajectory summary.
    # ------------------------------------------------------------------
    temporal_rows: list[dict[str, Any]] = []
    for run_id, group in trajectories.groupby("run_id", sort=True):
        group = group.sort_values("round")
        rounds = group["round"].to_numpy(int)
        macro = group["test_macro_f1"].to_numpy(float)
        bal = group["test_balanced_accuracy"].to_numpy(float)
        active = group["active_clients"].to_numpy(float)
        residual = group["mean_residual_energy"].to_numpy(float)
        row0 = group.iloc[0]
        by_round = group.set_index("round")
        post = group[group["round"] >= ATTACK_START_ROUND]
        nadir_index = post["test_macro_f1"].idxmin()
        nadir = group.loc[nadir_index]
        pressure = group["global_pressure"].astype(float)
        temporal_rows.append({
            "run_id": int(run_id),
            "condition_id": int(row0["condition_id"]),
            "outer_fold": int(row0["outer_fold"]),
            "alpha": float(row0["alpha"]),
            "scenario": str(row0["scenario"]),
            "method": str(row0["method"]),
            "method_label": METHOD_LABELS[str(row0["method"])],
            "fl_seed": int(row0["fl_seed"]),
            "macro_f1_auc_0_100": auc_normalized(rounds, macro, 0, 100),
            "macro_f1_auc_20_100": auc_normalized(rounds, macro, 20, 100),
            "balanced_accuracy_auc_0_100": auc_normalized(rounds, bal, 0, 100),
            "active_clients_auc_0_100": auc_normalized(rounds, active, 0, 100),
            "mean_residual_energy_auc_0_100": auc_normalized(rounds, residual, 0, 100),
            "macro_f1_round15": float(by_round.loc[15, "test_macro_f1"]),
            "macro_f1_round20": float(by_round.loc[20, "test_macro_f1"]),
            "macro_f1_round25": float(by_round.loc[25, "test_macro_f1"]),
            "macro_f1_round40": float(by_round.loc[40, "test_macro_f1"]),
            "macro_f1_round100": float(by_round.loc[100, "test_macro_f1"]),
            "macro_f1_postattack_nadir": float(nadir["test_macro_f1"]),
            "macro_f1_postattack_nadir_round": int(nadir["round"]),
            "macro_f1_delta_round25_minus15": float(by_round.loc[25, "test_macro_f1"] - by_round.loc[15, "test_macro_f1"]),
            "macro_f1_delta_round40_minus15": float(by_round.loc[40, "test_macro_f1"] - by_round.loc[15, "test_macro_f1"]),
            "macro_f1_delta_round100_minus15": float(by_round.loc[100, "test_macro_f1"] - by_round.loc[15, "test_macro_f1"]),
            "macro_f1_final_recovery_from_nadir": float(by_round.loc[100, "test_macro_f1"] - nadir["test_macro_f1"]),
            "active_clients_round100": float(by_round.loc[100, "active_clients"]),
            "mean_residual_energy_round100": float(by_round.loc[100, "mean_residual_energy"]),
            "global_pressure_round20": float(by_round.loc[20, "global_pressure"]) if pd.notna(by_round.loc[20, "global_pressure"]) else math.nan,
            "global_pressure_round40": float(by_round.loc[40, "global_pressure"]) if pd.notna(by_round.loc[40, "global_pressure"]) else math.nan,
            "global_pressure_round100": float(by_round.loc[100, "global_pressure"]) if pd.notna(by_round.loc[100, "global_pressure"]) else math.nan,
            "global_pressure_max": float(pressure.max()) if pressure.notna().any() else math.nan,
        })
    temporal_features = pd.DataFrame(temporal_rows)
    require(len(temporal_features) == EXPECTED_RUNS, "Temporal run-feature row count mismatch")
    temporal_features.to_csv(output_root / "PAMAP2_TEMPORAL_RUN_FEATURES_600.csv", index=False)

    temporal_metric_cols = [c for c in temporal_features.columns if c not in {"run_id", "condition_id", "outer_fold", "alpha", "scenario", "method", "method_label", "fl_seed"}]
    agg_spec: dict[str, list[str]] = {c: ["mean", "std", "median"] for c in temporal_metric_cols}
    temporal_summary = temporal_features.groupby(["alpha", "scenario", "method", "method_label"], as_index=False).agg(agg_spec)
    temporal_summary.columns = ["alpha", "scenario", "method", "method_label"] + [f"{a}_{b}" for a, b in temporal_summary.columns.tolist()[4:]]
    temporal_summary.to_csv(output_root / "PAMAP2_TEMPORAL_SUMMARY_60_CONDITIONS.csv", index=False)

    attack_summary = temporal_summary[temporal_summary["scenario"] != "clean"].copy()
    attack_summary.to_csv(output_root / "PAMAP2_ATTACK_ONSET_RESPONSE_48_ROWS.csv", index=False)

    trajectory_summary_rows: list[dict[str, Any]] = []
    traj_metrics = ("test_macro_f1", "test_balanced_accuracy", "active_clients", "mean_residual_energy", "global_pressure")
    for keys, group in trajectories.groupby(["alpha", "scenario", "method", "round"], sort=False):
        alpha, scenario, method, round_value = keys
        row: dict[str, Any] = {
            "alpha": float(alpha),
            "scenario": str(scenario),
            "method": str(method),
            "method_label": METHOD_LABELS[str(method)],
            "round": int(round_value),
            "n_runs": int(group["run_id"].nunique()),
        }
        for metric in traj_metrics:
            mean, std, low, high = mean_ci(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        trajectory_summary_rows.append(row)
    trajectory_summary = pd.DataFrame(trajectory_summary_rows)
    require(len(trajectory_summary) == 2 * 5 * 6 * 21, "Trajectory summary row-count mismatch")
    trajectory_summary.to_csv(output_root / "PAMAP2_TRAJECTORY_MEAN_CI_1260_ROWS.csv", index=False)

    # ------------------------------------------------------------------
    # Per-activity summaries.
    # ------------------------------------------------------------------
    per_class = per_class.copy()
    per_class["activity_id"] = per_class["class_index"].map(dict(enumerate(ACTIVITY_IDS)))
    per_class["activity_name"] = per_class["activity_id"].map(ACTIVITY_NAMES)
    require(per_class["activity_id"].notna().all() and per_class["activity_name"].notna().all(), "Activity mapping failed")

    class_overall = per_class.groupby(["class_index", "activity_id", "activity_name", "method"], as_index=False)["class_f1"].agg(["mean", "std", "median", "min", "max"]).reset_index()
    class_overall = class_overall.rename(columns={"mean": "class_f1_mean", "std": "class_f1_std", "median": "class_f1_median", "min": "class_f1_min", "max": "class_f1_max"})
    class_overall["method_label"] = class_overall["method"].map(METHOD_LABELS)
    class_overall["rank_within_activity"] = class_overall.groupby("class_index")["class_f1_mean"].rank(method="average", ascending=False)
    class_overall.to_csv(output_root / "PAMAP2_PER_ACTIVITY_OVERALL_72_ROWS.csv", index=False)

    class_condition = per_class.groupby(["alpha", "scenario", "class_index", "activity_id", "activity_name", "method"], as_index=False)["class_f1"].agg(["mean", "std", "median"]).reset_index()
    class_condition = class_condition.rename(columns={"mean": "class_f1_mean", "std": "class_f1_std", "median": "class_f1_median"})
    class_condition["method_label"] = class_condition["method"].map(METHOD_LABELS)
    class_condition["rank_within_condition_activity"] = class_condition.groupby(["alpha", "scenario", "class_index"])["class_f1_mean"].rank(method="average", ascending=False)
    require(len(class_condition) == 2 * 5 * 12 * 6, "Per-activity condition summary row-count mismatch")
    class_condition.to_csv(output_root / "PAMAP2_PER_ACTIVITY_CONDITION_720_ROWS.csv", index=False)

    difficulty = per_class.groupby(["class_index", "activity_id", "activity_name"], as_index=False)["class_f1"].agg(["mean", "std", "median", "min", "max"]).reset_index()
    difficulty = difficulty.rename(columns={"mean": "all_methods_f1_mean", "std": "all_methods_f1_std", "median": "all_methods_f1_median", "min": "all_methods_f1_min", "max": "all_methods_f1_max"})
    best_rows = class_overall.sort_values(["class_index", "class_f1_mean"], ascending=[True, False]).groupby("class_index", as_index=False).first()[["class_index", "method", "method_label", "class_f1_mean"]]
    best_rows = best_rows.rename(columns={"method": "best_method", "method_label": "best_method_label", "class_f1_mean": "best_method_f1_mean"})
    difficulty = difficulty.merge(best_rows, on="class_index", validate="one_to_one")
    difficulty["difficulty_rank_1_is_hardest"] = difficulty["all_methods_f1_mean"].rank(method="average", ascending=True)
    difficulty.to_csv(output_root / "PAMAP2_ACTIVITY_DIFFICULTY_12_ROWS.csv", index=False)

    # Condition-level method means for figure source and article tables.
    condition_methods = runs.groupby(["alpha", "scenario", "method"], as_index=False).agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        energy_consumed_mean=("total_normalized_energy_consumed", "mean"),
        final_active_clients_mean=("final_active_clients", "mean"),
        jain_fairness_mean=("jain_participation_fairness", "mean"),
    )
    condition_methods["method_label"] = condition_methods["method"].map(METHOD_LABELS)
    condition_methods["macro_f1_rank"] = condition_methods.groupby(["alpha", "scenario"])["macro_f1_mean"].rank(method="average", ascending=False)
    condition_methods.to_csv(output_root / "PAMAP2_ARTICLE_CONDITION_METHOD_TABLE_60_ROWS.csv", index=False)

    # Gate-150 evidence ledger restricted to article-relevant primary endpoints.
    evidence = gate150_article[gate150_article["endpoint_role"].astype(str).str.startswith("primary")].copy()
    evidence["direction_label"] = np.where(evidence["mean_benefit_difference_proposed_minus_comparator"] > 0, "proposed_better", np.where(evidence["mean_benefit_difference_proposed_minus_comparator"] < 0, "proposed_worse", "tie"))
    evidence.to_csv(output_root / "PAMAP2_ARTICLE_PRIMARY_EVIDENCE_LEDGER.csv", index=False)

    # ------------------------------------------------------------------
    # Figures.
    # ------------------------------------------------------------------
    plot_scatter(overall, output_root / "FIG151_01_OVERALL_MACRO_F1_VS_ENERGY.png")
    plot_trajectory(trajectory_summary, 0.1, "signflip_mu0p2", output_root / "FIG151_02_SIGNFLIP_MU0P2_ALPHA0P1_MACRO_F1_TRAJECTORY.png")
    plot_trajectory(trajectory_summary, 1.0, "signflip_mu0p4", output_root / "FIG151_03_SIGNFLIP_MU0P4_ALPHA1P0_MACRO_F1_TRAJECTORY.png")
    plot_active_clients(trajectory_summary, output_root / "FIG151_04_ACTIVE_CLIENTS_TRAJECTORY_OVERALL.png")
    plot_arl_pressure(trajectory_summary, output_root / "FIG151_05_ARL_GLOBAL_PRESSURE_ALPHA1P0.png")
    plot_class_heatmap(class_overall, output_root / "FIG151_06_PER_ACTIVITY_F1_HEATMAP.png")
    plot_condition_heatmap(condition_methods, output_root / "FIG151_07_MACRO_F1_CONDITION_HEATMAP.png")

    # ------------------------------------------------------------------
    # Report, audit, binding and manifest.
    # ------------------------------------------------------------------
    overall_sorted_eff = overall.sort_values("rank_effectiveness_energy")
    overall_sorted_bal = overall.sort_values("rank_balanced_four_objective")
    overall_sorted_ret = overall.sort_values("rank_retention_priority")
    pareto_2d = overall[overall["pareto_macro_f1_energy"]]["method_label"].tolist()
    pareto_4d = overall[overall["pareto_four_objective"]]["method_label"].tolist()
    hardest = difficulty.sort_values("difficulty_rank_1_is_hardest").head(3)
    easiest = difficulty.sort_values("difficulty_rank_1_is_hardest", ascending=False).head(3)

    primary_overall = gate150_article[(gate150_article["scope_id"] == "overall") & gate150_article["endpoint_role"].astype(str).str.startswith("primary")]
    tea_macro_sig = primary_overall[(primary_overall["proposed_method"] == "tea_fl") & (primary_overall["metric"] == "macro_f1")]["significant_holm_family_0p05"].sum()
    arl_macro_sig = primary_overall[(primary_overall["proposed_method"] == "arl_fl") & (primary_overall["metric"] == "macro_f1")]["significant_holm_family_0p05"].sum()
    tea_energy_sig = primary_overall[(primary_overall["proposed_method"] == "tea_fl") & (primary_overall["metric"] == "total_normalized_energy_consumed")]["significant_holm_family_0p05"].sum()
    arl_energy_sig = primary_overall[(primary_overall["proposed_method"] == "arl_fl") & (primary_overall["metric"] == "total_normalized_energy_consumed")]["significant_holm_family_0p05"].sum()

    report_lines = [
        "PAMAP2 GATE-151 MULTIOBJECTIVE, TEMPORAL AND PER-ACTIVITY ARTICLE ANALYSIS",
        "=" * 78,
        "",
        "STATUS",
        "-" * 78,
        "PASS",
        "",
        "VERIFIED PROVENANCE",
        "-" * 78,
        f"Gate-149R final binding: {EXPECTED_GATE149R_FINAL_BINDING}",
        f"Gate-150 final binding: {EXPECTED_GATE150_FINAL_BINDING}",
        f"Scientific runs verified: {len(runs)}/{EXPECTED_RUNS}",
        f"Accepted scientific optimizer steps: {int(runs['scientific_optimizer_steps_accounted'].sum())}",
        f"Evaluation trajectory rows: {len(trajectories)}",
        f"Per-activity F1 rows: {len(per_class)}",
        "",
        "GATE-150 PRIMARY OVERALL EVIDENCE CARRIED FORWARD",
        "-" * 78,
        f"TEA-FL Macro-F1 significant Holm comparisons vs four baselines: {int(tea_macro_sig)}/4",
        f"ARL-FL Macro-F1 significant Holm comparisons vs four baselines: {int(arl_macro_sig)}/4",
        f"TEA-FL energy significant Holm comparisons vs four baselines: {int(tea_energy_sig)}/4",
        f"ARL-FL energy significant Holm comparisons vs four baselines: {int(arl_energy_sig)}/4",
        "",
        "MULTIOBJECTIVE RESULTS",
        "-" * 78,
        f"Effectiveness-energy profile leader: {overall_sorted_eff.iloc[0]['method_label']}",
        f"Balanced four-objective profile leader: {overall_sorted_bal.iloc[0]['method_label']}",
        f"Retention-priority illustrative profile leader: {overall_sorted_ret.iloc[0]['method_label']}",
        f"Overall 2D Macro-F1-energy Pareto set: {', '.join(pareto_2d)}",
        f"Overall 4D Macro-F1-energy-active-clients-fairness Pareto set: {', '.join(pareto_4d)}",
        "Composite profiles are transparent decision aids, not inferential endpoints.",
        "",
        "TEMPORAL ANALYSIS",
        "-" * 78,
        f"Per-run temporal feature rows: {len(temporal_features)}",
        f"Condition-method temporal summary rows: {len(temporal_summary)}",
        f"Attack-condition response rows: {len(attack_summary)}",
        f"Trajectory mean/CI rows: {len(trajectory_summary)}",
        f"Attack start round: {ATTACK_START_ROUND}",
        "",
        "PER-ACTIVITY ANALYSIS",
        "-" * 78,
        f"Activity mapping IDs: {list(ACTIVITY_IDS)}",
        "Hardest activities by all-method mean F1: " + "; ".join(f"{r.activity_name}={r.all_methods_f1_mean:.4f}" for r in hardest.itertuples(index=False)),
        "Easiest activities by all-method mean F1: " + "; ".join(f"{r.activity_name}={r.all_methods_f1_mean:.4f}" for r in easiest.itertuples(index=False)),
        "",
        "SCIENTIFIC BOUNDARY",
        "-" * 78,
        "Scientific training executed by Gate-151: NO",
        "Scientific optimizer steps executed by Gate-151: 0",
        "New scientific runs started by Gate-151: 0",
        "Input scientific results modified by Gate-151: NO",
        "New inferential statistical tests executed by Gate-151: NO",
        "CICIoT2023 scientific training started by Gate-151: NO",
        "",
    ]
    report_path = output_root / "GATE151_REPORT.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8", newline="\n")

    data_dictionary = """PAMAP2 GATE-151 DATA DICTIONARY
===============================================================================

PAMAP2_OVERALL_MULTIOBJECTIVE_METHODS.csv
  Overall method means, benefit-oriented min-max normalizations, ranks, three
  transparent illustrative decision profiles, and 2D/4D Pareto membership.

PAMAP2_CONDITION_MULTIOBJECTIVE_PARETO.csv
  Condition-specific method means, normalized objective values, profile scores,
  and Pareto membership for every alpha-scenario condition.

PAMAP2_TEMPORAL_RUN_FEATURES_600.csv
  One row per scientific run. Includes normalized AUCs, pre/post-attack Macro-F1,
  post-attack nadir and recovery, final client/energy state, and ARL pressure.

PAMAP2_TEMPORAL_SUMMARY_60_CONDITIONS.csv
  Mean, standard deviation and median of temporal features for each
  alpha-scenario-method panel (10 matched runs per row).

PAMAP2_ATTACK_ONSET_RESPONSE_48_ROWS.csv
  Temporal summaries restricted to the eight attacked alpha-scenario panels.

PAMAP2_TRAJECTORY_MEAN_CI_1260_ROWS.csv
  Mean, standard deviation, and 95% Student-t intervals at 21 evaluation rounds
  for each alpha-scenario-method panel.

PAMAP2_PER_ACTIVITY_OVERALL_72_ROWS.csv
  Six methods x 12 PAMAP2 activities, with F1 summaries and within-activity rank.

PAMAP2_PER_ACTIVITY_CONDITION_720_ROWS.csv
  Ten alpha-scenario conditions x six methods x 12 activities.

PAMAP2_ACTIVITY_DIFFICULTY_12_ROWS.csv
  Overall activity difficulty and the best mean method for each activity.

PAMAP2_ARTICLE_CONDITION_METHOD_TABLE_60_ROWS.csv
  Compact article-ready means for the ten conditions and six methods.

PAMAP2_ARTICLE_PRIMARY_EVIDENCE_LEDGER.csv
  Gate-150 primary-endpoint comparisons carried forward without rerunning tests.

Benefit normalization
---------------------
Higher values always mean better. Energy consumption is direction-reversed.
The normalization is descriptive and relative to the six methods in this study.

Illustrative profiles
---------------------
Effectiveness-energy: 0.50 Macro-F1 + 0.50 energy efficiency.
Balanced four-objective: equal Macro-F1, energy efficiency, active clients,
and participation fairness.
Retention priority: 0.35 Macro-F1 + 0.20 energy efficiency + 0.30 active clients
+ 0.15 fairness. Profile weights are decision aids, not statistical endpoints.

PAMAP2 activity mapping
-----------------------
class_index 0..11 maps in order to activity IDs
1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24.
"""
    dictionary_path = output_root / "PAMAP2_GATE151_DATA_DICTIONARY.txt"
    dictionary_path.write_text(data_dictionary, encoding="utf-8", newline="\n")

    output_hashes_before_audit: dict[str, str] = {}
    for path in sorted(output_root.iterdir()):
        if path.is_file() and path.name not in {"GATE151_AUDIT.json", "GATE151_FINAL_BINDING.json", "MANIFEST_SHA256.csv"}:
            output_hashes_before_audit[path.name] = sha256_file(path)

    audit = {
        "gate_id": GATE_ID,
        "status": "PASS",
        "scope": SCOPE,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_gate149r_final_binding_sha256": EXPECTED_GATE149R_FINAL_BINDING,
        "input_gate150_final_binding_sha256": EXPECTED_GATE150_FINAL_BINDING,
        "input_run_level_results_sha256": EXPECTED_RUN_LEVEL_SHA256,
        "input_trajectory_sha256": EXPECTED_TRAJECTORY_SHA256,
        "input_per_class_sha256": EXPECTED_PER_CLASS_SHA256,
        "scientific_runs_verified": EXPECTED_RUNS,
        "accepted_scientific_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
        "trajectory_rows_verified": EXPECTED_TRAJECTORY_ROWS,
        "per_class_rows_verified": EXPECTED_PER_CLASS_ROWS,
        "temporal_run_feature_rows": len(temporal_features),
        "trajectory_summary_rows": len(trajectory_summary),
        "condition_pareto_rows": len(condition_pareto),
        "per_activity_overall_rows": len(class_overall),
        "per_activity_condition_rows": len(class_condition),
        "figure_count": len(list(output_root.glob("FIG151_*.png"))),
        "activity_ids": list(ACTIVITY_IDS),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
        "output_hashes_before_audit_and_binding": output_hashes_before_audit,
        "scientific_training_executed_by_gate151": False,
        "scientific_optimizer_steps_executed_by_gate151": 0,
        "new_scientific_runs_started_by_gate151": 0,
        "input_scientific_results_modified_by_gate151": False,
        "new_inferential_statistical_tests_executed_by_gate151": False,
        "ciciot2023_scientific_training_started_by_gate151": False,
    }
    audit_path = output_root / "GATE151_AUDIT.json"
    write_json(audit_path, audit)

    binding = {
        "gate_id": GATE_ID,
        "status": "PASS",
        "scope": SCOPE,
        "gate149r_final_binding_sha256": EXPECTED_GATE149R_FINAL_BINDING,
        "gate150_final_binding_sha256": EXPECTED_GATE150_FINAL_BINDING,
        "run_level_results_sha256": EXPECTED_RUN_LEVEL_SHA256,
        "trajectory_sha256": EXPECTED_TRAJECTORY_SHA256,
        "per_class_sha256": EXPECTED_PER_CLASS_SHA256,
        "scientific_runs_verified": EXPECTED_RUNS,
        "accepted_scientific_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
        "report_sha256": sha256_file(report_path),
        "data_dictionary_sha256": sha256_file(dictionary_path),
        "audit_sha256": sha256_file(audit_path),
        "overall_multiobjective_sha256": sha256_file(output_root / "PAMAP2_OVERALL_MULTIOBJECTIVE_METHODS.csv"),
        "condition_pareto_sha256": sha256_file(output_root / "PAMAP2_CONDITION_MULTIOBJECTIVE_PARETO.csv"),
        "temporal_features_sha256": sha256_file(output_root / "PAMAP2_TEMPORAL_RUN_FEATURES_600.csv"),
        "trajectory_summary_sha256": sha256_file(output_root / "PAMAP2_TRAJECTORY_MEAN_CI_1260_ROWS.csv"),
        "per_activity_overall_sha256": sha256_file(output_root / "PAMAP2_PER_ACTIVITY_OVERALL_72_ROWS.csv"),
        "per_activity_condition_sha256": sha256_file(output_root / "PAMAP2_PER_ACTIVITY_CONDITION_720_ROWS.csv"),
        "article_primary_evidence_sha256": sha256_file(output_root / "PAMAP2_ARTICLE_PRIMARY_EVIDENCE_LEDGER.csv"),
        "scientific_training_executed_by_gate151": False,
        "scientific_optimizer_steps_executed_by_gate151": 0,
        "new_scientific_runs_started_by_gate151": 0,
        "input_scientific_results_modified_by_gate151": False,
        "new_inferential_statistical_tests_executed_by_gate151": False,
        "ciciot2023_scientific_training_started_by_gate151": False,
    }
    binding_hash = canonical_sha256(binding)
    binding["gate151_final_binding_sha256"] = binding_hash
    binding_path = output_root / "GATE151_FINAL_BINDING.json"
    write_json(binding_path, binding)

    manifest_rows: list[dict[str, Any]] = []
    for path in sorted(output_root.iterdir()):
        if path.is_file() and path.name != "MANIFEST_SHA256.csv":
            manifest_rows.append({"filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_root / "MANIFEST_SHA256.csv", index=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_base = handoff_root / f"pamap2_gate151_article_analysis_{timestamp}"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=output_root.parent, base_dir=output_root.name))
    archive_hash = sha256_file(archive_path)

    print("=" * 78)
    print("GATE151_PASS")
    print("=" * 78)
    print(f"Scientific runs verified: {EXPECTED_RUNS}/{EXPECTED_RUNS}")
    print(f"Accepted scientific optimizer steps: {EXPECTED_OPTIMIZER_STEPS}")
    print(f"Trajectory rows verified: {EXPECTED_TRAJECTORY_ROWS}")
    print(f"Per-activity rows verified: {EXPECTED_PER_CLASS_ROWS}")
    print(f"Temporal run-feature rows: {len(temporal_features)}")
    print(f"Trajectory summary rows: {len(trajectory_summary)}")
    print(f"Condition multiobjective rows: {len(condition_pareto)}")
    print(f"Article figures created: {len(list(output_root.glob('FIG151_*.png')))}")
    print("Scientific training executed by Gate-151: NO")
    print("Scientific optimizer steps executed by Gate-151: 0")
    print("New scientific runs started by Gate-151: 0")
    print("Input scientific results modified by Gate-151: NO")
    print("New inferential statistical tests executed by Gate-151: NO")
    print("CICIoT2023 scientific training started by Gate-151: NO")
    print(f"Gate-151 final binding SHA256: {binding_hash}")
    print(f"Report: {report_path}")
    print(f"Evidence ZIP: {archive_path}")
    print(f"Evidence ZIP SHA256: {archive_hash}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"GATE151_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
