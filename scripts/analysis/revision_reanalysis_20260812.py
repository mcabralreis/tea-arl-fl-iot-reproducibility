from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "revised_analysis"
PAM_ROOT = ROOT / "results" / "campaign_outputs" / "pamap2"
CIC_ROOT = ROOT / "results" / "campaign_outputs" / "ciciot2023"
PART_ROOT = ROOT / "manifests" / "pamap2" / "partitions"

METHOD_ORDER_PAM = [
    "fedavg",
    "fedprox",
    "random_trimmed_mean",
    "fedle_adapted",
    "tea_fl",
    "arl_fl",
]
METHOD_LABEL_PAM = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "random_trimmed_mean": "Random Trimmed Mean",
    "fedle_adapted": "FedLE-adapted",
    "tea_fl": "TEA-FL",
    "arl_fl": "ARL-FL",
}


def rankdata(values: np.ndarray, method: str = "average") -> np.ndarray:
    if method != "average":
        raise ValueError("Only average ranks are implemented")
    x = np.asarray(values)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranked = np.empty(len(x), dtype=float)
    start = 0
    while start < len(x):
        stop = start + 1
        while stop < len(x) and sorted_x[stop] == sorted_x[start]:
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        ranked[order[start:stop]] = average_rank
        start = stop
    return ranked


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate(
        (len(p) - np.arange(len(p))) * p[order]
    )
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def hodges_lehmann_paired(differences: np.ndarray) -> float:
    d = np.asarray(differences, dtype=float)
    walsh = [(d[i] + d[j]) / 2.0 for i in range(len(d)) for j in range(i, len(d))]
    return float(np.median(walsh))


def exact_signed_rank(differences: np.ndarray) -> dict[str, float | int]:
    d = np.asarray(differences, dtype=float)
    d = d[~np.isclose(d, 0.0, rtol=0.0, atol=1e-15)]
    n = len(d)
    if n == 0:
        return {
            "n_nonzero": 0,
            "wins": 0,
            "ties": len(differences),
            "losses": 0,
            "rank_biserial": 0.0,
            "p_exact_two_sided": 1.0,
            "p_sign_two_sided": 1.0,
            "hodges_lehmann": 0.0,
        }
    ranks = rankdata(np.abs(d), method="average")
    total = float(ranks.sum())
    observed = float(np.dot(ranks, np.sign(d)))
    assignments = np.arange(1 << n, dtype=np.uint32)[:, None]
    signs = np.where(
        (assignments & (1 << np.arange(n, dtype=np.uint32))) != 0,
        1.0,
        -1.0,
    )
    signed_sums = signs @ ranks
    p_exact = float(
        np.mean(np.abs(signed_sums) >= abs(observed) - 1e-12)
    )
    wins = int(np.sum(d > 0))
    losses = int(np.sum(d < 0))
    tail = min(wins, losses)
    p_sign = min(
        1.0,
        2.0
        * sum(math.comb(n, k) for k in range(tail + 1))
        / (2**n),
    )
    return {
        "n_nonzero": n,
        "wins": wins,
        "ties": int(len(differences) - n),
        "losses": losses,
        "rank_biserial": observed / total,
        "p_exact_two_sided": p_exact,
        "p_sign_two_sided": p_sign,
        "hodges_lehmann": hodges_lehmann_paired(d),
    }


def bootstrap_mean_ci(
    differences: np.ndarray, seed: int, replicates: int = 5000
) -> tuple[float, float]:
    d = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(d), size=(replicates, len(d)))
    means = d[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def friedman_statistic(matrix: np.ndarray) -> tuple[float, float]:
    x = np.asarray(matrix, dtype=float)
    n, k = x.shape
    ranks = np.vstack([rankdata(row, method="average") for row in x])
    rank_sums = ranks.sum(axis=0)
    q = 12.0 / (n * k * (k + 1.0)) * np.sum(rank_sums**2) - 3.0 * n * (k + 1.0)
    tie_sum = 0.0
    for row in x:
        _, counts = np.unique(row, return_counts=True)
        tie_sum += float(np.sum(counts**3 - counts))
    correction = 1.0 - tie_sum / (n * (k**3 - k))
    if correction > 0:
        q /= correction
    return float(q), float(q / (n * (k - 1.0)))


def friedman_randomization_p(
    matrix: np.ndarray,
    seed: int,
    permutations: int = 200_000,
    batch_size: int = 5_000,
) -> tuple[float, int]:
    x = np.asarray(matrix, dtype=float)
    n, k = x.shape
    ranks = np.vstack([rankdata(row, method="average") for row in x])
    observed, _ = friedman_statistic(x)
    tie_sum = 0.0
    for row in x:
        _, counts = np.unique(row, return_counts=True)
        tie_sum += float(np.sum(counts**3 - counts))
    correction = 1.0 - tie_sum / (n * (k**3 - k))
    rng = np.random.default_rng(seed)
    exceed = 0
    done = 0
    while done < permutations:
        b = min(batch_size, permutations - done)
        rank_sums = np.zeros((b, k), dtype=float)
        for row_index in range(n):
            keys = rng.random((b, k))
            perm = np.argsort(keys, axis=1)
            rank_sums += ranks[row_index][perm]
        q = (
            12.0 / (n * k * (k + 1.0)) * np.sum(rank_sums**2, axis=1)
            - 3.0 * n * (k + 1.0)
        )
        if correction > 0:
            q /= correction
        exceed += int(np.sum(q >= observed - 1e-12))
        done += b
    return float((exceed + 1) / (permutations + 1)), exceed


def paired_comparisons(
    blocks: pd.DataFrame,
    block_column: str,
    metric: str,
    methods: list[str],
    higher_is_better: bool,
    seed_base: int,
) -> pd.DataFrame:
    wide = blocks.pivot(index=block_column, columns="method", values=metric)
    rows: list[dict[str, object]] = []
    for comparison_index, (a, b) in enumerate(itertools.combinations(methods, 2)):
        raw_difference = wide[a].to_numpy() - wide[b].to_numpy()
        benefit = raw_difference if higher_is_better else -raw_difference
        exact = exact_signed_rank(benefit)
        low, high = bootstrap_mean_ci(
            benefit, seed=seed_base + comparison_index
        )
        rows.append(
            {
                "metric": metric,
                "method_a": a,
                "method_a_label": METHOD_LABEL_PAM.get(a, a),
                "method_b": b,
                "method_b_label": METHOD_LABEL_PAM.get(b, b),
                "n_blocks": len(benefit),
                "mean_a": float(wide[a].mean()),
                "mean_b": float(wide[b].mean()),
                "mean_benefit_a_minus_b": float(benefit.mean()),
                "bootstrap_95ci_low": low,
                "bootstrap_95ci_high": high,
                **exact,
            }
        )
    adjusted = holm_adjust([float(row["p_exact_two_sided"]) for row in rows])
    for row, p_holm in zip(rows, adjusted):
        row["p_holm_15_pairs"] = p_holm
    return pd.DataFrame(rows)


def prepare_pamap2() -> dict[str, object]:
    runs = pd.read_csv(
        PAM_ROOT / "postcampaign_freeze_v1" / "PAMAP2_RUN_LEVEL_RESULTS_600.csv"
    )
    trajectories = pd.read_csv(
        PAM_ROOT
        / "postcampaign_freeze_v1"
        / "PAMAP2_EVALUATION_TRAJECTORIES_12600_ROWS.csv"
    )
    runs["energy_per_completed_round"] = (
        runs["total_normalized_energy_consumed"]
        / runs["completed_training_rounds"]
    )
    runs["completed_round_fraction"] = runs["completed_training_rounds"] / 100.0
    block_keys = ["outer_fold", "alpha", "scenario", "fl_seed"]
    minimum_round = (
        runs.groupby(block_keys, as_index=False)["completed_training_rounds"]
        .min()
        .rename(columns={"completed_training_rounds": "minimum_completed_round"})
    )
    minimum_round["common_horizon_round"] = (
        minimum_round["minimum_completed_round"] // 5 * 5
    ).astype(int)
    runs = runs.merge(minimum_round, on=block_keys, how="left", validate="many_to_one")
    initial = (
        trajectories.loc[trajectories["round"] == 0, ["run_id", "mean_residual_energy"]]
        .rename(columns={"mean_residual_energy": "initial_mean_residual_energy"})
    )
    horizon = trajectories.merge(
        runs[["run_id", "common_horizon_round"]], on="run_id", how="inner"
    )
    horizon = horizon.loc[
        horizon["round"] == horizon["common_horizon_round"],
        ["run_id", "test_macro_f1", "mean_residual_energy"],
    ].rename(
        columns={
            "test_macro_f1": "common_horizon_macro_f1",
            "mean_residual_energy": "common_horizon_mean_residual_energy",
        }
    )
    runs = runs.merge(initial, on="run_id", validate="one_to_one")
    runs = runs.merge(horizon, on="run_id", validate="one_to_one")
    runs["common_horizon_energy"] = 28.0 * (
        runs["initial_mean_residual_energy"]
        - runs["common_horizon_mean_residual_energy"]
    )
    runs.to_csv(OUT / "PAMAP2_RUNS_WITH_COMPARABILITY_ENDPOINTS.csv", index=False)
    minimum_round.to_csv(OUT / "PAMAP2_COMMON_HORIZONS_100_BLOCKS.csv", index=False)

    fold_metrics = [
        "macro_f1",
        "balanced_accuracy",
        "final_mean_residual_energy",
        "final_active_clients",
        "jain_participation_fairness",
        "completed_training_rounds",
        "completed_round_fraction",
        "total_normalized_energy_consumed",
        "energy_per_completed_round",
        "common_horizon_energy",
        "common_horizon_macro_f1",
    ]
    fold_blocks = (
        runs.groupby(["outer_fold", "method"], as_index=False)[fold_metrics]
        .mean()
        .sort_values(["outer_fold", "method"])
    )
    fold_blocks.to_csv(OUT / "PAMAP2_OVERALL_FIVE_FOLD_BLOCKS.csv", index=False)

    summary = (
        fold_blocks.groupby("method")[fold_metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "method" if column[0] == "method" else f"{column[0]}_{column[1]}"
        for column in summary.columns
    ]
    summary["method_label"] = summary["method"].map(METHOD_LABEL_PAM)
    completion_counts = (
        runs.groupby("method")["completed_training_rounds"]
        .agg(
            runs="size",
            completed_round_100=lambda x: int(np.sum(x == 100)),
            minimum="min",
            median="median",
            maximum="max",
        )
        .reset_index()
    )
    completion_counts["round_100_percent"] = (
        100.0
        * completion_counts["completed_round_100"]
        / completion_counts["runs"]
    )
    completion_counts.to_csv(OUT / "PAMAP2_COMPLETED_ROUNDS_BY_METHOD.csv", index=False)
    summary = summary.merge(completion_counts, on="method", how="left")
    summary.to_csv(OUT / "PAMAP2_TABLE5_REVISED_VALUES.csv", index=False)

    omnibus_rows = []
    pairwise_frames = []
    for metric_index, (metric, higher) in enumerate(
        [
            ("macro_f1", True),
            ("total_normalized_energy_consumed", False),
            ("energy_per_completed_round", False),
            ("common_horizon_macro_f1", True),
            ("common_horizon_energy", False),
            ("completed_round_fraction", True),
            ("final_active_clients", True),
            ("jain_participation_fairness", True),
        ]
    ):
        matrix = (
            fold_blocks.pivot(index="outer_fold", columns="method", values=metric)
            .reindex(columns=METHOD_ORDER_PAM)
            .to_numpy()
        )
        q, w = friedman_statistic(matrix)
        p_perm, exceed = friedman_randomization_p(
            matrix, seed=20260812 + metric_index
        )
        omnibus_rows.append(
            {
                "metric": metric,
                "n_fold_blocks": 5,
                "methods": 6,
                "friedman_q": q,
                "kendall_w": w,
                "randomization_p": p_perm,
                "randomizations": 200_000,
                "randomizations_at_least_as_extreme": exceed,
            }
        )
        pairwise_frames.append(
            paired_comparisons(
                fold_blocks,
                "outer_fold",
                metric,
                METHOD_ORDER_PAM,
                higher,
                seed_base=20261000 + 100 * metric_index,
            )
        )
    pd.DataFrame(omnibus_rows).to_csv(
        OUT / "PAMAP2_FIVE_FOLD_OMNIBUS.csv", index=False
    )
    pd.concat(pairwise_frames, ignore_index=True).to_csv(
        OUT / "PAMAP2_FIVE_FOLD_PAIRWISE.csv", index=False
    )

    overall = runs.groupby("method", as_index=False).agg(
        common_horizon_macro_f1=("common_horizon_macro_f1", "mean"),
        common_horizon_energy=("common_horizon_energy", "mean"),
        completed_round_fraction=("completed_round_fraction", "mean"),
        protocol_total_energy=("total_normalized_energy_consumed", "mean"),
    )
    dominated = []
    for _, row in overall.iterrows():
        is_dominated = False
        for _, other in overall.iterrows():
            if other["method"] == row["method"]:
                continue
            no_worse = (
                other["common_horizon_macro_f1"] >= row["common_horizon_macro_f1"]
                and other["common_horizon_energy"] <= row["common_horizon_energy"]
                and other["completed_round_fraction"] >= row["completed_round_fraction"]
            )
            strictly_better = (
                other["common_horizon_macro_f1"] > row["common_horizon_macro_f1"]
                or other["common_horizon_energy"] < row["common_horizon_energy"]
                or other["completed_round_fraction"] > row["completed_round_fraction"]
            )
            if no_worse and strictly_better:
                is_dominated = True
                break
        dominated.append(is_dominated)
    overall["pareto_nondominated_three_objective"] = ~np.asarray(dominated)
    overall["method_label"] = overall["method"].map(METHOD_LABEL_PAM)
    overall.to_csv(OUT / "PAMAP2_COMMON_HORIZON_PARETO.csv", index=False)
    return {"runs": runs, "fold_blocks": fold_blocks}


def prepare_ablation(primary_runs: pd.DataFrame) -> None:
    ablation = pd.read_csv(
        PAM_ROOT / "ablation_analysis_v1" / "PAMAP2_ABLATION_RUN_RESULTS_240.csv"
    )
    full = primary_runs[
        [
            "run_id",
            "macro_f1",
            "total_normalized_energy_consumed",
            "final_active_clients",
            "jain_participation_fairness",
        ]
    ].rename(
        columns={
            "run_id": "reference_run_id",
            "macro_f1": "full_macro_f1",
            "total_normalized_energy_consumed": "full_energy",
            "final_active_clients": "full_active_clients",
            "jain_participation_fairness": "full_fairness",
        }
    )
    paired = ablation.merge(full, on="reference_run_id", validate="many_to_one")
    paired["macro_f1_benefit"] = paired["full_macro_f1"] - paired["macro_f1"]
    paired["energy_benefit"] = (
        paired["total_normalized_energy_consumed"] - paired["full_energy"]
    )
    paired["active_client_benefit"] = (
        paired["full_active_clients"] - paired["final_active_clients"]
    )
    paired["fairness_benefit"] = paired["full_fairness"] - paired["jain_participation_fairness"]
    benefit_metrics = [
        "macro_f1_benefit",
        "energy_benefit",
        "active_client_benefit",
        "fairness_benefit",
    ]
    folds = (
        paired.groupby(
            [
                "variant",
                "variant_label",
                "component_isolated",
                "alpha",
                "scenario",
                "outer_fold",
            ],
            as_index=False,
        )[benefit_metrics]
        .mean()
    )
    folds.to_csv(OUT / "PAMAP2_ABLATION_FIVE_FOLD_BLOCKS.csv", index=False)
    rows = []
    for group_index, (keys, group) in enumerate(
        folds.groupby(
            ["variant", "variant_label", "component_isolated", "alpha", "scenario"],
            sort=True,
        )
    ):
        variant, variant_label, component, alpha, scenario = keys
        for metric_index, metric in enumerate(benefit_metrics):
            d = group[metric].to_numpy()
            exact = exact_signed_rank(d)
            low, high = bootstrap_mean_ci(
                d, seed=20262000 + group_index * 10 + metric_index
            )
            rows.append(
                {
                    "variant": variant,
                    "variant_label": variant_label,
                    "component_isolated": component,
                    "alpha": alpha,
                    "scenario": scenario,
                    "metric": metric,
                    "n_fold_blocks": 5,
                    "mean_benefit": float(d.mean()),
                    "bootstrap_95ci_low": low,
                    "bootstrap_95ci_high": high,
                    **exact,
                }
            )
    condition_tests = pd.DataFrame(rows)
    condition_tests.to_csv(
        OUT / "PAMAP2_ABLATION_FIVE_FOLD_CONDITION_TESTS.csv", index=False
    )
    component_rows = []
    for (variant, variant_label, component, metric), group in condition_tests.groupby(
        ["variant", "variant_label", "component_isolated", "metric"], sort=True
    ):
        component_rows.append(
            {
                "variant": variant,
                "variant_label": variant_label,
                "component_isolated": component,
                "metric": metric,
                "conditions": len(group),
                "mean_of_condition_mean_benefits": float(group["mean_benefit"].mean()),
                "conditions_positive_mean": int(np.sum(group["mean_benefit"] > 0)),
                "conditions_zero_mean": int(np.sum(np.isclose(group["mean_benefit"], 0))),
                "conditions_negative_mean": int(np.sum(group["mean_benefit"] < 0)),
                "conditions_raw_exact_p_below_0p05": int(
                    np.sum(group["p_exact_two_sided"] < 0.05)
                ),
                "minimum_exact_p": float(group["p_exact_two_sided"].min()),
            }
        )
    pd.DataFrame(component_rows).to_csv(
        OUT / "PAMAP2_ABLATION_REVISED_TABLE7_VALUES.csv", index=False
    )


def prepare_ciciot2023() -> None:
    master = pd.read_csv(
        CIC_ROOT / "scientific_campaign_final_audit_v1" / "MASTER_RUN_RESULTS.csv"
    )
    master["method"] = master["method_id"]
    methods = [
        "FEDAVG",
        "FEDPROX",
        "RANDOM_TRIMMED_MEAN",
        "FEDLE_ADAPTED",
        "TEA_FL",
        "ARL_FL",
    ]
    seed_blocks = (
        master.groupby(
            ["scenario_id", "experimental_seed", "method"], as_index=False
        )["test_macro_f1"]
        .mean()
        .rename(columns={"test_macro_f1": "macro_f1_alpha_average"})
    )
    seed_blocks.to_csv(OUT / "CICIOT2023_PRIMARY_FIVE_SEED_BLOCKS.csv", index=False)
    omnibus_rows = []
    pairwise_rows = []
    for scenario_index, scenario in enumerate(
        ["CLEAN", "SIGNFLIP_MU0P2", "SIGNFLIP_MU0P4"]
    ):
        scenario_blocks = seed_blocks.loc[seed_blocks["scenario_id"] == scenario]
        wide = (
            scenario_blocks.pivot(
                index="experimental_seed", columns="method", values="macro_f1_alpha_average"
            )
            .reindex(columns=methods)
        )
        matrix = wide.to_numpy()
        q, w = friedman_statistic(matrix)
        p_perm, exceed = friedman_randomization_p(
            matrix, seed=20263000 + scenario_index
        )
        omnibus_rows.append(
            {
                "scenario": scenario,
                "n_seed_blocks": 5,
                "friedman_q": q,
                "kendall_w": w,
                "randomization_p": p_perm,
                "randomizations": 200_000,
                "randomizations_at_least_as_extreme": exceed,
            }
        )
        scenario_pairs = []
        for comparison_index, baseline in enumerate(methods[:-1]):
            d = wide["ARL_FL"].to_numpy() - wide[baseline].to_numpy()
            exact = exact_signed_rank(d)
            low, high = bootstrap_mean_ci(
                d, seed=20263100 + 10 * scenario_index + comparison_index
            )
            scenario_pairs.append(
                {
                    "scenario": scenario,
                    "comparison": f"ARL_FL - {baseline}",
                    "baseline": baseline,
                    "n_seed_blocks": 5,
                    "arl_mean": float(wide["ARL_FL"].mean()),
                    "baseline_mean": float(wide[baseline].mean()),
                    "mean_difference": float(d.mean()),
                    "bootstrap_95ci_low": low,
                    "bootstrap_95ci_high": high,
                    **exact,
                }
            )
        adjusted = holm_adjust(
            [float(row["p_exact_two_sided"]) for row in scenario_pairs]
        )
        for row, p_holm in zip(scenario_pairs, adjusted):
            row["p_holm_five_planned"] = p_holm
        pairwise_rows.extend(scenario_pairs)
    pd.DataFrame(omnibus_rows).to_csv(
        OUT / "CICIOT2023_FIVE_SEED_OMNIBUS.csv", index=False
    )
    pd.DataFrame(pairwise_rows).to_csv(
        OUT / "CICIOT2023_FIVE_SEED_ARL_PAIRWISE.csv", index=False
    )


def prepare_virtual_clients() -> None:
    manifest = pd.read_csv(PART_ROOT / "outer_fold_client_manifest.csv")
    rows = []
    for (fold, test_subject, alpha), group in manifest.groupby(
        ["outer_fold", "outer_test_subject", "alpha"], sort=True
    ):
        windows = group["windows"].to_numpy()
        entropy = group["normalized_label_entropy"].to_numpy()
        divergence = group["js_divergence_from_fold"].to_numpy()
        rows.append(
            {
                "outer_fold": fold,
                "outer_test_subject": test_subject,
                "alpha": alpha,
                "clients": len(group),
                "window_min": float(np.min(windows)),
                "window_q1": float(np.quantile(windows, 0.25)),
                "window_median": float(np.median(windows)),
                "window_q3": float(np.quantile(windows, 0.75)),
                "window_max": float(np.max(windows)),
                "entropy_min": float(np.min(entropy)),
                "entropy_q1": float(np.quantile(entropy, 0.25)),
                "entropy_median": float(np.median(entropy)),
                "entropy_q3": float(np.quantile(entropy, 0.75)),
                "entropy_max": float(np.max(entropy)),
                "js_divergence_median": float(np.median(divergence)),
                "unique_original_subjects_per_client_min": 1,
                "unique_original_subjects_per_client_max": 1,
                "largest_subject_share_per_client": 1.0,
                "partition_depends_on_fl_seed": False,
            }
        )
    pd.DataFrame(rows).to_csv(
        OUT / "PAMAP2_VIRTUAL_CLIENT_CHARACTERIZATION.csv", index=False
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pam = prepare_pamap2()
    prepare_ablation(pam["runs"])
    prepare_ciciot2023()
    prepare_virtual_clients()
    (OUT / "ANALYSIS_METADATA.json").write_text(
        json.dumps(
            {
                "analysis_date": "2026-08-12",
                "purpose": "Coauthor-requested inferential-unit and unequal-horizon reanalysis",
                "no_training_repeated": True,
                "pamap2_primary_unit": "outer fold after averaging federated seeds",
                "ciciot2023_primary_unit": "experimental seed after averaging alpha conditions",
                "friedman_randomizations": 200000,
                "bootstrap_replicates": 5000,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Revision analysis written to {OUT}")


if __name__ == "__main__":
    main()
