from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GATE113_BINDING = "7649BB9E5EA9452C53D26948D3A45D057C85A693A06896D6D556E6D6952BCE6A"
GATE114_BINDING = "237B16492B01942CACE6101A5A1DA4BE947BE44AFFBE2778EBE09DD68B656DC5"
METHOD_ORDER = [
    "FEDAVG",
    "FEDPROX",
    "RANDOM_TRIMMED_MEAN",
    "FEDLE_ADAPTED",
    "TEA_FL",
    "ARL_FL",
]
METHOD_LABEL = {
    "FEDAVG": "FedAvg",
    "FEDPROX": "FedProx",
    "RANDOM_TRIMMED_MEAN": "Random+Trimmed Mean",
    "FEDLE_ADAPTED": "FedLE-adapted",
    "TEA_FL": "TEA-FL",
    "ARL_FL": "ARL-FL",
}
SCENARIO_ORDER = ["CLEAN", "SIGNFLIP_MU0P2", "SIGNFLIP_MU0P4"]
SCENARIO_LABEL = {
    "CLEAN": "Clean",
    "SIGNFLIP_MU0P2": "Sign-flip μ=0.2",
    "SIGNFLIP_MU0P4": "Sign-flip μ=0.4",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def canonical_json_sha256(payload: dict) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest().upper()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def preserve_previous(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    index = 1
    while True:
        candidate = output_dir.with_name(f"{output_dir.name}_previous_v{index}")
        if not candidate.exists():
            shutil.move(str(output_dir), str(candidate))
            return candidate
        index += 1


def read_binding(path: Path, key: str, expected: str) -> dict:
    require(path.is_file(), f"Missing binding file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    observed = str(payload.get(key, "")).upper()
    require(observed == expected, f"Binding mismatch in {path.name}: {observed} != {expected}")
    return payload


def format_mean_sd(mean: float, std: float, decimals: int = 4) -> str:
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def write_latex_table(path: Path, headers: list[str], rows: Iterable[Iterable[object]], caption: str, label: str) -> None:
    rows = [list(row) for row in rows]
    colspec = "l" + "c" * (len(headers) - 1)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(latex_escape(h) for h in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(x) for x in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_main_performance_table(desc: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    d = desc[desc["metric"] == "test_macro_f1"].copy()
    d["method_id"] = pd.Categorical(d["method_id"], METHOD_ORDER, ordered=True)
    d["scenario_id"] = pd.Categorical(d["scenario_id"], SCENARIO_ORDER, ordered=True)
    d = d.sort_values(["alpha", "method_id", "scenario_id"])
    d["mean_sd"] = [format_mean_sd(m, s) for m, s in zip(d["mean"], d["std"])]

    records = []
    for alpha in sorted(d["alpha"].unique()):
        for method in METHOD_ORDER:
            sub = d[(d["alpha"] == alpha) & (d["method_id"] == method)]
            row = {"alpha": alpha, "method_id": method, "method_label": METHOD_LABEL[method]}
            for scenario in SCENARIO_ORDER:
                cell = sub[sub["scenario_id"] == scenario]
                require(len(cell) == 1, f"Missing performance cell: alpha={alpha}, method={method}, scenario={scenario}")
                row[scenario] = cell.iloc[0]["mean_sd"]
                row[f"{scenario}_mean"] = float(cell.iloc[0]["mean"])
                row[f"{scenario}_std"] = float(cell.iloc[0]["std"])
            records.append(row)
    table = pd.DataFrame(records)
    table.to_csv(output_dir / "TABLE1_MACRO_F1_BY_ALPHA_METHOD_SCENARIO.csv", index=False)

    latex_rows = []
    for alpha in sorted(table["alpha"].unique()):
        for _, row in table[table["alpha"] == alpha].iterrows():
            latex_rows.append([
                f"{alpha:.1f}", row["method_label"], row["CLEAN"], row["SIGNFLIP_MU0P2"], row["SIGNFLIP_MU0P4"]
            ])
    write_latex_table(
        output_dir / "TABLE1_MACRO_F1_BY_ALPHA_METHOD_SCENARIO.tex",
        ["α", "Method", "Clean", "Sign-flip μ=0.2", "Sign-flip μ=0.4"],
        latex_rows,
        "Test Macro-F1 (mean ± standard deviation across five seeds) by heterogeneity level, method, and scenario.",
        "tab:ciciot_macro_f1",
    )
    return table


def build_statistical_table(friedman: pd.DataFrame, pairwise: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIO_ORDER:
        f = friedman[friedman["scenario_id"] == scenario].iloc[0]
        sig = pairwise[(pairwise["scenario_id"] == scenario) & (pairwise["exact_signflip_holm_p_value"] < 0.05)]
        sig_text = "; ".join(
            f"ARL-FL vs {METHOD_LABEL[m]} (Δ={d:+.4f}, Holm p={p:.4g})"
            for m, d, p in zip(sig["baseline_method"], sig["mean_difference"], sig["exact_signflip_holm_p_value"])
        ) or "None"
        rows.append({
            "scenario_id": scenario,
            "scenario_label": SCENARIO_LABEL[scenario],
            "friedman_chi_square": float(f["friedman_chi_square"]),
            "friedman_df": int(f["df"]),
            "friedman_p_value": float(f["p_value"]),
            "kendalls_w": float(f["kendalls_w"]),
            "significant_arl_pairwise_after_holm": sig_text,
        })
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "TABLE2_PRIMARY_STATISTICAL_TESTS.csv", index=False)
    write_latex_table(
        output_dir / "TABLE2_PRIMARY_STATISTICAL_TESTS.tex",
        ["Scenario", "Friedman χ²(5)", "p", "Kendall W", "Significant ARL-FL comparisons"],
        [[
            r["scenario_label"], f'{r["friedman_chi_square"]:.3f}', f'{r["friedman_p_value"]:.4g}',
            f'{r["kendalls_w"]:.3f}', r["significant_arl_pairwise_after_holm"]
        ] for _, r in table.iterrows()],
        "Primary repeated-measures tests over ten frozen (α, seed) blocks.",
        "tab:ciciot_primary_stats",
    )
    return table


def build_robustness_table(robust: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    r = robust.copy()
    r["method_id"] = pd.Categorical(r["method_id"], METHOD_ORDER, ordered=True)
    r["attack_scenario"] = pd.Categorical(r["attack_scenario"], SCENARIO_ORDER[1:], ordered=True)
    r = r.sort_values(["alpha", "method_id", "attack_scenario"])
    out = r[[
        "alpha", "method_id", "attack_scenario", "absolute_drop_mean", "absolute_drop_std",
        "retention_ratio_mean", "retention_ratio_std"
    ]].copy()
    out["method_label"] = out["method_id"].map(METHOD_LABEL)
    out["attack_label"] = out["attack_scenario"].map(SCENARIO_LABEL)
    out.to_csv(output_dir / "TABLE3_ROBUSTNESS_DROP_AND_RETENTION.csv", index=False)
    return out


def build_floor_table(floor: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    f = floor.copy()
    f["method_id"] = pd.Categorical(f["method_id"], METHOD_ORDER, ordered=True)
    f["scenario_id"] = pd.Categorical(f["scenario_id"], SCENARIO_ORDER, ordered=True)
    f = f.sort_values(["alpha", "method_id", "scenario_id"])
    f.to_csv(output_dir / "TABLE4_MODAL_FLOOR_RATES.csv", index=False)
    return f


def style_axes(ax: plt.Axes, ylabel: str, title: str) -> None:
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_macro_f1(table: pd.DataFrame, output_dir: Path, alpha: float, stem: str) -> None:
    sub = table[table["alpha"] == alpha]
    x = np.arange(len(SCENARIO_ORDER), dtype=float)
    width = 0.12
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    for i, method in enumerate(METHOD_ORDER):
        row = sub[sub["method_id"] == method].iloc[0]
        means = [row[f"{s}_mean"] for s in SCENARIO_ORDER]
        stds = [row[f"{s}_std"] for s in SCENARIO_ORDER]
        ax.bar(x + (i - 2.5) * width, means, width, yerr=stds, capsize=2.5, label=METHOD_LABEL[method])
    ax.set_xticks(x, [SCENARIO_LABEL[s] for s in SCENARIO_ORDER])
    ax.set_ylim(0, 0.75)
    style_axes(ax, "Test Macro-F1", f"CICIoT2023 performance at α={alpha:.1f} (mean ± SD, n=5)")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    save_figure(fig, output_dir, stem)


def figure_robustness(robust: pd.DataFrame, output_dir: Path, alpha: float, stem: str) -> None:
    sub = robust[robust["alpha"] == alpha].copy()
    x = np.arange(len(METHOD_ORDER), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    for i, scenario in enumerate(SCENARIO_ORDER[1:]):
        s = sub[sub["attack_scenario"] == scenario].set_index("method_id").reindex(METHOD_ORDER)
        ax.bar(x + (i - 0.5) * width, s["absolute_drop_mean"], width,
               yerr=s["absolute_drop_std"], capsize=3, label=SCENARIO_LABEL[scenario])
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x, [METHOD_LABEL[m] for m in METHOD_ORDER], rotation=18, ha="right")
    style_axes(ax, "Absolute Macro-F1 drop from clean", f"Robustness loss at α={alpha:.1f} (mean ± SD, n=5)")
    ax.legend(frameon=False)
    save_figure(fig, output_dir, stem)


def figure_average_ranks(ranks: pd.DataFrame, output_dir: Path) -> None:
    r = ranks[ranks["analysis_level"] == "PRIMARY_POOLED_ALPHA_SEED_BLOCKS"].copy()
    x = np.arange(len(SCENARIO_ORDER), dtype=float)
    width = 0.12
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    for i, method in enumerate(METHOD_ORDER):
        s = r[r["method_id"] == method].set_index("scenario_id").reindex(SCENARIO_ORDER)
        ax.bar(x + (i - 2.5) * width, s["average_rank"], width, label=METHOD_LABEL[method])
    ax.set_xticks(x, [SCENARIO_LABEL[s] for s in SCENARIO_ORDER])
    ax.set_ylim(1, 6.3)
    ax.invert_yaxis()
    style_axes(ax, "Average rank (1 = best)", "Average method ranks across ten frozen (α, seed) blocks")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    save_figure(fig, output_dir, "FIGURE3_AVERAGE_RANKS_BY_SCENARIO")


def figure_floor_heatmap(floor: pd.DataFrame, output_dir: Path) -> None:
    ordered_rows = []
    labels = []
    for alpha in sorted(floor["alpha"].unique()):
        for method in METHOD_ORDER:
            s = floor[(floor["alpha"] == alpha) & (floor["method_id"] == method)].set_index("scenario_id").reindex(SCENARIO_ORDER)
            ordered_rows.append(s["floor_rate"].to_numpy(dtype=float))
            labels.append(f"α={alpha:.1f} · {METHOD_LABEL[method]}")
    matrix = np.vstack(ordered_rows)
    fig, ax = plt.subplots(figsize=(8.2, 8.5))
    image = ax.imshow(matrix, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(SCENARIO_ORDER)), [SCENARIO_LABEL[s] for s in SCENARIO_ORDER])
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_title("Rate of runs at the modal Macro-F1 floor")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Floor rate")
    save_figure(fig, output_dir, "FIGURE4_MODAL_FLOOR_RATE_HEATMAP")


def make_summary(
    friedman: pd.DataFrame,
    pairwise: pd.DataFrame,
    robust_pairwise: pd.DataFrame,
    floor: pd.DataFrame,
) -> str:
    lines = [
        "CICIoT2023 PUBLICATION RESULTS SUMMARY",
        "=" * 78,
        "",
        "PRIMARY GLOBAL TESTS",
        "--------------------",
    ]
    for scenario in SCENARIO_ORDER:
        row = friedman[friedman["scenario_id"] == scenario].iloc[0]
        lines.append(
            f"{SCENARIO_LABEL[scenario]}: Friedman chi-square(5)={row['friedman_chi_square']:.6f}, "
            f"p={row['p_value']:.9g}, Kendall W={row['kendalls_w']:.6f}."
        )
    lines.extend(["", "ARL-FL PRIMARY PAIRWISE RESULTS AFTER HOLM", "------------------------------------------"])
    for scenario in SCENARIO_ORDER:
        s = pairwise[pairwise["scenario_id"] == scenario]
        significant = s[s["exact_signflip_holm_p_value"] < 0.05]
        if significant.empty:
            lines.append(f"{SCENARIO_LABEL[scenario]}: no planned comparison significant after Holm correction.")
        else:
            for _, row in significant.iterrows():
                lines.append(
                    f"{SCENARIO_LABEL[scenario]}: ARL-FL vs {METHOD_LABEL[row['baseline_method']]}: "
                    f"mean paired difference={row['mean_difference']:+.6f}, "
                    f"wins/ties/losses={int(row['wins'])}/{int(row['ties'])}/{int(row['losses'])}, "
                    f"exact sign-flip Holm p={row['exact_signflip_holm_p_value']:.6g}."
                )
    lines.extend(["", "ROBUSTNESS-DROP COMPARISONS", "---------------------------"])
    for _, row in robust_pairwise[robust_pairwise["exact_signflip_holm_p_value"] < 0.05].iterrows():
        direction = "smaller" if row["mean_difference"] > 0 else "larger"
        lines.append(
            f"{SCENARIO_LABEL[row['scenario_id']]}: ARL-FL has a {direction} clean-to-attack drop than "
            f"{METHOD_LABEL[row['baseline_method']]} (transformed mean difference={row['mean_difference']:+.6f}, "
            f"Holm p={row['exact_signflip_holm_p_value']:.6g})."
        )
    total_floor = int(floor["floor_count"].sum())
    total_runs = int(floor["n_runs"].sum())
    lines.extend([
        "",
        "MODAL FLOOR",
        "-----------",
        f"Runs at the modal Macro-F1 floor: {total_floor}/{total_runs} ({total_floor/total_runs:.1%}).",
        "",
        "INTERPRETATION BOUNDARY",
        "-----------------------",
        "Alpha-stratified tests use only five seeds per cell and are exploratory. Primary conclusions should be based on the ten-block pooled paired analysis.",
        "A RUN_STATE status of IN_PROGRESS is an auxiliary stale field; acceptance is grounded in RUN_COMPLETE.json and the Gate-113 binding replay.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    cic_root = project_root / "outputs" / "ciciot2023"
    gate113 = cic_root / "scientific_campaign_final_audit_v1"
    gate114 = cic_root / "scientific_campaign_statistical_analysis_v1"
    output_dir = cic_root / "scientific_campaign_publication_artifacts_v1"

    preserved = preserve_previous(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    read_binding(gate113 / "GATE113_FINAL_BINDING.json", "gate113_final_binding_sha256", GATE113_BINDING)
    read_binding(gate114 / "GATE114_FINAL_BINDING.json", "gate114_final_binding_sha256", GATE114_BINDING)

    master_path = gate113 / "MASTER_RUN_RESULTS.csv"
    require(master_path.is_file(), f"Missing master results: {master_path}")
    master = pd.read_csv(master_path)
    require(len(master) == 180, f"Expected 180 master rows, found {len(master)}")
    require(master["run_index"].nunique() == 180, "Run indices are not unique")
    require(master["run_result_binding_sha256"].nunique() == 180, "Run result bindings are not unique")

    expected_cells = pd.MultiIndex.from_product(
        [sorted(master["alpha"].unique()), sorted(master["experimental_seed"].unique()), METHOD_ORDER, SCENARIO_ORDER],
        names=["alpha", "experimental_seed", "method_id", "scenario_id"],
    )
    observed_cells = pd.MultiIndex.from_frame(master[["alpha", "experimental_seed", "method_id", "scenario_id"]])
    require(len(observed_cells) == 180 and set(observed_cells) == set(expected_cells), "Experimental matrix is incomplete")

    desc = pd.read_csv(gate114 / "DESCRIPTIVE_BY_ALPHA_METHOD_SCENARIO.csv")
    friedman = pd.read_csv(gate114 / "FRIEDMAN_PRIMARY_BY_SCENARIO.csv")
    pairwise = pd.read_csv(gate114 / "ARL_PAIRWISE_PRIMARY_BY_SCENARIO.csv")
    ranks = pd.read_csv(gate114 / "AVERAGE_RANKS.csv")
    robust = pd.read_csv(gate114 / "ROBUSTNESS_DESCRIPTIVE_BY_ALPHA_METHOD.csv")
    robust_pairwise = pd.read_csv(gate114 / "ROBUSTNESS_DROP_ARL_PAIRWISE.csv")
    floor = pd.read_csv(gate114 / "MODAL_FLOOR_COUNTS.csv")

    require(len(friedman) == 3, "Expected 3 primary Friedman tests")
    require(len(pairwise) == 15, "Expected 15 primary ARL-FL comparisons")
    require(int(floor["floor_count"].sum()) == 30, "Expected 30 modal-floor runs")

    table1 = build_main_performance_table(desc, output_dir)
    build_statistical_table(friedman, pairwise, output_dir)
    build_robustness_table(robust, output_dir)
    build_floor_table(floor, output_dir)

    figure_macro_f1(table1, output_dir, 0.1, "FIGURE1A_MACRO_F1_ALPHA_0P1")
    figure_macro_f1(table1, output_dir, 1.0, "FIGURE1B_MACRO_F1_ALPHA_1P0")
    figure_robustness(robust, output_dir, 0.1, "FIGURE2A_ROBUSTNESS_DROP_ALPHA_0P1")
    figure_robustness(robust, output_dir, 1.0, "FIGURE2B_ROBUSTNESS_DROP_ALPHA_1P0")
    figure_average_ranks(ranks, output_dir)
    figure_floor_heatmap(floor, output_dir)

    summary = make_summary(friedman, pairwise, robust_pairwise, floor)
    (output_dir / "PUBLICATION_RESULTS_SUMMARY.txt").write_text(summary, encoding="utf-8")

    audit = {
        "status": "PASS",
        "gate_id": "GATE-115",
        "scope": "PUBLICATION_TABLES_AND_FIGURES_ONLY",
        "scientific_training_executed_by_gate115": False,
        "optimizer_steps_executed_by_gate115": 0,
        "pamap2_600_run_campaign_started": False,
        "gate113_final_binding_sha256": GATE113_BINDING,
        "gate114_final_binding_sha256": GATE114_BINDING,
        "master_run_results_sha256": sha256_file(master_path),
        "rows_verified": int(len(master)),
        "matrix_cells_verified": 180,
        "modal_floor_count_verified": int(floor["floor_count"].sum()),
        "preserved_previous_output": str(preserved) if preserved else None,
    }
    audit_path = output_dir / "GATE115_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    generated_before_binding = sorted(p for p in output_dir.iterdir() if p.is_file())
    audit["output_file_sha256"] = {p.name: sha256_file(p) for p in generated_before_binding}
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    audit_sha = sha256_file(audit_path)

    binding_payload = dict(audit)
    binding_payload["gate115_audit_sha256"] = audit_sha
    binding_payload["gate115_final_binding_sha256"] = canonical_json_sha256(binding_payload)
    binding_path = output_dir / "GATE115_FINAL_BINDING.json"
    binding_path.write_text(json.dumps(binding_payload, indent=2, sort_keys=True), encoding="utf-8")

    manifest_rows = []
    for p in sorted(output_dir.iterdir()):
        if p.is_file() and p.name != "MANIFEST_SHA256.csv":
            manifest_rows.append({"filename": p.name, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})
    pd.DataFrame(manifest_rows).to_csv(output_dir / "MANIFEST_SHA256.csv", index=False)

    report = f"""CICIoT2023 GATE-115 PUBLICATION TABLES AND FIGURES
==============================================================================

STATUS
------
PASS

SCOPE
-----
PUBLICATION_TABLES_AND_FIGURES_ONLY
Scientific training executed by Gate-115: NO
Optimizer steps executed by Gate-115: 0
PAMAP2 600-run campaign started: NO

INPUT AUDIT
-----------
Gate-113 final binding: {GATE113_BINDING}
Gate-114 final binding: {GATE114_BINDING}
Rows verified: 180/180
Experimental matrix cells verified: 180/180
Modal Macro-F1 floor count verified: 30/180

OUTPUTS
-------
Publication tables: 4 CSV tables and 2 LaTeX tables
Publication figures: 6 figures, each in PNG and PDF
Publication results summary: PUBLICATION_RESULTS_SUMMARY.txt

FINAL BINDING
-------------
{binding_payload['gate115_final_binding_sha256']}
"""
    (output_dir / "GATE115_REPORT.txt").write_text(report, encoding="utf-8")

    # Refresh manifest after report creation.
    manifest_rows = []
    for p in sorted(output_dir.iterdir()):
        if p.is_file() and p.name != "MANIFEST_SHA256.csv":
            manifest_rows.append({"filename": p.name, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})
    pd.DataFrame(manifest_rows).to_csv(output_dir / "MANIFEST_SHA256.csv", index=False)

    print("=" * 78)
    print("GATE-115 PASS")
    print("=" * 78)
    print("Rows verified: 180/180")
    print("Experimental matrix cells verified: 180/180")
    print("Publication tables generated: 4 CSV + 2 LaTeX")
    print("Publication figures generated: 6 PNG + 6 PDF")
    print("Modal Macro-F1 floor count verified: 30/180")
    print(f"Gate-115 final binding SHA256: {binding_payload['gate115_final_binding_sha256']}")
    print("Scientific training executed by Gate-115: NO")
    print("Optimizer steps executed by Gate-115: 0")
    print("PAMAP2 600-run campaign started: NO")
    print("")
    print("Gate-115 report:")
    print(f"  {output_dir / 'GATE115_REPORT.txt'}")
    print("Publication artifacts:")
    print(f"  {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
