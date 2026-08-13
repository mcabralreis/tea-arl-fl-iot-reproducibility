from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "revised_analysis" / "PAMAP2_TABLE5_REVISED_VALUES.csv"
OUT = ROOT / "results" / "figures"

METHOD_ORDER = [
    "fedavg",
    "fedprox",
    "random_trimmed_mean",
    "fedle_adapted",
    "tea_fl",
    "arl_fl",
]

METHOD_STYLE = {
    "fedavg": {"color": "#0072B2", "marker": "o"},
    "fedprox": {"color": "#E69F00", "marker": "s"},
    "random_trimmed_mean": {"color": "#009E73", "marker": "^"},
    "fedle_adapted": {"color": "#D55E00", "marker": "D"},
    "tea_fl": {"color": "#CC79A7", "marker": "P"},
    "arl_fl": {"color": "#56B4E9", "marker": "X"},
}

LABEL_OFFSETS = {
    "protocol": {
        "fedavg": (7, -15),
        "fedprox": (7, 8),
        "random_trimmed_mean": (7, 8),
        "fedle_adapted": (7, -4),
        "tea_fl": (7, -12),
        "arl_fl": (7, 8),
    },
    "common": {
        "fedavg": (7, -15),
        "fedprox": (7, 8),
        "random_trimmed_mean": (7, 8),
        "fedle_adapted": (7, -6),
        "tea_fl": (7, -12),
        "arl_fl": (7, 8),
    },
}


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def load_data() -> pd.DataFrame:
    frame = pd.read_csv(DATA).set_index("method").loc[METHOD_ORDER].reset_index()
    numeric = [
        "macro_f1_mean",
        "macro_f1_std",
        "total_normalized_energy_consumed_mean",
        "total_normalized_energy_consumed_std",
        "common_horizon_macro_f1_mean",
        "common_horizon_macro_f1_std",
        "common_horizon_energy_mean",
        "common_horizon_energy_std",
        "completed_round_100",
    ]
    frame[numeric] = frame[numeric].apply(pd.to_numeric)
    return frame


def format_axes(ax: plt.Axes, *, panel_title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(panel_title, loc="left", fontweight="bold", pad=7)
    ax.set_xlabel(xlabel, labelpad=5)
    ax.set_ylabel(ylabel, labelpad=5)
    ax.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)
    ax.tick_params(direction="out", length=3, width=0.7, color="#333333")


def draw_panel(ax: plt.Axes, frame: pd.DataFrame, *, mode: str) -> None:
    if mode == "protocol":
        x_mean = "total_normalized_energy_consumed_mean"
        x_std = "total_normalized_energy_consumed_std"
        y_mean = "macro_f1_mean"
        y_std = "macro_f1_std"
        panel_title = "(a) Protocol endpoint"
        xlabel = "Protocol-total normalized energy\n(lower is better)"
        ylabel = "Protocol-endpoint Macro-F1\n(higher is better)"
        xlim = (14.82, 18.05)
        ylim = (0.305, 0.555)
    elif mode == "common":
        x_mean = "common_horizon_energy_mean"
        x_std = "common_horizon_energy_std"
        y_mean = "common_horizon_macro_f1_mean"
        y_std = "common_horizon_macro_f1_std"
        panel_title = "(b) Matched common horizon"
        xlabel = "Common-horizon normalized energy\n(lower is better)"
        ylabel = "Common-horizon Macro-F1\n(higher is better)"
        xlim = (14.82, 17.85)
        ylim = (0.305, 0.565)
    else:
        raise ValueError(mode)

    format_axes(ax, panel_title=panel_title, xlabel=xlabel, ylabel=ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    for row in frame.itertuples(index=False):
        style = METHOD_STYLE[row.method]
        x = float(getattr(row, x_mean))
        y = float(getattr(row, y_mean))
        xe = float(getattr(row, x_std))
        ye = float(getattr(row, y_std))
        proposed = row.method in {"tea_fl", "arl_fl"}

        ax.errorbar(
            x,
            y,
            xerr=xe,
            yerr=ye,
            fmt="none",
            ecolor=style["color"],
            elinewidth=0.9,
            capsize=2.4,
            capthick=0.9,
            alpha=0.48,
            zorder=2,
        )
        ax.scatter(
            [x],
            [y],
            s=62 if proposed else 48,
            c=[style["color"]],
            marker=style["marker"],
            edgecolors="#1A1A1A",
            linewidths=0.65,
            zorder=3,
        )

        completed = int(row.completed_round_100)
        label = f"{row.method_label} [{completed}/100 R100]"
        dx, dy = LABEL_OFFSETS[mode][row.method]
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.2,
            fontweight="bold" if proposed else "normal",
            color="#111111",
            annotation_clip=False,
            zorder=4,
        )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.035,
        metadata={"Software": "Matplotlib", "Title": stem},
    )
    fig.savefig(
        OUT / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.035,
        metadata={
            "Creator": "Matplotlib",
            "Title": stem,
            "Subject": "PAMAP2 effectiveness-energy comparison",
        },
    )


def make_combined(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.35), constrained_layout=True)
    draw_panel(axes[0], frame, mode="protocol")
    draw_panel(axes[1], frame, mode="common")
    fig.suptitle(
        "PAMAP2 effectiveness-energy comparison",
        fontsize=10.5,
        fontweight="bold",
    )
    save_figure(fig, "Figure_2_PAMAP2_Effectiveness_Energy_Revised")
    plt.close(fig)


def make_single(frame: pd.DataFrame, *, mode: str, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(5.35, 4.25), constrained_layout=True)
    draw_panel(ax, frame, mode=mode)
    save_figure(fig, stem)
    plt.close(fig)


def main() -> None:
    frame = load_data()
    make_combined(frame)
    make_single(
        frame,
        mode="protocol",
        stem="Figure_2A_PAMAP2_Protocol_Endpoint",
    )
    make_single(
        frame,
        mode="common",
        stem="Figure_2B_PAMAP2_Common_Horizon",
    )


if __name__ == "__main__":
    main()
