"""
Generate publication-quality figures (Nature/NEJM AI style) from architecture_eval_metrics.csv.

Figures produced:
- figure1_primary_results.{png,pdf,tiff}: 2x2 panels highlighting sensitivity (primary) and specificity.
- figure2_degradation_optimized.{png,pdf,tiff}: Sensitivity degradation from physician-created to real-world messages.
- figure3_comprehensive_metrics.{png,pdf}: F1, MCC, AUROC, and sensitivity degradation.

Assumes architecture_eval_metrics.csv is present in the same directory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Global style (Nature/NEJM AI-like)
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "lines.linewidth": 0.8,
        "patch.linewidth": 0.5,
        "figure.figsize": (7.2, 4.5),
    }
)

COLORS = {
    "constellation": "#0173B2",
    "guardrail": "#DE8F05",
    "rl_controller": "#029E73",
    "hybrid": "#CC78BC",
    "gpt": "#CA9161",
    "deepseek": "#949494",
    "sensitivity": "#D55E00",
    "specificity": "#56B4E9",
    "f1": "#009E73",
}

ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "results" / "architecture_eval_metrics.csv"


def calculate_ci_halfwidth(row: pd.Series, metric: str):
    lower = f"{metric}_ci_lower"
    upper = f"{metric}_ci_upper"
    if lower in row and upper in row and pd.notnull(row[lower]) and pd.notnull(row[upper]):
        mid = (row[upper] + row[lower]) / 2
        return (mid - row[lower], row[upper] - mid)
    return (0, 0)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(METRICS_PATH)
    # Normalize system names for display
    name_map = {
        "Constellation": "Constellation",
        "Guardrail": "Guardrail",
        "DecisionPolicy": "RL Controller",
        "Hybrid_RAG_Rules": "Hybrid RAG+Rules",
        "openai_LLM_Safety": "GPT-5.1 Safety",
        "openai_RAG": "GPT-5.1 RAG",
        "openai_LLM_Null": "GPT-5.1 Null",
        "deepseek_LLM_Null": "DeepSeek-R1",
        "deepseek_LLM_Safety": "DeepSeek-R1 Safety",
    }
    df["system_display"] = df["system"].map(name_map).fillna(df["system"])
    return df


def subset_architectures(df: pd.DataFrame, systems: List[str]) -> pd.DataFrame:
    return df[df["system_display"].isin(systems)].copy()


def create_figure1_primary_results(df: pd.DataFrame):
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)

    architectures = [
        "Constellation",
        "Guardrail",
        "RL Controller",
        "Hybrid RAG+Rules",
        "GPT-5.1 Safety",
        "GPT-5.1 RAG",
        "DeepSeek-R1",
    ]

    # Helper to draw a panel
    def panel(ax, data: pd.DataFrame, metric: str, title: str, color: str, best=True):
        data = data.set_index("system_display")
        y_pos = np.arange(len(architectures))
        vals = [data.loc[arch, metric] if arch in data.index else np.nan for arch in architectures]
        errs = [
            calculate_ci_halfwidth(data.loc[arch], metric) if arch in data.index else (0, 0)
            for arch in architectures
        ]
        bars = ax.barh(
            y_pos,
            vals,
            xerr=np.array(errs).T if errs else None,
            color=color,
            alpha=0.7,
            error_kw={"linewidth": 0.5, "capsize": 2, "capthick": 0.5},
        )
        if best:
            best_idx = int(np.nanargmax(vals))
            bars[best_idx].set_edgecolor("black")
            bars[best_idx].set_linewidth(1.2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(architectures, fontsize=6)
        ax.set_xlabel(metric.capitalize(), fontweight="bold")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=8)
        ax.set_xlim([0.5, 1.0])
        ax.axvline(x=0.80, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.grid(axis="x", alpha=0.2, linewidth=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    phys = df[df["dataset"] == "Physician_Test"]
    real = df[df["dataset"] == "RealWorld_Test"]

    ax1 = fig.add_subplot(gs[0, 0])
    panel(ax1, phys, "sensitivity", "a  Physician-created (n=41)", COLORS["sensitivity"], best=True)

    ax2 = fig.add_subplot(gs[0, 1])
    panel(ax2, real, "sensitivity", "b  Real-world messages (n=201)", COLORS["sensitivity"], best=True)

    ax3 = fig.add_subplot(gs[1, 0])
    panel(ax3, phys, "specificity", "c  Physician-created (n=41)", COLORS["specificity"], best=False)

    ax4 = fig.add_subplot(gs[1, 1])
    panel(ax4, real, "specificity", "d  Real-world messages (n=201)", COLORS["specificity"], best=False)

    fig.savefig(ROOT / "figure1_primary_results.png", dpi=600, bbox_inches="tight")
    fig.savefig(ROOT / "figure1_primary_results.pdf", dpi=600, bbox_inches="tight")
    fig.savefig(
        ROOT / "figure1_primary_results.tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    return fig


def create_figure2_degradation_optimized(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    phys = df[df["dataset"] == "Physician_Test"].set_index("system_display")
    real = df[df["dataset"] == "RealWorld_Test"].set_index("system_display")
    architectures = [
        "Constellation",
        "Guardrail",
        "RL Controller",
        "Hybrid RAG+Rules",
        "GPT-5.1 Safety",
        "GPT-5.1 RAG",
        "DeepSeek-R1",
    ]
    style_map = {
        "Constellation": {"color": COLORS["constellation"], "marker": "o", "size": 120, "label": "Constellation"},
        "Guardrail": {"color": COLORS["guardrail"], "marker": "s", "size": 100, "label": "Rule-based"},
        "RL Controller": {"color": COLORS["rl_controller"], "marker": "^", "size": 100, "label": "Decision-theoretic"},
        "Hybrid RAG+Rules": {"color": COLORS["hybrid"], "marker": "D", "size": 90, "label": "Hybrid"},
        "GPT-5.1 Safety": {"color": COLORS["gpt"], "marker": "v", "size": 100, "label": "GPT-5.1"},
        "GPT-5.1 RAG": {"color": COLORS["gpt"], "marker": "v", "size": 100, "label": None},
        "DeepSeek-R1": {"color": COLORS["deepseek"], "marker": "v", "size": 100, "label": "DeepSeek-R1"},
    }
    added = set()
    for arch in architectures:
        if arch not in phys.index or arch not in real.index:
            continue
        x = phys.loc[arch, "sensitivity"]
        y = real.loc[arch, "sensitivity"]
        st = style_map[arch]
        lab = st["label"] if st["label"] and st["label"] not in added else None
        if st["label"]:
            added.add(st["label"])
        ax.scatter(x, y, s=st["size"], c=st["color"], marker=st["marker"], alpha=0.8, edgecolors="black", linewidths=1.0, label=lab, zorder=3)
        ax.annotate(
            arch,
            (x, y),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5.5,
            ha="left",
            va="bottom",
        )

    ax.plot([0.45, 0.95], [0.45, 0.95], "k--", alpha=0.3, linewidth=1.0, label="Perfect concordance", zorder=1)
    ax.set_xlabel("Sensitivity on physician-created scenarios", fontweight="bold", fontsize=7)
    ax.set_ylabel("Sensitivity on real-world messages", fontweight="bold", fontsize=7)
    ax.set_xlim([0.48, 0.93])
    ax.set_ylim([0.48, 0.93])
    ax.grid(alpha=0.2, linewidth=0.3, zorder=0)
    ax.legend(loc="lower right", frameon=True, fontsize=5.5, framealpha=0.9, edgecolor="black")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "figure2_degradation_optimized.png", dpi=600, bbox_inches="tight")
    fig.savefig(ROOT / "figure2_degradation_optimized.pdf", dpi=600, bbox_inches="tight")
    fig.savefig(
        ROOT / "figure2_degradation_optimized.tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    return fig


def create_figure3_comprehensive_metrics(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    architectures = [
        "Constellation",
        "Guardrail",
        "RL Controller",
        "Hybrid RAG+Rules",
        "GPT-5.1 Safety",
        "GPT-5.1 RAG",
        "DeepSeek-R1",
    ]
    datasets = [("Physician_Test", "Physician (n=41)"), ("RealWorld_Test", "Real-world (n=201)")]
    metrics = [("f1", "F1 Score", COLORS["f1"], (0, 0)), ("mcc", "Matthews Correlation", COLORS["constellation"], (0, 1)), ("auroc", "AUROC", COLORS["guardrail"], (1, 0))]
    for metric, title, color, pos in metrics:
        ax = axes[pos[0], pos[1]]
        x = np.arange(len(architectures))
        width = 0.35
        for i, (dataset, label) in enumerate(datasets):
            data = df[df["dataset"] == dataset].set_index("system_display")
            vals = [data.loc[arch, metric] if arch in data.index else 0 for arch in architectures]
            offset = width / 2 if i == 0 else -width / 2
            bars = ax.bar(x + offset, vals, width, label=label, alpha=0.7, color=color if i == 0 else sns.desaturate(color, 0.5))
            best_idx = int(np.nanargmax(vals))
            bars[best_idx].set_edgecolor("black")
            bars[best_idx].set_linewidth(1.2)
        ax.set_ylabel(title, fontweight="bold", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(architectures, rotation=45, ha="right", fontsize=5.5)
        ax.legend(loc="lower right", fontsize=5.5, frameon=True)
        ax.grid(axis="y", alpha=0.2, linewidth=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim([0, 1.0] if metric != "mcc" else [0, 0.8])

    # Sensitivity degradation
    ax4 = axes[1, 1]
    phys = df[df["dataset"] == "Physician_Test"].set_index("system_display")
    real = df[df["dataset"] == "RealWorld_Test"].set_index("system_display")
    degradation = [phys.loc[arch, "sensitivity"] - real.loc[arch, "sensitivity"] if arch in phys.index and arch in real.index else np.nan for arch in architectures]
    bars = ax4.bar(range(len(architectures)), degradation, color=COLORS["sensitivity"], alpha=0.7)
    ax4.set_ylabel("Sensitivity degradation (pp)", fontweight="bold", fontsize=7)
    ax4.set_xlabel("Architecture", fontweight="bold", fontsize=7)
    ax4.set_xticks(range(len(architectures)))
    ax4.set_xticklabels(architectures, rotation=45, ha="right", fontsize=5.5)
    ax4.axhline(y=0.10, color="orange", linestyle="--", linewidth=0.8, alpha=0.5)
    ax4.grid(axis="y", alpha=0.2, linewidth=0.3)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(ROOT / "figure3_comprehensive_metrics.png", dpi=600, bbox_inches="tight")
    fig.savefig(ROOT / "figure3_comprehensive_metrics.pdf", dpi=600, bbox_inches="tight")
    return fig


def main():
    df = load_data()
    # Use only relevant systems for plotting
    systems = [
        "Constellation",
        "Guardrail",
        "RL Controller",
        "Hybrid RAG+Rules",
        "GPT-5.1 Safety",
        "GPT-5.1 RAG",
        "DeepSeek-R1",
    ]
    df = subset_architectures(df, systems)
    create_figure1_primary_results(df)
    create_figure2_degradation_optimized(df)
    create_figure3_comprehensive_metrics(df)
    print("Saved figures: figure1_primary_results.*, figure2_degradation_optimized.*, figure3_comprehensive_metrics.*")


if __name__ == "__main__":
    main()
