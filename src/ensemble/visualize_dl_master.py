"""Presentation figures for the consolidated single-DL model results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


FIGURE_DPI = 220
FIGURE_DIR = Path("results") / "figures" / "dl_master"

MODELS = ["LSTM", "TST", "1DCNN", "TCN"]
MODEL_COLORS = {
    "LSTM": "#3B82F6",
    "TST": "#8B5CF6",
    "1DCNN": "#F97316",
    "TCN": "#10B981",
}
MODEL_ROLES = {
    "LSTM": "Persistence\nsequential memory",
    "TST": "Global attention\nimportant dates",
    "1DCNN": "Local shocks\nshort windows",
    "TCN": "Long-lag patterns\ndilated history",
}

BLUE = "#2563EB"
GREEN = "#059669"
GRAY = "#64748B"
DARK = "#111827"
LIGHT_GRID = "#E5E7EB"


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": DARK,
            "axes.titlecolor": DARK,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


def load_dl_master(project_root: str | Path = ".") -> pd.DataFrame:
    """Load Full Test rows from the consolidated DL result master."""
    root = Path(project_root)
    df = pd.read_csv(root / "results" / "dl_results_master.csv")
    full = df[df["phase"] == "Full Test"].copy()
    full["model"] = pd.Categorical(full["model"], categories=MODELS, ordered=True)
    return full


def _best_by_metric(full: pd.DataFrame, metric: str) -> pd.DataFrame:
    idx = full.groupby(["regime", "country", "feature_set", "protocol"], observed=False)[metric].idxmin()
    return full.loc[idx].copy()


def plot_qlike_winner_map(full: pd.DataFrame, output_dir: Path = FIGURE_DIR) -> Path:
    """Figure 1: QLIKE best single model changes by condition."""
    _set_style()
    best = _best_by_metric(full, "QLIKE")
    regime_order = ["normal", "911", "gfc", "covid"]
    country_order = ["US", "KR", "JP"]
    tier_order = ["core", "momentum", "extended"]
    protocol_order = ["static", "expanding"]

    best["row"] = pd.Categorical(
        best["regime"] + " / " + best["country"],
        categories=[f"{r} / {c}" for r in regime_order for c in country_order],
        ordered=True,
    )
    best["col"] = pd.Categorical(
        best["feature_set"] + "\n" + best["protocol"],
        categories=[f"{t}\n{p}" for t in tier_order for p in protocol_order],
        ordered=True,
    )
    model_to_code = {model: i for i, model in enumerate(MODELS)}
    matrix = (
        best.assign(code=best["model"].map(model_to_code).astype(int))
        .pivot(index="row", columns="col", values="code")
        .sort_index()
    )

    cmap = ListedColormap([MODEL_COLORS[m] for m in MODELS])
    fig, ax = plt.subplots(figsize=(11.6, 7.2))
    ax.imshow(matrix.to_numpy(), cmap=cmap, vmin=-0.5, vmax=len(MODELS) - 0.5, aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, fontsize=9)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=10)
    fig.suptitle(
        "No Single DL Model Dominates Every Market Condition",
        x=0.12,
        y=0.98,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.12,
        0.915,
        "Each cell shows the best single DL model by QLIKE within the same regime-country-tier-protocol condition.",
        color=GRAY,
        fontsize=10,
    )

    code_to_model = {i: model for model, i in model_to_code.items()}
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            model = code_to_model[int(matrix.iloc[i, j])]
            ax.text(j, i, model, ha="center", va="center", fontsize=9, fontweight="bold", color="white")

    legend = [Patch(facecolor=MODEL_COLORS[m], label=m) for m in MODELS]
    ax.legend(handles=legend, title="Best single DL", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.tick_params(length=0)
    ax.set_xlabel("Feature tier / protocol")
    ax.set_ylabel("Regime / country")
    fig.tight_layout(rect=(0, 0, 0.88, 0.86))
    return _save(fig, output_dir, "01_single_dl_winner_map_qlike.png")


def plot_regime_winner_distribution(full: pd.DataFrame, output_dir: Path = FIGURE_DIR) -> Path:
    """Figure 2: QLIKE best single DL composition by market regime."""
    _set_style()
    best = _best_by_metric(full, "QLIKE")
    regime_order = ["normal", "911", "gfc", "covid"]
    counts = pd.crosstab(best["regime"], best["model"]).reindex(regime_order).reindex(columns=MODELS, fill_value=0)

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    bottom = np.zeros(len(counts))
    x = np.arange(len(counts))
    for model in MODELS:
        values = counts[model].to_numpy()
        bars = ax.bar(x, values, bottom=bottom, label=model, color=MODEL_COLORS[model], width=0.64)
        for i, (bar, value) in enumerate(zip(bars, values)):
            if value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom[i] + value / 2,
                    str(int(value)),
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                    fontsize=11,
                )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(regime_order, fontweight="bold")
    ax.set_ylim(0, 18.8)
    ax.set_ylabel("Number of QLIKE-best cells")
    fig.suptitle(
        "Different Market Regimes Favor Different DL Structures",
        x=0.12,
        y=0.98,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.12,
        0.91,
        "Each regime has 18 cells. Normal favors LSTM; 911 favors TST; GFC shows more local-shock diversity.",
        color=GRAY,
        fontsize=10,
    )
    ax.grid(axis="y", color=LIGHT_GRID)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Best single DL", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout(rect=(0, 0, 0.86, 0.86))
    return _save(fig, output_dir, "02_regime_winner_distribution_qlike.png")


def plot_model_role_summary(full: pd.DataFrame, output_dir: Path = FIGURE_DIR) -> Path:
    """Figure 3: connect model structure hypotheses with QLIKE winner counts."""
    _set_style()
    best_qlike = _best_by_metric(full, "QLIKE")
    counts = pd.DataFrame(
        {
            "QLIKE wins": best_qlike["model"].value_counts().reindex(MODELS, fill_value=0),
            "Median QLIKE": full.groupby("model", observed=False)["QLIKE"].median().reindex(MODELS),
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw={"width_ratios": [1.1, 1.2]})

    axes[0].axis("off")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Model structures imply different strengths", loc="left", pad=12)
    y_positions = np.linspace(0.82, 0.18, len(MODELS))
    for y, model in zip(y_positions, MODELS):
        axes[0].scatter(0.14, y, s=360, color=MODEL_COLORS[model], edgecolor="white", linewidth=1.5)
        axes[0].text(0.24, y + 0.045, model, ha="left", va="center", fontweight="bold", fontsize=12, color=DARK)
        axes[0].text(0.24, y - 0.035, MODEL_ROLES[model], ha="left", va="center", fontsize=10.5, color=DARK)
    axes[0].text(
        0.02,
        0.02,
        "Interpretation: if models specialize in different temporal patterns,\n"
        "the best single model should vary across metrics and market conditions.",
        ha="left",
        va="bottom",
        fontsize=9.5,
        color=GRAY,
    )

    x = np.arange(len(MODELS))
    bars1 = axes[1].bar(x, counts["QLIKE wins"], color=[MODEL_COLORS[m] for m in MODELS], width=0.62)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(MODELS, fontweight="bold")
    axes[1].set_ylabel("QLIKE-best count out of 72 cells")
    axes[1].set_title("Empirical QLIKE winners are distributed", loc="left", pad=12)
    axes[1].grid(axis="y", color=LIGHT_GRID)
    axes[1].spines[["top", "right"]].set_visible(False)
    for bar in bars1:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    fig.suptitle(
        "Single-DL Results Motivate an Ensemble Rather Than One Fixed Winner",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.915,
        "Different architectures win under different market conditions, supporting the complementarity hypothesis.",
        color=GRAY,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return _save(fig, output_dir, "03_model_role_summary.png")


def make_all_dl_master_figures(project_root: str | Path = ".", output_dir: str | Path | None = None) -> list[Path]:
    """Generate the three single-DL master figures."""
    root = Path(project_root)
    out_dir = root / FIGURE_DIR if output_dir is None else Path(output_dir)
    full = load_dl_master(root)
    paths = [
        plot_qlike_winner_map(full, out_dir),
        plot_regime_winner_distribution(full, out_dir),
        plot_model_role_summary(full, out_dir),
    ]
    pd.DataFrame({"figure": [p.name for p in paths], "path": [p.as_posix() for p in paths]}).to_csv(
        out_dir / "figure_manifest.csv", index=False, encoding="utf-8-sig"
    )
    return paths
