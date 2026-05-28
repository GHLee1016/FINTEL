"""Presentation figures for the consolidated single-DL model results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Patch


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
    "LSTM": "persistence",
    "TST": "global attention",
    "1DCNN": "local shocks",
    "TCN": "long lags",
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


def plot_regime_rank_profile(full: pd.DataFrame, output_dir: Path = FIGURE_DIR) -> Path:
    """Figure 2: average QLIKE rank profile by market regime."""
    _set_style()
    regime_order = ["normal", "911", "gfc", "covid"]

    ranked = full.copy()
    ranked["qlike_rank"] = ranked.groupby(
        ["regime", "country", "feature_set", "protocol"], observed=False
    )["QLIKE"].rank(method="min", ascending=True)
    profile = (
        ranked.groupby(["regime", "model"], observed=False)["qlike_rank"]
        .mean()
        .unstack("model")
        .reindex(regime_order)
        .reindex(columns=MODELS)
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    image = ax.imshow(profile.to_numpy(), cmap="RdYlGn_r", vmin=1, vmax=4, aspect="auto")

    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_xticklabels(MODELS, fontweight="bold")
    ax.set_yticks(np.arange(len(regime_order)))
    ax.set_yticklabels(regime_order, fontweight="bold")
    ax.set_xlabel("DL architecture")
    ax.set_ylabel("Market regime")
    fig.suptitle(
        "Model Ranking Changes With Market Regime",
        x=0.1,
        y=0.98,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.1,
        0.91,
        "Average QLIKE rank within identical country-tier-protocol cells. 1.0 is best; 4.0 is worst.",
        color=GRAY,
        fontsize=10,
    )

    for i, regime in enumerate(regime_order):
        for j, model in enumerate(MODELS):
            value = profile.loc[regime, model]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=12, fontweight="bold", color=DARK)

    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.035)
    cbar.set_label("Average rank")
    cbar.set_ticks([1, 2, 3, 4])
    fig.tight_layout(rect=(0, 0, 0.98, 0.86))
    return _save(fig, output_dir, "02_regime_model_rank_profile.png")


def plot_dl_to_ensemble_bridge(full: pd.DataFrame, output_dir: Path = FIGURE_DIR) -> Path:
    """Figure 3: compact slide bridge from single-DL results to ensemble."""
    _set_style()
    best_qlike = _best_by_metric(full, "QLIKE")
    counts = best_qlike["model"].value_counts().reindex(MODELS, fill_value=0)
    top_two = counts.sort_values(ascending=False).head(2)

    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle(
        "From Single-DL Instability to Ensemble",
        x=0.05,
        y=0.96,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )

    cards = [
        (
            0.04,
            "Observed",
            f"QLIKE winners split\nLSTM {counts['LSTM']} | TST {counts['TST']}\n1DCNN {counts['1DCNN']} | TCN {counts['TCN']}",
            BLUE,
        ),
        (
            0.37,
            "Why",
            "Each architecture sees\na different time pattern\n"
            f"{top_two.index[0]} and {top_two.index[1]} lead,\nbut not everywhere",
            "#7C3AED",
        ),
        (
            0.70,
            "Next",
            "Equal-weight ensemble\nreduces dependence on\none fragile model choice",
            GREEN,
        ),
    ]

    for x0, label, body, color in cards:
        box = FancyBboxPatch(
            (x0, 0.18),
            0.26,
            0.58,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.2,
            edgecolor="#CBD5E1",
            facecolor="#FFFFFF",
        )
        ax.add_patch(box)
        ax.text(x0 + 0.03, 0.67, label, ha="left", va="center", color=color, fontsize=13, fontweight="bold")
        ax.text(x0 + 0.03, 0.47, body, ha="left", va="center", color=DARK, fontsize=11.5, linespacing=1.35)

    for x0 in [0.32, 0.65]:
        ax.annotate(
            "",
            xy=(x0 + 0.03, 0.48),
            xytext=(x0 - 0.03, 0.48),
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.6),
        )

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, output_dir, "03_dl_to_ensemble_bridge.png")


def make_all_dl_master_figures(project_root: str | Path = ".", output_dir: str | Path | None = None) -> list[Path]:
    """Generate the three single-DL master figures."""
    root = Path(project_root)
    out_dir = root / FIGURE_DIR if output_dir is None else Path(output_dir)
    full = load_dl_master(root)
    paths = [
        plot_qlike_winner_map(full, out_dir),
        plot_regime_rank_profile(full, out_dir),
        plot_dl_to_ensemble_bridge(full, out_dir),
    ]
    pd.DataFrame({"figure": [p.name for p in paths], "path": [p.as_posix() for p in paths]}).to_csv(
        out_dir / "figure_manifest.csv", index=False, encoding="utf-8-sig"
    )
    return paths
