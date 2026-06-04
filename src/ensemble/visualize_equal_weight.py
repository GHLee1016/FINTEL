"""Presentation figures for the DL equal-weight ensemble experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE_DPI = 220
FIGURE_DIR = Path("results") / "figures" / "ensemble"

MODEL_COLORS = {
    "LSTM": "#3B82F6",
    "1DCNN": "#F97316",
    "TCN": "#10B981",
    "TST": "#8B5CF6",
}

BLUE = "#2563EB"
GREEN = "#059669"
ORANGE = "#EA580C"
RED = "#DC2626"
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


def load_outputs(project_root: str | Path = ".") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ensemble result, comparison, and date-level prediction files."""
    root = Path(project_root)
    results = pd.read_csv(root / "results" / "ensemble_results_equal_weight.csv")
    comparison = pd.read_csv(root / "results" / "ensemble_best_comparison.csv")
    predictions = pd.read_csv(root / "log" / "Ensemble" / "ensemble_prediction_log.csv")

    comparison["ensemble_better_RMSE_CV"] = comparison["RMSE_CV_improvement"] > 0
    comparison["cell"] = (
        comparison["regime"].astype(str)
        + "_"
        + comparison["country"].astype(str)
        + "_"
        + comparison["feature_set"].astype(str)
        + "_"
        + comparison["protocol"].astype(str)
    )
    return results, comparison, predictions


def plot_overall_gain(comparison: pd.DataFrame, output_dir: Path = FIGURE_DIR) -> Path:
    """Figure 1: overall win rate against the best single DL model."""
    _set_style()
    metrics = [
        ("QLIKE", int(comparison["ensemble_better_QLIKE"].sum()), len(comparison), BLUE),
        ("RMSE_CV", int(comparison["ensemble_better_RMSE_CV"].sum()), len(comparison), GREEN),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    y = np.arange(len(metrics))
    rates = [wins / total for _, wins, total, _ in metrics]
    colors = [c for _, _, _, c in metrics]

    ax.barh(y, [1, 1], color="#EEF2F7", height=0.5)
    ax.barh(y, rates, color=colors, height=0.5)
    ax.set_xlim(0, 1)
    ax.set_yticks(y)
    ax.set_yticklabels([m for m, _, _, _ in metrics], fontweight="bold")
    ax.set_xlabel("Share of cells where ensemble beats best single DL")
    fig.suptitle(
        "Equal-Weight Ensemble Improves Most Experiment Cells",
        x=0.13,
        y=0.98,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.13,
        0.9,
        "Comparison unit: 72 cells = regime x country x feature tier x protocol",
        color=GRAY,
        fontsize=10,
    )
    ax.grid(axis="x", color=LIGHT_GRID, linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    for i, (metric, wins, total, color) in enumerate(metrics):
        ax.text(
            min(rates[i] + 0.025, 0.96),
            i,
            f"{wins}/{total} ({rates[i]:.1%})",
            va="center",
            ha="left" if rates[i] < 0.9 else "right",
            color=DARK,
            fontweight="bold",
            fontsize=13,
        )

    return _save(fig, output_dir, "01_overall_ensemble_gain.png")


def plot_improvement_distribution(
    comparison: pd.DataFrame,
    output_dir: Path = FIGURE_DIR,
) -> Path:
    """Figure 2: sorted cell-level improvements against the best single DL."""
    _set_style()
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.5), sharex=True)
    panels = [
        ("QLIKE", "QLIKE_improvement", "ensemble_better_QLIKE", BLUE),
        ("RMSE_CV", "RMSE_CV_improvement", "ensemble_better_RMSE_CV", GREEN),
    ]

    for ax, (label, value_col, flag_col, color) in zip(axes, panels):
        frame = comparison.sort_values(value_col, ascending=True).reset_index(drop=True)
        values = frame[value_col].to_numpy()
        bar_colors = np.where(values >= 0, color, RED)
        ax.bar(np.arange(len(values)), values, color=bar_colors, width=0.86)
        ax.axhline(0, color="#475569", linewidth=1.0)
        wins = int(comparison[flag_col].sum())
        losses = len(comparison) - wins
        ax.set_title(f"{label}: {wins} wins, {losses} losses out of 72 cells", loc="left")
        ax.set_ylabel("Improvement")
        ax.text(
            0.01,
            0.88,
            "Above zero = ensemble beats the best single DL",
            transform=ax.transAxes,
            color=GRAY,
            fontsize=10,
        )
        ax.grid(color=LIGHT_GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Experiment cells sorted from weakest to strongest improvement")
    fig.suptitle(
        "Cell-Level Improvements Are Broad, Not Driven by a Few Outliers",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.935,
        "Each bar is one regime-country-tier-protocol cell; positive bars mean the best ensemble outperformed the best single DL.",
        color=GRAY,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, output_dir, "02_cell_level_improvement_distribution.png")


def plot_market_condition_map(comparison: pd.DataFrame, output_dir: Path = FIGURE_DIR) -> Path:
    """Figure 3: QLIKE win coverage by regime and country."""
    _set_style()
    data = comparison.copy()
    regime_order = ["normal", "911", "gfc", "covid"]
    country_order = ["US", "KR", "JP"]

    grouped = (
        data.groupby(["regime", "country"])
        .agg(
            wins=("ensemble_better_QLIKE", "sum"),
            cells=("ensemble_better_QLIKE", "count"),
            mean_improvement=("QLIKE_improvement", "mean"),
        )
        .reset_index()
    )
    grouped["rate"] = grouped["wins"] / grouped["cells"]
    win_matrix = grouped.pivot(index="regime", columns="country", values="rate").loc[regime_order, country_order]
    wins = grouped.pivot(index="regime", columns="country", values="wins").loc[regime_order, country_order]
    mean_imp = (
        grouped.pivot(index="regime", columns="country", values="mean_improvement")
        .loc[regime_order, country_order]
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    im = ax.imshow(win_matrix.to_numpy(), cmap="YlGn", vmin=0.65, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(country_order)))
    ax.set_xticklabels(country_order, fontweight="bold")
    ax.set_yticks(np.arange(len(regime_order)))
    ax.set_yticklabels(regime_order, fontweight="bold")
    ax.set_title("Ensemble Gains Appear Across Markets and Crisis Regimes", loc="left", pad=14)
    fig.text(
        0.125,
        0.9,
        "Cell label: QLIKE wins out of six tier-protocol settings; parenthesis is average QLIKE improvement x1,000.",
        color=GRAY,
        fontsize=10,
    )
    for i, regime in enumerate(regime_order):
        for j, country in enumerate(country_order):
            ax.text(
                j,
                i,
                f"{int(wins.loc[regime, country])}/6\n({mean_imp.loc[regime, country] * 1000:.1f})",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="white" if win_matrix.loc[regime, country] >= 0.95 else DARK,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.035)
    cbar.set_label("QLIKE win rate")
    ax.tick_params(length=0)
    ax.set_xlabel("Country")
    ax.set_ylabel("Regime")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return _save(fig, output_dir, "03_market_condition_win_map.png")


def plot_axis_robustness(comparison: pd.DataFrame, output_dir: Path = FIGURE_DIR) -> Path:
    """Figure 4: win rates by protocol and feature tier."""
    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))

    protocol = comparison.groupby("protocol").agg(
        QLIKE=("ensemble_better_QLIKE", "mean"),
        RMSE_CV=("ensemble_better_RMSE_CV", "mean"),
        n=("ensemble_better_QLIKE", "count"),
        QLIKE_wins=("ensemble_better_QLIKE", "sum"),
        RMSE_CV_wins=("ensemble_better_RMSE_CV", "sum"),
    )
    protocol = protocol.loc[["static", "expanding"]]

    tier = comparison.groupby("feature_set").agg(
        QLIKE=("ensemble_better_QLIKE", "mean"),
        RMSE_CV=("ensemble_better_RMSE_CV", "mean"),
        n=("ensemble_better_QLIKE", "count"),
        QLIKE_wins=("ensemble_better_QLIKE", "sum"),
        RMSE_CV_wins=("ensemble_better_RMSE_CV", "sum"),
    )
    tier = tier.loc[["core", "momentum", "extended"]]

    for ax, frame, title in [
        (axes[0], protocol, "By protocol"),
        (axes[1], tier, "By feature tier"),
    ]:
        x = np.arange(len(frame.index))
        width = 0.35
        bars1 = ax.bar(x - width / 2, frame["QLIKE"], width, label="QLIKE", color=BLUE)
        bars2 = ax.bar(x + width / 2, frame["RMSE_CV"], width, label="RMSE_CV", color=GREEN)
        ax.set_xticks(x)
        ax.set_xticklabels(frame.index)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Win rate vs. best single DL")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        for bars, wins_col in [(bars1, "QLIKE_wins"), (bars2, "RMSE_CV_wins")]:
            for k, bar in enumerate(bars):
                wins = int(frame.iloc[k][wins_col])
                total = int(frame.iloc[k]["n"])
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.025,
                    f"{wins}/{total}\n{bar.get_height():.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    axes[0].legend(loc="lower right")
    fig.suptitle("Robustness Is Strongest Under Expanding Evaluation and Simpler Feature Sets", x=0.02, ha="left", fontsize=16, fontweight="bold")
    fig.text(
        0.02,
        0.91,
        "Bars show how often the best ensemble beats the best single DL within each experimental axis.",
        color=GRAY,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return _save(fig, output_dir, "04_robustness_by_axis.png")


def plot_model_complementarity(comparison: pd.DataFrame, output_dir: Path = FIGURE_DIR) -> Path:
    """Figure 5: best combination frequency and model inclusion frequency."""
    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.35, 1]})

    combo_counts = comparison["best_ensemble_models"].value_counts().sort_values(ascending=True)
    combo_colors = [BLUE if combo == "LSTM+TCN+TST" else "#93C5FD" for combo in combo_counts.index]
    axes[0].barh(combo_counts.index, combo_counts.values, color=combo_colors)
    axes[0].set_xlabel("Number of cells where combination is best")
    axes[0].set_title("Best ensemble combinations", loc="left")
    axes[0].grid(axis="x", color=LIGHT_GRID, linewidth=0.8)
    axes[0].spines[["top", "right", "left"]].set_visible(False)
    axes[0].tick_params(axis="y", length=0)
    for y, v in enumerate(combo_counts.values):
        axes[0].text(v + 0.3, y, str(v), va="center", fontweight="bold", color=DARK)

    models = ["LSTM", "1DCNN", "TCN", "TST"]
    inclusion = pd.Series(
        {model: int(comparison["best_ensemble_models"].str.contains(model, regex=False).sum()) for model in models}
    ).sort_values(ascending=False)
    colors = [MODEL_COLORS[m] for m in inclusion.index]
    axes[1].bar(inclusion.index, inclusion.values, color=colors)
    axes[1].set_ylim(0, len(comparison) + 5)
    axes[1].set_ylabel("Best-combo inclusion count")
    axes[1].set_title("Model structures that repeatedly contribute", loc="left")
    axes[1].grid(axis="y", color=LIGHT_GRID, linewidth=0.8)
    axes[1].spines[["top", "right"]].set_visible(False)
    for x, v in enumerate(inclusion.values):
        axes[1].text(x, v + 1.2, str(v), ha="center", fontweight="bold", color=DARK)

    fig.suptitle(
        "Complementary Temporal Structures Drive Ensemble Gains",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.91,
        "LSTM+TCN+TST is the most frequent best combo, suggesting persistence, long-lag patterns, and global attention are complementary.",
        color=GRAY,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return _save(fig, output_dir, "05_model_complementarity.png")


def make_all_figures(project_root: str | Path = ".", output_dir: str | Path | None = None) -> list[Path]:
    """Generate the five core presentation figures."""
    root = Path(project_root)
    out_dir = root / FIGURE_DIR if output_dir is None else Path(output_dir)
    _results, comparison, _predictions = load_outputs(root)

    paths = [
        plot_overall_gain(comparison, out_dir),
        plot_improvement_distribution(comparison, out_dir),
        plot_market_condition_map(comparison, out_dir),
        plot_axis_robustness(comparison, out_dir),
        plot_model_complementarity(comparison, out_dir),
    ]

    summary = pd.DataFrame(
        {
            "figure": [path.name for path in paths],
            "path": [str(path.as_posix()) for path in paths],
        }
    )
    summary.to_csv(out_dir / "figure_manifest.csv", index=False, encoding="utf-8-sig")
    return paths
