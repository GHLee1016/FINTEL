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
            "axes.titlesize": 16,
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
    ax.set_title("Equal-Weight Ensemble Improves Most Experiment Cells", loc="left", pad=16)
    ax.text(
        0,
        1.05,
        "Comparison unit: 72 cells = regime x country x feature tier x protocol",
        transform=ax.transAxes,
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


def plot_single_vs_ensemble_scatter(
    comparison: pd.DataFrame,
    output_dir: Path = FIGURE_DIR,
) -> Path:
    """Figure 2: best single DL versus best ensemble, QLIKE and RMSE_CV."""
    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    panels = [
        ("QLIKE", "single_QLIKE", "ensemble_QLIKE", "ensemble_better_QLIKE", BLUE),
        ("RMSE_CV", "single_RMSE_CV", "ensemble_RMSE_CV", "ensemble_better_RMSE_CV", GREEN),
    ]
    protocol_markers = {"static": "o", "expanding": "s"}

    for ax, (label, x_col, y_col, flag_col, color) in zip(axes, panels):
        lo = min(comparison[x_col].min(), comparison[y_col].min())
        hi = max(comparison[x_col].max(), comparison[y_col].max())
        pad = (hi - lo) * 0.08
        lo -= pad
        hi += pad
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#94A3B8", linewidth=1.2)

        for protocol, marker in protocol_markers.items():
            sub = comparison[comparison["protocol"] == protocol]
            ax.scatter(
                sub[x_col],
                sub[y_col],
                s=54,
                marker=marker,
                c=np.where(sub[flag_col], color, RED),
                edgecolor="white",
                linewidth=0.8,
                alpha=0.88,
                label=protocol,
            )

        wins = int(comparison[flag_col].sum())
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Best single DL {label}")
        ax.set_ylabel(f"Best ensemble {label}")
        ax.set_title(f"{label}: {wins}/72 cells improved", loc="left")
        ax.text(
            0.03,
            0.94,
            "Below diagonal = ensemble wins",
            transform=ax.transAxes,
            color=GRAY,
            fontsize=9,
        )
        ax.grid(color=LIGHT_GRID, linewidth=0.8)

    axes[0].legend(title="Protocol", loc="lower right")
    fig.suptitle("Best Ensemble vs. Best Single DL", x=0.02, ha="left", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, output_dir, "02_single_vs_ensemble_scatter.png")


def plot_improvement_heatmap(comparison: pd.DataFrame, output_dir: Path = FIGURE_DIR) -> Path:
    """Figure 3: QLIKE improvement by market condition and experiment axis."""
    _set_style()
    data = comparison.copy()
    regime_order = ["normal", "911", "gfc", "covid"]
    country_order = ["US", "KR", "JP"]
    tier_order = ["core", "momentum", "extended"]
    protocol_order = ["static", "expanding"]

    data["row"] = pd.Categorical(
        data["regime"] + " / " + data["country"],
        categories=[f"{r} / {c}" for r in regime_order for c in country_order],
        ordered=True,
    )
    data["col"] = pd.Categorical(
        data["feature_set"] + "\n" + data["protocol"],
        categories=[f"{t}\n{p}" for t in tier_order for p in protocol_order],
        ordered=True,
    )
    matrix = data.pivot(index="row", columns="col", values="QLIKE_improvement").sort_index()
    values = matrix.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(values)))

    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    im = ax.imshow(values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, fontsize=9)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=10)
    ax.set_title("QLIKE Improvement Is Broadly Distributed Across Market Conditions", loc="left", pad=16)
    ax.text(
        0,
        1.04,
        "Positive values mean best ensemble has lower QLIKE than the best single DL model. Cell labels are x1,000.",
        transform=ax.transAxes,
        color=GRAY,
        fontsize=10,
    )

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isnan(val):
                continue
            label = f"{val * 1000:.1f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=DARK)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.025)
    cbar.set_label("QLIKE improvement x 1.0")
    ax.tick_params(length=0)
    ax.set_xlabel("Feature tier / protocol")
    ax.set_ylabel("Regime / country")
    return _save(fig, output_dir, "03_qlike_improvement_heatmap.png")


def plot_axis_robustness(comparison: pd.DataFrame, output_dir: Path = FIGURE_DIR) -> Path:
    """Figure 4: win rates by protocol and feature tier."""
    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    protocol = comparison.groupby("protocol").agg(
        QLIKE=("ensemble_better_QLIKE", "mean"),
        RMSE_CV=("ensemble_better_RMSE_CV", "mean"),
    )
    protocol = protocol.loc[["static", "expanding"]]

    tier = comparison.groupby("feature_set").agg(
        QLIKE=("ensemble_better_QLIKE", "mean"),
        RMSE_CV=("ensemble_better_RMSE_CV", "mean"),
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
        for bars in [bars1, bars2]:
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.025,
                    f"{bar.get_height():.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

    axes[0].legend(loc="lower right")
    fig.suptitle("Where the Ensemble Is Most Robust", x=0.02, ha="left", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
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
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.93,
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
        plot_single_vs_ensemble_scatter(comparison, out_dir),
        plot_improvement_heatmap(comparison, out_dir),
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
