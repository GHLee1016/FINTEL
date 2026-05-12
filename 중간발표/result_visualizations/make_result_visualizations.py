"""Create midterm presentation visualizations from FINTEL result CSVs.

The script reads existing result files only. It supports both repository layouts:

1. FINTEL/results, FINTEL/dataset
2. FINTEL/Project/results, FINTEL/Project/dataset

Outputs are written under this script directory:
중간발표/result_visualizations/outputs
중간발표/result_visualizations/summary_tables
"""

from __future__ import annotations

import os
from pathlib import Path
from textwrap import fill

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
OUTPUT_DIR = SCRIPT_DIR / "outputs"
SUMMARY_DIR = SCRIPT_DIR / "summary_tables"
MPL_CONFIG_DIR = SCRIPT_DIR / ".matplotlib"

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REGIME_ORDER = ["normal", "911", "gfc", "covid"]
REGIME_LABELS = {
    "normal": "Normal",
    "911": "9/11",
    "gfc": "GFC",
    "covid": "COVID",
}
COUNTRY_ORDER = ["US", "KR", "JP"]
COUNTRY_LABELS = {
    "US": "S&P 500 (US)",
    "KR": "KOSPI (KR)",
    "JP": "Nikkei 225 (JP)",
}
MODEL_COLORS = {
    "GARCH": "#9ca3af",
    "HAR_RV": "#6b7280",
    "Ridge": "#2563eb",
    "ElasticNet": "#0f766e",
    "Huber": "#7c3aed",
    "LightGBM": "#16a34a",
    "XGBoost": "#dc2626",
}


def find_project_dir() -> Path:
    """Return the directory containing results/ and dataset/."""
    candidates = [REPO_ROOT, REPO_ROOT / "Project"]
    required = [
        "financial_results.csv",
        "ml_results_core.csv",
        "ml_results_momentum.csv",
        "ml_results_extended.csv",
    ]
    for candidate in candidates:
        results_dir = candidate / "results"
        dataset_dir = candidate / "dataset"
        if results_dir.exists() and dataset_dir.exists():
            if all((results_dir / name).exists() for name in required):
                return candidate
    raise FileNotFoundError(
        "Could not find FINTEL result files in either ./results or ./Project/results."
    )


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    project_dir = find_project_dir()
    results_dir = project_dir / "results"
    dataset_dir = project_dir / "dataset"

    financial = pd.read_csv(results_dir / "financial_results.csv")
    financial["feature_set"] = "financial"
    financial["model_group"] = "Financial"

    ml_frames = []
    for feature_set in ["core", "momentum", "extended"]:
        frame = pd.read_csv(results_dir / f"ml_results_{feature_set}.csv")
        frame["model_group"] = "ML"
        ml_frames.append(frame)
    ml = pd.concat(ml_frames, ignore_index=True)

    all_results = pd.concat([financial, ml], ignore_index=True, sort=False)
    dataset_summary_path = dataset_dir / "dataset_summary.csv"
    dataset_summary = (
        pd.read_csv(dataset_summary_path) if dataset_summary_path.exists() else pd.DataFrame()
    )
    return all_results, ml, dataset_summary


def prepare_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def full_test(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["phase"].eq("Full Test")].copy()


def best_by(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    idx = df.groupby(keys, dropna=False)["RMSE_CV"].idxmin()
    return df.loc[idx].copy()


def ordered_label(values: pd.Series, mapping: dict[str, str]) -> list[str]:
    return [mapping.get(value, str(value)) for value in values]


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.savefig(OUTPUT_DIR / filename, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_experiment_flow() -> None:
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.axis("off")

    boxes = [
        ("Dataset", "US / KR / JP\nNormal, 9/11, GFC, COVID"),
        ("Feature Tier", "Core (10)\nMomentum (14)\nExtended (28)"),
        ("Model Groups", "Financial: HAR-RV, GARCH\nML: Ridge, EN, Huber,\nLightGBM, XGBoost"),
        ("Protocol", "Static\nExpanding"),
        ("Metric", "RMSE_CV\nlower is better"),
    ]
    x_positions = np.linspace(0.06, 0.82, len(boxes))
    width = 0.15
    height = 0.55

    for i, ((title, body), x) in enumerate(zip(boxes, x_positions)):
        patch = FancyBboxPatch(
            (x, 0.23),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=1.3,
            edgecolor="#1f2937",
            facecolor="#f8fafc",
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, 0.68, title, ha="center", va="center", fontsize=13, weight="bold")
        ax.text(x + width / 2, 0.47, body, ha="center", va="center", fontsize=10, linespacing=1.35)
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(x_positions[i + 1] - 0.02, 0.505),
                xytext=(x + width + 0.015, 0.505),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#334155"),
            )

    ax.text(
        0.5,
        0.08,
        "Same splits, same regimes, same protocols, and same normalized metric make the comparison fair.",
        ha="center",
        va="center",
        fontsize=12,
        color="#334155",
    )
    save_figure(fig, "01_experiment_flow.png")


def make_financial_vs_ml(all_results: pd.DataFrame) -> None:
    full = full_test(all_results)
    best_group = best_by(full, ["country", "regime", "model_group"])
    summary = best_group[
        ["country", "regime", "model_group", "model", "feature_set", "protocol", "RMSE_CV"]
    ].sort_values(["country", "regime", "model_group"])
    summary.to_csv(SUMMARY_DIR / "financial_vs_ml_summary.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True)
    bar_width = 0.34
    x = np.arange(len(REGIME_ORDER))
    colors = {"Financial": "#94a3b8", "ML": "#2563eb"}

    for ax, country in zip(axes, COUNTRY_ORDER):
        subset = best_group[best_group["country"].eq(country)]
        for offset, group in [(-bar_width / 2, "Financial"), (bar_width / 2, "ML")]:
            values = []
            labels = []
            for regime in REGIME_ORDER:
                row = subset[
                    subset["regime"].eq(regime) & subset["model_group"].eq(group)
                ]
                values.append(row["RMSE_CV"].iloc[0] if not row.empty else np.nan)
                labels.append(row["model"].iloc[0] if not row.empty else "")
            bars = ax.bar(x + offset, values, bar_width, label=group, color=colors[group])
            for bar, label in zip(bars, labels):
                if label:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.015,
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90,
                    )

        ax.set_title(COUNTRY_LABELS[country], fontsize=12, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([REGIME_LABELS[r] for r in REGIME_ORDER])
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(0.85, np.nanmax(best_group["RMSE_CV"]) * 1.18))
    axes[0].set_ylabel("RMSE_CV (lower is better)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Best Financial vs Best ML Performance", fontsize=16, weight="bold")
    fig.text(
        0.5,
        -0.02,
        "Meaning: ML models generally use broader and nonlinear information better than HAR-RV/GARCH.",
        ha="center",
        fontsize=11,
        color="#334155",
    )
    save_figure(fig, "02_best_financial_vs_ml.png")


def make_best_model_matrix(all_results: pd.DataFrame) -> None:
    full = full_test(all_results)
    best = best_by(full, ["country", "regime"])
    best = best[["country", "regime", "model", "feature_set", "protocol", "RMSE_CV"]]
    best.to_csv(SUMMARY_DIR / "best_model_matrix.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_xlim(0, len(REGIME_ORDER))
    ax.set_ylim(0, len(COUNTRY_ORDER))
    ax.invert_yaxis()
    ax.axis("off")

    for i, country in enumerate(COUNTRY_ORDER):
        for j, regime in enumerate(REGIME_ORDER):
            row = best[best["country"].eq(country) & best["regime"].eq(regime)]
            if row.empty:
                model, feature, protocol, score = "N/A", "", "", np.nan
            else:
                item = row.iloc[0]
                model = item["model"]
                feature = item["feature_set"]
                protocol = item["protocol"]
                score = item["RMSE_CV"]
            color = MODEL_COLORS.get(model, "#e5e7eb")
            patch = FancyBboxPatch(
                (j + 0.05, i + 0.08),
                0.9,
                0.78,
                boxstyle="round,pad=0.02,rounding_size=0.03",
                linewidth=1,
                edgecolor="#ffffff",
                facecolor=color,
                alpha=0.88,
            )
            ax.add_patch(patch)
            text_color = "white" if model not in {"GARCH", "HAR_RV"} else "#111827"
            ax.text(j + 0.5, i + 0.34, model, ha="center", va="center", fontsize=13, weight="bold", color=text_color)
            ax.text(
                j + 0.5,
                i + 0.58,
                f"{feature} / {protocol}\nRMSE_CV {score:.3f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color=text_color,
            )

    for j, regime in enumerate(REGIME_ORDER):
        ax.text(j + 0.5, -0.15, REGIME_LABELS[regime], ha="center", va="center", fontsize=12, weight="bold")
    for i, country in enumerate(COUNTRY_ORDER):
        ax.text(-0.18, i + 0.47, country, ha="right", va="center", fontsize=12, weight="bold")

    fig.suptitle("Best Model Matrix by Market and Regime", fontsize=16, weight="bold")
    fig.text(
        0.5,
        0.02,
        "Meaning: there is no single universal winner; model choice depends on market and shock type.",
        ha="center",
        fontsize=11,
        color="#334155",
    )
    save_figure(fig, "03_best_model_matrix.png")


def make_feature_tier_effect(ml: pd.DataFrame) -> None:
    full = full_test(ml)
    best_tier = best_by(full, ["country", "regime", "feature_set"])
    pivot = best_tier.pivot_table(
        index=["country", "regime"],
        columns="feature_set",
        values="RMSE_CV",
        aggfunc="min",
    ).reset_index()
    pivot["core_to_extended_improvement_pct"] = (
        (pivot["core"] - pivot["extended"]) / pivot["core"] * 100
    )
    pivot["core_to_momentum_improvement_pct"] = (
        (pivot["core"] - pivot["momentum"]) / pivot["core"] * 100
    )
    pivot["momentum_to_extended_improvement_pct"] = (
        (pivot["momentum"] - pivot["extended"]) / pivot["momentum"] * 100
    )
    pivot["country_order"] = pivot["country"].map({value: i for i, value in enumerate(COUNTRY_ORDER)})
    pivot["regime_order"] = pivot["regime"].map({value: i for i, value in enumerate(REGIME_ORDER)})
    pivot = pivot.sort_values(["country_order", "regime_order"]).drop(
        columns=["country_order", "regime_order"]
    )
    pivot.to_csv(SUMMARY_DIR / "feature_tier_effect.csv", index=False, encoding="utf-8-sig")

    labels = [
        f"{country}-{REGIME_LABELS.get(regime, regime)}"
        for country, regime in zip(pivot["country"], pivot["regime"])
    ]
    values = pivot["core_to_extended_improvement_pct"].to_numpy()
    y = np.arange(len(labels))
    colors = np.where(values >= 0, "#16a34a", "#dc2626")

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.barh(y, values, color=colors, alpha=0.88)
    ax.axvline(0, color="#111827", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Core -> Extended RMSE_CV improvement (%)")
    ax.set_title("Feature Tier Effect", fontsize=16, weight="bold")
    ax.grid(axis="x", alpha=0.25)
    min_value = min(values.min(), 0)
    max_value = max(values.max(), 0)
    ax.set_xlim(min_value - 0.7, max_value + 0.7)
    for pos, value in zip(y, values):
        if value >= 0:
            ax.text(value + 0.25, pos, f"{value:+.1f}%", va="center", ha="left", fontsize=9)
        elif abs(value) >= 0.8:
            ax.text(value / 2, pos, f"{value:+.1f}%", va="center", ha="center", fontsize=9, color="white")
        else:
            ax.text(value - 0.25, pos, f"{value:+.1f}%", va="center", ha="right", fontsize=9)
    fig.text(
        0.5,
        -0.02,
        "Meaning: more variables are useful only when their information matches the market-regime context.",
        ha="center",
        fontsize=11,
        color="#334155",
    )
    save_figure(fig, "04_feature_tier_effect.png")


def make_market_insight_cards(all_results: pd.DataFrame, ml: pd.DataFrame) -> None:
    full = full_test(all_results)
    best = best_by(full, ["country", "regime"])
    ml_full = full_test(ml)
    tier_best = best_by(ml_full, ["country", "regime", "feature_set"])
    tier_pivot = tier_best.pivot_table(
        index=["country", "regime"],
        columns="feature_set",
        values="RMSE_CV",
        aggfunc="min",
    ).reset_index()
    tier_pivot["improvement"] = (tier_pivot["core"] - tier_pivot["extended"]) / tier_pivot["core"] * 100

    insights = {}
    for country in COUNTRY_ORDER:
        country_best = best[best["country"].eq(country)]
        model_counts = country_best["model"].value_counts()
        top_models = ", ".join(model_counts.index[:2])
        avg_improvement = tier_pivot[tier_pivot["country"].eq(country)]["improvement"].mean()
        best_regime_row = tier_pivot[tier_pivot["country"].eq(country)].sort_values("improvement", ascending=False).head(1)
        worst_regime_row = tier_pivot[tier_pivot["country"].eq(country)].sort_values("improvement").head(1)
        best_regime = REGIME_LABELS[best_regime_row["regime"].iloc[0]]
        worst_regime = REGIME_LABELS[worst_regime_row["regime"].iloc[0]]
        insights[country] = [
            f"Frequent winners: {top_models}",
            f"Average Core->Extended effect: {avg_improvement:+.1f}%",
            f"Largest feature gain: {best_regime}",
            f"Weakest feature gain: {worst_regime}",
        ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    card_colors = {"US": "#eff6ff", "KR": "#ecfdf5", "JP": "#fff7ed"}
    edge_colors = {"US": "#2563eb", "KR": "#16a34a", "JP": "#ea580c"}
    subtitles = {
        "US": "Macro and nonlinear signals matter in selected shocks.",
        "KR": "Spillover-sensitive market; best model varies strongly.",
        "JP": "External information can be helpful, but also noisy.",
    }

    for ax, country in zip(axes, COUNTRY_ORDER):
        ax.axis("off")
        patch = FancyBboxPatch(
            (0.05, 0.08),
            0.9,
            0.84,
            boxstyle="round,pad=0.03,rounding_size=0.035",
            linewidth=1.6,
            edgecolor=edge_colors[country],
            facecolor=card_colors[country],
        )
        ax.add_patch(patch)
        ax.text(0.5, 0.78, country, ha="center", va="center", fontsize=22, weight="bold", color=edge_colors[country])
        ax.text(0.5, 0.65, fill(subtitles[country], 34), ha="center", va="center", fontsize=11, color="#334155")
        y = 0.47
        for line in insights[country]:
            ax.text(0.13, y, "- " + fill(line, 34), ha="left", va="top", fontsize=10.5, color="#111827")
            y -= 0.13

    fig.suptitle("Market-Level Interpretation Cards", fontsize=16, weight="bold")
    fig.text(
        0.5,
        0.02,
        "Meaning: performance differences are interpreted as market-regime differences, not only algorithm differences.",
        ha="center",
        fontsize=11,
        color="#334155",
    )
    save_figure(fig, "05_market_insight_cards.png")


def main() -> None:
    prepare_dirs()
    all_results, ml, _dataset_summary = load_results()
    make_experiment_flow()
    make_financial_vs_ml(all_results)
    make_best_model_matrix(all_results)
    make_feature_tier_effect(ml)
    make_market_insight_cards(all_results, ml)
    print(f"Saved visualizations to: {OUTPUT_DIR}")
    print(f"Saved summary tables to: {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
