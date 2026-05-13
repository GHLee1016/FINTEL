"""Feature Attention visualization — what features each best ML model attends to.

For each (country, regime, tier) cell, identify the best ML model by lowest RMSE_CV
on phase="Full Test" (protocol-agnostic), load the corresponding static-fit pkl,
extract |coef_| (linear) or feature_importances_ (tree), and visualize top-K
features as 4x3 small multiples per tier.

Outputs:
  outputs/06_feature_attention_core.png
  outputs/07_feature_attention_momentum.png
  outputs/08_feature_attention_extended.png
  summary_tables/feature_attention.csv
  summary_tables/feature_attention_best_models.csv

Note: pkl objects only contain the static fit (per 02_ml.ipynb cell 10).
For cells where expanding was the best protocol, we still load the static pkl
of the same model family — the model family/features/hyperparameters are
identical, so attention patterns are similar but not identical.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import joblib
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
from matplotlib.patches import Patch


# --- Constants (consistent with make_result_visualizations.py) ---
REGIME_ORDER = ["normal", "911", "gfc", "covid"]
REGIME_LABELS = {"normal": "Normal", "911": "9/11", "gfc": "GFC", "covid": "COVID"}
COUNTRY_ORDER = ["US", "KR", "JP"]
MODEL_COLORS = {
    "Ridge": "#2563eb",
    "ElasticNet": "#0f766e",
    "Huber": "#7c3aed",
    "LightGBM": "#16a34a",
    "XGBoost": "#dc2626",
}
LINEAR_MODELS = {"Ridge", "ElasticNet", "Huber"}
TREE_MODELS = {"LightGBM", "XGBoost"}

TIER_ORDER = ["core", "momentum", "extended"]
TIER_LABELS = {
    "core": "Core (10 features)",
    "momentum": "Momentum (14 features)",
    "extended": "Extended (28 features)",
}
TIER_PNG_INDEX = {"core": 6, "momentum": 7, "extended": 8}
TOP_K = 5


def find_project_dir() -> Path:
    """Return the directory containing results/ (and optionally dataset/)."""
    candidates = [REPO_ROOT, REPO_ROOT / "Project"]
    for candidate in candidates:
        results_dir = candidate / "results"
        if (
            (results_dir / "best_models").exists()
            and (results_dir / "ml_results_core.csv").exists()
        ):
            return candidate
    raise FileNotFoundError(
        "Could not find FINTEL results with best_models/ in ./results or ./Project/results."
    )


def prepare_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.savefig(OUTPUT_DIR / filename, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def best_by(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    idx = df.groupby(keys, dropna=False)["RMSE_CV"].idxmin()
    return df.loc[idx].copy()


def load_ml_results(results_dir: Path) -> pd.DataFrame:
    frames = []
    for tier in TIER_ORDER:
        frame = pd.read_csv(results_dir / f"ml_results_{tier}.csv")
        if "feature_set" not in frame.columns:
            frame["feature_set"] = tier
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def identify_best_models(ml: pd.DataFrame) -> pd.DataFrame:
    """Best model per (country, regime, feature_set) on phase=Full Test, protocol-agnostic."""
    full = ml[ml["phase"] == "Full Test"].copy()
    best = best_by(full, ["country", "regime", "feature_set"])
    return (
        best[["country", "regime", "feature_set", "model", "protocol", "RMSE_CV"]]
        .reset_index(drop=True)
    )


def extract_attention(best36: pd.DataFrame, best_models_dir: Path) -> pd.DataFrame:
    """Load each best model's pkl and extract per-feature attention scores."""
    rows = []
    missing: list[str] = []

    # sklearn version mismatch warnings are loud — suppress them once verified
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        try:
            from sklearn.exceptions import InconsistentVersionWarning
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        except ImportError:
            pass

        for _, b in best36.iterrows():
            pkl_path = best_models_dir / b.model / f"{b.regime}_{b.country}_{b.feature_set}.pkl"
            if not pkl_path.exists():
                missing.append(str(pkl_path))
                continue

            data = joblib.load(pkl_path)
            wrapper = data["model"]
            feats = list(data["feature_cols"])
            est = wrapper.estimator_

            if b.model in LINEAR_MODELS:
                raw = np.abs(np.asarray(est.coef_, dtype=float))
            elif b.model in TREE_MODELS:
                raw = np.asarray(est.feature_importances_, dtype=float)
            else:
                raise ValueError(f"unknown model family: {b.model!r}")

            if raw.shape[0] != len(feats):
                raise ValueError(
                    f"shape mismatch for {b.model} {b.regime}_{b.country}_{b.feature_set}: "
                    f"raw={raw.shape}, feats={len(feats)}"
                )

            denom = raw.max()
            norm = raw / denom if denom > 0 else raw.copy()

            for f, r, n in zip(feats, raw, norm):
                rows.append({
                    "regime": b.regime,
                    "country": b.country,
                    "tier": b.feature_set,
                    "model": b.model,
                    "feature": f,
                    "importance_raw": float(r),
                    "importance_norm": float(n),
                })

    if missing:
        print(f"[warn] {len(missing)} pkl missing. examples: {missing[:3]}")

    attn = pd.DataFrame(rows)
    if not attn.empty:
        attn["rank"] = (
            attn.groupby(["regime", "country", "tier"])["importance_norm"]
                .rank(ascending=False, method="first")
                .astype(int)
        )
    return attn


def plot_tier(attn_tier: pd.DataFrame, tier: str) -> None:
    """Render a 4x3 grid of top-K horizontal bars for one tier."""
    n_rows = len(REGIME_ORDER)
    n_cols = len(COUNTRY_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.5, 13.5))

    for i, regime in enumerate(REGIME_ORDER):
        for j, country in enumerate(COUNTRY_ORDER):
            ax = axes[i, j]
            cell = attn_tier[
                (attn_tier.regime == regime) & (attn_tier.country == country)
            ]
            if cell.empty:
                ax.axis("off")
                ax.set_title(f"{country} - {REGIME_LABELS[regime]}\n(N/A)", fontsize=10)
                continue

            top = cell.sort_values("rank").head(TOP_K)
            model = top["model"].iloc[0]
            color = MODEL_COLORS.get(model, "#9ca3af")

            y = np.arange(len(top))
            ax.barh(y, top["importance_norm"].to_numpy(), color=color, alpha=0.88)
            ax.set_yticks(y)
            ax.set_yticklabels(top["feature"].tolist(), fontsize=9)
            ax.invert_yaxis()
            ax.set_xlim(0, 1.05)
            ax.set_xticks([0, 0.5, 1.0])
            ax.tick_params(axis="x", labelsize=8)
            ax.grid(axis="x", alpha=0.25)
            ax.set_title(
                f"{country} - {REGIME_LABELS[regime]}\n{model}",
                fontsize=10, weight="bold",
            )

            # annotate normalized value at the end of each bar
            for pos, value in zip(y, top["importance_norm"].to_numpy()):
                ax.text(
                    min(value + 0.03, 1.03), pos, f"{value:.2f}",
                    va="center", ha="left", fontsize=8, color="#334155",
                )

    fig.suptitle(
        f"Feature Attention — {TIER_LABELS[tier]} (top-{TOP_K} per cell, normalized)",
        fontsize=15, weight="bold", y=0.995,
    )

    present_models = sorted(attn_tier["model"].unique().tolist())
    handles = [
        Patch(facecolor=MODEL_COLORS.get(m, "#9ca3af"), label=m) for m in present_models
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=len(present_models),
        bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=10,
    )

    fig.text(
        0.5, -0.025,
        "Linear: |coef| in scaled space.   Tree: feature_importances_.   "
        "Each panel max-normalized to 1.",
        ha="center", fontsize=10, color="#334155",
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_figure(fig, f"{TIER_PNG_INDEX[tier]:02d}_feature_attention_{tier}.png")


def main() -> None:
    project = find_project_dir()
    results_dir = project / "results"
    best_models_dir = results_dir / "best_models"

    # pkl objects reference src.models.ml.* classes — make them importable
    project_abs = str(project.resolve())
    if project_abs not in sys.path:
        sys.path.insert(0, project_abs)

    prepare_dirs()

    ml = load_ml_results(results_dir)
    best36 = identify_best_models(ml)
    print(f"[ok] best models identified: {len(best36)} rows (expected 36)")
    print("[info] best model distribution:")
    print(best36.groupby(["model", "protocol"]).size().to_string())

    attn = extract_attention(best36, best_models_dir)
    if attn.empty:
        raise RuntimeError("Failed to extract any feature attention rows.")
    print(f"[ok] attention rows: {len(attn)}")

    best36.to_csv(
        SUMMARY_DIR / "feature_attention_best_models.csv",
        index=False, encoding="utf-8-sig",
    )
    attn.to_csv(
        SUMMARY_DIR / "feature_attention.csv",
        index=False, encoding="utf-8-sig",
    )
    print(f"[ok] wrote summary CSVs to: {SUMMARY_DIR}")

    for tier in TIER_ORDER:
        plot_tier(attn[attn.tier == tier], tier)
        print(f"[ok] saved 0{TIER_PNG_INDEX[tier]}_feature_attention_{tier}.png")

    print(f"Done. Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
