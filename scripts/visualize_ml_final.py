"""Generate final-report ML comparison figures.

The script intentionally uses only pandas/numpy/Pillow so it can run in a
minimal environment without matplotlib. Outputs are written under
results/figures/ml_final and do not overwrite existing figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "figures" / "ml_final"

W, H = 1600, 1000
BG = "#FFFFFF"
DARK = "#111827"
TEXT = "#334155"
MUTED = "#64748B"
GRID = "#E5E7EB"
GRID_DARK = "#CBD5E1"
SOFT = "#F8FAFC"

REGIMES = ["normal", "911", "gfc", "covid"]
COUNTRIES = ["US", "KR", "JP"]
TIERS = ["core", "momentum", "extended"]
PROTOCOLS = ["static", "expanding"]
ML_MODELS = ["Ridge", "ElasticNet", "Huber", "LightGBM", "XGBoost"]
FIN_MODELS = ["GARCH", "HAR_RV"]
DL_MODELS = ["LSTM", "TST", "1DCNN", "TCN"]

MODEL_COLORS = {
    "Ridge": "#2563EB",
    "ElasticNet": "#7C3AED",
    "Huber": "#92400E",
    "LightGBM": "#059669",
    "XGBoost": "#DC2626",
    "Financial": "#64748B",
    "ML": "#2563EB",
    "Single DL": "#8B5CF6",
    "Ensemble": "#0F766E",
}

REGIME_COLORS = {
    "normal": "#2563EB",
    "911": "#F97316",
    "gfc": "#DC2626",
    "covid": "#059669",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


F_TITLE = font(42, True)
F_SUB = font(22)
F_H = font(24, True)
F_BODY = font(21)
F_SMALL = font(17)
F_TINY = font(14)


@dataclass
class Canvas:
    image: Image.Image
    draw: ImageDraw.ImageDraw


def new_canvas(title: str, subtitle: str) -> Canvas:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((80, 54), title, fill=DARK, font=F_TITLE)
    d.text((80, 114), subtitle, fill=MUTED, font=F_SUB)
    return Canvas(img, d)


def save(canvas: Canvas, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    canvas.image.save(path, quality=95)
    return path


def text_center(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fill: str, fnt: ImageFont.ImageFont) -> None:
    bbox = d.multiline_textbbox((0, 0), text, font=fnt, spacing=3, align="center")
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - tw) / 2
    y = box[1] + (box[3] - box[1] - th) / 2
    d.multiline_text((x, y), text, fill=fill, font=fnt, spacing=3, align="center")


def draw_legend(d: ImageDraw.ImageDraw, items: Iterable[tuple[str, str]], x: int, y: int, cols: int = 4) -> None:
    for idx, (label, color) in enumerate(items):
        col = idx % cols
        row = idx // cols
        xx = x + col * 230
        yy = y + row * 34
        d.rounded_rectangle((xx, yy, xx + 24, yy + 24), radius=5, fill=color)
        d.text((xx + 34, yy - 1), label, fill=TEXT, font=F_SMALL)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fin = pd.read_csv(RESULTS / "financial_results.csv")
    fin = fin[fin["phase"].eq("Full Test")].copy()

    ml_frames = []
    for tier in TIERS:
        df = pd.read_csv(RESULTS / f"ml_results_{tier}.csv")
        ml_frames.append(df)
    ml = pd.concat(ml_frames, ignore_index=True)
    ml = ml[ml["phase"].eq("Full Test")].copy()

    dl = pd.read_csv(RESULTS / "dl_results_master.csv")
    dl = dl[dl["phase"].eq("Full Test")].copy()

    ens_path = RESULTS / "ensemble_best_comparison.csv"
    ensemble = pd.read_csv(ens_path) if ens_path.exists() else pd.DataFrame()
    return fin, ml, dl, ensemble


def best_by(df: pd.DataFrame, keys: list[str], metric: str = "QLIKE") -> pd.DataFrame:
    idx = df.groupby(keys, observed=False)[metric].idxmin()
    return df.loc[idx].copy()


def pct_improvement(base: float, challenger: float) -> float:
    if pd.isna(base) or base == 0:
        return np.nan
    return (base - challenger) / base * 100


def value_to_color(value: float, lo: float, hi: float, c1=(239, 246, 255), c2=(37, 99, 235)) -> str:
    if hi <= lo:
        t = 0.5
    else:
        t = max(0, min(1, (value - lo) / (hi - lo)))
    rgb = tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def plot_ml_winner_map(ml: pd.DataFrame) -> Path:
    c = new_canvas(
        "ML Winner Map by QLIKE",
        "Best ML model changes across market regime, country, feature tier, and evaluation protocol.",
    )
    d = c.draw
    best = best_by(ml, ["regime", "country", "feature_set", "protocol"])

    left, top = 170, 190
    cell_w = 170
    row_h = 48
    header_h = 50
    cols = [(tier, protocol) for tier in TIERS for protocol in PROTOCOLS]
    rows = [(regime, country) for regime in REGIMES for country in COUNTRIES]

    for j, (tier, protocol) in enumerate(cols):
        x = left + j * cell_w
        d.rounded_rectangle((x, top, x + cell_w - 4, top + header_h), radius=8, fill=SOFT, outline=GRID_DARK)
        text_center(d, (x, top, x + cell_w - 4, top + header_h), f"{tier}\n{protocol}", DARK, F_SMALL)

    for i, (regime, country) in enumerate(rows):
        y = top + header_h + i * row_h
        d.text((70, y + 13), f"{regime} / {country}", fill=DARK, font=F_SMALL)
        for j, (tier, protocol) in enumerate(cols):
            x = left + j * cell_w
            rec = best[
                best["regime"].eq(regime)
                & best["country"].eq(country)
                & best["feature_set"].eq(tier)
                & best["protocol"].eq(protocol)
            ].iloc[0]
            model = rec["model"]
            color = MODEL_COLORS[model]
            d.rounded_rectangle((x, y, x + cell_w - 4, y + row_h - 5), radius=7, fill=color)
            text_center(d, (x, y, x + cell_w - 4, y + row_h - 5), str(model), "white", F_SMALL)

    d.text((80, 858), "Each cell selects the lowest QLIKE among Ridge, ElasticNet, Huber, LightGBM, and XGBoost.", fill=MUTED, font=F_SMALL)
    draw_legend(d, [(m, MODEL_COLORS[m]) for m in ML_MODELS], 170, 910, cols=5)
    return save(c, "01_ml_winner_map_qlike.png")


def plot_feature_tier_ablation(ml: pd.DataFrame) -> Path:
    c = new_canvas(
        "Feature Tier Ablation for Best ML",
        "Adding more features is not uniformly better; the useful information set depends on market condition.",
    )
    d = c.draw
    best = best_by(ml, ["regime", "country", "feature_set", "protocol"])

    chart = (170, 230, 1430, 800)
    lo, hi = best["QLIKE"].min() * 0.92, best["QLIKE"].max() * 1.05
    d.line((chart[0], chart[3], chart[2], chart[3]), fill=GRID_DARK, width=2)
    d.line((chart[0], chart[1], chart[0], chart[3]), fill=GRID_DARK, width=2)

    for k in range(6):
        val = lo + (hi - lo) * k / 5
        y = chart[3] - (val - lo) / (hi - lo) * (chart[3] - chart[1])
        d.line((chart[0], y, chart[2], y), fill=GRID, width=1)
        d.text((95, y - 12), f"{val:.3f}", fill=MUTED, font=F_TINY)

    tier_colors = {"core": "#2563EB", "momentum": "#059669", "extended": "#7C3AED"}
    x_centers = np.linspace(chart[0] + 190, chart[2] - 190, len(TIERS))
    rng = np.random.default_rng(42)

    for x, tier in zip(x_centers, TIERS):
        vals = best[best["feature_set"].eq(tier)]["QLIKE"].to_numpy()
        for val in vals:
            jitter = rng.uniform(-42, 42)
            y = chart[3] - (val - lo) / (hi - lo) * (chart[3] - chart[1])
            d.ellipse((x + jitter - 6, y - 6, x + jitter + 6, y + 6), fill=tier_colors[tier], outline="white", width=1)
        mean = vals.mean()
        med = np.median(vals)
        y_mean = chart[3] - (mean - lo) / (hi - lo) * (chart[3] - chart[1])
        y_med = chart[3] - (med - lo) / (hi - lo) * (chart[3] - chart[1])
        d.rounded_rectangle((x - 75, y_mean - 11, x + 75, y_mean + 11), radius=8, fill=tier_colors[tier])
        text_center(d, (int(x - 75), int(y_mean - 13), int(x + 75), int(y_mean + 13)), f"mean {mean:.3f}", "white", F_TINY)
        d.line((x - 88, y_med, x + 88, y_med), fill=DARK, width=3)
        text_center(d, (int(x - 105), chart[3] + 24, int(x + 105), chart[3] + 65), tier, DARK, F_H)

    d.text((80, 870), "Dots are best ML QLIKE values for regime-country-protocol cells within each feature tier. Lower is better.", fill=MUTED, font=F_SMALL)
    return save(c, "02_ml_feature_tier_ablation.png")


def plot_ml_vs_financial(fin: pd.DataFrame, ml: pd.DataFrame) -> Path:
    c = new_canvas(
        "Best ML Improvement over Financial Baselines",
        "Best ML models substantially reduce QLIKE relative to the best HAR-RV/GARCH baseline.",
    )
    d = c.draw
    best_fin = best_by(fin, ["regime", "country", "protocol"]).rename(columns={"QLIKE": "financial_QLIKE"})
    best_ml = best_by(ml, ["regime", "country", "feature_set", "protocol"])
    best_ml = best_by(best_ml, ["regime", "country", "protocol"]).rename(columns={"QLIKE": "ml_QLIKE"})
    merged = best_fin[["regime", "country", "protocol", "financial_QLIKE"]].merge(
        best_ml[["regime", "country", "protocol", "model", "feature_set", "ml_QLIKE"]],
        on=["regime", "country", "protocol"],
    )
    merged["improvement_pct"] = merged.apply(lambda r: pct_improvement(r["financial_QLIKE"], r["ml_QLIKE"]), axis=1)
    agg = merged.groupby(["regime", "country"], observed=False)["improvement_pct"].mean().reset_index()

    left, top = 260, 240
    cell_w, cell_h = 260, 120
    vals = agg["improvement_pct"].to_numpy()
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    for j, country in enumerate(COUNTRIES):
        text_center(d, (left + j * cell_w, top - 55, left + (j + 1) * cell_w - 10, top - 5), country, DARK, F_H)
    for i, regime in enumerate(REGIMES):
        y = top + i * cell_h
        d.text((90, y + 42), regime, fill=DARK, font=F_H)
        for j, country in enumerate(COUNTRIES):
            x = left + j * cell_w
            val = float(agg[(agg["regime"].eq(regime)) & (agg["country"].eq(country))]["improvement_pct"].iloc[0])
            fill = value_to_color(val, lo, hi, c1=(236, 253, 245), c2=(5, 150, 105))
            d.rounded_rectangle((x, y, x + cell_w - 14, y + cell_h - 16), radius=12, fill=fill, outline=GRID_DARK)
            label = f"{val:.1f}%"
            color = "white" if val > (lo + hi) / 2 else DARK
            text_center(d, (x, y + 10, x + cell_w - 14, y + cell_h - 35), label, color, font(34, True))
            d.text((x + 35, y + cell_h - 38), "QLIKE reduction", fill=MUTED, font=F_TINY)

    d.text((80, 810), "Values are average percentage QLIKE reduction across static and expanding protocols.", fill=MUTED, font=F_SMALL)
    draw_legend(d, [("lower improvement", "#ECFDF5"), ("higher improvement", "#059669")], 80, 860, cols=2)
    return save(c, "03_ml_vs_financial_improvement.png")


def plot_ml_rank_robustness(ml: pd.DataFrame) -> Path:
    c = new_canvas(
        "ML Model Robustness by QLIKE Rank",
        "Robustness is measured by average rank within identical regime-country-tier-protocol cells.",
    )
    d = c.draw
    ranked = ml.copy()
    keys = ["regime", "country", "feature_set", "protocol"]
    ranked["rank"] = ranked.groupby(keys, observed=False)["QLIKE"].rank(method="min", ascending=True)
    summary = ranked.groupby("model", observed=False)["rank"].agg(["mean", "median"]).reindex(ML_MODELS)
    win_counts = best_by(ml, keys)["model"].value_counts().reindex(ML_MODELS, fill_value=0)

    chart = (210, 250, 1420, 750)
    d.line((chart[0], chart[3], chart[2], chart[3]), fill=GRID_DARK, width=2)
    for r in range(1, 6):
        y = chart[1] + (r - 1) / 4 * (chart[3] - chart[1])
        d.line((chart[0], y, chart[2], y), fill=GRID, width=1)
        d.text((145, y - 12), f"rank {r}", fill=MUTED, font=F_TINY)

    x_centers = np.linspace(chart[0] + 105, chart[2] - 105, len(ML_MODELS))
    for x, model in zip(x_centers, ML_MODELS):
        mean_rank = float(summary.loc[model, "mean"])
        bar_top = chart[1] + (mean_rank - 1) / 4 * (chart[3] - chart[1])
        color = MODEL_COLORS[model]
        d.rounded_rectangle((x - 60, bar_top, x + 60, chart[3]), radius=10, fill=color)
        text_center(d, (int(x - 80), int(bar_top - 44), int(x + 80), int(bar_top - 8)), f"{mean_rank:.2f}", DARK, F_H)
        text_center(d, (int(x - 100), chart[3] + 20, int(x + 100), chart[3] + 72), model, DARK, F_SMALL)
        text_center(d, (int(x - 100), chart[3] + 70, int(x + 100), chart[3] + 110), f"{int(win_counts[model])} wins", MUTED, F_TINY)

    d.text((80, 880), "Lower average rank means more stable performance. Win count alone can hide rank consistency.", fill=MUTED, font=F_SMALL)
    return save(c, "04_ml_model_rank_robustness.png")


def plot_best_ml_vs_best_dl(ml: pd.DataFrame, dl: pd.DataFrame) -> Path:
    c = new_canvas(
        "Best ML vs. Best Single-DL by Condition",
        "Single DL complements ML in selected cells, but does not uniformly replace the strongest ML baseline.",
    )
    d = c.draw
    keys = ["regime", "country", "feature_set", "protocol"]
    best_ml = best_by(ml, keys).rename(columns={"QLIKE": "ml_QLIKE", "model": "ml_model"})
    best_dl = best_by(dl[dl["model"].isin(DL_MODELS)], keys).rename(columns={"QLIKE": "dl_QLIKE", "model": "dl_model"})
    m = best_ml[keys + ["ml_model", "ml_QLIKE"]].merge(best_dl[keys + ["dl_model", "dl_QLIKE"]], on=keys)

    chart = (210, 230, 1360, 800)
    vals = np.r_[m["ml_QLIKE"].to_numpy(), m["dl_QLIKE"].to_numpy()]
    lo, hi = vals.min() * 0.92, vals.max() * 1.05

    d.rectangle(chart, outline=GRID_DARK, width=2)
    for k in range(6):
        val = lo + (hi - lo) * k / 5
        x = chart[0] + (val - lo) / (hi - lo) * (chart[2] - chart[0])
        y = chart[3] - (val - lo) / (hi - lo) * (chart[3] - chart[1])
        d.line((x, chart[1], x, chart[3]), fill=GRID, width=1)
        d.line((chart[0], y, chart[2], y), fill=GRID, width=1)
        d.text((x - 20, chart[3] + 12), f"{val:.3f}", fill=MUTED, font=F_TINY)
        d.text((145, y - 12), f"{val:.3f}", fill=MUTED, font=F_TINY)

    d.line((chart[0], chart[3], chart[2], chart[1]), fill="#94A3B8", width=3)
    d.text((chart[2] - 240, chart[1] + 22), "tie line", fill=MUTED, font=F_TINY)

    for _, row in m.iterrows():
        x = chart[0] + (row["ml_QLIKE"] - lo) / (hi - lo) * (chart[2] - chart[0])
        y = chart[3] - (row["dl_QLIKE"] - lo) / (hi - lo) * (chart[3] - chart[1])
        color = REGIME_COLORS[row["regime"]]
        d.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline="white", width=2)

    d.text(((chart[0] + chart[2]) // 2 - 120, 845), "Best ML QLIKE", fill=DARK, font=F_H)
    d.text((45, 505), "Best single-DL QLIKE", fill=DARK, font=F_H)
    dl_better = int((m["dl_QLIKE"] < m["ml_QLIKE"]).sum())
    d.text((80, 885), f"Single DL beats best ML in {dl_better}/{len(m)} comparable cells. Lower-left is better.", fill=MUTED, font=F_SMALL)
    draw_legend(d, [(r, REGIME_COLORS[r]) for r in REGIMES], 1050, 860, cols=2)
    return save(c, "05_best_ml_vs_best_single_dl.png")


def _box_stats(values: np.ndarray) -> tuple[float, float, float, float, float]:
    return tuple(np.percentile(values, [5, 25, 50, 75, 95]))


def plot_model_family_robustness(fin: pd.DataFrame, ml: pd.DataFrame, dl: pd.DataFrame, ensemble: pd.DataFrame) -> Path:
    c = new_canvas(
        "Model Family Robustness",
        "Financial baselines, ML, single-DL, and ensemble serve different roles in the final forecasting framework.",
    )
    d = c.draw

    data = {
        "Financial": best_by(fin, ["regime", "country", "protocol"])["QLIKE"].to_numpy(),
        "ML": best_by(ml, ["regime", "country", "feature_set", "protocol"])["QLIKE"].to_numpy(),
        "Single DL": best_by(dl[dl["model"].isin(DL_MODELS)], ["regime", "country", "feature_set", "protocol"])["QLIKE"].to_numpy(),
    }
    if not ensemble.empty:
        data["Ensemble"] = ensemble["ensemble_QLIKE"].to_numpy()

    chart = (210, 240, 1400, 770)
    vals = np.concatenate(list(data.values()))
    lo, hi = np.nanmin(vals) * 0.9, np.nanmax(vals) * 1.08
    d.line((chart[0], chart[3], chart[2], chart[3]), fill=GRID_DARK, width=2)
    d.line((chart[0], chart[1], chart[0], chart[3]), fill=GRID_DARK, width=2)

    for k in range(6):
        val = lo + (hi - lo) * k / 5
        y = chart[3] - (val - lo) / (hi - lo) * (chart[3] - chart[1])
        d.line((chart[0], y, chart[2], y), fill=GRID, width=1)
        d.text((135, y - 12), f"{val:.3f}", fill=MUTED, font=F_TINY)

    x_centers = np.linspace(chart[0] + 150, chart[2] - 150, len(data))
    for x, (name, values) in zip(x_centers, data.items()):
        p5, q1, med, q3, p95 = _box_stats(values)
        def y_of(v: float) -> float:
            return chart[3] - (v - lo) / (hi - lo) * (chart[3] - chart[1])

        color = MODEL_COLORS[name]
        y5, y1, ym, y3, y95 = map(y_of, [p5, q1, med, q3, p95])
        d.line((x, y5, x, y95), fill=color, width=4)
        d.line((x - 45, y5, x + 45, y5), fill=color, width=3)
        d.line((x - 45, y95, x + 45, y95), fill=color, width=3)
        d.rounded_rectangle((x - 75, y3, x + 75, y1), radius=8, fill="#FFFFFF", outline=color, width=4)
        d.line((x - 75, ym, x + 75, ym), fill=color, width=5)
        text_center(d, (int(x - 105), chart[3] + 24, int(x + 105), chart[3] + 70), name, DARK, F_H)
        text_center(d, (int(x - 105), chart[3] + 70, int(x + 105), chart[3] + 106), f"median {med:.3f}", MUTED, F_TINY)

    d.text((80, 880), "Boxes show 25th-75th percentile, line is median, whiskers show 5th-95th percentile. Lower QLIKE is better.", fill=MUTED, font=F_SMALL)
    return save(c, "06_model_family_robustness.png")


def write_manifest(paths: list[Path]) -> None:
    descriptions = {
        "01_ml_winner_map_qlike.png": "Best ML model by QLIKE for each regime-country-feature-protocol condition.",
        "02_ml_feature_tier_ablation.png": "Best ML QLIKE distribution across core, momentum, and extended feature tiers.",
        "03_ml_vs_financial_improvement.png": "Average percentage QLIKE reduction of best ML over best financial baseline.",
        "04_ml_model_rank_robustness.png": "Average QLIKE rank and winner counts for ML models.",
        "05_best_ml_vs_best_single_dl.png": "Comparable-cell scatter of best ML QLIKE against best single-DL QLIKE.",
        "06_model_family_robustness.png": "QLIKE distribution by model family: financial, ML, single-DL, ensemble.",
    }
    rows = [
        {
            "figure": p.name,
            "path": p.relative_to(ROOT).as_posix(),
            "report_use": descriptions[p.name],
        }
        for p in paths
    ]
    pd.DataFrame(rows).to_csv(OUT / "figure_manifest.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    fin, ml, dl, ensemble = load_data()
    paths = [
        plot_ml_winner_map(ml),
        plot_feature_tier_ablation(ml),
        plot_ml_vs_financial(fin, ml),
        plot_ml_rank_robustness(ml),
        plot_best_ml_vs_best_dl(ml, dl),
        plot_model_family_robustness(fin, ml, dl, ensemble),
    ]
    write_manifest(paths)
    print("Generated figures:")
    for p in paths:
        print(" -", p.relative_to(ROOT).as_posix())
    print(" -", (OUT / "figure_manifest.csv").relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
