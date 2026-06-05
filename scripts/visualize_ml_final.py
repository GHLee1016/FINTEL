"""Generate final-report ML/DL visualization assets for FINTEL.

The output is intentionally chart-first: each PNG is a clean analytical graph
that can be inserted into the final report or a slide, with minimal narrative.
Only pandas/numpy/Pillow are required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "figures" / "ml_final"

W, H = 1600, 1000
BG = "#FFFFFF"
INK = "#111827"
TEXT = "#334155"
MUTED = "#64748B"
GRID = "#E5E7EB"
BLUE = "#2563EB"
BLUE_L = "#DBEAFE"
GREEN = "#059669"
GREEN_L = "#D1FAE5"
ORANGE = "#EA580C"
ORANGE_L = "#FFEDD5"
RED = "#DC2626"
RED_L = "#FEE2E2"
PURPLE = "#7C3AED"
PURPLE_L = "#EDE9FE"
TEAL = "#0F766E"
SLATE = "#475569"

REGIMES = ["normal", "911", "gfc", "covid"]
COUNTRIES = ["US", "KR", "JP"]
TIERS = ["core", "momentum", "extended"]
ML_MODELS = ["Ridge", "ElasticNet", "Huber", "LightGBM", "XGBoost"]
MODEL_COLORS = {
    "Ridge": BLUE,
    "ElasticNet": PURPLE,
    "Huber": ORANGE,
    "LightGBM": GREEN,
    "XGBoost": RED,
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


F_TITLE = font(38, True)
F_SUB = font(20)
F_AXIS = font(18)
F_LABEL = font(21)
F_SMALL = font(16)
F_NUM = font(18, True)


def text_size(d: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = d.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((70, 46), title, fill=INK, font=F_TITLE)
    d.text((70, 94), subtitle, fill=MUTED, font=F_SUB)
    return img, d


def footer(d: ImageDraw.ImageDraw, note: str) -> None:
    d.text((70, 952), note, fill=MUTED, font=F_SMALL)


def save(img: Image.Image, name: str) -> None:
    img.save(OUT / name, quality=95)


def qlike_axis(values: pd.Series, lo_pad: float = 0.0) -> tuple[float, float]:
    vals = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    lo, hi = float(vals.min()), float(vals.max())
    pad = max((hi - lo) * 0.08, 1e-6)
    return max(0.0, lo - pad - lo_pad), hi + pad


def pct_axis(values: pd.Series, symmetric: bool = False) -> tuple[float, float]:
    vals = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    lo, hi = float(vals.min()), float(vals.max())
    if symmetric:
        m = max(abs(lo), abs(hi), 1.0)
        return -m * 1.12, m * 1.12
    pad = max((hi - lo) * 0.1, 1.0)
    return lo - pad, hi + pad


def draw_grid_y(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], y_ticks: list[float], y_min: float, y_max: float, suffix: str = "") -> None:
    x0, y0, x1, y1 = box
    for tick in y_ticks:
        if tick < y_min or tick > y_max:
            continue
        y = y1 - (tick - y_min) / (y_max - y_min) * (y1 - y0)
        d.line((x0, y, x1, y), fill=GRID, width=1)
        label = f"{tick:g}{suffix}"
        tw, th = text_size(d, label, F_AXIS)
        d.text((x0 - tw - 12, y - th / 2), label, fill=MUTED, font=F_AXIS)
    d.line((x0, y1, x1, y1), fill="#CBD5E1", width=2)
    d.line((x0, y0, x0, y1), fill="#CBD5E1", width=2)


def grouped_bar_chart(
    d: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    data: pd.DataFrame,
    groups: list[str],
    series: list[str],
    colors: dict[str, str],
    y_min: float,
    y_max: float,
    y_ticks: list[float],
    suffix: str = "",
    zero_line: bool = False,
) -> None:
    x0, y0, x1, y1 = box
    draw_grid_y(d, box, y_ticks, y_min, y_max, suffix=suffix)
    if zero_line and y_min < 0 < y_max:
        y = y1 - (0 - y_min) / (y_max - y_min) * (y1 - y0)
        d.line((x0, y, x1, y), fill=SLATE, width=2)

    group_w = (x1 - x0) / len(groups)
    bar_w = min(42, group_w / (len(series) + 1.4))
    for gi, group in enumerate(groups):
        center = x0 + group_w * (gi + 0.5)
        for si, name in enumerate(series):
            value = float(data.loc[group, name])
            bx0 = center - (len(series) * bar_w) / 2 + si * bar_w
            bx1 = bx0 + bar_w * 0.82
            y_val = y1 - (value - y_min) / (y_max - y_min) * (y1 - y0)
            y_zero = y1 - (0 - y_min) / (y_max - y_min) * (y1 - y0) if y_min < 0 < y_max else y1
            d.rounded_rectangle((bx0, min(y_val, y_zero), bx1, max(y_val, y_zero)), radius=3, fill=colors[name])
            label = f"{value:.1f}{suffix}"
            tw, th = text_size(d, label, F_SMALL)
            ly = min(y_val, y_zero) - th - 6 if value >= 0 else max(y_val, y_zero) + 4
            d.text((bx0 + (bx1 - bx0 - tw) / 2, ly), label, fill=TEXT, font=F_SMALL)
        tw, th = text_size(d, group, F_LABEL)
        d.text((center - tw / 2, y1 + 22), group, fill=TEXT, font=F_LABEL)

    lx = x1 - 360
    for i, name in enumerate(series):
        display_name = {
            "momentum_vs_core": "Momentum vs core",
            "extended_vs_core": "Extended vs core",
            "ensemble_gain_pct": "Ensemble gain",
            "dl_win_rate": "DL win rate",
        }.get(name, name)
        d.rounded_rectangle((lx, y0 - 48 + i * 30, lx + 22, y0 - 29 + i * 30), radius=3, fill=colors[name])
        d.text((lx + 32, y0 - 51 + i * 30), display_name, fill=TEXT, font=F_AXIS)


def dot_distribution(
    d: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    rows: pd.DataFrame,
    group_col: str,
    value_col: str,
    groups: list[str],
    y_min: float,
    y_max: float,
    ticks: list[float],
    color: str,
    suffix: str = "%",
    zero_line: bool = False,
) -> None:
    x0, y0, x1, y1 = box
    draw_grid_y(d, box, ticks, y_min, y_max, suffix=suffix)
    if zero_line and y_min < 0 < y_max:
        y = y1 - (0 - y_min) / (y_max - y_min) * (y1 - y0)
        d.line((x0, y, x1, y), fill=SLATE, width=2)
    group_w = (x1 - x0) / len(groups)
    offsets = [-33, -22, -11, 0, 11, 22, 33, -27, -16, -5, 5, 16, 27]
    for gi, group in enumerate(groups):
        vals = rows.loc[rows[group_col] == group, value_col].dropna().astype(float).tolist()
        center = x0 + group_w * (gi + 0.5)
        for i, value in enumerate(vals):
            x = center + offsets[i % len(offsets)]
            y = y1 - (value - y_min) / (y_max - y_min) * (y1 - y0)
            d.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline=BG, width=2)
        mean = float(np.mean(vals))
        y_mean = y1 - (mean - y_min) / (y_max - y_min) * (y1 - y0)
        d.line((center - 50, y_mean, center + 50, y_mean), fill=INK, width=4)
        label = f"mean {mean:.1f}{suffix}"
        tw, th = text_size(d, label, F_SMALL)
        d.text((center - tw / 2, y_mean - th - 8), label, fill=INK, font=F_SMALL)
        tw, th = text_size(d, group, F_LABEL)
        d.text((center - tw / 2, y1 + 22), group, fill=TEXT, font=F_LABEL)


def heat_color(value: float, max_value: float, base: str) -> str:
    if max_value <= 0:
        return "#F8FAFC"
    alpha = value / max_value
    if base == "green":
        start, end = np.array([236, 253, 245]), np.array([5, 150, 105])
    else:
        start, end = np.array([239, 246, 255]), np.array([37, 99, 235])
    rgb = (start * (1 - alpha) + end * alpha).astype(int)
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def heatmap_counts(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], matrix: pd.DataFrame, base: str = "blue") -> None:
    x0, y0, x1, y1 = box
    rows = list(matrix.index)
    cols = list(matrix.columns)
    cw = (x1 - x0) / len(cols)
    ch = (y1 - y0) / len(rows)
    max_v = float(matrix.max().max())
    for ci, col in enumerate(cols):
        tw, th = text_size(d, col, F_AXIS)
        d.text((x0 + cw * (ci + 0.5) - tw / 2, y0 - 36), col, fill=TEXT, font=F_AXIS)
    for ri, row in enumerate(rows):
        tw, th = text_size(d, row, F_LABEL)
        d.text((x0 - tw - 18, y0 + ch * (ri + 0.5) - th / 2), row, fill=TEXT, font=F_LABEL)
        for ci, col in enumerate(cols):
            val = float(matrix.loc[row, col])
            fill = heat_color(val, max_v, base)
            rx0 = x0 + ci * cw + 4
            ry0 = y0 + ri * ch + 4
            rx1 = x0 + (ci + 1) * cw - 4
            ry1 = y0 + (ri + 1) * ch - 4
            d.rounded_rectangle((rx0, ry0, rx1, ry1), radius=5, fill=fill, outline=BG)
            label = f"{int(val)}"
            tw, th = text_size(d, label, F_NUM)
            d.text((rx0 + (rx1 - rx0 - tw) / 2, ry0 + (ry1 - ry0 - th) / 2), label, fill=INK, font=F_NUM)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ml = pd.concat(
        [pd.read_csv(RESULTS / f"ml_results_{tier}.csv") for tier in TIERS],
        ignore_index=True,
    )
    dl = pd.read_csv(RESULTS / "dl_results_master.csv")
    financial = pd.read_csv(RESULTS / "financial_results.csv")
    ensemble = pd.read_csv(RESULTS / "ensemble_best_comparison.csv")
    for df in [ml, dl, financial, ensemble]:
        for col in ["QLIKE", "single_QLIKE", "ensemble_QLIKE", "QLIKE_improvement"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return ml, dl, financial, ensemble


def best(df: pd.DataFrame, keys: list[str], value: str = "QLIKE") -> pd.DataFrame:
    return df.loc[df.groupby(keys)[value].idxmin()].reset_index(drop=True)


def fig_01(ml: pd.DataFrame, financial: pd.DataFrame) -> tuple[str, str]:
    best_ml = best(ml, ["regime", "country", "protocol"])
    best_fin = best(financial, ["regime", "country", "protocol"])
    merged = best_ml.merge(
        best_fin,
        on=["regime", "country", "protocol"],
        suffixes=("_ml", "_fin"),
    )
    merged["improvement_pct"] = (merged["QLIKE_fin"] - merged["QLIKE_ml"]) / merged["QLIKE_fin"] * 100
    y_min, y_max = pct_axis(merged["improvement_pct"])
    ticks = [30, 40, 50, 60, 70, 80]
    img, d = canvas(
        "Financial Baseline vs Best ML",
        "Each dot is one regime-country-protocol cell; value is QLIKE reduction vs best financial baseline.",
    )
    dot_distribution(d, (190, 185, 1490, 810), merged, "regime", "improvement_pct", REGIMES, y_min, y_max, ticks, BLUE)
    footer(d, f"n={len(merged)} cells. Positive values mean lower QLIKE than GARCH/HAR-RV.")
    name = "01_financial_baseline_vs_best_ml.png"
    save(img, name)
    return name, "Distribution of QLIKE improvement from best financial baseline to best ML."


def fig_02(ml: pd.DataFrame) -> tuple[str, str]:
    best_ml = best(ml, ["regime", "country", "feature_set", "protocol"])
    counts = (
        best_ml.groupby(["regime", "model"]).size().unstack(fill_value=0).reindex(index=REGIMES, columns=ML_MODELS, fill_value=0)
    )
    img, d = canvas(
        "Best ML Model Frequency by Market Regime",
        "Counts are based on the winning model within each regime-country-feature-protocol cell.",
    )
    heatmap_counts(d, (300, 230, 1450, 770), counts, base="blue")
    footer(d, f"n={len(best_ml)} cells. Darker cells indicate more frequent model wins.")
    name = "02_ml_winner_frequency_heatmap.png"
    save(img, name)
    return name, "Heatmap showing which ML model wins most often in each market regime."


def fig_03(ml: pd.DataFrame) -> tuple[str, str]:
    tier_best = best(ml, ["regime", "country", "protocol", "feature_set"])
    pivot = tier_best.pivot_table(index=["regime", "country", "protocol"], columns="feature_set", values="QLIKE")
    rows = pivot.reset_index()
    rows["momentum_vs_core"] = (rows["core"] - rows["momentum"]) / rows["core"] * 100
    rows["extended_vs_core"] = (rows["core"] - rows["extended"]) / rows["core"] * 100
    means = rows.groupby("regime")[["momentum_vs_core", "extended_vs_core"]].mean().reindex(REGIMES)
    y_min, y_max = pct_axis(pd.concat([means["momentum_vs_core"], means["extended_vs_core"]]), symmetric=True)
    ticks = [-20, -10, 0, 10, 20, 30, 40]
    img, d = canvas(
        "Feature Expansion Gain Relative to Core Features",
        "Bars show average QLIKE reduction by regime; positive means the feature tier improves on core.",
    )
    grouped_bar_chart(
        d,
        (180, 190, 1480, 800),
        means,
        REGIMES,
        ["momentum_vs_core", "extended_vs_core"],
        {"momentum_vs_core": TEAL, "extended_vs_core": ORANGE},
        y_min,
        y_max,
        ticks,
        suffix="%",
        zero_line=True,
    )
    footer(d, "Computed from best model within each feature tier for every regime-country-protocol cell.")
    name = "03_feature_tier_gain_vs_core.png"
    save(img, name)
    return name, "Grouped bars showing feature-tier gains over the core feature set."


def fig_04(ml: pd.DataFrame) -> tuple[str, str]:
    cell_best = best(ml, ["regime", "country", "feature_set", "protocol"])
    pivot = cell_best.pivot_table(index=["regime", "country", "feature_set"], columns="protocol", values="QLIKE").reset_index()
    pivot["expanding_gain_pct"] = (pivot["static"] - pivot["expanding"]) / pivot["static"] * 100
    y_min, y_max = pct_axis(pivot["expanding_gain_pct"], symmetric=True)
    ticks = [-30, -20, -10, 0, 10, 20, 30]
    img, d = canvas(
        "Expanding-Window Training Gain",
        "Each dot compares best expanding-window ML against best static ML for the same regime-country-feature cell.",
    )
    dot_distribution(d, (190, 185, 1490, 810), pivot, "regime", "expanding_gain_pct", REGIMES, y_min, y_max, ticks, GREEN, zero_line=True)
    footer(d, f"n={len(pivot)} cells. Positive values mean expanding-window training lowers QLIKE.")
    name = "04_expanding_window_gain_distribution.png"
    save(img, name)
    return name, "Dot distribution of expanding-window gain over static ML."


def fig_05(ml: pd.DataFrame, dl: pd.DataFrame) -> tuple[str, str]:
    best_ml = best(ml, ["regime", "country", "feature_set", "protocol"])
    best_dl = best(dl, ["regime", "country", "feature_set", "protocol"])
    merged = best_ml.merge(
        best_dl,
        on=["regime", "country", "feature_set", "protocol"],
        suffixes=("_ml", "_dl"),
    )
    merged["dl_gain_pct"] = (merged["QLIKE_ml"] - merged["QLIKE_dl"]) / merged["QLIKE_ml"] * 100
    summary = (
        merged.assign(dl_win=merged["dl_gain_pct"] > 0)
        .groupby("regime")
        .agg(dl_win_rate=("dl_win", "mean"), mean_gain=("dl_gain_pct", "mean"), wins=("dl_win", "sum"), cells=("dl_win", "size"))
        .reindex(REGIMES)
    )
    summary["dl_win_rate"] = summary["dl_win_rate"] * 100
    img, d = canvas(
        "Best Single DL Win Rate against Best ML",
        "Bars show the share of matched cells where a single DL model lowers QLIKE relative to best ML.",
    )
    grouped_bar_chart(
        d,
        (180, 190, 1480, 800),
        summary[["dl_win_rate"]],
        REGIMES,
        ["dl_win_rate"],
        {"dl_win_rate": PURPLE},
        0,
        100,
        [0, 20, 40, 60, 80, 100],
        suffix="%",
    )
    x0, y0, x1, y1 = (180, 190, 1480, 800)
    group_w = (x1 - x0) / len(REGIMES)
    for gi, regime in enumerate(REGIMES):
        row = summary.loc[regime]
        center = x0 + group_w * (gi + 0.5)
        label = f"{int(row['wins'])}/{int(row['cells'])} cells, mean gain {row['mean_gain']:.1f}%"
        tw, th = text_size(d, label, F_SMALL)
        d.text((center - tw / 2, 845), label, fill=MUTED, font=F_SMALL)
    wins = int((merged["dl_gain_pct"] > 0).sum())
    footer(d, f"DL improves {wins}/{len(merged)} cells overall. Mean gain is negative when DL's average QLIKE is higher than ML.")
    name = "05_best_single_dl_vs_best_ml.png"
    save(img, name)
    return name, "Bar chart showing where best single DL beats best ML at the matched-cell level."


def fig_06(ensemble: pd.DataFrame) -> tuple[str, str]:
    ensemble = ensemble.copy()
    ensemble["ensemble_gain_pct"] = ensemble["QLIKE_improvement"] / ensemble["single_QLIKE"] * 100
    means = ensemble.groupby("regime")["ensemble_gain_pct"].mean().reindex(REGIMES).to_frame("ensemble_gain_pct")
    y_min = 0
    y_max = max(float(means["ensemble_gain_pct"].max()) * 1.35, 20)
    ticks = [0, 5, 10, 15, 20]
    img, d = canvas(
        "Ensemble Gain over Best Single DL",
        "Bars show average QLIKE reduction; labels show cell-level ensemble win rate.",
    )
    grouped_bar_chart(
        d,
        (180, 190, 1480, 800),
        means,
        REGIMES,
        ["ensemble_gain_pct"],
        {"ensemble_gain_pct": BLUE},
        y_min,
        y_max,
        ticks,
        suffix="%",
    )
    x0, y0, x1, y1 = (180, 190, 1480, 800)
    group_w = (x1 - x0) / len(REGIMES)
    for gi, regime in enumerate(REGIMES):
        sub = ensemble[ensemble["regime"] == regime]
        win_rate = sub["ensemble_better_QLIKE"].astype(str).str.lower().eq("true").mean() * 100
        label = f"win rate {win_rate:.0f}%"
        tw, th = text_size(d, label, F_SMALL)
        d.text((x0 + group_w * (gi + 0.5) - tw / 2, 845), label, fill=MUTED, font=F_SMALL)
    footer(d, f"n={len(ensemble)} cells. Ensemble comparison is against the best single DL model, not against ML.")
    name = "06_ensemble_gain_over_single_dl.png"
    save(img, name)
    return name, "Bar chart of ensemble QLIKE gain over best single DL."


def write_manifest(items: list[tuple[str, str]]) -> None:
    pd.DataFrame(items, columns=["figure", "description"]).to_csv(OUT / "figure_manifest.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in OUT.iterdir():
        if path.is_file():
            path.unlink()

    ml, dl, financial, ensemble = load_data()
    items = [
        fig_01(ml, financial),
        fig_02(ml),
        fig_03(ml),
        fig_04(ml),
        fig_05(ml, dl),
        fig_06(ensemble),
    ]
    write_manifest(items)


if __name__ == "__main__":
    main()
