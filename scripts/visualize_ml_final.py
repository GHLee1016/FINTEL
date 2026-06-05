"""Generate final-report chart assets for FINTEL ML/DL results.

The figures are designed as report-ready analytical charts, not slide-like
explainers. Each chart uses direct comparisons, win/loss matrices, or summary
bars so the result is readable without scattered point clouds.
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
AXIS = "#CBD5E1"
BLUE = "#2563EB"
BLUE_L = "#DBEAFE"
GREEN = "#059669"
GREEN_L = "#D1FAE5"
TEAL = "#0F766E"
TEAL_L = "#CCFBF1"
ORANGE = "#EA580C"
ORANGE_L = "#FFEDD5"
RED = "#DC2626"
RED_L = "#FEE2E2"
PURPLE = "#7C3AED"
PURPLE_L = "#EDE9FE"
SLATE = "#475569"
SLATE_L = "#F1F5F9"

REGIMES = ["normal", "911", "gfc", "covid"]
COUNTRIES = ["US", "KR", "JP"]
TIERS = ["core", "momentum", "extended"]
ML_MODELS = ["Ridge", "ElasticNet", "Huber", "LightGBM", "XGBoost"]


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
F_LABEL_B = font(21, True)
F_SMALL = font(16)
F_NUM = font(18, True)
F_BIG = font(28, True)


def text_size(d: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = d.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((70, 46), title, fill=INK, font=F_TITLE)
    d.text((70, 96), subtitle, fill=MUTED, font=F_SUB)
    return img, d


def footer(d: ImageDraw.ImageDraw, note: str) -> None:
    d.text((70, 952), note, fill=MUTED, font=F_SMALL)


def save(img: Image.Image, name: str) -> None:
    img.save(OUT / name, quality=95)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ml = pd.concat([pd.read_csv(RESULTS / f"ml_results_{tier}.csv") for tier in TIERS], ignore_index=True)
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


def draw_y_grid(
    d: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    ticks: list[float],
    y_min: float,
    y_max: float,
    fmt: str = "{:.3f}",
) -> None:
    x0, y0, x1, y1 = box
    for tick in ticks:
        if tick < y_min or tick > y_max:
            continue
        y = y1 - (tick - y_min) / (y_max - y_min) * (y1 - y0)
        d.line((x0, y, x1, y), fill=GRID, width=1)
        label = fmt.format(tick)
        tw, th = text_size(d, label, F_AXIS)
        d.text((x0 - tw - 12, y - th / 2), label, fill=MUTED, font=F_AXIS)
    d.line((x0, y1, x1, y1), fill=AXIS, width=2)
    d.line((x0, y0, x0, y1), fill=AXIS, width=2)


def draw_legend(d: ImageDraw.ImageDraw, items: list[tuple[str, str]], x: int, y: int) -> None:
    for i, (label, color) in enumerate(items):
        yy = y + i * 32
        d.rounded_rectangle((x, yy, x + 22, yy + 20), radius=3, fill=color)
        d.text((x + 32, yy - 3), label, fill=TEXT, font=F_AXIS)


def grouped_bars(
    d: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    data: pd.DataFrame,
    groups: list[str],
    series: list[str],
    labels: dict[str, str],
    colors: dict[str, str],
    y_min: float,
    y_max: float,
    ticks: list[float],
    fmt: str = "{:.3f}",
    label_fmt: str = "{:.3f}",
    zero_line: bool = False,
) -> None:
    x0, y0, x1, y1 = box
    draw_y_grid(d, box, ticks, y_min, y_max, fmt)
    if zero_line and y_min < 0 < y_max:
        y = y1 - (0 - y_min) / (y_max - y_min) * (y1 - y0)
        d.line((x0, y, x1, y), fill=SLATE, width=2)
    group_w = (x1 - x0) / len(groups)
    bar_w = min(58, group_w / (len(series) + 1.3))
    for gi, group in enumerate(groups):
        center = x0 + group_w * (gi + 0.5)
        for si, key in enumerate(series):
            value = float(data.loc[group, key])
            bx0 = center - (len(series) * bar_w) / 2 + si * bar_w
            bx1 = bx0 + bar_w * 0.82
            yv = y1 - (value - y_min) / (y_max - y_min) * (y1 - y0)
            yz = y1 - (0 - y_min) / (y_max - y_min) * (y1 - y0) if y_min < 0 < y_max else y1
            d.rounded_rectangle((bx0, min(yv, yz), bx1, max(yv, yz)), radius=4, fill=colors[key])
            label = label_fmt.format(value)
            tw, th = text_size(d, label, F_SMALL)
            ly = min(yv, yz) - th - 7 if value >= 0 else max(yv, yz) + 5
            d.text((bx0 + (bx1 - bx0 - tw) / 2, ly), label, fill=TEXT, font=F_SMALL)
        tw, th = text_size(d, group, F_LABEL)
        d.text((center - tw / 2, y1 + 24), group, fill=TEXT, font=F_LABEL)
    draw_legend(d, [(labels[key], colors[key]) for key in series], x1 - 360, y0 - 50)


def heat_shade(value: float, max_value: float, base: str = "blue") -> str:
    if max_value <= 0:
        return SLATE_L
    alpha = value / max_value
    palettes = {
        "blue": (np.array([239, 246, 255]), np.array([37, 99, 235])),
        "green": (np.array([209, 250, 229]), np.array([5, 150, 105])),
        "red": (np.array([254, 226, 226]), np.array([220, 38, 38])),
    }
    start, end = palettes.get(base, palettes["blue"])
    rgb = (start * (1 - alpha) + end * alpha).astype(int)
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def count_heatmap(
    d: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    matrix: pd.DataFrame,
    highlight_max: bool = True,
) -> None:
    x0, y0, x1, y1 = box
    rows = list(matrix.index)
    cols = list(matrix.columns)
    cw = (x1 - x0) / len(cols)
    ch = (y1 - y0) / len(rows)
    max_v = float(matrix.max().max())
    for ci, col in enumerate(cols):
        tw, th = text_size(d, col, F_AXIS)
        d.text((x0 + cw * (ci + 0.5) - tw / 2, y0 - 38), col, fill=TEXT, font=F_AXIS)
    for ri, row in enumerate(rows):
        tw, th = text_size(d, row, F_LABEL)
        d.text((x0 - tw - 18, y0 + ch * (ri + 0.5) - th / 2), row, fill=TEXT, font=F_LABEL)
        row_max = float(matrix.loc[row].max())
        for ci, col in enumerate(cols):
            val = float(matrix.loc[row, col])
            rx0 = x0 + ci * cw + 4
            ry0 = y0 + ri * ch + 4
            rx1 = x0 + (ci + 1) * cw - 4
            ry1 = y0 + (ri + 1) * ch - 4
            outline = INK if highlight_max and val == row_max and row_max > 0 else BG
            width = 4 if outline == INK else 1
            d.rounded_rectangle((rx0, ry0, rx1, ry1), radius=5, fill=heat_shade(val, max_v), outline=outline, width=width)
            label = f"{int(val)}"
            tw, th = text_size(d, label, F_NUM)
            d.text((rx0 + (rx1 - rx0 - tw) / 2, ry0 + (ry1 - ry0 - th) / 2), label, fill=INK, font=F_NUM)


def win_loss_heatmap(
    d: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    matrix: pd.DataFrame,
    value_matrix: pd.DataFrame,
    row_labels: list[str],
    col_labels: list[str],
) -> None:
    x0, y0, x1, y1 = box
    cw = (x1 - x0) / len(col_labels)
    ch = (y1 - y0) / len(row_labels)
    for ci, col in enumerate(col_labels):
        tw, th = text_size(d, col, F_LABEL_B)
        d.text((x0 + cw * (ci + 0.5) - tw / 2, y0 - 40), col, fill=TEXT, font=F_LABEL_B)
    for ri, row in enumerate(row_labels):
        tw, th = text_size(d, row, F_LABEL_B)
        d.text((x0 - tw - 24, y0 + ch * (ri + 0.5) - th / 2), row, fill=TEXT, font=F_LABEL_B)
        for ci, col in enumerate(col_labels):
            win_rate = float(matrix.loc[row, col])
            mean_gain = float(value_matrix.loc[row, col])
            fill = GREEN_L if win_rate >= 50 else RED_L
            stroke = GREEN if win_rate >= 50 else RED
            rx0 = x0 + ci * cw + 6
            ry0 = y0 + ri * ch + 6
            rx1 = x0 + (ci + 1) * cw - 6
            ry1 = y0 + (ri + 1) * ch - 6
            d.rounded_rectangle((rx0, ry0, rx1, ry1), radius=6, fill=fill, outline=stroke, width=3)
            label = f"{win_rate:.0f}%"
            tw, th = text_size(d, label, F_BIG)
            d.text((rx0 + (rx1 - rx0 - tw) / 2, ry0 + 33), label, fill=INK, font=F_BIG)
            sub = f"gain {mean_gain:+.1f}%"
            tw, th = text_size(d, sub, F_SMALL)
            d.text((rx0 + (rx1 - rx0 - tw) / 2, ry0 + 78), sub, fill=TEXT, font=F_SMALL)


def fig_01(ml: pd.DataFrame, financial: pd.DataFrame) -> tuple[str, str]:
    best_ml = best(ml, ["regime", "country", "protocol"])
    best_fin = best(financial, ["regime", "country", "protocol"])
    merged = best_ml.merge(best_fin, on=["regime", "country", "protocol"], suffixes=("_ml", "_fin"))
    summary = merged.groupby("regime")[["QLIKE_fin", "QLIKE_ml"]].mean().reindex(REGIMES)
    summary["reduction_pct"] = (summary["QLIKE_fin"] - summary["QLIKE_ml"]) / summary["QLIKE_fin"] * 100
    max_y = float(summary[["QLIKE_fin", "QLIKE_ml"]].max().max()) * 1.2
    ticks = [0, 0.02, 0.04, 0.06, 0.08, 0.10]
    img, d = canvas(
        "Average QLIKE: Financial Baseline vs Best ML",
        "Lower is better. Bars compare regime-level averages across country-protocol cells.",
    )
    grouped_bars(
        d,
        (190, 190, 1480, 790),
        summary,
        REGIMES,
        ["QLIKE_fin", "QLIKE_ml"],
        {"QLIKE_fin": "Best financial baseline", "QLIKE_ml": "Best ML"},
        {"QLIKE_fin": SLATE, "QLIKE_ml": BLUE},
        0,
        max_y,
        ticks,
    )
    x0, _, x1, y1 = (190, 190, 1480, 790)
    group_w = (x1 - x0) / len(REGIMES)
    for gi, regime in enumerate(REGIMES):
        label = f"{summary.loc[regime, 'reduction_pct']:.1f}% lower"
        tw, th = text_size(d, label, F_LABEL_B)
        d.text((x0 + group_w * (gi + 0.5) - tw / 2, y1 + 68), label, fill=BLUE, font=F_LABEL_B)
    footer(d, f"n={len(merged)} matched cells. Financial baseline is the better of GARCH and HAR-RV.")
    name = "01_baseline_vs_ml_grouped_bar.png"
    save(img, name)
    return name, "Grouped bars comparing average QLIKE of best financial baseline and best ML by regime."


def fig_02(ml: pd.DataFrame) -> tuple[str, str]:
    best_ml = best(ml, ["regime", "country", "feature_set", "protocol"])
    counts = best_ml.groupby(["regime", "model"]).size().unstack(fill_value=0).reindex(index=REGIMES, columns=ML_MODELS, fill_value=0)
    img, d = canvas(
        "ML Winner Frequency by Regime",
        "Each cell counts how often a model is the lowest-QLIKE ML model; row maxima are outlined.",
    )
    count_heatmap(d, (300, 230, 1450, 770), counts)
    footer(d, f"n={len(best_ml)} cells. This chart supports regime-aware model selection instead of one universal ML model.")
    name = "02_ml_winner_regime_heatmap.png"
    save(img, name)
    return name, "Heatmap of winning ML model frequency by regime."


def fig_03(ml: pd.DataFrame) -> tuple[str, str]:
    best_tier = best(ml, ["regime", "country", "protocol"])
    counts = best_tier.groupby(["regime", "feature_set"]).size().unstack(fill_value=0).reindex(index=REGIMES, columns=TIERS, fill_value=0)
    img, d = canvas(
        "Best Feature Tier Frequency by Regime",
        "Each cell counts how often a feature tier produces the lowest QLIKE; row maxima are outlined.",
    )
    count_heatmap(d, (390, 230, 1350, 770), counts)
    totals = counts.sum(axis=0)
    total_text = "Overall wins: " + " | ".join(f"{tier} {int(totals[tier])}" for tier in TIERS)
    tw, th = text_size(d, total_text, F_LABEL_B)
    d.text((390 + (960 - tw) / 2, 825), total_text, fill=TEXT, font=F_LABEL_B)
    footer(d, f"n={len(best_tier)} cells. This chart checks whether adding more features consistently improves QLIKE.")
    name = "03_feature_tier_winner_heatmap.png"
    save(img, name)
    return name, "Heatmap showing how often each feature tier is the best option by regime."


def fig_04(ml: pd.DataFrame) -> tuple[str, str]:
    cell_best = best(ml, ["regime", "country", "feature_set", "protocol"])
    pivot = cell_best.pivot_table(index=["regime", "country", "feature_set"], columns="protocol", values="QLIKE").reset_index()
    pivot["expanding_gain_pct"] = (pivot["static"] - pivot["expanding"]) / pivot["static"] * 100
    summary = pivot.assign(exp_win=pivot["expanding_gain_pct"] > 0).groupby("regime").agg(
        win_rate=("exp_win", "mean"),
        mean_gain=("expanding_gain_pct", "mean"),
        wins=("exp_win", "sum"),
        cells=("exp_win", "size"),
    ).reindex(REGIMES)
    summary["win_rate"] *= 100
    img, d = canvas(
        "Static vs Expanding-Window ML",
        "Bars show expanding-window win rate; signed labels show average QLIKE reduction vs static.",
    )
    grouped_bars(
        d,
        (190, 190, 1480, 790),
        summary[["win_rate"]],
        REGIMES,
        ["win_rate"],
        {"win_rate": "Expanding win rate"},
        {"win_rate": GREEN},
        0,
        100,
        [0, 20, 40, 60, 80, 100],
        fmt="{:.0f}%",
        label_fmt="{:.0f}%",
    )
    x0, _, x1, y1 = (190, 190, 1480, 790)
    group_w = (x1 - x0) / len(REGIMES)
    for gi, regime in enumerate(REGIMES):
        row = summary.loc[regime]
        label = f"{int(row['wins'])}/{int(row['cells'])} wins | mean gain {row['mean_gain']:+.1f}%"
        tw, th = text_size(d, label, F_SMALL)
        d.text((x0 + group_w * (gi + 0.5) - tw / 2, y1 + 68), label, fill=TEXT, font=F_SMALL)
    footer(d, f"n={len(pivot)} cells. Positive mean gain means expanding-window training lowers QLIKE.")
    name = "04_static_vs_expanding_summary.png"
    save(img, name)
    return name, "Summary bar chart of expanding-window win rate and mean QLIKE gain against static training."


def fig_05(ml: pd.DataFrame, dl: pd.DataFrame) -> tuple[str, str]:
    best_ml = best(ml, ["regime", "country", "feature_set", "protocol"])
    best_dl = best(dl, ["regime", "country", "feature_set", "protocol"])
    merged = best_ml.merge(best_dl, on=["regime", "country", "feature_set", "protocol"], suffixes=("_ml", "_dl"))
    merged["dl_gain_pct"] = (merged["QLIKE_ml"] - merged["QLIKE_dl"]) / merged["QLIKE_ml"] * 100
    matrix = merged.assign(dl_win=merged["dl_gain_pct"] > 0).groupby(["regime", "country"])["dl_win"].mean().unstack().reindex(index=REGIMES, columns=COUNTRIES) * 100
    gains = merged.groupby(["regime", "country"])["dl_gain_pct"].mean().unstack().reindex(index=REGIMES, columns=COUNTRIES)
    img, d = canvas(
        "Best Single DL vs Best ML: Win/Loss Matrix",
        "Each tile shows DL win rate across feature-protocol cells; green means DL wins at least half.",
    )
    win_loss_heatmap(d, (310, 235, 1410, 785), matrix, gains, REGIMES, COUNTRIES)
    draw_legend(d, [("DL win rate >= 50%", GREEN), ("DL win rate < 50%", RED)], 1120, 185)
    wins = int((merged["dl_gain_pct"] > 0).sum())
    footer(d, f"DL wins {wins}/{len(merged)} matched cells overall. Gain is QLIKE reduction relative to best ML.")
    name = "05_ml_vs_dl_win_loss_heatmap.png"
    save(img, name)
    return name, "Win/loss heatmap comparing best single DL against best ML by regime and country."


def fig_06(ensemble: pd.DataFrame) -> tuple[str, str]:
    ensemble = ensemble.copy()
    summary = ensemble.groupby("regime")[["single_QLIKE", "ensemble_QLIKE"]].mean().reindex(REGIMES)
    summary["reduction_pct"] = (summary["single_QLIKE"] - summary["ensemble_QLIKE"]) / summary["single_QLIKE"] * 100
    max_y = float(summary[["single_QLIKE", "ensemble_QLIKE"]].max().max()) * 1.2
    ticks = np.linspace(0, max_y, 6).tolist()
    img, d = canvas(
        "Best Single DL vs DL Ensemble",
        "Lower is better. Bars compare regime-level average QLIKE within the DL model family.",
    )
    grouped_bars(
        d,
        (190, 190, 1480, 790),
        summary,
        REGIMES,
        ["single_QLIKE", "ensemble_QLIKE"],
        {"single_QLIKE": "Best single DL", "ensemble_QLIKE": "DL ensemble"},
        {"single_QLIKE": SLATE, "ensemble_QLIKE": BLUE},
        0,
        max_y,
        ticks,
    )
    x0, _, x1, y1 = (190, 190, 1480, 790)
    group_w = (x1 - x0) / len(REGIMES)
    for gi, regime in enumerate(REGIMES):
        win_rate = ensemble.loc[ensemble["regime"] == regime, "ensemble_better_QLIKE"].astype(str).str.lower().eq("true").mean() * 100
        label = f"{summary.loc[regime, 'reduction_pct']:.1f}% lower | win rate {win_rate:.0f}%"
        tw, th = text_size(d, label, F_SMALL)
        d.text((x0 + group_w * (gi + 0.5) - tw / 2, y1 + 68), label, fill=BLUE, font=F_SMALL)
    footer(d, f"n={len(ensemble)} DL ensemble cells. This comparison is within DL, not against ML.")
    name = "06_single_dl_vs_ensemble_grouped_bar.png"
    save(img, name)
    return name, "Grouped bars comparing best single DL and DL ensemble average QLIKE by regime."


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
