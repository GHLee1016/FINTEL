"""One-shot audit: Feature Tier Effect — before vs after protocol-locking.

Compares the original best_by(full, [country, regime, feature_set]) with the
protocol-locked version (Core's best protocol is fixed per (country, regime)
and all tiers must use that protocol).

Output:
  summary_tables/feature_tier_lock_diff.csv

US/COVID row should show core_to_ext_after_% ~= 10.6 (down from ~12.1) to
confirm the patch.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SUMMARY_DIR = SCRIPT_DIR / "summary_tables"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def find_results_dir() -> Path:
    for candidate in [REPO_ROOT, REPO_ROOT / "Project"]:
        results_dir = candidate / "results"
        if (results_dir / "ml_results_core.csv").exists():
            return results_dir
    raise FileNotFoundError("Could not locate FINTEL results directory.")


def best_by(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    idx = df.groupby(keys, dropna=False)["RMSE_CV"].idxmin()
    return df.loc[idx].copy()


def main() -> None:
    results_dir = find_results_dir()
    frames = []
    for tier in ["core", "momentum", "extended"]:
        frame = pd.read_csv(results_dir / f"ml_results_{tier}.csv")
        if "feature_set" not in frame.columns:
            frame["feature_set"] = tier
        frames.append(frame)
    ml = pd.concat(frames, ignore_index=True)
    full = ml[ml["phase"] == "Full Test"].copy()

    # --- Before: free protocol per (country, regime, feature_set) ---
    before = (
        best_by(full, ["country", "regime", "feature_set"])
        .pivot_table(
            index=["country", "regime"],
            columns="feature_set",
            values="RMSE_CV",
            aggfunc="min",
        )
        .add_suffix("_before")
    )

    # --- After: protocol locked to Core's best protocol per (country, regime) ---
    core = full[full["feature_set"] == "core"].copy()
    core_winner_idx = core.groupby(["country", "regime"])["RMSE_CV"].idxmin()
    locked = (
        core.loc[core_winner_idx, ["country", "regime", "protocol"]]
            .rename(columns={"protocol": "locked_protocol"})
    )
    full_locked = (
        full.merge(locked, on=["country", "regime"])
            .query("protocol == locked_protocol")
            .drop(columns=["locked_protocol"])
    )
    after = (
        best_by(full_locked, ["country", "regime", "feature_set"])
        .pivot_table(
            index=["country", "regime"],
            columns="feature_set",
            values="RMSE_CV",
            aggfunc="min",
        )
        .add_suffix("_after")
    )

    # Also capture best-model identity per cell (before/after) for sanity check
    best_before = (
        best_by(full, ["country", "regime", "feature_set"])
        [["country", "regime", "feature_set", "model", "protocol"]]
    )
    bm_before = best_before.pivot_table(
        index=["country", "regime"],
        columns="feature_set",
        values="model",
        aggfunc="first",
    ).add_suffix("_model_before")
    bp_before = best_before.pivot_table(
        index=["country", "regime"],
        columns="feature_set",
        values="protocol",
        aggfunc="first",
    ).add_suffix("_protocol_before")

    best_after = (
        best_by(full_locked, ["country", "regime", "feature_set"])
        [["country", "regime", "feature_set", "model", "protocol"]]
    )
    bm_after = best_after.pivot_table(
        index=["country", "regime"],
        columns="feature_set",
        values="model",
        aggfunc="first",
    ).add_suffix("_model_after")

    diff = (
        before.join(after, how="outer")
              .join(locked.set_index(["country", "regime"]))
              .join(bm_before, how="left")
              .join(bp_before, how="left")
              .join(bm_after, how="left")
    )
    diff["core_to_ext_before_%"] = (
        (diff["core_before"] - diff["extended_before"]) / diff["core_before"] * 100
    )
    diff["core_to_ext_after_%"] = (
        (diff["core_after"] - diff["extended_after"]) / diff["core_after"] * 100
    )
    diff["core_to_mom_before_%"] = (
        (diff["core_before"] - diff["momentum_before"]) / diff["core_before"] * 100
    )
    diff["core_to_mom_after_%"] = (
        (diff["core_after"] - diff["momentum_after"]) / diff["core_after"] * 100
    )

    # Order rows for readability
    country_order = {"US": 0, "KR": 1, "JP": 2}
    regime_order = {"normal": 0, "911": 1, "gfc": 2, "covid": 3}
    diff = diff.reset_index()
    diff["_co"] = diff["country"].map(country_order)
    diff["_ro"] = diff["regime"].map(regime_order)
    diff = (
        diff.sort_values(["_co", "_ro"])
            .drop(columns=["_co", "_ro"])
            .set_index(["country", "regime"])
    )

    out = SUMMARY_DIR / "feature_tier_lock_diff.csv"
    diff.round(4).to_csv(out, encoding="utf-8-sig")
    print(f"[ok] Wrote: {out}")
    print()

    cols_to_show = [
        "core_before", "core_after",
        "extended_before", "extended_after",
        "locked_protocol",
        "core_to_ext_before_%", "core_to_ext_after_%",
    ]
    print(diff[cols_to_show].round(4).to_string())
    print()

    # Sanity checks
    core_eq = (diff["core_before"].round(6) == diff["core_after"].round(6)).all()
    print(f"[check] core_before == core_after for every cell? {core_eq}")
    monotone = (
        diff["extended_after"].fillna(diff["extended_before"])
        >= diff["extended_before"] - 1e-9
    ).all()
    print(f"[check] extended_after >= extended_before for every cell? {monotone}")
    us_covid = diff.loc[("US", "covid")]
    print(
        f"[check] US/COVID core_to_ext: "
        f"{us_covid['core_to_ext_before_%']:.2f}% -> {us_covid['core_to_ext_after_%']:.2f}%"
    )


if __name__ == "__main__":
    main()
