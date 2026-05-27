"""Equal-weight ensemble evaluation for the four DL prediction logs.

The ensemble combines model predictions only within an identical
(regime, country, feature_set, protocol) cell. It never mixes feature tiers
or static/expanding protocols.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..eval.metrics import evaluate
from ..eval.phases import iter_phases

MODELS = ["LSTM", "1DCNN", "TCN", "TST"]
REGIMES = ["normal", "911", "gfc", "covid"]
COUNTRIES = ["US", "KR", "JP"]
TIERS = ["core", "momentum", "extended"]
PROTOCOLS = ["static", "expanding"]

SOURCE_LOGS = {
    "LSTM": ("LSTM_HS", "log/LSTM/log_LSTM_{regime}_{country}.csv"),
    "1DCNN": ("CNNBased_GH", "log/1DCNN/log_1DCNN_{regime}_{country}.csv"),
    "TCN": ("CNNBased_GH", "log/TCN/log_TCN_{regime}_{country}.csv"),
    "TST": ("TST_JH", "log/TST/log_TST_{regime}_{country}.csv"),
}

LOG_COLUMNS = ["date", "feature_set", "L", "protocol", "RV_target", "RV_pred"]
CELL_COLUMNS = ["regime", "country", "feature_set", "protocol"]
DATE_KEY_COLUMNS = CELL_COLUMNS + ["date"]
TARGET_TOLERANCE = 1e-6


def _raw_log_url(repo: str, branch: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"


def collect_prediction_master(repo: str = "GHLee1016/FINTEL") -> pd.DataFrame:
    """Download the latest date-level DL predictions from their source branches."""
    frames = []
    for model, (branch, pattern) in SOURCE_LOGS.items():
        for regime in REGIMES:
            for country in COUNTRIES:
                path = pattern.format(regime=regime, country=country)
                url = _raw_log_url(repo, branch, path)
                try:
                    df = pd.read_csv(url)
                except Exception as exc:
                    raise FileNotFoundError(
                        f"Could not read {model} prediction log: {branch}/{path}"
                    ) from exc

                missing = set(LOG_COLUMNS) - set(df.columns)
                if missing:
                    raise ValueError(f"{model} {path} missing columns: {sorted(missing)}")

                df = df[LOG_COLUMNS].copy()
                df.insert(0, "country", country)
                df.insert(0, "regime", regime)
                df.insert(0, "model", model)
                df["source_branch"] = branch
                df["source_file"] = path
                frames.append(df)

    master = pd.concat(frames, ignore_index=True)
    master["date"] = pd.to_datetime(master["date"])
    master["RV_pred"] = np.clip(master["RV_pred"].astype(float), 1e-12, None)
    master["RV_target"] = master["RV_target"].astype(float)
    return master.sort_values(["model", *DATE_KEY_COLUMNS]).reset_index(drop=True)


def coverage_table(master: pd.DataFrame) -> pd.DataFrame:
    """Return prediction row counts for each expected model and experiment cell."""
    expected = pd.MultiIndex.from_product(
        [MODELS, REGIMES, COUNTRIES, TIERS, PROTOCOLS],
        names=["model", *CELL_COLUMNS],
    ).to_frame(index=False)
    counts = (
        master.groupby(["model", *CELL_COLUMNS], as_index=False)
        .size()
        .rename(columns={"size": "n_rows"})
    )
    return expected.merge(counts, how="left").fillna({"n_rows": 0}).astype({"n_rows": int})


def validate_full_coverage(master: pd.DataFrame) -> pd.DataFrame:
    """Validate that all four models can be compared fairly in every cell."""
    required = {"model", *DATE_KEY_COLUMNS, "L", "RV_target", "RV_pred"}
    missing_columns = required - set(master.columns)
    if missing_columns:
        raise ValueError(f"Prediction master missing columns: {sorted(missing_columns)}")

    duplicate_mask = master.duplicated(["model", *DATE_KEY_COLUMNS], keep=False)
    if duplicate_mask.any():
        duplicated = master.loc[duplicate_mask, ["model", *DATE_KEY_COLUMNS]].head()
        raise ValueError(f"Duplicated model/date predictions found:\n{duplicated}")

    coverage = coverage_table(master)
    missing_cells = coverage[coverage["n_rows"] == 0]
    if not missing_cells.empty:
        text = missing_cells[["model", *CELL_COLUMNS]].to_string(index=False)
        raise ValueError(
            "Prediction logs are incomplete. Upload missing logs before running the "
            f"four-model ensemble:\n{text}"
        )

    for cell_key, cell in master.groupby(CELL_COLUMNS, sort=False):
        counts = cell.groupby("model").size()
        if counts.nunique() != 1:
            raise ValueError(f"Date row count mismatch in cell {cell_key}: {counts.to_dict()}")

        date_counts = cell.groupby("date")["model"].nunique()
        if not (date_counts == len(MODELS)).all():
            raise ValueError(f"Models do not share identical dates in cell {cell_key}")

        target_spread = cell.groupby("date")["RV_target"].agg(lambda x: x.max() - x.min())
        if (target_spread > TARGET_TOLERANCE).any():
            raise ValueError(f"RV_target mismatch across models in cell {cell_key}")

    return coverage


def _model_combinations(models: Iterable[str] = MODELS) -> list[tuple[str, ...]]:
    ordered = list(models)
    return [
        combo
        for n_models in range(2, len(ordered) + 1)
        for combo in combinations(ordered, n_models)
    ]


def build_equal_weight_predictions(master: pd.DataFrame) -> pd.DataFrame:
    """Create date-level equal-weight predictions for every DL model combination."""
    validate_full_coverage(master)
    combos = _model_combinations()
    rows = []

    for cell_key, cell in master.groupby(CELL_COLUMNS, sort=False):
        regime, country, feature_set, protocol = cell_key
        pred_wide = cell.pivot(index="date", columns="model", values="RV_pred")
        target = cell.groupby("date")["RV_target"].first().sort_index()
        l_map = cell.groupby("model")["L"].unique().to_dict()
        if any(len(values) != 1 for values in l_map.values()):
            raise ValueError(f"Multiple lookback lengths found in cell {cell_key}")

        for combo in combos:
            member_ls = "|".join(f"{model}:L{int(l_map[model][0])}" for model in combo)
            pred = pred_wide[list(combo)].mean(axis=1).sort_index()
            out = pd.DataFrame(
                {
                    "ensemble_method": "equal_weight",
                    "models": "+".join(combo),
                    "n_models": len(combo),
                    "regime": regime,
                    "country": country,
                    "feature_set": feature_set,
                    "protocol": protocol,
                    "member_Ls": member_ls,
                    "date": pred.index,
                    "RV_target": target.loc[pred.index].values,
                    "RV_pred": np.clip(pred.values, 1e-12, None),
                }
            )
            rows.append(out)

    return pd.concat(rows, ignore_index=True)


def evaluate_ensemble_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate existing project metrics for Full Test and configured phases."""
    result_rows = []
    group_cols = [
        "ensemble_method",
        "models",
        "n_models",
        "regime",
        "country",
        "feature_set",
        "protocol",
        "member_Ls",
    ]
    for keys, group in predictions.groupby(group_cols, sort=False):
        info = dict(zip(group_cols, keys))
        indexed = group.sort_values("date").set_index("date")
        for phase, mask, _color in iter_phases(indexed, info["regime"]):
            if mask.sum() == 0:
                continue
            metrics = evaluate(indexed["RV_target"].values[mask], indexed["RV_pred"].values[mask])
            result_rows.append({**info, "phase": phase, **metrics, "n_obs": int(mask.sum())})

    return pd.DataFrame(result_rows)


def build_single_vs_ensemble_summary(
    single_results_path: str | Path,
    ensemble_results: pd.DataFrame,
) -> pd.DataFrame:
    """Compare the best single DL model and best equal-weight ensemble by QLIKE."""
    single = pd.read_csv(single_results_path)
    single = single[(single["phase"] == "Full Test") & single["model"].isin(MODELS)].copy()
    ensemble = ensemble_results[ensemble_results["phase"] == "Full Test"].copy()
    key = CELL_COLUMNS

    best_single = single.loc[single.groupby(key)["QLIKE"].idxmin(), key + ["model", "QLIKE", "RMSE_CV"]]
    best_single = best_single.rename(
        columns={"model": "best_single_model", "QLIKE": "single_QLIKE", "RMSE_CV": "single_RMSE_CV"}
    )

    best_ensemble = ensemble.loc[
        ensemble.groupby(key)["QLIKE"].idxmin(),
        key + ["models", "n_models", "QLIKE", "RMSE_CV"],
    ]
    best_ensemble = best_ensemble.rename(
        columns={
            "models": "best_ensemble_models",
            "n_models": "ensemble_n_models",
            "QLIKE": "ensemble_QLIKE",
            "RMSE_CV": "ensemble_RMSE_CV",
        }
    )

    summary = best_single.merge(best_ensemble, on=key, how="inner")
    summary["QLIKE_improvement"] = summary["single_QLIKE"] - summary["ensemble_QLIKE"]
    summary["RMSE_CV_improvement"] = summary["single_RMSE_CV"] - summary["ensemble_RMSE_CV"]
    summary["ensemble_better_QLIKE"] = summary["QLIKE_improvement"] > 0
    return summary.sort_values(key).reset_index(drop=True)
