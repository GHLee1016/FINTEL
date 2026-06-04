"""DL 학습용 cell 로드 wrapper.

dataset_DL/L{L}/{regime}_{country}_*.{meta.csv,X.npy,y.npy}에서
한 cell의 train/valid/test split을 scaling 적용 후 numpy로 반환.

핵심 함수:
- `load_one(regime, country, L, tier)` → splits dict (X_*, y_*, meta_*, scaler, feature_cols)
- `SequenceDataset(X, y)` → 간단한 torch.utils.data.Dataset wrapper

사용:
    from src.dl import load_one, SequenceDataset
    splits = load_one('normal', 'US', L=22, tier='core')
    # splits['train_X'].shape == (N_train, 22, 10)
    train_ds = SequenceDataset(splits['train_X'], splits['train_y'])
    loader = DataLoader(train_ds, batch_size=64, shuffle=False)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..preprocess.features import get_feature_list
from .scaler import fit_transform_splits


# Project root: src/dl/dataset.py → parents[2] = project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DL_DIR: Path = PROJECT_ROOT / "dataset_DL"


def load_one(
    regime: str,
    country: str,
    L: int,
    tier: str,
    dataset_dir: Optional[Path] = None,
) -> Dict:
    """한 (regime, country, L, tier) cell을 로드하고 train/valid/test로 split + scaling.

    Parameters
    ----------
    regime : {'normal', '911', 'gfc', 'covid'}
    country : {'US', 'KR', 'JP'}
    L : {22, 60, 252}  (사전 변환된 lookback 길이 중 하나)
    tier : {'core', 'momentum', 'extended'}
    dataset_dir : default PROJECT_ROOT/dataset_DL

    Returns
    -------
    dict :
        feature_cols : List[str]            # 길이 F (tier에 따라 10/14/28)
        scaler : CustomScaler               # train에서 fit됨
        train_X : (N_train, L, F) float32   # scaled
        train_y : (N_train,) float32
        train_meta : pd.DataFrame           # sample_id, prediction_date, split, RV_target
        valid_X, valid_y, valid_meta : same shape pattern
        test_X, test_y, test_meta : same shape pattern

    Raises
    ------
    FileNotFoundError
        dataset_DL/L{L}/ 안에 해당 cell 파일이 없을 때.
    """
    if dataset_dir is None:
        dataset_dir = DEFAULT_DATASET_DL_DIR
    dataset_dir = Path(dataset_dir)

    L_dir = dataset_dir / f"L{L}"
    meta_path = L_dir / f"{regime}_{country}_meta.csv"
    X_path = L_dir / f"{regime}_{country}_X.npy"
    y_path = L_dir / f"{regime}_{country}_y.npy"
    feat_path = dataset_dir / f"feature_columns_{country}.txt"

    for p in [meta_path, X_path, y_path, feat_path]:
        if not p.exists():
            raise FileNotFoundError(f"missing required file: {p}")

    # 1. 로드
    meta_all = pd.read_csv(meta_path)
    X_all = np.load(str(X_path))           # (N, L, 28)
    y_all = np.load(str(y_path))           # (N,)
    feat_all = feat_path.read_text(encoding="utf-8").strip().splitlines()
    if X_all.shape[2] != len(feat_all):
        raise ValueError(
            f"feature dim mismatch: X.shape[2]={X_all.shape[2]} vs feature_columns={len(feat_all)}"
        )

    # 2. tier 따라 feature column subset
    # get_feature_list는 DataFrame 받으므로 dummy DataFrame로 컬럼 셋만 전달
    dummy_df = pd.DataFrame(0.0, index=[0], columns=feat_all)
    feature_cols = get_feature_list(dummy_df, country, tier)
    col_idx = [feat_all.index(c) for c in feature_cols]
    X_all_tier = X_all[:, :, col_idx]      # (N, L, F_tier)

    # 3. split 분리
    splits_data = {}
    for split_name in ["train", "valid", "test"]:
        mask = (meta_all["split"] == split_name).values
        splits_data[split_name] = {
            "X": X_all_tier[mask],
            "y": y_all[mask],
            "meta": meta_all[mask].reset_index(drop=True),
        }

    # 4. scaling — train에서만 fit, valid/test는 transform (look-ahead 방지)
    scaler, X_train_s, X_valid_s, X_test_s = fit_transform_splits(
        splits_data["train"]["X"],
        splits_data["valid"]["X"],
        splits_data["test"]["X"],
        feature_cols,
    )

    return {
        "feature_cols": feature_cols,
        "scaler": scaler,
        "train_X": X_train_s,
        "train_y": splits_data["train"]["y"],
        "train_meta": splits_data["train"]["meta"],
        "valid_X": X_valid_s,
        "valid_y": splits_data["valid"]["y"],
        "valid_meta": splits_data["valid"]["meta"],
        "test_X": X_test_s,
        "test_y": splits_data["test"]["y"],
        "test_meta": splits_data["test"]["meta"],
    }


class SequenceDataset(Dataset):
    """numpy (N, L, F) + (N,) → torch Dataset wrapper.

    DataLoader와 함께 사용:
        ds = SequenceDataset(X, y)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        for x_batch, y_batch in loader:
            # x_batch: (B, L, F),  y_batch: (B,)
            ...
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        if X.ndim != 3:
            raise ValueError(f"X must be 3D (N, L, F), got shape {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"X/y len mismatch: {len(X)} vs {len(y)}")
        self.X = torch.from_numpy(np.ascontiguousarray(X)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(y)).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
