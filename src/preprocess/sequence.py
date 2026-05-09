"""
sequence.py
-----------
DataFrame → (X, y) 시퀀스 텐서 변환 유틸리티.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


# ──────────────────────────────────────────────
# 1. 시퀀스 생성
# ──────────────────────────────────────────────
def make_sequences(
    df          : pd.DataFrame,
    feature_cols: list[str],
    seq_len     : int,
    target_col  : str = 'RV_target',
) -> Tuple[torch.Tensor, torch.Tensor, pd.DatetimeIndex]:
    """
    슬라이딩 윈도우로 (X, y) 시퀀스 생성.

    Parameters
    ----------
    df           : 시계열 DataFrame (index = DatetimeIndex)
    feature_cols : 입력 피처 컬럼 리스트
    seq_len      : 입력 윈도우 길이
    target_col   : 예측 대상 컬럼

    Returns
    -------
    X   : FloatTensor  (N, seq_len, n_features)
    y   : FloatTensor  (N,)
    idx : DatetimeIndex  길이 N  (각 샘플의 예측 날짜)
    """
    if len(df) <= seq_len:
        raise ValueError(
            f'데이터 길이({len(df)})가 seq_len({seq_len})보다 짧거나 같습니다.'
        )

    feat_arr = df[feature_cols].values.astype(np.float32)  # (T, F)
    tgt_arr  = df[target_col].values.astype(np.float32)    # (T,)
    dates    = df.index

    N = len(df) - seq_len
    # numpy stride tricks로 빠르게 생성
    idx_arr  = np.arange(N)
    X = np.stack([feat_arr[i : i + seq_len] for i in idx_arr])  # (N, seq_len, F)
    y = tgt_arr[seq_len:]                                         # (N,)
    date_idx = dates[seq_len:]                                    # DatetimeIndex

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        date_idx,
    )


# ──────────────────────────────────────────────
# 2. Dataset / DataLoader
# ──────────────────────────────────────────────
class SequenceDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        assert len(X) == len(y)
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loader(
    X          : torch.Tensor,
    y          : torch.Tensor,
    batch_size : int  = 64,
    shuffle    : bool = False,
) -> DataLoader:
    ds = SequenceDataset(X, y)
    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = 0,                          # Colab 호환
        pin_memory  = torch.cuda.is_available(),
    )


# ──────────────────────────────────────────────
# 3. 3분할 → Loader 빌드
# ──────────────────────────────────────────────
def build_loaders(
    train_df    : pd.DataFrame,
    valid_df    : pd.DataFrame,
    test_df     : pd.DataFrame,
    feature_cols: list[str],
    seq_len     : int,
    batch_size  : int = 64,
    target_col  : str = 'RV_target',
) -> Tuple[DataLoader, DataLoader, DataLoader, pd.DatetimeIndex]:
    """
    3분할 DataFrame → (train_loader, valid_loader, test_loader, test_idx).

    valid / test 앞에 이전 구간 끝 (seq_len-1)행을 이어붙여
    윈도우 경계 문제를 해결합니다.

    Returns
    -------
    train_loader, valid_loader, test_loader : DataLoader
    test_idx : DatetimeIndex
    """
    pad = seq_len - 1

    # valid: train 끝 pad행 + valid 전체
    valid_ext = pd.concat([train_df.iloc[-pad:], valid_df]) if pad > 0 else valid_df

    # test: (train+valid) 끝 pad행 + test 전체
    combined  = pd.concat([train_df, valid_df])
    test_ext  = pd.concat([combined.iloc[-pad:], test_df]) if pad > 0 else test_df

    X_tr, y_tr, _        = make_sequences(train_df,  feature_cols, seq_len, target_col)
    X_va, y_va, _        = make_sequences(valid_ext, feature_cols, seq_len, target_col)
    X_te, y_te, test_idx = make_sequences(test_ext,  feature_cols, seq_len, target_col)

    return (
        make_loader(X_tr, y_tr, batch_size, shuffle=True),
        make_loader(X_va, y_va, batch_size, shuffle=False),
        make_loader(X_te, y_te, batch_size, shuffle=False),
        test_idx,
    )