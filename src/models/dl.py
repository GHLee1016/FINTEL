"""
dl.py
-----
DL 모델 정의.

모델      seq_len   비고
------    -------   -------------------------------
1DCNN       30      1D Conv + GlobalAvgPool
TCN         60      Dilated causal residual blocks

공통 인터페이스:
    forward(x: FloatTensor[B, T, F]) -> FloatTensor[B]   (양수, softplus)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D(nn.Module):
    """
    Conv1D x 2 -> BatchNorm -> ReLU -> GlobalAvgPool -> FC.

    Optuna 탐색 파라미터
    --------------------
    num_filters : Conv 채널 수       [32, 64, 128]
    kernel_size : 합성곱 커널 크기   [3, 5, 7]
    hidden_dim  : FC 은닉 크기       [32, 64, 128]
    dropout     : Dropout 비율       [0.0, 0.3]
    """

    def __init__(
        self,
        input_dim: int,
        num_filters: int = 64,
        kernel_size: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(input_dim, num_filters, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(num_filters * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        x = x.mean(dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).squeeze(-1)
        return F.softplus(x)


class _CausalConv1d(nn.Module):
    """왼쪽 패딩만 적용해 미래 정보 누수를 막는 1D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class _TCNResidualBlock(nn.Module):
    """TCN 기본 residual block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        self.conv1 = _CausalConv1d(channels, channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = _CausalConv1d(channels, channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        return F.relu(x + residual)


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network.

    Input projection -> dilated causal residual blocks -> GlobalAvgPool -> FC.

    Optuna 탐색 파라미터
    --------------------
    hidden_dim  : 블록 채널 수       [32, 64, 128]
    kernel_size : 합성곱 커널 크기   [2, 3, 5]
    num_layers  : residual block 수  [2, 3, 4]
    dropout     : Dropout 비율       [0.0, 0.3]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                _TCNResidualBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2 ** layer_idx,
                    dropout=dropout,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.drop(x.mean(dim=-1))
        x = F.relu(self.fc1(x))
        x = self.fc2(x).squeeze(-1)
        return F.softplus(x)


DL_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "1DCNN": CNN1D,
    "TCN": TCNModel,
}

DL_SEQ_LEN: dict[str, int] = {
    "1DCNN": 30,
    "TCN": 60,
}

ALL_DL_MODEL_NAMES: list[str] = list(DL_MODEL_REGISTRY)


def make_dl_model(model_name: str, input_dim: int, **kwargs) -> nn.Module:
    """모델 이름과 파라미터로 DL 모델 인스턴스를 생성한다."""
    if model_name not in DL_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown DL model: {model_name}. "
            f"Available: {list(DL_MODEL_REGISTRY)}"
        )
    return DL_MODEL_REGISTRY[model_name](input_dim=input_dim, **kwargs)
