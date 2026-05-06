"""
dl.py
-----
DL 모델 4종 정의.

모델        seq_len   비고
--------    -------   ----
CNN1D         30      1D Conv + GlobalAvgPool
LSTMModel     60      양방향 LSTM
TFTModel      60      경량 TFT (GRN + MultiheadAttention)
MambaModel   120      Selective SSM — parallel scan (순수 PyTorch)

공통 인터페이스:
    forward(x: FloatTensor[B, T, F]) → FloatTensor[B]   (양수, softplus)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════
# 1. 1D-CNN
# ══════════════════════════════════════════════
class CNN1D(nn.Module):
    """
    Conv1D × 2 → BatchNorm → ReLU → GlobalAvgPool → FC.

    Optuna 탐색 파라미터
    --------------------
    num_filters : Conv 채널 수       [32, 64, 128]
    kernel_size : 합성곱 커널 크기   [3, 5, 7]
    hidden_dim  : FC 은닉 크기       [32, 64, 128]
    dropout     : Dropout 비율       [0.0, 0.3]
    """

    def __init__(
        self,
        input_dim  : int,
        num_filters: int   = 64,
        kernel_size: int   = 3,
        hidden_dim : int   = 64,
        dropout    : float = 0.1,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(input_dim,    num_filters,     kernel_size, padding=pad)
        self.bn1   = nn.BatchNorm1d(num_filters)
        self.conv2 = nn.Conv1d(num_filters,  num_filters * 2, kernel_size, padding=pad)
        self.bn2   = nn.BatchNorm1d(num_filters * 2)
        self.drop  = nn.Dropout(dropout)
        self.fc1   = nn.Linear(num_filters * 2, hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, F) → Conv1d 요구 형식 (B, F, T)
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        x = x.mean(dim=-1)               # GlobalAvgPool → (B, num_filters*2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).squeeze(-1)      # (B,)
        return F.softplus(x)             # 양수 보장


# ══════════════════════════════════════════════
# 2. LSTM
# ══════════════════════════════════════════════
class LSTMModel(nn.Module):
    """
    양방향 LSTM → 마지막 hidden → FC.

    Optuna 탐색 파라미터
    --------------------
    hidden_dim  : LSTM 은닉 크기   [32, 64, 128]
    num_layers  : LSTM 레이어 수   [1, 2, 3]
    dropout     : Dropout 비율     [0.0, 0.3]
    """

    def __init__(
        self,
        input_dim : int,
        hidden_dim: int   = 64,
        num_layers: int   = 2,
        dropout   : float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
        )
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_dim * 2, 1)   # ×2 : 양방향

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)                       # (B, T, hidden*2)
        out    = self.drop(out[:, -1])              # 마지막 시점 → (B, hidden*2)
        out    = self.fc(out).squeeze(-1)
        return F.softplus(out)


# ══════════════════════════════════════════════
# 3. TFT (경량)
# ══════════════════════════════════════════════
class _GRN(nn.Module):
    """Gated Residual Network — TFT 핵심 블록."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1  = nn.Linear(dim, dim)
        self.fc2  = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h    = F.elu(self.fc1(x))
        h    = self.drop(h)
        gate = torch.sigmoid(self.gate(h))
        out  = gate * self.fc2(h)
        return self.norm(x + out)


class TFTModel(nn.Module):
    """
    경량 Temporal Fusion Transformer.
    Linear projection → GRN stack → MultiheadAttention → GRN → FC.

    Optuna 탐색 파라미터
    --------------------
    hidden_dim  : 내부 은닉 크기   [32, 64, 128]
    num_heads   : Attention 헤드   [2, 4, 8]
    num_layers  : GRN 레이어 수    [1, 2, 3]
    dropout     : Dropout 비율     [0.0, 0.3]

    Notes
    -----
    hidden_dim 은 num_heads 의 배수로 자동 보정됩니다.
    """

    def __init__(
        self,
        input_dim : int,
        hidden_dim: int   = 64,
        num_heads : int   = 4,
        num_layers: int   = 2,
        dropout   : float = 0.1,
    ):
        super().__init__()
        # num_heads 배수 보정
        hidden_dim = max(num_heads, (hidden_dim // num_heads) * num_heads)

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.grn_in     = nn.ModuleList(
            [_GRN(hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm  = nn.LayerNorm(hidden_dim)
        self.grn_out    = _GRN(hidden_dim, dropout)
        self.drop       = nn.Dropout(dropout)
        self.fc         = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)                     # (B, T, H)
        for grn in self.grn_in:
            x = grn(x)
        attn_out, _ = self.attn(x, x, x)
        x = self.attn_norm(x + attn_out)           # residual
        x = self.grn_out(x)
        x = self.drop(x[:, -1])                    # 마지막 시점 → (B, H)
        return F.softplus(self.fc(x).squeeze(-1))


# ══════════════════════════════════════════════
# 4. Mamba (Selective SSM — parallel scan)
# ══════════════════════════════════════════════
def _parallel_scan(
    log_a : torch.Tensor,   # (B, T, D)  log-scale decay
    b     : torch.Tensor,   # (B, T, D)  input contribution
) -> torch.Tensor:
    """
    선형 재귀  h[t] = a[t] * h[t-1] + b[t]  를
    log-space cumsum으로 병렬 계산.

    Returns
    -------
    h : (B, T, D)
    """
    # log-space: log_h[t] = log_a[t] + log_h[t-1]  (단, b 항은 근사)
    # 정확한 associative scan 대신 안정적인 근사:
    # h[t] = exp(cumsum(log_a)[t]) * sum_s( b[s] * exp(-cumsum(log_a)[s]) )
    # → prefix product 방식
    T = log_a.shape[1]

    # cumulative log_a  (B, T, D)
    cum_log_a = torch.cumsum(log_a, dim=1)

    # exp(cum_log_a - cum_log_a[t]) * b 의 누적합
    # = exp(cum_log_a) * cumsum( b * exp(-cum_log_a) )
    exp_cum   = torch.exp(cum_log_a)                        # (B, T, D)
    b_scaled  = b * torch.exp(-cum_log_a)                   # (B, T, D)
    h         = exp_cum * torch.cumsum(b_scaled, dim=1)     # (B, T, D)
    return h


class _MambaBlock(nn.Module):
    """
    Mamba 블록 (순수 PyTorch, parallel scan).

    Parameters
    ----------
    d_model : 모델 차원
    d_state : SSM 상태 차원
    d_conv  : depthwise conv 커널 크기
    expand  : 내부 확장 비율 (d_inner = d_model × expand)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv : int = 4,
        expand : int = 2,
    ):
        super().__init__()
        d_inner = d_model * expand

        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(
            d_inner, d_inner, d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True,
        )
        # dt, B_ssm, C_ssm projection
        self.x_proj   = nn.Linear(d_inner, d_state * 2 + d_inner, bias=False)
        self.dt_proj  = nn.Linear(d_inner, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)

        # A: (d_inner, d_state) — 고정 log 파라미터
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.register_buffer(
            'log_A',
            A.log().unsqueeze(0).expand(d_inner, -1),   # (d_inner, d_state)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        B, T, _ = x.shape
        residual = x

        # 1) Input projection + gating
        xz    = self.in_proj(x)                          # (B, T, d_inner*2)
        x_in, z = xz.chunk(2, dim=-1)                    # 각 (B, T, d_inner)

        # 2) Depthwise conv (causal: 앞 d_conv-1 패딩 → 뒤 자르기)
        x_conv = self.conv1d(
            x_in.permute(0, 2, 1)                        # (B, d_inner, T)
        )[..., :T].permute(0, 2, 1)                      # (B, T, d_inner)
        x_conv = F.silu(x_conv)

        # 3) SSM 파라미터 계산
        d_state = self.log_A.shape[1]
        ssm_in  = self.x_proj(x_conv)                    # (B, T, d_state*2 + d_inner)
        B_ssm   = ssm_in[..., :d_state]                  # (B, T, d_state)
        C_ssm   = ssm_in[..., d_state : d_state * 2]     # (B, T, d_state)
        dt      = F.softplus(
            self.dt_proj(ssm_in[..., d_state * 2:])      # (B, T, d_inner)
        )

        # 4) Selective decay: log_a = -dt * exp(log_A)
        #    log_A: (d_inner, d_state) → (1, 1, d_inner, d_state)
        log_A   = self.log_A.unsqueeze(0).unsqueeze(0)
        A_pos   = torch.exp(log_A)                        # (1, 1, d_inner, d_state)
        # dt: (B, T, d_inner) → (B, T, d_inner, 1)
        log_a   = -(dt.unsqueeze(-1) * A_pos)             # (B, T, d_inner, d_state)

        # b = dt ⊗ B_ssm: (B, T, d_inner, d_state)
        b_ssm   = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)  # (B, T, d_inner, d_state)

        # 5) Parallel scan — 각 (d_inner, d_state) 채널 독립
        #    reshape: (B, T, d_inner*d_state)
        d_inner = dt.shape[-1]
        log_a_flat = log_a.reshape(B, T, d_inner * d_state)
        b_flat     = b_ssm.reshape(B, T, d_inner * d_state)

        h_flat = _parallel_scan(log_a_flat, b_flat)       # (B, T, d_inner*d_state)
        h      = h_flat.reshape(B, T, d_inner, d_state)   # (B, T, d_inner, d_state)

        # 6) Output: y[t] = C_ssm[t] · h[t]
        #    C_ssm: (B, T, d_state) → (B, T, 1, d_state)
        y = (h * C_ssm.unsqueeze(2)).sum(-1)              # (B, T, d_inner)

        # 7) Gating + output projection
        y   = y * F.silu(z)
        out = self.out_proj(y)                            # (B, T, d_model)
        return self.norm(out + residual)


class MambaModel(nn.Module):
    """
    Mamba (Selective SSM) 기반 시계열 예측 모델.

    Optuna 탐색 파라미터
    --------------------
    d_model    : 모델 차원          [32, 64, 128]
    d_state    : SSM 상태 차원      [8, 16, 32]
    d_conv     : depthwise 커널     [2, 4]
    expand     : 내부 확장 비율     [1, 2]
    num_layers : Mamba 블록 수      [1, 2, 3]
    dropout    : Dropout 비율       [0.0, 0.3]
    """

    def __init__(
        self,
        input_dim : int,
        d_model   : int   = 64,
        d_state   : int   = 16,
        d_conv    : int   = 4,
        expand    : int   = 2,
        num_layers: int   = 2,
        dropout   : float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks     = nn.ModuleList([
            _MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)          # (B, T, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.drop(x[:, -1])         # 마지막 시점 → (B, d_model)
        return F.softplus(self.fc(x).squeeze(-1))


# ══════════════════════════════════════════════
# 5. Registry
# ══════════════════════════════════════════════
DL_MODEL_REGISTRY: dict[str, type] = {
    '1DCNN': CNN1D,
    'LSTM' : LSTMModel,
    'TFT'  : TFTModel,
    'Mamba': MambaModel,
}

DL_SEQ_LEN: dict[str, int] = {
    '1DCNN': 30,
    'LSTM' : 60,
    'TFT'  : 60,
    'Mamba': 120,
}

ALL_DL_MODEL_NAMES: list[str] = list(DL_MODEL_REGISTRY)


def make_dl_model(model_name: str, input_dim: int, **kwargs) -> nn.Module:
    """
    모델 이름 + 하이퍼파라미터로 DL 모델 인스턴스 생성.

    Parameters
    ----------
    model_name : '1DCNN' | 'LSTM' | 'TFT' | 'Mamba'
    input_dim  : 입력 피처 수 (F)
    **kwargs   : 모델별 하이퍼파라미터
    """
    if model_name not in DL_MODEL_REGISTRY:
        raise ValueError(
            f'Unknown DL model: {model_name}. '
            f'Available: {list(DL_MODEL_REGISTRY)}'
        )
    return DL_MODEL_REGISTRY[model_name](input_dim=input_dim, **kwargs)