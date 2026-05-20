"""DL 모델 패키지 — PyTorch 기반 신경망 모델들.

ML 모델(src/models/ml.py)과 분리. 학습 인터페이스는 비슷하지만 (fit/predict)
내부는 PyTorch nn.Module + 학습 루프.

모델 종류:
- 1D-CNN (cnn1d.py)     — Conv1d + AdaptiveAvgPool1d
- TCN (tcn.py)          — Dilated causal conv + residual (Bai et al., 2018)

모든 모델 동일 인터페이스:
    model.fit(X_train, y_train, X_valid, y_valid)
    y_pred = model.predict(X_test)
    model.history_df()
    model.save_checkpoint(path, extra=...)
    ModelClass.from_checkpoint(path)
"""

from .cnn1d import CNN1DNet, CNN1DModel
from .tcn import TCNNet, TCNModel

__all__ = [
    "CNN1DNet", "CNN1DModel",
    "TCNNet", "TCNModel",
]
