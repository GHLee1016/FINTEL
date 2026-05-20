# FINTEL DL 모델 작업 — 컨텍스트 + 정합성 원칙

성균관대 데이터사이언스 캡스톤 FINTEL 팀. 
일별 RV(Realized Volatility) 예측 프로젝트.
ML 모델(Ridge/EN/Huber/LightGBM/XGBoost) 5종은 이미 완성. 이제 DL 단계(LSTM/1DCNN/TCN/TST/그룹구조 NN) 진행 중.
이 프롬프트는 DL 인프라 설계 원칙 + 새 DL 모델 추가 시 따라야 할 규칙을 정리한 것.

## 1. 프로젝트 구조 (필독)

```
FINTEL/
├── dataset/                      # 원본 12 CSV ({regime}_{country}_dataset.csv)
├── dataset_DL/                   # DL용 사전 변환된 sliding window (Meta+npy)
│   ├── feature_columns_{US,KR,JP}.txt
│   └── L{22,60,252}/
│       └── {regime}_{country}_{X.npy, y.npy, meta.csv}
├── src/
│   ├── config.py                 # REGIMES, COUNTRIES, SPLITS, PHASES
│   ├── data_loader.py            # 원본 CSV 로드 (ML/금융용)
│   ├── preprocess/
│   │   ├── features.py           # CORE/MOMENTUM/EXTENDED + get_feature_list, SPILLOVER_MAP
│   │   └── scaler.py             # SCALING_CONFIG (ML), SCALING_CONFIG_DL, CustomScaler
│   ├── dl/                       # ★ DL 전용 모듈
│   │   ├── scaler.py             # 3D 시퀀스용 fit_transform_splits 등
│   │   └── dataset.py            # load_one(regime, country, L, tier), SequenceDataset
│   ├── models/
│   │   ├── ml.py                 # ML 5종 wrapper (_BaseMLModel)
│   │   └── dl/                   # ★ DL 모델 패키지
│   │       └── lstm.py           # LSTMNet, LSTMModel
│   └── eval/
│       ├── metrics.py            # rmse, mae, qlike, rmse_cv, evaluate (재사용)
│       ├── phases.py             # iter_phases, get_phase_mask (재사용)
│       └── protocols.py          # run_expanding (ML용, DL은 별도)
├── notebooks/
│   ├── 02_ml.ipynb               # ML 학습 (참고용)
│   ├── 04_build_dl_dataset.ipynb # dataset_DL 생성 (1회 실행 후 사용)
│   └── 05_lstm.ipynb             # LSTM 학습 (★ 새 모델 추가 시 이 패턴 따라가기)
├── scripts/
│   └── build_lstm_notebook.py    # 05_lstm.ipynb 자동 생성 스크립트
└── results/
    ├── dl_hp_config.json
    ├── dl_tuned_L.csv            # 36 cells × 3 L의 valid metrics + best L 표시
    ├── dl_results_{tier}.csv     # ML schema + L, is_best_L 컬럼
    └── best_dl_models/*.pt       # self-contained checkpoint (state_dict + hp + metadata)
```

## 2. 핵심 정합성 원칙 (절대 어기지 말 것)

### 2-1. 결과 schema는 ML과 동일
`make_result_visualizations.py` 등 기존 분석 도구가 자동 통합되려면 컬럼이 동일해야 함.

```python
# 모든 DL 모델의 결과 row schema (필수)
{
    'model': 'LSTM' (또는 모델 이름),
    'regime': str, 'country': str,
    'feature_set': str,           # 'core' / 'momentum' / 'extended'
    'L': int,                     # ★ DL 전용 추가 컬럼
    'protocol': 'static' or 'expanding',
    'phase': str,                 # 'Full Test' + sub-phase
    'RMSE': float, 'MAE': float, 'QLIKE': float, 'RMSE_CV': float,
    # 추가 메타 (있으면)
    'tuning_val_qlike': float,
    'tuning_best_epoch': int,
    'final_epochs': int,
    'n_features': int,
    'is_best_L': bool,            # ★ DL 전용
}
```

### 2-2. Look-ahead 방지 (절대 위반 금지)
- Scaler는 **train으로만 fit**, valid/test에는 transform만
- `src/dl/scaler.py`의 `fit_transform_splits()`가 강제로 이걸 보장
- 새 모델도 반드시 이 함수 사용

### 2-3. Best L tuning은 valid QLIKE 기준
- ML도 QLIKE 기준 (Optuna objective)
- DL은 매 epoch valid MSE + QLIKE 둘 다 측정, **best_epoch = valid QLIKE 최저**
- 새 모델도 `early_stop_metric='qlike'` 옵션 지원 권장

### 2-4. Final fit on combined (train+valid)
ML과 정합되는 표준 흐름:
```
1. Tuning: L별로 train fit + valid early stop → best L 선택 (valid QLIKE)
2. Final fit: best L로 (train + valid) combined fit
   - max_epochs = tuning에서 측정한 best_epoch (early stop 비활성)
3. Test predict → 결과 저장
```

### 2-5. Expanding은 static의 best L 그대로 사용
- `dl_tuned_L.csv`에서 `is_best_L=True` 행 가져옴
- cell당 1 L만 학습 (108 → 36, 시간 ~1/3)
- 매 refit은 full retrain (warm start 아님), `REFIT_EVERY=20`

## 3. 데이터 인프라 사용 (1줄로 끝)

```python
from src.dl import load_one
from src.models.dl import LSTMModel  # 또는 새 모델

# 한 cell 로드 (tier subset + scaling 다 처리됨)
splits = load_one('normal', 'US', L=22, tier='core')
# splits = {
#   'feature_cols': List[str],
#   'scaler': CustomScaler,
#   'train_X': (N_train, L, F) np.float32, 'train_y': (N_train,),
#   'valid_X': ..., 'valid_y': ...,
#   'test_X': ..., 'test_y': ...,
#   'train_meta': pd.DataFrame, 'valid_meta': ..., 'test_meta': ...,
# }

# 학습 (LSTMModel 패턴)
model = LSTMModel(feature_cols=splits['feature_cols'], L=22, **HP)
model.fit(splits['train_X'], splits['train_y'],
          splits['valid_X'], splits['valid_y'])

# Test 예측
y_pred = model.predict(splits['test_X'])

# 평가
from src.eval.metrics import evaluate
metrics = evaluate(splits['test_y'], y_pred)
# {'RMSE': ..., 'MAE': ..., 'QLIKE': ..., 'RMSE_CV': ...}

# Phase별 평가
from src.eval.phases import iter_phases
test_dates = pd.to_datetime(splits['test_meta']['prediction_date'])
dummy_df = pd.DataFrame(index=pd.DatetimeIndex(test_dates))
for phase_name, mask, _ in iter_phases(dummy_df, regime='normal'):
    if mask.sum() > 0:
        m = evaluate(splits['test_y'][mask], y_pred[mask])
        ...
```

## 4. 새 DL 모델 wrapper 추가 시 체크리스트

LSTMModel(`src/models/dl/lstm.py`)을 패턴으로 사용. 필수 메서드:

```python
class NewDLModel:
    name = "ModelName"   # 'LSTM', '1DCNN', 'TCN', 'TST', 'GroupNN' 등

    def __init__(
        self,
        feature_cols: List[str],
        L: int,                        # lookback (모델 특화 hp는 별도)
        # ... 모델 고유 hp ...
        # 공통:
        lr=1e-3, weight_decay=1e-5, batch_size=64,
        max_epochs=100, early_stop_patience=10,
        early_stop_metric='qlike',     # 'mse' or 'qlike'
        lr_patience=5, lr_factor=0.5, lr_min=1e-6,
        grad_clip=1.0, seed=42,
        device=None, verbose=False,
    ):
        ...
        # device auto: torch.cuda.is_available() 시 'cuda'

    def fit(self, X_train, y_train, X_valid=None, y_valid=None) -> 'NewDLModel':
        # AdamW + ReduceLROnPlateau + MSE loss + early stop + grad clip
        # 매 epoch valid MSE + QLIKE 둘 다 측정 (LSTMModel._eval_valid 참고)
        # best_state는 early_stop_metric 기준
        # X_valid 없으면 valid_loss = train_loss (early stop 비활성)
        # attributes 채우기: net_, best_val_loss_, best_val_mse_, best_val_qlike_,
        #                    best_epoch_, epochs_used_,
        #                    train_loss_history_, valid_loss_history_,
        #                    valid_mse_history_, valid_qlike_history_, lr_history_
        return self

    def predict(self, X_test) -> np.ndarray:
        # net.eval() + torch.no_grad()
        # np.clip(y_pred, 1e-8, None)  # QLIKE floor (필수)
        return y_pred

    def history_df(self) -> pd.DataFrame:
        # 컬럼: epoch, train_loss, valid_loss, valid_mse, valid_qlike, lr
        ...

    def save_checkpoint(self, path, extra=None) -> None:
        # state_dict + best_val_* + epochs + hp + feature_cols + L + extra
        ...

    @classmethod
    def from_checkpoint(cls, path, device=None) -> 'NewDLModel':
        # 학습 환경 없이 복원
        ...
```

### Forward 패턴 (모델별 차이)

- **LSTM**: `nn.LSTM(input → hidden, batch_first=True)` → `out[:, -1, :]` (마지막 timestep)
- **1DCNN**: input `(B, L, F)` → `permute(0, 2, 1)` → `nn.Conv1d` 여러 layer + `AdaptiveAvgPool1d(1)` → fc
- **TCN**: 1DCNN + dilated causal conv (L에 맞춰 dilation 자동 조정 권장)
- **TST**: positional encoding 추가 → `nn.TransformerEncoder` → mean pool 또는 last token → fc
- **그룹구조 NN**: feature 차원을 group별로 slicing → sub-network ModuleDict → 후단 결합

모든 모델 `(B, L, F)` 받음. 모델 내부에서 자기 형식으로 변환.

## 5. HP 정책

현재 fixed (Optuna tuning 안 함):

```python
HP = dict(
    hidden_size=64, num_layers=1, dropout=0.2,   # architecture (모델별 다름)
    lr=1e-3, weight_decay=1e-5, batch_size=64,
    max_epochs=100,
    early_stop_patience=10, early_stop_min_delta=1e-5,
    early_stop_metric='qlike',                   # ★ ML 정합
    lr_patience=5, lr_factor=0.5, lr_min=1e-6,
    grad_clip=1.0,                               # LSTM exploding 방지 (모든 RNN 필수)
    seed=42,
)

# L grid (categorical hyperparameter, 유일한 tuning 차원)
LS = [22, 60, 252]
```

새 모델 추가 시:
- 위 공통 HP는 그대로
- 모델 특화 HP만 추가 (예: TCN의 dilation_base, TST의 n_heads)

## 6. 학습 단위

```
4 regime × 3 country × 3 tier = 36 cells

Static (cell 10 단계 1~4):
- cell당 3 L tuning + best L 선택 + final fit on combined = 4 fit
- 36 × 4 = 144 fit (CPU 40~70분, Colab GPU 7~17분)

Expanding (cell 14):
- cell당 best L만 (108 → 36 cells)
- REFIT_EVERY=20, walk-forward full retrain
- cell당 ~30 refit × LSTM fit = 30분 (CPU)
- 36 × 30분 = ~6~12시간 CPU / ~1~2시간 Colab GPU
```

## 7. 흔한 함정 (피해야 할 것)

### 7-1. Scaler를 train+valid 합쳐서 fit
❌ `scaler.fit_transform(np.concat([train_X, valid_X], ...))` — look-ahead!
✅ `scaler, X_train, X_valid, X_test = fit_transform_splits(train_X, valid_X, test_X, feature_cols)`

### 7-2. tier subset 누락
원본 X.npy는 28 features. tier별로 column 선택해야 함.
`load_one()`이 이미 처리 — 직접 X.npy 로드하지 말고 wrapper 사용.

### 7-3. shuffle=True
시계열인데 batch shuffle하면 의미 깨짐.
`DataLoader(..., shuffle=False)` 항상.

### 7-4. predict 출력에 floor 안 함
QLIKE이 log(y_pred) 사용 → y_pred ≤ 0이면 폭발.
`np.clip(y_pred, 1e-8, None)` 항상.

### 7-5. Final fit에서 valid 안 합침
ML과 다름. final fit은 train+valid combined로 가야 정합.

### 7-6. Gradient clipping 안 함
LSTM 특히 exploding gradient. `torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)` 필수.

### 7-7. Random seed 안 맞춤
재현성 깨짐. `torch.manual_seed(42)`, `np.random.seed(42)` 매 fit 시작 시.

### 7-8. State_dict만 저장
나중에 복원할 때 hp/feature_cols 알 수 없음.
`save_checkpoint(path, extra={...})`로 self-contained 저장.

### 7-9. tier 컬럼 이름이 'feature_set'
ML 결과와 정합 위해 csv에선 `feature_set` (코드 변수는 `tier` OK).

### 7-10. dataset_DL을 매번 새로 생성
`04_build_dl_dataset.ipynb`는 1회만 실행. 결과(dataset_DL/L*/*.npy)는 .gitignore.
새 모델 추가 시는 그대로 사용.

## 8. 노트북 패턴 (새 모델 노트북 만들 때)

`notebooks/05_lstm.ipynb`를 템플릿으로 (cell 구조 동일):

```
1. md  : 제목 + 산출물
2. code: import + project root + DEVICE
3. md  : 설정표
4. code: HP + LS + 출력 디렉토리 + HP json 저장
5. md  : 모델 설명
6. code: from src.models.dl import NewDLModel
7. md  : helper 설명
8. code: eval_phases helper + sanity test
9. md  : ★ STATIC
10. code: 36 cell × 4 fit loop (tuning → best L → final fit → test)
11. md  : 저장
12. code: dl_tuned_*.csv + dl_results_*.csv + log/{Model}/ 저장
13. md  : ★ EXPANDING (별도)
14. code: 36 cell × walk-forward (best L만)
15. md  : append
16. code: 결과 append
17. md  : 검증
18. code: tier별 best L 표 + ML 비교
19. md  : 학습 곡선
20. code: plot_one() + plot_grid()
```

`build_<model>_notebook.py` 스크립트로 자동 생성 (build_lstm_notebook.py 참고).

## 9. 디버깅 / 검증 체크리스트

새 모델 만들면 smoke test:

```python
from src.dl import load_one
from src.models.dl import NewModel

splits = load_one('normal', 'US', L=22, tier='core')
m = NewModel(feature_cols=splits['feature_cols'], L=22, max_epochs=10, **HP_mini)
m.fit(splits['train_X'], splits['train_y'], splits['valid_X'], splits['valid_y'])
y_pred = m.predict(splits['test_X'])

# 검증:
# 1. shape 확인: y_pred.shape == (1502,)
# 2. NaN/Inf 없음
# 3. RMSE_CV ~ 0.2~0.4 (ML 결과 범위)
# 4. best_val_qlike_ < initial val (학습됨)
# 5. checkpoint round-trip: predictions identical
```

## 10. 작업 시 자주 묻는 패턴

- **"왜 valid가 best epoch 기준?"** → ML도 valid 기반 hp tuning, 정합성 위해
- **"왜 final fit?"** → ML의 cell 9 변경 (train+valid combined)과 정합
- **"왜 expanding에 best L 그대로?"** → tuning에서 정한 best params를 production에 적용하는 표준
- **"왜 grad_clip?"** → RNN exploding gradient 방지 (LSTM 필수)
- **"왜 dataset_DL/ 사전 변환?"** → 모든 환경에서 동일 데이터 보장, 디스크 1.3GB라 .gitignore

---

위 원칙을 따르면 새 모델이 ML 결과와 자동 비교되고, `make_result_visualizations.py`로 통합 차트가 자동 생성됨.