# Version History

## 프로젝트 정보
- **폴더명**: `d_BTCn`
- **앱 설명**: BTC(비트코인) 가격 예측 MLOps 앱. Streamlit 대시보드에서 TimeSformer 딥러닝과 전통 ML 앙상블로 1일~365일 다중 호라이즌 예측 제공. GitHub Actions 매일 자동 학습·커밋, Render 클라우드 서비스.

---

## v1.0.7 (2026-06-04)
모델 품질 3대 개선 (학습 안정화 · 365d 재설계 · 중기 중심 전략)
- **① 학습 안정화** (`src/train_transformer.py`): Adam weight_decay(1e-4) L2 정규화, dropout 0.1→0.2, ReduceLROnPlateau 스케줄러(검증손실 정체 시 LR 절반), MIN_EPOCHS=5 가드(1-epoch 노이즈성 조기종료 방지)
- **② 365d 재설계 — 정직한 평가** (`src/train_transformer.py`, `app.py`): 방향 skill(=방향정확도−다수클래스 baseline) 지표 추가로 한쪽 쏠린 구간의 착시 제거(365d 방향98%→skill≈0). degenerate/low_confidence 플래그(표본<200·양수비율<10%/>90%·R²<0&skill≤0)를 val_metrics.json 저장, 대시보드에 skill·저신뢰 경고 표시
- **③ 중기 중심 전략 재편** (`src/predictor.py`): predictor가 신뢰도 플래그를 읽어, 저신뢰 장기 horizon(예: 365d degenerate)을 신뢰 horizon 추세(원점 통과 최소제곱 기울기)로 외삽 재구성. 하드코딩 방향 없이 모델 자체 중기 신호 추종, 라벨로 투명 표시

---

## v1.0.6 (2026-06-04)
예측 신뢰도 정밀 개선 (데이터 누수 제거 · 편향 수정 · 학습 안정화)
- **#1 시계열 hold-out 도입** (`src/train_transformer.py`): 검증 시작일(운영 phase, 2025-01-01) 이전 데이터로만 별도 **검증 모델** 학습 + 타깃 누수 방지 embargo, 스케일러도 train 구간만 fit. 정직한 out-of-sample 지표를 `val_metrics.json`에 저장. 운영 모델은 전체 데이터 학습 유지. → 대시보드 검증 R²/방향정확도가 실제 예측력 반영
- **app.py**: `load_transformer_val_metrics`가 운영 평가 phase에서 저장된 정직한 지표 우선 사용 (기존 in-sample 재계산 대체)
- **#2 소스별 ffill 한도** (`src/data_collector.py`): 무제한 `ffill()` → `_forward_fill_by_source` (거시 95일 / 시장가 4일 / 온체인·심리 7일). 장기 공백 시 낡은 값 동결 방지
- **#3 180일 하드코딩 보정 삭제** (`src/predictor.py`): 모델 출력 강제 클리핑 제거 → 약세 신호 보존
- **#4 `timezone` 임포트 복구** (`app.py`): `NameError`로 무력화됐던 모델 노후화 경고(`stale_days`) 복구
- **#5 early stopping + 결측 처리** (`src/train_transformer.py`): 내부 시간분할 val 기반 best epoch 탐색 → 운영 모델 재사용으로 과적합 억제. `fillna(0)` → 전일값 ffill 후 잔여만 0

---

## v1.0.5 (2026-06-04)
- **GitHub Actions** (`daily_update.yml`): `train_transformer.py` + `verify_reliability.py` 단계 추가 — 풀 파이프라인(데이터→학습→Transformer→Reality Check) 실행
- **`scripts/run_daily.sh`**: 로컬 학습 완료 후 `git add data/ models/ && git push` 자동 수행 → Render 배포 앱 최신화
- **`scripts/run_daily_guard.sh`** (신규): LaunchAgent용 일일 1회 가드. 오전 9시 이후 오늘 학습 미완료 시만 실행
- **LaunchAgent** (`com.btcn.daily.plist`): 자정 cron → 매시간 체크 방식으로 교체 (Mac 수면 후 깨어나도 당일 학습 보장)

---

## v1.0.4 (2026-05-29)
- **Button Font Optimization**: Adjusted the font-size of streamlit buttons (`st.button`) in Custom CSS from `0.77rem`/`0.8rem` to `0.68rem`/`0.7rem` (88% size reduction) to prevent text wrapping on narrow columns (e.g. "추세보\n기").

## v1.0.3 (2026-05-29)
- **NASDAQ Card Integration**: Added NASDAQ (^IXIC) index card between S&P 500 and USD/KRW Exchange Rate.
- **Aesthetic Alignment**: Expanded the grid header columns to 8-col format to seamlessly support NASDAQ.
- **Trend Charts**: Integrated NASDAQ (^IXIC) into individual 30-year trend view and the combined 6-asset comparison tool fallback mappings.

## v1.0.2 (2026-05-28)
- **Label Clean**: Modified yfinance fallback source label format from `Yahoo Finance/yfinance` to `Yahoo Finance` in `src/data_fetcher.py`.

## v1.0.1 (2026-05-28)
- **KOSDAQ Card Integration**: Added KOSDAQ (^KQ11) index card next to KOSPI.
- **Data Fetching Integration**: Created `fetch_naver_kosdaq` in `src/data_fetcher.py` and linked it via Naver Mobile JSON API.
- **Trend Charts**: Integrated KOSDAQ into individual 30-year trend view and the combined 6-asset comparison tool.
