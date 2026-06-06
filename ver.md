# Version History

## 프로젝트 정보
- **폴더명**: `d_BTCn`
- **앱 설명**: BTC(비트코인) 가격 예측 MLOps 앱. Streamlit 대시보드에서 TimeSformer 딥러닝과 전통 ML 앙상블로 1일~365일 다중 호라이즌 예측 제공. GitHub Actions 매일 자동 학습·커밋, Render 클라우드 서비스.

---

## v1.1.2 (2026-06-06)
자동학습 재개 — LaunchAgent 권한 오류 해결
- **원인**: LaunchAgent `Operation not permitted` 오류 → 2일 이상 학습 미실행
- **진단**: stderr.log에서 권한 거부 반복 확인, plist 재로드 필요 판단
- **조치**: `launchctl unload/load ~/Library/LaunchAgents/com.btcn.daily.plist` → 권한 캐시 초기화
- **검증**: 수동 테스트 후 스탐프 파일(2026-06-06) 생성 확인, 자동 커밋(14:44 KST) 푸시 확인
- **다음 학습**: 매시간 체크, 오전 9시 이후 자동 실행 재개

---

## v1.1.1 (2026-06-04)
- **단어 줄바꿈 금지** (`app.py` 전역 CSS): 모든 텍스트에 `word-break: keep-all`(한글 단어 보존) + `overflow-wrap/word-wrap: normal`(영문 단어 중간 분리 금지) + `hyphens: none` 적용 → 한 단어가 절대 두 라인에 걸쳐 나타나지 않음

---

## v1.1.0 (2026-06-04)
제품 방향 전환: 점 예측 → 시나리오 밴드("맞히기" 대신 "범위 보여주기")
- **`src/scenario.py` 신규**: 모델-free 시나리오 밴드. 각 horizon의 역사적 h일 로그수익률 실증 분위수(p5/p25/p50/p75/p95)를 **현재 변동성 국면**(최근 vol/역사 vol)으로 스케일 → 팻테일·왜도 반영한 "가능한 범위" 콘. 중앙선은 예측이 아닌 역사적 중앙 경로
- **근거**: 다중시드 검증서 방향 예측력이 통계적으로 없음을 확인(v1.0.9). 반면 변동성은 지속적·추정 가능 → 정직하게 범위 제시
- **대시보드** (`app.py`): 미래예측[종합] 탭 상단에 부채꼴(fan) 콘 차트 + 변동성 국면 지표(최근/역사 연율 변동성, 배수) + 핵심 기간(30·90·180·365일) 범위 표. 현재 변동성 낮음(35% vs 역사 67%)이라 밴드가 적절히 좁아짐
- 모델 재학습 불필요(런타임 계산)

---

## v1.0.9 (2026-06-04)
피처 품질 점검(②) · 불확실성 정직 전달(③) — 3대 후속개선 완료
- **② 피처 품질 — 증거 기반 "변경 안 함" 결정** (`scripts/feature_sweep.py` 신규): 진단 결과 상관>0.95 중복 45개·비정상 가격레벨 다수 발견. 그러나 다중 시드(3) 검증에서 정상성/프루닝 피처셋이 전체 대비 **평균적으로 개선 없음**(15d: 전체 -1.3% vs 정상성 -4.0%). 단일시드 "+5.8%"는 노이즈로 확인 → 피처셋 미변경(리스크 회피). 진단 스크립트만 보존
- **③ 불확실성 정직 전달** (`src/train_transformer.py`, `app.py`): 방향 skill을 **다중 시드(3) 평균±편차**로 산출, `skill_significant`(평균이 편차보다 큰가) 플래그 추가. 어떤 horizon도 통계적 유의성 없음을 대시보드에 명시. 미래예측 안내문을 과대광고("신뢰도 향상")에서 정직한 문구("유의미한 예측력 미확인, 참고용")로 교체
- **재구성 앵커 분리** (`src/predictor.py`): 장기 horizon 외삽 기준을 `low_confidence`→`degenerate`(표본부족/편중)로 변경. skill 비유의성이 거의 모든 horizon에 걸려도 365d 외삽이 정상 작동하도록 분리
- 핵심 결론: 구조·피처·평가를 정직하게 만들었으나 **수익률 방향 예측은 본질적으로 신호가 약함**(시드 변동 > 실제 신호). 정직성·안정성·투명성 확보가 성과

---

## v1.0.8 (2026-06-04)
모델 구조 재설계 (과적합 해소) — 3대 후속개선 ①
- **아키텍처 스윕** (`scripts/arch_sweep.py` 신규): 정직한 hold-out으로 5개 구조를 15d·90d에서 비교. 승자 `small_seq90`(d_model 64→32, 레이어 2→1, seq 60→90, dropout 0.2→0.3)
- **구조 적용** (`src/transformer_model.py`, `src/train_transformer.py`): 파라미터 ~100K→18K로 축소. 진단이었던 ep=1 즉시 과적합 해소 → 대부분 horizon이 ep=7~20까지 실제 학습. R² 전반 개선(5d -0.51→-0.04, 90d -0.14→+0.007), 30d skill +5.1%
- **metadata 기반 로딩** (`src/predictor.py`): 학습 구조를 metadata.json에 기록, 로더가 재구성 → 구조 변경 시 추론 불일치 방지
- 한계: skill은 여전히 작고 run간 변동 큼(신호 자체가 약함). 180d/365d는 ep=1 유지→중기 추세 외삽

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
