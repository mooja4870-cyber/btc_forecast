# UNIFIED_DEV.md - 통합 개발 기법 가이드

## 통합 기법 개요
이 프로젝트는 오케스트레이션 + 하네스 통합 기법으로 개발됩니다.
에이전트가 작업 성격을 분석하여 최적의 기법을 자동 선택합니다.

두 가지 모드:
- 🚀 오케스트레이션 모드: 기존 코드를 서브태스크로 분할하여 안전하게 수정
- 🛡️ 하네스 모드: 격리 환경에서 먼저 개발/검증 후 실제 앱에 통합


## 프로젝트 정보
- 프로젝트명: BTC_Price_Prediction_MLOps (BTCn)
- 기술 스택: Python, Streamlit, PyTorch, scikit-learn, XGBoost, LightGBM, Plotly
- 언어: Python
- 상태관리: config.py + src/config.py 기반 설정 객체 방식
- 패키지 매니저: pip (.venv)
- 테스트 프레임워크: unittest
- 주요 라이브러리: pandas, numpy, torch, streamlit, plotly, scikit-learn, yfinance


## 📂 프로젝트 구조 맵
__harness__/              ← 하네스 전용 폴더 (🛡️ 모드에서 사용)
├── README.md
└── [기능별 하네스 폴더들]

BTCn/
├── app.py (메인 대시보드)
├── config.py (전역 설정)
├── config/ (YAML 설정)
├── data/ (데이터 및 로그)
├── models/ (학습된 AI 모델 아카이브)
├── scripts/ (운영 스크립트)
├── src/ (핵심 모듈)
│   ├── data_collector.py
│   ├── feature_engineer.py
│   ├── predictor.py
│   ├── train_transformer.py
│   └── ...
├── tests/ (유닛 테스트)
└── __harness__/ (하네스 환경)


## 🔗 모듈 의존성
- `app.py` → `config` + `src.predictor` 로 대시보드 렌더링
- `src.predictor` → `config` + `src.config` + `src.feature_engineer` + 저장된 모델 아티팩트 사용
- `src.train_transformer` → `config` + `src.config` + `src.transformer_model` + `src.feature_engineer` 사용
- `src.data_collector` / `src.run_pipeline` → 데이터 수집 후 feature_engineer, train_transformer 순으로 연결


## 🛠️ 명령어
- 빌드: `bash render_start.sh`
- 전체 테스트: `python3 -m unittest discover -s tests -p 'test_*.py'`
- 하네스 테스트: `python3 -m unittest discover -s __harness__ -p 'test_*.py'`
- 타입체크: N/A
- 린트: N/A
- 개발서버: `streamlit run app.py`


## 🏷️ 코드 스타일
- 함수/변수명은 snake_case 사용
- 파일 상단에 긴 docstring과 섹션 구분 주석(---) 유지
- `sys.path.insert(...)`로 프로젝트 루트 경로 명시적 추가
- 테스트는 unittest 클래스 기반 스타일 유지


## 🎯 기법 자동 선택 기준

### 🚀 오케스트레이션 모드 → 이런 작업일 때
- 기존 코드 수정, 버그 수정, 핫픽스
- 텍스트/스타일/UI 변경
- 기존 기능에 옵션/파라미터 추가
- 설정 변경, 단순 리팩토링
- 판단 기준: "기존 코드를 직접 수정해도 안전한 작업"

### 🛡️ 하네스 모드 → 이런 작업일 때
- 신규 기능 추가
- 기존 로직 대폭 변경
- 새로운 API 연동
- 외부 라이브러리 통합
- 실험적/불확실한 기능
- 판단 기준: "기존 코드를 잘못 건드리면 위험한 작업"


## 🚀 오케스트레이션 모드 상세

### 서브태스크 분할 패턴

레이어 기반 분할 (권장):
Task 1: UI 레이어 (app.py, dashboard.py)
Task 2: 로직 레이어 (src/feature_engineer.py, src/predictor.py)
Task 3: API 레이어 (src/data_collector.py, data_fetcher.py)
Task 4: 상태 레이어 (config.py, config/config.yaml)

의존성 순서 기반 분할 (권장):
Task 1: 수집 로직 (collector)
Task 2: 전처리 로직 (engineer)
Task 3: 학습/추론 로직 (trainer, predictor)
Task 4: UI 연동 (app)

### 작업 규모별 기준
- 소규모 (파일 1-2개): 서브태스크 없이 직접 처리
- 중규모 (파일 3-5개): 2-3개 서브태스크
- 대규모 (파일 5개+): 3개 이상 서브태스크


## 🛡️ 하네스 모드 상세

### 4 Phase 프로세스
Phase 1: 하네스 구성 (__harness__/[기능명]/ 생성, Mock/Stub 준비)
Phase 2: 하네스 안에서 기능 개발 (실제 앱 미접근)
Phase 3: 하네스 검증 (테스트 + 사용자 승인 대기)
Phase 4: 실제 앱 통합 (승인 후, 정제하여 통합)

### 하네스 폴더 구조
__harness__/[기능명]/
├── index.harness.py
├── mock/data.py
├── mock/api_mock.py
├── stub/
├── [기능명]_dev.py
└── [기능명]_test.py


## 📋 통합 체크리스트

### 🚀 오케스트레이션 모드 체크리스트
- [ ] 작업 규모 판단 (소/중/대)
- [ ] 영향 파일 목록 파악
- [ ] 서브태스크 분할 계획 수립
- [ ] 사용자 확인 완료
- [ ] 지정 파일만 접근
- [ ] 이전 태스크 결과 재활용
- [ ] 완료 후 수정 파일 목록 및 결과 요약 보고

### 🛡️ 하네스 모드 체크리스트
Phase 1: __harness__/[기능명]/ 폴더 및 Mock/Stub 구성
Phase 2: 하네스 안에서만 코드 작성, 실제 앱 미수정
Phase 3: 정상/엣지 케이스 테스트 통과 후 사용자 승인
Phase 4: 코드 정제 후 최소 파일 수정으로 실제 앱 통합


## ⛔ 절대 금지
1. 전체 프로젝트 grep/find 검색
2. 계획 없이 즉시 코드 수정
3. 사용자 확인 전 실행
4. 서브태스크당 3개 이상 파일 동시 수정
5. 수정 대상 아닌 파일 읽기
6. 의존성 패키지 임의 설치
7. 기존 작동 코드 구조 임의 변경
8. 요청 범위 밖 코드 수정
9. 하네스 검증 없이 실제 앱 직접 수정 (🛡️ 모드 시)
10. 하네스 코드를 정제 없이 실제 앱에 그대로 복붙 (🛡️ 모드 시)


## 💰 토큰 절약 효과

| 작업 유형 | 일반 방식 | 통합 기법 | 적용 모드 |
|----------|----------|----------|----------|
| 단순 UI 수정 | ~50K | ~8K | 🚀 |
| 버그 수정 | ~100K | ~12K | 🚀 |
| API 추가 | ~80K | ~15K | 🚀 or 🛡️ |
| 신규 기능 추가 | ~200K | ~40K | 🛡️ |
| 기존 로직 대폭 변경 | ~150K | ~35K | 🛡️ |
| 평균 절약률 | 기준 | 70-80% 절약 | |
