# PROJECT_STATUS.md

## 📅 마지막 업데이트: 2026-03-24 (현재 진행 중)

## 🚀 프로젝트 개요: BTCn (Bitcoin Forecast Engine)
이 프로젝트는 **비트코인(BTC)의 미래 가격을 예측**하는 인공지능 머신러닝 시스템입니다. 
다양한 경제 데이터(금리, 지질학적 리스크 등)를 수집하고, 최신 **Transformer(TimeSformer)** 모델을 사용하여 여러 시점(1일, 7일, 30일 등)의 수익률을 예측합니다.

---

## 🏗 프로젝트 구조 (Project Structure)

### 1. 폴더 구조
```text
BTCn/
├── config/              # 시스템 설정 파일 (.yaml)
├── data/                # 수집된 원본 및 가공된 데이터
├── logs/                # 실행 로그
├── models/              # 학습된 AI 모델 저장소 (transformer/latest)
├── notebooks/           # 분석용 주피터 노트북
├── scripts/             # 실행 스크립트 (run_daily.sh 등)
├── src/                 # 핵심 소스 코드
│   ├── config.py        # 설정 로더
│   ├── predictor.py     # 예측 엔진
│   ├── dashboard.py     # (✨신규) 시각화 대시보드
│   └── ...              # 기타 데이터 처리 및 학습 모듈
└── tests/               # 테스트 코드
```

### 2. 주요 모듈 및 역할
| 모듈명 | 역할 |
| :--- | :--- |
| `data_fetcher.py` | 야후 파이낸스, 인베스팅닷컴 등에서 최신 데이터 수집 |
| `feature_engineer.py` | AI가 이해하기 쉽게 데이터를 가공 (수익률, 이동평균 등) |
| `train_transformer.py` | Transformer 모델을 사용하여 기계학습 수행 |
| `predictor.py` | 학습된 모델을 불러와 미래 가격을 정교하게 예측 |
| `monitoring.py` | 예측이 실제와 얼마나 맞았는지 체크하고 보고서 생성 |
| `dashboard.py` | 사용자가 웹 화면에서 예측 결과를 한눈에 볼 수 있게 함 |

### 3. 모듈 협력 관계 (Relationships)
1. `run_pipeline.py`가 전체 과정을 조율합니다.
2. `data_fetcher` → `feature_engineer` 순서로 데이터를 준비합니다.
3. `train_transformer`가 데이터를 먹고 똑똑한 모델(`models/`)을 만듭니다.
4. `predictor`가 이 모델을 사용해 미래를 점칩니다.
5. **(New)** `dashboard`가 `predictor`의 결과와 `reports`를 가져와 화려하게 보여줍니다.

---

## ✅ 현재 작업 상태
- [x] 데이터 수집 및 전처리 파이프라인 구축 완료
- [x] Transformer 기반 정밀 예측 모델 구현 완료
- [x] 일일 자동화 스크립트(`run_daily.sh`) 연동 완료
- [x] **(완료) Streamlit 기반 시각화 대시보드 구축**
    - [x] 화려한 '무지개 반짝이' 효과 적용 (글로벌 스타일 전파)
    - [x] 예측 결과 테이블 및 그래프 시각화
    - [x] 대시보드 바탕 배경을 이중 물결(Dual-layered wave) 애니메이션으로 고도화

---

## 🛠 향후 계획
1. 대시보드에 실시간 비트코인 시세 연동
2. 모델 예측 오차(Error)를 줄이기 위한 추가 피처 도입
3. 모바일 알림 연동 기능 검토
