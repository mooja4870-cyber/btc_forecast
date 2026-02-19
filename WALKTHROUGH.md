# BTC MLOps Pipeline Walkthrough

## Overview
The BTC forecasting project has been refactored into a configuration-driven MLOps pipeline. All parameters are now centralized in `config/config.yaml`.

## Directory Structure
```
BTC_04_MLOPs/
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── run_20260212_232209/   # Latest run
│   ├── run_20260212_231356/   # Verification run
│   └── latest -> ...          # Symlink to best run
├── src/
│   ├── config.py
│   ├── run_pipeline.py
│   ├── data_collector.py
│   ├── feature_engineer.py
│   ├── model_trainer.py
│   ├── backtester.py
│   └── ...
└── requirements.txt
```

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure
Edit `config/config.yaml` to set date ranges, features, or model hyperparameters.

### 3. Run Pipeline
Run the full end-to-end pipeline:
```bash
python3 src/run_pipeline.py
```

**Common Flags:**
- `--skip-data`: Skip data collection (use existing data).
- `--skip-train`: Run only data collection and feature engineering.
- `--phases 2`: Run only specific training phases (e.g., Phase 2).

Example validation run:
```bash
python3 src/run_pipeline.py --phases 2 --skip-data
```

## Artifacts
After a run, check the `models/<run_id>/` directory:
- `manifest.json`: Full metadata of the run (config, metrics, best models).
- `phaseX/horizon_Yd/`:
  - `best_model_*.joblib`: Trained model.
  - `metrics.json`: Train/Val metrics.
  - `val_predictions.csv`: Validation predictions vs actuals.

## Model Promotion
The pipeline automatically checks if the new run performs better than the `latest` run (based on R2 score). If it improves, `models/latest` is updated to point to the new run.

## Monitoring & Reporting
The system tracks predictions and compares them with actual prices as they become available.

### 1. Logs
- `data/logs/predictions_log.csv`: Log of every prediction made by the production model (latest configured phase, currently Phase 6).
- `data/logs/eval_log.csv`: Predictions joined with actuals (where available).

### 2. Reports
Every time the pipeline runs, it checks for new actuals and generates a report:
- `data/reports/monitoring_report.md`: Markdown report with metrics, drift analysis, and recent error rates.

To run **only** the monitoring step (e.g., as a daily cron job):
```bash
python3 src/run_pipeline.py --monitor-only
```

## 자동화 (Cron Job 설정)
매일 자정에 파이프라인을 자동으로 실행하려면 다음 단계를 따르세요:

1. **스크립트 실행 권한 부여:**
터미널에서 다음 명령어를 실행합니다:
```bash
chmod +x scripts/run_daily.sh
```

2. **Crontab 등록:**
Crontab 에디터를 엽니다:
```bash
crontab -e
```
파일의 맨 아래에 다음 줄을 추가합니다 (매일 00:00 실행):
```cron
0 0 * * * /Users/mooja/AI_Study/Project/BTC_04_MLOPs/scripts/run_daily.sh
```
*(참고: `vi` 에디터가 열리면 `i`를 눌러 입력 모드로 전환하고, 붙여넣기 후 `Esc` -> `:wq` -> `Enter`를 눌러 저장하세요.)*

3. **로그 확인:**
자동 실행 기록은 아래 파일에 저장됩니다:
- `data/logs/cron_job.log`

## 대시보드 실행 (Dashboard)
예측 결과를 시각적으로 확인하려면 Streamlit 대시보드를 실행하세요:

```bash
streamlit run app.py
```

- **상태 확인**: 모델 성능 요약, 가격 추이, 미래 예측 경로를 탭별로 확인할 수 있습니다.
- **실시간 가격**: 대시보드 상단에서 현재 BTC 가격과 주요 자산 지표를 실시간으로 보여줍니다.
