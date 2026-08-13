# 📊 BTC Model Monitoring Report
**Generated:** 2026-08-13 14:50:24
**Run ID:** run_20260813_235023

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Recent MAPE (22.7%) exceeds threshold (15.0%)
- ⚠️ Expansion features quality issue count: 10
- ⚠️ Expansion feature drift detected: 3 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $5218.56 | $14505.10 |
| **RMSE** | $6653.77 | $14521.87 |
| **MAPE** | 7.3% | 22.7% |
| **Count** | 273 | 21 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-08-13 00:00:00 | 180d      |           49653.1 |        63854.2 |    -22.2398 |
| 2026-08-13 00:00:00 | 180d      |           49283   |        63854.2 |    -22.8194 |
| 2026-08-13 00:00:00 | 180d      |           48418.7 |        63854.2 |    -24.173  |
| 2026-08-13 00:00:00 | 180d      |           49327.8 |        63854.2 |    -22.7493 |
| 2026-08-13 00:00:00 | 180d      |           49434.5 |        63854.2 |    -22.5822 |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 3

### Quality Snapshot (Top 15 by missing/staleness)
| feature                  |   missing_pct_recent_30d |   stale_days |
|:-------------------------|-------------------------:|-------------:|
| rate_fvx_close_ret30d    |                   100    |   1000000000 |
| rate_tnx_close_ret30d    |                   100    |   1000000000 |
| rate_fvx_close_ret7d     |                    56.67 |            0 |
| rate_tnx_close_ret7d     |                    56.67 |            0 |
| rate_fvx_close_ret1d     |                    36.67 |            0 |
| rate_tnx_close_ret1d     |                    36.67 |            0 |
| curve_2y10y_spread_proxy |                    33.33 |            0 |
| expected_policy_rate_6m  |                    33.33 |            0 |
| rate_fvx_close           |                    33.33 |            0 |
| rate_tnx_close           |                    33.33 |            0 |
| commodity_shock_score    |                     0    |            0 |
| corn_fut_close           |                     0    |            0 |
| corn_fut_close_ret1d     |                     0    |            0 |
| corn_fut_close_ret30d    |                     0    |            0 |
| corn_fut_close_ret7d     |                     0    |            0 |

### Drift Snapshot (Top 15 by z-score)
| feature                     |   z_score |   current_mean |   ref_mean |
|:----------------------------|----------:|---------------:|-----------:|
| rate_irx_close              |   3.34601 |        3.72433 |    3.60733 |
| expected_policy_rate_3m     |   3.34601 |        3.72433 |    3.60733 |
| gold_fut_oi_change_7d_proxy |   2.01749 |       96.1376  |    9.94865 |