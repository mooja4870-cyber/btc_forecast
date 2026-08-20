# 📊 BTC Model Monitoring Report
**Generated:** 2026-08-20 14:24:13
**Run ID:** run_20260820_232412

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Recent MAPE (34.4%) exceeds threshold (15.0%)
- ⚠️ Expansion features quality issue count: 6
- ⚠️ Expansion feature drift detected: 1 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $7403.20 | $22021.74 |
| **RMSE** | $12689.72 | $28399.79 |
| **MAPE** | 10.8% | 34.4% |
| **Count** | 303 | 51 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-08-17 00:00:00 | 180d      |            121097 |        64506.3 |     87.7291 |
| 2026-08-17 00:00:00 | 180d      |            121661 |        64506.3 |     88.6034 |
| 2026-08-17 00:00:00 | 180d      |            120565 |        64506.3 |     86.9048 |
| 2026-08-18 00:00:00 | 180d      |            119377 |        64680.7 |     84.5637 |
| 2026-08-18 00:00:00 | 180d      |            118828 |        64680.7 |     83.7142 |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 1

### Quality Snapshot (Top 15 by missing/staleness)
| feature                  |   missing_pct_recent_30d |   stale_days |
|:-------------------------|-------------------------:|-------------:|
| rate_fvx_close_ret30d    |                   100    |   1000000000 |
| rate_irx_close_ret30d    |                   100    |   1000000000 |
| rate_tnx_close_ret30d    |                   100    |   1000000000 |
| rate_fvx_close_ret7d     |                    33.33 |            0 |
| rate_irx_close_ret7d     |                    33.33 |            0 |
| rate_tnx_close_ret7d     |                    33.33 |            0 |
| rate_fvx_close_ret1d     |                    13.33 |            0 |
| rate_irx_close_ret1d     |                    13.33 |            0 |
| rate_tnx_close_ret1d     |                    13.33 |            0 |
| curve_2y10y_spread_proxy |                    10    |            0 |
| expected_policy_rate_3m  |                    10    |            0 |
| expected_policy_rate_6m  |                    10    |            0 |
| rate_fvx_close           |                    10    |            0 |
| rate_irx_close           |                    10    |            0 |
| rate_tnx_close           |                    10    |            0 |

### Drift Snapshot (Top 15 by z-score)
| feature                     |   z_score |   current_mean |   ref_mean |
|:----------------------------|----------:|---------------:|-----------:|
| gold_fut_oi_change_7d_proxy |    2.0402 |        97.9242 |    10.3964 |