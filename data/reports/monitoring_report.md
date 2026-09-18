# 📊 BTC Model Monitoring Report
**Generated:** 2026-09-18 17:21:45
**Run ID:** run_20260919_022144

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 9 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $7403.20 | $0.00 |
| **RMSE** | $12689.72 | $0.00 |
| **MAPE** | 10.8% | 0.0% |
| **Count** | 303 | 0 |

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
- Drifted features (30d vs prev 180d): 9

### Quality Snapshot (Top 15 by missing/staleness)
| feature                          |   missing_pct_recent_30d |   stale_days |
|:---------------------------------|-------------------------:|-------------:|
| commodity_shock_score            |                        0 |            0 |
| corn_fut_close                   |                        0 |            0 |
| corn_fut_close_ret1d             |                        0 |            0 |
| corn_fut_close_ret30d            |                        0 |            0 |
| corn_fut_close_ret7d             |                        0 |            0 |
| corn_fut_days_to_expiry          |                        0 |            0 |
| corn_fut_expiry_week             |                        0 |            0 |
| corn_fut_front_next_spread_proxy |                        0 |            0 |
| corn_fut_oi_change_7d_proxy      |                        0 |            0 |
| corn_fut_roll_return_20d         |                        0 |            0 |
| corn_fut_volume                  |                        0 |            0 |
| curve_2y10y_spread_proxy         |                        0 |            0 |
| days_to_fomc                     |                        0 |            0 |
| expected_policy_rate_3m          |                        0 |            0 |
| expected_policy_rate_6m          |                        0 |            0 |

### Drift Snapshot (Top 15 by z-score)
| feature                 |   z_score |   current_mean |     ref_mean |
|:------------------------|----------:|---------------:|-------------:|
| corn_fut_close          |   3.73933 |     508.408    | 444.692      |
| log_corn_fut_close      |   3.46453 |       6.23084  |   6.09664    |
| wheat_fut_close         |   3.26054 |     720.183    | 614.86       |
| log_wheat_fut_close     |   3.06803 |       6.57876  |   6.42005    |
| rate_irx_close          |   2.5111  |       3.7759   |   3.63514    |
| expected_policy_rate_3m |   2.5111  |       3.7759   |   3.63514    |
| expected_policy_rate_6m |   2.23359 |       4.08489  |   3.81837    |
| rate_tnx_close          |   2.04321 |       4.79437  |   4.42218    |
| corn_fut_close_ret30d   |   2.01653 |       0.129025 |   0.00994239 |