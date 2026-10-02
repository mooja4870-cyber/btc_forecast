# 📊 BTC Model Monitoring Report
**Generated:** 2026-10-02 18:47:38
**Run ID:** run_20261003_034738

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 11 features

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
- Drifted features (30d vs prev 180d): 11

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
| rate_irx_close          |   4.81751 |      3.92283   |   3.64453    |
| expected_policy_rate_3m |   4.81751 |      3.92283   |   3.64453    |
| rate_irx_close_ret30d   |   4.14371 |      0.0575048 |   0.00551152 |
| expected_policy_rate_6m |   3.90076 |      4.27229   |   3.84941    |
| rate_tnx_close          |   3.20464 |      4.98177   |   4.47405    |
| rate_fvx_close          |   3.19306 |      4.79647   |   4.15672    |
| corn_fut_close          |   3.13778 |    520.308     | 449.876      |
| log_corn_fut_close      |   2.96054 |      6.25421   |   6.10775    |
| rate_irx_close_ret7d    |   2.21645 |      0.0212048 |   0.00139964 |
| wheat_fut_close         |   2.06841 |    712.825     | 625.996      |
| log_wheat_fut_close     |   2.03715 |      6.56899   |   6.43721    |