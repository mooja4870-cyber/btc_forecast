# 📊 BTC Model Monitoring Report
**Generated:** 2026-09-26 17:27:28
**Run ID:** run_20260927_022727

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
| corn_fut_close          |   3.77671 |    519.575     | 447.296      |
| expected_policy_rate_3m |   3.74366 |      3.85377   |   3.64022    |
| rate_irx_close          |   3.74366 |      3.85377   |   3.64022    |
| log_corn_fut_close      |   3.50499 |      6.25283   |   6.10231    |
| wheat_fut_close         |   3.09752 |    726.783     | 619.965      |
| expected_policy_rate_6m |   3.05152 |      4.18399   |   3.83511    |
| log_wheat_fut_close     |   2.92642 |      6.58824   |   6.42816    |
| rate_irx_close_ret30d   |   2.80033 |      0.0392792 |   0.00553906 |
| rate_fvx_close          |   2.52752 |      4.67933   |   4.12743    |
| rate_tnx_close          |   2.47471 |      4.8789    |   4.45081    |
| corn_fut_close_ret30d   |   2.02502 |      0.134865  |   0.0125698  |