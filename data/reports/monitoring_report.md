# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-24 14:51:38
**Run ID:** run_20260324_235137

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 14 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $2504.75 | $3595.34 |
| **RMSE** | $3072.96 | $3884.79 |
| **MAPE** | 3.6% | 5.1% |
| **Count** | 150 | 89 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-03-20 00:00:00 | 30d       |           65625.2 |        70522.6 |    -6.94435 |
| 2026-03-20 00:00:00 | 30d       |           65697.1 |        70522.6 |    -6.84244 |
| 2026-03-20 00:00:00 | 30d       |           66811.2 |        70522.6 |    -5.26275 |
| 2026-03-21 00:00:00 | 30d       |           66381.1 |        68711.5 |    -3.39156 |
| 2026-03-21 00:00:00 | 30d       |           66685.3 |        68711.5 |    -2.94887 |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 14

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
| feature                         |   z_score |   current_mean |        ref_mean |
|:--------------------------------|----------:|---------------:|----------------:|
| oil_fut_close                   |   8.87352 |     84.2837    |     60.7176     |
| geo_ovx_close                   |   7.51942 |     87.5317    |     37.1201     |
| log_oil_fut_close               |   7.25418 |      4.42151   |      4.10529    |
| oil_fut_roll_return_20d         |   6.54073 |      0.291989  |      0.00167993 |
| oil_fut_close_ret30d            |   5.90363 |      0.323434  |     -0.00151864 |
| oil_fut_front_next_spread_proxy |   4.99117 |      0.13693   |      0.00081357 |
| wheat_fut_close                 |   4.65842 |    591.417     |    522.079      |
| log_wheat_fut_close             |   4.40513 |      6.38215   |      6.25742    |
| gold_fut_oi_change_7d_proxy     |   4.03834 |     81.598     |      5.49357    |
| oil_fut_close_ret7d             |   3.09648 |      0.10314   |      0.00118579 |
| geo_ovx_close_ret30d            |   3.02002 |      0.791158  |      0.081968   |
| geo_vix_close                   |   2.65371 |     23.7383    |     17.1208     |
| oil_fut_volume                  |   2.36583 | 511126         | 258146          |
| wheat_fut_close_ret30d          |   2.26038 |      0.0978735 |      0.00904214 |