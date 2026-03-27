# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-27 14:45:01
**Run ID:** run_20260327_234500

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 13 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $2504.75 | $4015.68 |
| **RMSE** | $3072.96 | $4253.14 |
| **MAPE** | 3.6% | 5.6% |
| **Count** | 150 | 61 |

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
- Drifted features (30d vs prev 180d): 13

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
| oil_fut_close                   |   9.61487 |     86.829     |     60.7527     |
| log_oil_fut_close               |   7.83552 |      4.45405   |      4.10583    |
| geo_ovx_close                   |   7.45217 |     90.798     |     37.5333     |
| oil_fut_roll_return_20d         |   6.64336 |      0.299824  |      0.00238302 |
| oil_fut_close_ret30d            |   6.33308 |      0.353698  |      0.00060292 |
| oil_fut_front_next_spread_proxy |   4.76458 |      0.132198  |      0.00117662 |
| wheat_fut_close                 |   4.4329  |    593.975     |    523.126      |
| log_wheat_fut_close             |   4.20363 |      6.38654   |      6.25937    |
| geo_ovx_close_ret30d            |   3.14836 |      0.82601   |      0.086379   |
| geo_vix_close                   |   2.87145 |     24.3797    |     17.2037     |
| oil_fut_volume                  |   2.78502 | 557872         | 259992          |
| oil_fut_close_ret7d             |   2.71038 |      0.0928485 |      0.00183395 |
| wheat_fut_close_ret30d          |   2.11992 |      0.0946848 |      0.0111974  |