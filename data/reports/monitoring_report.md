# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-17 14:49:34
**Run ID:** run_20260317_234933

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 15 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $2359.67 | $2371.66 |
| **RMSE** | $3020.76 | $3031.72 |
| **MAPE** | 3.4% | 3.4% |
| **Count** | 128 | 127 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-03-17 00:00:00 | 30d       |           68924.5 |        74356.7 |    -7.30563 |
| 2026-03-17 00:00:00 | 30d       |           69357.1 |        74356.7 |    -6.72384 |
| 2026-03-17 00:00:00 | 30d       |           68797.2 |        74356.7 |    -7.47672 |
| 2026-03-17 00:00:00 | 30d       |           69174.3 |        74356.7 |    -6.96968 |
| 2026-03-17 00:00:00 | 30d       |           68726.6 |        74356.7 |    -7.57177 |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 15

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
| feature                         |   z_score |   current_mean |         ref_mean |
|:--------------------------------|----------:|---------------:|-----------------:|
| geo_ovx_close                   |   6.33664 |     76.5323    |     36.4916      |
| oil_fut_close                   |   6.3157  |     76.9767    |     60.6751      |
| log_oil_fut_close               |   5.29102 |      4.32936   |      4.10463     |
| wheat_fut_close                 |   4.61662 |    581.983     |    520.097       |
| oil_fut_close_ret30d            |   4.46885 |      0.229354  |     -0.00617915  |
| oil_fut_front_next_spread_proxy |   4.38284 |      0.118018  |     -5.16988e-06 |
| log_wheat_fut_close             |   4.35666 |      6.36571   |      6.25369     |
| oil_fut_roll_return_20d         |   4.33653 |      0.198921  |     -0.00104335  |
| oil_fut_close_ret7d             |   3.1093  |      0.100697  |      0.000816525 |
| oil_fut_volume                  |   2.59038 | 531545         | 256553           |
| wheat_fut_close_ret30d          |   2.45215 |      0.0990985 |      0.00327177  |
| geo_vix_close                   |   2.31436 |     22.5937    |     16.9234      |
| geo_ovx_close_ret7d             |   2.13883 |      0.252091  |      0.018034    |
| geo_ovx_close_ret30d            |   2.11955 |      0.573756  |      0.0772109   |
| gold_fut_close                  |   2.08682 |   5109.87      |   4168.03        |