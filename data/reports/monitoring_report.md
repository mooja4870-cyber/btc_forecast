# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-14 14:20:14
**Run ID:** run_20260314_232013

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 13 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $1635.93 | $1643.76 |
| **RMSE** | $2075.09 | $2083.59 |
| **MAPE** | 2.5% | 2.5% |
| **Count** | 103 | 102 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-03-06 00:00:00 | 15d       |           66683.9 |        68136.5 |    -2.13181 |
| 2026-03-12 00:00:00 | 30d       |           69044.4 |        70493.5 |    -2.05565 |
| 2026-03-12 00:00:00 | 30d       |           69044.4 |        70493.5 |    -2.05565 |
| 2026-03-12 00:00:00 | 30d       |           69044.4 |        70493.5 |    -2.05565 |
| 2026-03-12 00:00:00 | 30d       |           69044.4 |        70493.5 |    -2.05565 |

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
| feature                         |   z_score |   current_mean |         ref_mean |
|:--------------------------------|----------:|---------------:|-----------------:|
| geo_ovx_close                   |   5.19109 |     68.914     |     36.3979      |
| oil_fut_close                   |   4.91184 |     73.3663    |     60.6773      |
| wheat_fut_close                 |   4.30759 |    575.158     |    519.351       |
| log_oil_fut_close               |   4.21806 |      4.28397   |      4.10467     |
| log_wheat_fut_close             |   4.0862  |      6.35402   |      6.25227     |
| oil_fut_front_next_spread_proxy |   3.47616 |      0.0943911 |     -0.000426389 |
| oil_fut_close_ret30d            |   3.46485 |      0.175464  |     -0.00784575  |
| oil_fut_roll_return_20d         |   3.31952 |      0.152176  |     -0.0025271   |
| oil_fut_close_ret7d             |   2.8476  |      0.0923094 |      0.000736664 |
| wheat_fut_close_ret30d          |   2.39581 |      0.094263  |      0.000877646 |
| oil_fut_volume                  |   2.339   | 503884         | 254381           |
| gold_fut_close                  |   2.14297 |   5106.54      |   4140.46        |
| geo_vix_close                   |   2.05898 |     21.8437    |     16.8468      |