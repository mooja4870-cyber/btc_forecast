# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-11 14:40:52
**Run ID:** run_20260311_234052

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 13 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $1643.47 | $1651.70 |
| **RMSE** | $2096.46 | $2105.43 |
| **MAPE** | 2.5% | 2.5% |
| **Count** | 99 | 98 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-03-05 00:00:00 | 15d       |           66206.7 |        70841.1 |    -6.54196 |
| 2026-03-05 00:00:00 | 15d       |           66926.2 |        70841.1 |    -5.52635 |
| 2026-03-05 00:00:00 | 15d       |           66182.4 |        70841.1 |    -6.5763  |
| 2026-03-06 00:00:00 | 15d       |           65970.1 |        68136.5 |    -3.17955 |
| 2026-03-06 00:00:00 | 15d       |           66683.9 |        68136.5 |    -2.13181 |

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
| geo_ovx_close                   |   4.57152 |     64.17      |     36.2367      |
| oil_fut_close                   |   4.09142 |     71.1873    |     60.6671      |
| wheat_fut_close                 |   4.02201 |    571.15      |    519.043       |
| log_wheat_fut_close             |   3.82341 |      6.34689   |      6.25168     |
| log_oil_fut_close               |   3.58927 |      4.25641   |      4.10451     |
| oil_fut_close_ret30d            |   3.0003  |      0.147172  |     -0.00929298  |
| oil_fut_front_next_spread_proxy |   2.88574 |      0.0781317 |     -0.00100343  |
| oil_fut_roll_return_20d         |   2.82537 |      0.126256  |     -0.00362995  |
| oil_fut_close_ret7d             |   2.57317 |      0.0834    |      0.000627273 |
| wheat_fut_close_ret30d          |   2.27098 |      0.0891458 |     -0.000235974 |
| oil_fut_volume                  |   2.25622 | 493844         | 253451           |
| gold_fut_close                  |   2.18963 |   5103.48      |   4121.54        |
| log_gold_fut_close              |   2.01728 |      8.53748   |      8.31811     |