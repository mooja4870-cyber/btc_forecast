# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-10 14:39:53
**Run ID:** run_20260310_233952

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
| feature                         |   z_score |   current_mean |       ref_mean |
|:--------------------------------|----------:|---------------:|---------------:|
| geo_ovx_close                   |   4.22356 |     60.758     |   35.9886      |
| wheat_fut_close                 |   3.78051 |    567.858     |  518.786       |
| oil_fut_close                   |   3.69277 |     70.1033    |   60.6555      |
| log_wheat_fut_close             |   3.59567 |      6.34089   |    6.25118     |
| log_oil_fut_close               |   3.24178 |      4.24087   |    4.10433     |
| oil_fut_close_ret30d            |   2.84067 |      0.135693  |   -0.0109894   |
| oil_fut_front_next_spread_proxy |   2.72652 |      0.0737925 |   -0.00170445  |
| gold_fut_oi_change_7d_proxy     |   2.69032 |     56.1975    |    5.51471     |
| oil_fut_roll_return_20d         |   2.663   |      0.116366  |   -0.00484569  |
| oil_fut_close_ret7d             |   2.44558 |      0.0788723 |    0.000253785 |
| gold_fut_close                  |   2.21157 |   5089.03      | 4103.24        |
| wheat_fut_close_ret30d          |   2.1536  |      0.0844709 |   -0.00113933  |
| log_gold_fut_close              |   2.03347 |      8.53465   |    8.31367     |