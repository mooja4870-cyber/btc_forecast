# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-09 14:42:26
**Run ID:** run_20260309_234225

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 12 features

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
- Drifted features (30d vs prev 180d): 12

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
| feature                         |   z_score |   current_mean |      ref_mean |
|:--------------------------------|----------:|---------------:|--------------:|
| geo_ovx_close                   |   4.05301 |     59.0773    |   35.8824     |
| wheat_fut_close                 |   3.60062 |    565.358     |  518.704      |
| log_wheat_fut_close             |   3.43124 |      6.33648   |    6.25102    |
| oil_fut_close                   |   3.19672 |     68.8453    |   60.6578     |
| log_oil_fut_close               |   2.86906 |      4.22533   |    4.10436    |
| oil_fut_close_ret30d            |   2.56564 |      0.119239  |   -0.0119107  |
| oil_fut_front_next_spread_proxy |   2.28749 |      0.0614604 |   -0.00197557 |
| oil_fut_roll_return_20d         |   2.28294 |      0.0978356 |   -0.00541863 |
| gold_fut_close                  |   2.22869 |   5085.4       | 4094.36       |
| wheat_fut_close_ret30d          |   2.058   |      0.0805062 |   -0.0015299  |
| log_gold_fut_close              |   2.04666 |      8.53392   |    8.3115     |
| gold_fut_oi_change_7d_proxy     |   2.03124 |     43.7935    |    5.53799    |