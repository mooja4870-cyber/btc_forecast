# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-16 14:55:16
**Run ID:** run_20260316_235513

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 15 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $2049.15 | $2059.33 |
| **RMSE** | $2586.14 | $2595.84 |
| **MAPE** | 3.0% | 3.0% |
| **Count** | 120 | 119 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-03-16 00:00:00 | 30d       |           69065.2 |        73304.4 |    -5.78305 |
| 2026-03-16 00:00:00 | 30d       |           69733.6 |        73304.4 |    -4.87123 |
| 2026-03-16 00:00:00 | 30d       |           68931.4 |        73304.4 |    -5.96553 |
| 2026-03-16 00:00:00 | 30d       |           68935.4 |        73304.4 |    -5.96001 |
| 2026-03-16 00:00:00 | 30d       |           68424.4 |        73304.4 |    -6.65717 |

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
| geo_ovx_close                   |   5.94897 |     73.971     |     36.464       |
| oil_fut_close                   |   5.88978 |     75.897     |     60.678       |
| gold_fut_oi_change_7d_proxy     |   5.39401 |    107.135     |      5.50844     |
| log_oil_fut_close               |   4.95862 |      4.31551   |      4.10468     |
| wheat_fut_close                 |   4.51025 |    579.8       |    519.842       |
| log_wheat_fut_close             |   4.26285 |      6.36197   |      6.2532      |
| oil_fut_close_ret30d            |   4.20423 |      0.21427   |     -0.00684983  |
| oil_fut_front_next_spread_proxy |   4.16558 |      0.112325  |     -9.26159e-05 |
| oil_fut_roll_return_20d         |   4.01468 |      0.184855  |     -0.00167368  |
| oil_fut_close_ret7d             |   3.09294 |      0.100172  |      0.000827318 |
| oil_fut_volume                  |   2.44258 | 516623         | 255661           |
| wheat_fut_close_ret30d          |   2.42898 |      0.0976131 |      0.0024131   |
| geo_vix_close                   |   2.24802 |     22.374     |     16.8928      |
| gold_fut_close                  |   2.11213 |   5111.31      |   4158.64        |
| geo_ovx_close_ret7d             |   2.03701 |      0.240235  |      0.0196986   |