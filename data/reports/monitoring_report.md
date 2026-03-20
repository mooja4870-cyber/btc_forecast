# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-20 14:36:47
**Run ID:** run_20260320_233646

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 15 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $2489.56 | $2591.02 |
| **RMSE** | $3056.05 | $3148.10 |
| **MAPE** | 3.6% | 3.7% |
| **Count** | 148 | 138 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-03-20 00:00:00 | 30d       |           65877.1 |        70159.9 |    -6.10442 |
| 2026-03-20 00:00:00 | 30d       |           66410.3 |        70159.9 |    -5.34446 |
| 2026-03-20 00:00:00 | 30d       |           65697.1 |        70159.9 |    -6.36091 |
| 2026-03-20 00:00:00 | 30d       |           66811.2 |        70159.9 |    -4.77305 |
| 2026-03-20 00:00:00 | 30d       |           65625.2 |        70159.9 |    -6.46335 |

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
| oil_fut_close                   |   7.61596 |      80.288    |     60.6697      |
| geo_ovx_close                   |   7.17785 |      82.1533   |     36.6518      |
| log_oil_fut_close               |   6.30333 |       4.37175  |      4.10455     |
| oil_fut_roll_return_20d         |   5.41663 |       0.243324 |      0.000537721 |
| oil_fut_close_ret30d            |   5.32047 |       0.27668  |     -0.00429688  |
| oil_fut_front_next_spread_proxy |   4.93387 |       0.132403 |      0.000109096 |
| wheat_fut_close                 |   4.88819 |     587.342    |    520.788       |
| log_wheat_fut_close             |   4.6034  |       6.37509  |      6.255       |
| oil_fut_close_ret7d             |   3.42634 |       0.110928 |      0.000551032 |
| geo_ovx_close_ret30d            |   2.69505 |       0.708482 |      0.0766201   |
| oil_fut_volume                  |   2.58477 |  530730        | 257720           |
| wheat_fut_close_ret30d          |   2.57754 |       0.103718 |      0.00543501  |
| geo_vix_close                   |   2.36872 |      22.8993   |     17.0098      |
| geo_ovx_close_ret7d             |   2.19682 |       0.261714 |      0.0143761   |
| gold_fut_close                  |   2.0003  |    5094.92     |   4195.44        |