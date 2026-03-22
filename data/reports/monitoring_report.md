# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-22 14:19:01
**Run ID:** run_20260322_231901

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 14 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $2504.75 | $2992.33 |
| **RMSE** | $3072.96 | $3456.58 |
| **MAPE** | 3.6% | 4.3% |
| **Count** | 150 | 115 |

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
| feature                         |   z_score |   current_mean |         ref_mean |
|:--------------------------------|----------:|---------------:|-----------------:|
| oil_fut_close                   |   8.3147  |      82.4307   |     60.6936      |
| geo_ovx_close                   |   7.53024 |      85.17     |     36.8303      |
| log_oil_fut_close               |   6.82672 |       4.39826  |      4.10492     |
| oil_fut_roll_return_20d         |   6.12595 |       0.273584 |      0.00122777  |
| oil_fut_close_ret30d            |   5.67513 |       0.302642 |     -0.00286396  |
| oil_fut_front_next_spread_proxy |   5.06579 |       0.137367 |      0.000496297 |
| wheat_fut_close                 |   4.94306 |     590.25     |    521.326       |
| log_wheat_fut_close             |   4.65383 |       6.38015  |      6.25602     |
| oil_fut_close_ret7d             |   3.30967 |       0.108216 |      0.000764395 |
| geo_ovx_close_ret30d            |   2.94385 |       0.767994 |      0.078078    |
| oil_fut_volume                  |   2.64091 |  538123        | 256511           |
| wheat_fut_close_ret30d          |   2.54122 |       0.103906 |      0.00702677  |
| geo_vix_close                   |   2.48468 |      23.2637   |     17.0603      |
| geo_ovx_close_ret7d             |   2.03931 |       0.244889 |      0.0163477   |