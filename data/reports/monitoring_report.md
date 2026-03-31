# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-31 14:55:22
**Run ID:** run_20260331_235521

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 17 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $2504.75 | $4110.76 |
| **RMSE** | $3072.96 | $4318.85 |
| **MAPE** | 3.6% | 5.7% |
| **Count** | 150 | 59 |

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
- Drifted features (30d vs prev 180d): 17

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
| feature                          |   z_score |   current_mean |        ref_mean |
|:---------------------------------|----------:|---------------:|----------------:|
| oil_fut_close                    |  10.9588  |     91.3823    |     60.8007     |
| log_oil_fut_close                |   8.83348 |      4.50956   |      4.10656    |
| oil_fut_close_ret30d             |   7.44446 |      0.415573  |      0.00320807 |
| geo_ovx_close                    |   7.26653 |     95.4653    |     38.1466     |
| oil_fut_roll_return_20d          |   6.78486 |      0.30984   |      0.00322097 |
| oil_fut_front_next_spread_proxy  |   4.85861 |      0.136423  |      0.00156458 |
| wheat_fut_close                  |   4.0351  |    597.6       |    524.504      |
| log_wheat_fut_close              |   3.84892 |      6.39268   |      6.26187    |
| geo_vix_close                    |   3.43854 |     25.867     |     17.2896     |
| geo_ovx_close_ret30d             |   3.32465 |      0.871394  |      0.0946249  |
| oil_fut_close_ret7d              |   2.90367 |      0.0994088 |      0.00183306 |
| oil_fut_volume                   |   2.7284  | 549198         | 269323          |
| corn_fut_close                   |   2.16277 |    452.75      |    428.011      |
| gold_fut_front_next_spread_proxy |   2.14769 |     -0.0350498 |      0.0227434  |
| log_corn_fut_close               |   2.08314 |      6.11502   |      6.05879    |