# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-29 14:25:05
**Run ID:** run_20260329_232504

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 14 features

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
| feature                         |   z_score |   current_mean |        ref_mean |
|:--------------------------------|----------:|---------------:|----------------:|
| oil_fut_close                   |  10.382   |     89.1413    |     60.7672     |
| log_oil_fut_close               |   8.40926 |      4.48246   |      4.10605    |
| geo_ovx_close                   |   7.40366 |     93.172     |     37.8176     |
| oil_fut_close_ret30d            |   6.90001 |      0.384959  |      0.00218945 |
| oil_fut_roll_return_20d         |   6.72674 |      0.304505  |      0.00269249 |
| oil_fut_front_next_spread_proxy |   4.86284 |      0.1351    |      0.00124885 |
| wheat_fut_close                 |   4.3719  |    596.608     |    523.69       |
| log_wheat_fut_close             |   4.14609 |      6.39102   |      6.2604     |
| geo_ovx_close_ret30d            |   3.24407 |      0.850687  |      0.0903303  |
| geo_vix_close                   |   3.14218 |     25.08      |     17.2464     |
| oil_fut_volume                  |   2.81575 | 559714         | 264467          |
| oil_fut_close_ret7d             |   2.76087 |      0.0944878 |      0.00169044 |
| wheat_fut_close_ret30d          |   2.02304 |      0.0934688 |      0.0122889  |
| corn_fut_close                  |   2.01221 |    451.317     |    427.558      |