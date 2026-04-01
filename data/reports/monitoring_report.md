# 📊 BTC Model Monitoring Report
**Generated:** 2026-04-01 14:58:24
**Run ID:** run_20260401_235824

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 19 features

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
- Drifted features (30d vs prev 180d): 19

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
| oil_fut_close                    |  11.3422  |     92.6383    |     60.8086     |
| log_oil_fut_close                |   9.11088 |      4.52443   |      4.10668    |
| oil_fut_close_ret30d             |   7.76253 |      0.433388  |      0.00350585 |
| geo_ovx_close                    |   7.16397 |     96.504     |     38.3247     |
| oil_fut_roll_return_20d          |   6.9074  |      0.315151  |      0.00318987 |
| oil_fut_front_next_spread_proxy  |   4.91985 |      0.138174  |      0.00157284 |
| wheat_fut_close                  |   3.91842 |    598.467     |    524.939      |
| log_wheat_fut_close              |   3.74401 |      6.39411   |      6.26266    |
| geo_vix_close                    |   3.56774 |     26.2253    |     17.3146     |
| geo_ovx_close_ret30d             |   3.37272 |      0.881847  |      0.0965101  |
| oil_fut_close_ret7d              |   3.04101 |      0.103572  |      0.00168024 |
| oil_fut_volume                   |   2.71295 | 550136         | 269972          |
| gold_fut_front_next_spread_proxy |   2.2163  |     -0.0368661 |      0.0226515  |
| corn_fut_close                   |   2.20413 |    453.142     |    428.21       |
| gold_fut_close_ret30d            |   2.19262 |     -0.0300578 |      0.0679178  |