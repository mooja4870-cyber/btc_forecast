# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-30 14:58:04
**Run ID:** run_20260330_235803

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 16 features

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
- Drifted features (30d vs prev 180d): 16

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
| oil_fut_close                    |  10.6606  |     90.2287    |     60.7839     |
| log_oil_fut_close                |   8.61557 |      4.49568   |      4.10631    |
| geo_ovx_close                    |   7.38273 |     94.4       |     37.9686     |
| oil_fut_close_ret30d             |   7.14104 |      0.399174  |      0.00279826 |
| oil_fut_roll_return_20d          |   6.69289 |      0.304398  |      0.0029915  |
| oil_fut_front_next_spread_proxy  |   4.84673 |      0.13543   |      0.00141396 |
| gold_fut_oi_change_7d_proxy      |   4.31325 |     87.7441    |      5.7255     |
| wheat_fut_close                  |   4.19398 |    597.142     |    524.097      |
| log_wheat_fut_close              |   3.98989 |      6.39191   |      6.26114    |
| geo_vix_close                    |   3.30348 |     25.494     |     17.2646     |
| geo_ovx_close_ret30d             |   3.29973 |      0.864376  |      0.0920983  |
| oil_fut_volume                   |   2.7848  | 555823         | 266895          |
| oil_fut_close_ret7d              |   2.7638  |      0.094619  |      0.00171261 |
| corn_fut_close                   |   2.09065 |    452.092     |    427.785      |
| gold_fut_front_next_spread_proxy |   2.0363  |     -0.0320656 |      0.0227122  |