# 📊 BTC Model Monitoring Report
**Generated:** 2026-02-24 16:08:06
**Run ID:** run_20260224_160806

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 6 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $1475.92 | $1483.90 |
| **RMSE** | $1886.55 | $1896.00 |
| **MAPE** | 2.2% | 2.3% |
| **Count** | 81 | 80 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-02-24 00:00:00 | 7d        |           68143.5 |        64246.7 |     6.0654  |
| 2026-02-24 00:00:00 | 7d        |           68410.9 |        64246.7 |     6.4816  |
| 2026-02-24 00:00:00 | 7d        |           68318.1 |        64246.7 |     6.33718 |
| 2026-02-24 00:00:00 | 7d        |           67687.9 |        64246.7 |     5.3563  |
| 2026-02-24 00:00:00 | 5d        |           66709.6 |        64246.7 |     3.83361 |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 6

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
| feature               |   z_score |   current_mean |     ref_mean |
|:----------------------|----------:|---------------:|-------------:|
| geo_ovx_close         |   4.49619 |     50.3273    |   34.7859    |
| geo_ovx_close_ret30d  |   3.24139 |      0.426552  |    0.0107304 |
| oil_fut_close_ret30d  |   2.76943 |      0.0904848 |   -0.0203893 |
| gold_fut_close        |   2.43425 |   4980.87      | 3981.32      |
| log_gold_fut_close    |   2.19643 |      8.51288   |    8.28401   |
| rate_irx_close_ret30d |   2.0109  |      0.0122625 |   -0.0289215 |