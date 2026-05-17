# 📊 BTC Model Monitoring Report
**Generated:** 2026-05-17 14:49:37
**Run ID:** run_20260517_234936

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 2 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $4387.29 | $7735.24 |
| **RMSE** | $5460.34 | $8278.36 |
| **MAPE** | 5.9% | 9.8% |
| **Count** | 235 | 51 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-05-17 00:00:00 | 90d       |           70637.7 |        78034.9 |    -9.47933 |
| 2026-05-17 00:00:00 | 90d       |           70242.9 |        78034.9 |    -9.98527 |
| 2026-05-17 00:00:00 | 90d       |           71358.5 |        78034.9 |    -8.55561 |
| 2026-05-17 00:00:00 | 90d       |           70501.2 |        78034.9 |    -9.65425 |
| 2026-05-17 00:00:00 | 90d       |           70150.7 |        78034.9 |   -10.1034  |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 2

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
| feature             |   z_score |   current_mean |   ref_mean |
|:--------------------|----------:|---------------:|-----------:|
| wheat_fut_close     |   2.17243 |      618.917   |  546.608   |
| log_wheat_fut_close |   2.09938 |        6.42746 |    6.30193 |