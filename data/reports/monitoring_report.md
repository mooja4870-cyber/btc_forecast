# 📊 BTC Model Monitoring Report
**Generated:** 2026-05-16 14:47:23
**Run ID:** run_20260516_234722

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 2 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $4322.37 | $7990.61 |
| **RMSE** | $5411.03 | $8543.42 |
| **MAPE** | 5.8% | 10.2% |
| **Count** | 230 | 51 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-05-16 00:00:00 | 90d       |           70301.6 |        78183.2 |   -10.0809  |
| 2026-05-16 00:00:00 | 90d       |           69129.2 |        78183.2 |   -11.5805  |
| 2026-05-16 00:00:00 | 90d       |           70339.9 |        78183.2 |   -10.032   |
| 2026-05-16 00:00:00 | 90d       |           70523.5 |        78183.2 |    -9.79707 |
| 2026-05-16 00:00:00 | 90d       |           70132.5 |        78183.2 |   -10.2973  |

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
| wheat_fut_close     |   2.1521  |      617.55    |  546.082   |
| log_wheat_fut_close |   2.08225 |        6.42525 |    6.30098 |