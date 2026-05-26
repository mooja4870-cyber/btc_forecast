# 📊 BTC Model Monitoring Report
**Generated:** 2026-05-26 17:19:06
**Run ID:** run_20260527_021906

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 9 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $4444.68 | $7771.95 |
| **RMSE** | $5512.55 | $8271.02 |
| **MAPE** | 6.0% | 9.8% |
| **Count** | 252 | 51 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-05-19 00:00:00 | 90d       |           79909.5 |        76750.9 |    4.11537  |
| 2026-05-19 00:00:00 | 90d       |           80084.3 |        76750.9 |    4.34312  |
| 2026-05-19 00:00:00 | 90d       |           80858.8 |        76750.9 |    5.35224  |
| 2026-05-20 00:00:00 | 90d       |           76766.3 |        77457.8 |   -0.89271  |
| 2026-05-20 00:00:00 | 90d       |           77838.2 |        77457.8 |    0.491192 |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 9

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
| feature                     |   z_score |   current_mean |   ref_mean |
|:----------------------------|----------:|---------------:|-----------:|
| rate_fvx_close              |   2.81543 |       4.10827  |   3.76254  |
| rate_tnx_close              |   2.66006 |       4.45587  |   4.17737  |
| wheat_fut_close             |   2.43427 |     634.192    | 551.126    |
| log_wheat_fut_close         |   2.32107 |       6.45187  |   6.31009  |
| corn_fut_oi_change_7d_proxy |   2.29237 |     370.224    |  30.2088   |
| curve_2y10y_spread_proxy    |   2.08913 |       0.868167 |   0.562644 |
| corn_fut_close              |   2.05168 |     462.567    | 438.918    |
| expected_policy_rate_6m     |   2.03515 |       3.79593  |   3.67385  |
| log_corn_fut_close          |   2.02287 |       6.13669  |   6.08397  |