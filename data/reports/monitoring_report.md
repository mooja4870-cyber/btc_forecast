# 📊 BTC Model Monitoring Report
**Generated:** 2026-05-15 15:54:51
**Run ID:** run_20260516_005450

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 2 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $4196.22 | $7963.83 |
| **RMSE** | $5303.85 | $8524.32 |
| **MAPE** | 5.7% | 10.2% |
| **Count** | 222 | 51 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-05-15 00:00:00 | 90d       |           70170.7 |        79203.9 |    -11.405  |
| 2026-05-15 00:00:00 | 90d       |           70786.4 |        79203.9 |    -10.6276 |
| 2026-05-15 00:00:00 | 90d       |           69547.3 |        79203.9 |    -12.1921 |
| 2026-05-15 00:00:00 | 90d       |           70513.9 |        79203.9 |    -10.9717 |
| 2026-05-15 00:00:00 | 90d       |           69322.2 |        79203.9 |    -12.4763 |

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
| wheat_fut_close     |   2.13638 |      616.433   |  545.582   |
| log_wheat_fut_close |   2.06877 |        6.42341 |    6.30006 |