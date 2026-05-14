# 📊 BTC Model Monitoring Report
**Generated:** 2026-05-14 16:03:20
**Run ID:** run_20260515_010319

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 2 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $4026.23 | $7658.32 |
| **RMSE** | $5133.86 | $8252.92 |
| **MAPE** | 5.5% | 9.9% |
| **Count** | 214 | 51 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-05-14 00:00:00 | 90d       |           71287.3 |        81339.9 |    -12.3587 |
| 2026-05-14 00:00:00 | 90d       |           70636.4 |        81339.9 |    -13.159  |
| 2026-05-14 00:00:00 | 90d       |           70133.4 |        81339.9 |    -13.7774 |
| 2026-05-14 00:00:00 | 90d       |           72180.6 |        81339.9 |    -11.2605 |
| 2026-05-14 00:00:00 | 90d       |           69565.3 |        81339.9 |    -14.4758 |

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
| wheat_fut_close     |   2.10521 |       614.842  |  545.085   |
| log_wheat_fut_close |   2.04131 |         6.4208 |    6.29915 |