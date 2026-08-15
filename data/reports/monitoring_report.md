# 📊 BTC Model Monitoring Report
**Generated:** 2026-08-15 14:11:26
**Run ID:** run_20260815_231125

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Recent MAPE (22.8%) exceeds threshold (15.0%)
- ⚠️ Expansion feature drift detected: 4 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $5634.07 | $14449.56 |
| **RMSE** | $7211.07 | $14566.30 |
| **MAPE** | 8.0% | 22.8% |
| **Count** | 286 | 34 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-08-15 00:00:00 | 180d      |           46201.1 |        62969.2 |   -26.629   |
| 2026-08-15 00:00:00 | 180d      |           47524.8 |        62969.2 |   -24.5269  |
| 2026-08-15 00:00:00 | 180d      |           45913.2 |        62969.2 |   -27.0863  |
| 2026-08-15 00:00:00 | 180d      |           68415.2 |        62969.2 |     8.64872 |
| 2026-08-15 00:00:00 | 180d      |           45829.9 |        62969.2 |   -27.2185  |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 4

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
| rate_irx_close              |   3.1514  |        3.7238  |    3.60902 |
| expected_policy_rate_3m     |   3.1514  |        3.7238  |    3.60902 |
| expected_policy_rate_6m     |   2.20435 |        3.98195 |    3.75729 |
| gold_fut_oi_change_7d_proxy |   2.097   |       99.673   |   10.0746  |