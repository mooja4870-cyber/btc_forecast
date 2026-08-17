# 📊 BTC Model Monitoring Report
**Generated:** 2026-08-17 14:17:24
**Run ID:** run_20260817_231722

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Recent MAPE (32.8%) exceeds threshold (15.0%)
- ⚠️ Expansion feature drift detected: 4 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $7113.85 | $20840.98 |
| **RMSE** | $12041.95 | $27101.36 |
| **MAPE** | 10.3% | 32.8% |
| **Count** | 301 | 49 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-08-17 00:00:00 | 180d      |            113849 |        63638.4 |     78.8997 |
| 2026-08-17 00:00:00 | 180d      |            122895 |        63638.4 |     93.1147 |
| 2026-08-17 00:00:00 | 180d      |            124946 |        63638.4 |     96.3379 |
| 2026-08-17 00:00:00 | 180d      |            120565 |        63638.4 |     89.4537 |
| 2026-08-17 00:00:00 | 180d      |            121097 |        63638.4 |     90.2892 |

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
| rate_irx_close              |   3.06607 |        3.7241  |    3.61052 |
| expected_policy_rate_3m     |   3.06607 |        3.7241  |    3.61052 |
| gold_fut_oi_change_7d_proxy |   2.55277 |      119.921   |   10.4147  |
| expected_policy_rate_6m     |   2.18113 |        3.98462 |    3.76016 |