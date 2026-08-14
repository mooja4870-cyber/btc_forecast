# 📊 BTC Model Monitoring Report
**Generated:** 2026-08-14 14:45:36
**Run ID:** run_20260814_234535

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Recent MAPE (22.7%) exceeds threshold (15.0%)
- ⚠️ Expansion features quality issue count: 5
- ⚠️ Expansion feature drift detected: 2 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $5470.75 | $14386.96 |
| **RMSE** | $6975.56 | $14402.10 |
| **MAPE** | 7.7% | 22.7% |
| **Count** | 281 | 29 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-08-14 00:00:00 | 180d      |           48989.2 |        62699.4 |    -21.8666 |
| 2026-08-14 00:00:00 | 180d      |           48774.7 |        62699.4 |    -22.2086 |
| 2026-08-14 00:00:00 | 180d      |           47921.4 |        62699.4 |    -23.5696 |
| 2026-08-14 00:00:00 | 180d      |           47684.1 |        62699.4 |    -23.9481 |
| 2026-08-14 00:00:00 | 180d      |           47510.6 |        62699.4 |    -24.2249 |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 2

### Quality Snapshot (Top 15 by missing/staleness)
| feature                          |   missing_pct_recent_30d |   stale_days |
|:---------------------------------|-------------------------:|-------------:|
| rate_fvx_close_ret30d            |                   100    |   1000000000 |
| rate_fvx_close_ret7d             |                    53.33 |            0 |
| rate_fvx_close_ret1d             |                    33.33 |            0 |
| expected_policy_rate_6m          |                    30    |            0 |
| rate_fvx_close                   |                    30    |            0 |
| commodity_shock_score            |                     0    |            0 |
| corn_fut_close                   |                     0    |            0 |
| corn_fut_close_ret1d             |                     0    |            0 |
| corn_fut_close_ret30d            |                     0    |            0 |
| corn_fut_close_ret7d             |                     0    |            0 |
| corn_fut_days_to_expiry          |                     0    |            0 |
| corn_fut_expiry_week             |                     0    |            0 |
| corn_fut_front_next_spread_proxy |                     0    |            0 |
| corn_fut_oi_change_7d_proxy      |                     0    |            0 |
| corn_fut_roll_return_20d         |                     0    |            0 |

### Drift Snapshot (Top 15 by z-score)
| feature                 |   z_score |   current_mean |   ref_mean |
|:------------------------|----------:|---------------:|-----------:|
| rate_irx_close          |   3.21127 |        3.72363 |    3.60827 |
| expected_policy_rate_3m |   3.21127 |        3.72363 |    3.60827 |