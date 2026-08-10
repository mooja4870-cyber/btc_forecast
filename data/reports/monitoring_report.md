# 📊 BTC Model Monitoring Report
**Generated:** 2026-08-10 14:50:15
**Run ID:** run_20260810_235014

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Recent MAPE (23.6%) exceeds threshold (15.0%)
- ⚠️ Expansion feature drift detected: 4 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $4614.67 | $15323.78 |
| **RMSE** | $5795.04 | $15323.78 |
| **MAPE** | 6.2% | 23.6% |
| **Count** | 256 | 4 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-05-20 00:00:00 | 90d       |           76766.3 |        77457.8 |    -0.89271 |
| 2026-08-09 00:00:00 | 180d      |           49521.1 |        64844.9 |   -23.6314  |
| 2026-08-09 00:00:00 | 180d      |           49521.1 |        64844.9 |   -23.6314  |
| 2026-08-09 00:00:00 | 180d      |           49521.1 |        64844.9 |   -23.6314  |
| 2026-08-09 00:00:00 | 180d      |           49521.1 |        64844.9 |   -23.6314  |

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
| rate_irx_close              |   3.41283 |        3.7219  |    3.60467 |
| expected_policy_rate_3m     |   3.41283 |        3.7219  |    3.60467 |
| expected_policy_rate_6m     |   2.32507 |        3.97771 |    3.74833 |
| gold_fut_oi_change_7d_proxy |   2.31757 |      108.952   |    9.93756 |