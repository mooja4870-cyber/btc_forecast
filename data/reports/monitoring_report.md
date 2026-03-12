# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-12 14:41:43
**Run ID:** run_20260312_234142

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 13 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $1601.27 | $1608.76 |
| **RMSE** | $2058.27 | $2066.67 |
| **MAPE** | 2.4% | 2.4% |
| **Count** | 103 | 102 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-03-06 00:00:00 | 15d       |           66683.9 |        68136.5 |   -2.13181  |
| 2026-03-12 00:00:00 | 30d       |           69044.4 |        69601   |   -0.799825 |
| 2026-03-12 00:00:00 | 30d       |           69044.4 |        69601   |   -0.799825 |
| 2026-03-12 00:00:00 | 30d       |           69044.4 |        69601   |   -0.799825 |
| 2026-03-12 00:00:00 | 30d       |           69044.4 |        69601   |   -0.799825 |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 13

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
| feature                         |   z_score |   current_mean |         ref_mean |
|:--------------------------------|----------:|---------------:|-----------------:|
| geo_ovx_close                   |   4.57152 |     64.17      |     36.2367      |
| oil_fut_close                   |   4.1307  |     71.2883    |     60.6671      |
| wheat_fut_close                 |   4.02008 |    571.125     |    519.043       |
| log_wheat_fut_close             |   3.82172 |      6.34685   |      6.25168     |
| log_oil_fut_close               |   3.61734 |      4.2576    |      4.10451     |
| oil_fut_close_ret30d            |   3.03039 |      0.148742  |     -0.00929298  |
| oil_fut_front_next_spread_proxy |   2.93243 |      0.0794121 |     -0.00100343  |
| oil_fut_roll_return_20d         |   2.85845 |      0.127776  |     -0.00362995  |
| oil_fut_close_ret7d             |   2.61523 |      0.0847528 |      0.000627273 |
| oil_fut_volume                  |   2.33384 | 502114         | 253451           |
| wheat_fut_close_ret30d          |   2.26978 |      0.0890985 |     -0.000235974 |
| gold_fut_close                  |   2.18895 |   5103.18      |   4121.54        |
| log_gold_fut_close              |   2.01674 |      8.53743   |      8.31811     |