# 📊 BTC Model Monitoring Report
**Generated:** 2026-02-24 00:13:49
**Run ID:** run_20260224_001348

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 6 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $1025.64 | $1028.29 |
| **RMSE** | $1236.27 | $1240.98 |
| **MAPE** | 1.5% | 1.5% |
| **Count** | 72 | 71 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-02-23 00:00:00 | 5d        |           66951.9 |        65950.7 |     1.51797 |
| 2026-02-23 00:00:00 | 5d        |           67543.3 |        65950.7 |     2.41478 |
| 2026-02-23 00:00:00 | 5d        |           68027.5 |        65950.7 |     3.14897 |
| 2026-02-23 00:00:00 | 7d        |           67878.3 |        65950.7 |     2.9228  |
| 2026-02-23 00:00:00 | 7d        |           68398   |        65950.7 |     3.7107  |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 6

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
| feature               |   z_score |   current_mean |      ref_mean |
|:----------------------|----------:|---------------:|--------------:|
| geo_ovx_close         |   4.63828 |     49.664     |   34.6383     |
| geo_ovx_close_ret30d  |   3.5755  |      0.440638  |    0.00365739 |
| oil_fut_close_ret30d  |   2.73915 |      0.0874192 |   -0.0208369  |
| gold_fut_close        |   2.50261 |   4970.45      | 3962.94       |
| log_gold_fut_close    |   2.24662 |      8.51083   |    8.27952    |
| rate_irx_close_ret30d |   2.02681 |      0.0127149 |   -0.0288955  |