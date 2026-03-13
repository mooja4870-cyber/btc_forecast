# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-13 14:35:48
**Run ID:** run_20260313_233548

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 13 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $1635.93 | $1643.76 |
| **RMSE** | $2075.09 | $2083.59 |
| **MAPE** | 2.5% | 2.5% |
| **Count** | 103 | 102 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-03-06 00:00:00 | 15d       |           66683.9 |        68136.5 |    -2.13181 |
| 2026-03-12 00:00:00 | 30d       |           69044.4 |        70493.5 |    -2.05565 |
| 2026-03-12 00:00:00 | 30d       |           69044.4 |        70493.5 |    -2.05565 |
| 2026-03-12 00:00:00 | 30d       |           69044.4 |        70493.5 |    -2.05565 |
| 2026-03-12 00:00:00 | 30d       |           69044.4 |        70493.5 |    -2.05565 |

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
| geo_ovx_close                   |   4.86479 |     66.5073    |     36.3309      |
| oil_fut_close                   |   4.55085 |     72.3687    |     60.6671      |
| wheat_fut_close                 |   4.21369 |    573.625     |    519.181       |
| log_wheat_fut_close             |   3.99917 |      6.35126   |      6.25194     |
| log_oil_fut_close               |   3.94085 |      4.27129   |      4.10451     |
| oil_fut_close_ret30d            |   3.26444 |      0.162901  |     -0.00862436  |
| oil_fut_front_next_spread_proxy |   3.2333  |      0.0877771 |     -0.00078039  |
| oil_fut_roll_return_20d         |   3.12385 |      0.140992  |     -0.00322154  |
| oil_fut_close_ret7d             |   2.79741 |      0.0907027 |      0.000686234 |
| oil_fut_volume                  |   2.39538 | 508839         | 253588           |
| wheat_fut_close_ret30d          |   2.3757  |      0.0931227 |      0.00023843  |
| gold_fut_close                  |   2.17272 |   5107.12      |   4130.81        |
| log_gold_fut_close              |   2.00478 |      8.53821   |      8.32036     |