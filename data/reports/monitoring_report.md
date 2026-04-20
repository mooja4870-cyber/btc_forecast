# 📊 BTC Model Monitoring Report
**Generated:** 2026-04-20 15:09:08
**Run ID:** run_20260421_000907

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 11 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $3590.12 | $6782.40 |
| **RMSE** | $4543.14 | $7319.33 |
| **MAPE** | 5.0% | 9.0% |
| **Count** | 201 | 51 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-04-19 00:00:00 | 60d       |           71313   |        73856.4 |    -3.44369 |
| 2026-04-19 00:00:00 | 60d       |           72725.8 |        73856.4 |    -1.53081 |
| 2026-04-19 00:00:00 | 60d       |           72459.5 |        73856.4 |    -1.89131 |
| 2026-04-20 00:00:00 | 60d       |           70386.1 |        74834.4 |    -5.94429 |
| 2026-04-20 00:00:00 | 60d       |           70555   |        74834.4 |    -5.71853 |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 11

### Quality Snapshot (Top 15 by missing/staleness)
| feature                          |   missing_pct_recent_30d |   stale_days |
|:---------------------------------|-------------------------:|-------------:|
| corn_fut_oi_change_7d_proxy      |                    13.33 |            0 |
| gold_fut_oi_change_7d_proxy      |                    13.33 |            0 |
| oil_fut_oi_change_7d_proxy       |                    13.33 |            0 |
| wheat_fut_oi_change_7d_proxy     |                    13.33 |            0 |
| commodity_shock_score            |                     0    |            0 |
| corn_fut_close                   |                     0    |            0 |
| corn_fut_close_ret1d             |                     0    |            0 |
| corn_fut_close_ret30d            |                     0    |            0 |
| corn_fut_close_ret7d             |                     0    |            0 |
| corn_fut_days_to_expiry          |                     0    |            0 |
| corn_fut_expiry_week             |                     0    |            0 |
| corn_fut_front_next_spread_proxy |                     0    |            0 |
| corn_fut_roll_return_20d         |                     0    |            0 |
| corn_fut_volume                  |                     0    |            0 |
| curve_2y10y_spread_proxy         |                     0    |            0 |

### Drift Snapshot (Top 15 by z-score)
| feature                     |   z_score |   current_mean |     ref_mean |
|:----------------------------|----------:|---------------:|-------------:|
| oil_fut_close               |   3.47298 |     98.0947    |  63.6777     |
| log_oil_fut_close           |   3.28838 |      4.58203   |   4.14406    |
| rate_fvx_close              |   3.13898 |      3.96147   |   3.70544    |
| gold_fut_close_ret30d       |   2.9443  |     -0.0811828 |   0.0618369  |
| gold_fut_oi_change_7d_proxy |   2.65609 |     55.8025    |   5.80388    |
| rate_tnx_close              |   2.57352 |      4.33453   |   4.12787    |
| rate_fvx_close_ret30d       |   2.21376 |      0.071353  |   0.00218853 |
| wheat_fut_close             |   2.11237 |    594.167     | 534.061      |
| log_wheat_fut_close         |   2.08955 |      6.38698   |   6.27916    |
| corn_fut_close              |   2.01226 |    453.217     | 432.347      |
| geo_ovx_close               |   2.00994 |     87.017     |  45.2603     |