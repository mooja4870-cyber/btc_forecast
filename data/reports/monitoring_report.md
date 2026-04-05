# 📊 BTC Model Monitoring Report
**Generated:** 2026-04-05 14:26:49
**Run ID:** run_20260405_232649

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 19 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $2504.75 | $4176.63 |
| **RMSE** | $3072.96 | $4386.58 |
| **MAPE** | 3.6% | 5.8% |
| **Count** | 150 | 51 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-03-20 00:00:00 | 30d       |           65625.2 |        70522.6 |    -6.94435 |
| 2026-03-20 00:00:00 | 30d       |           65697.1 |        70522.6 |    -6.84244 |
| 2026-03-20 00:00:00 | 30d       |           66811.2 |        70522.6 |    -5.26275 |
| 2026-03-21 00:00:00 | 30d       |           66381.1 |        68711.5 |    -3.39156 |
| 2026-03-21 00:00:00 | 30d       |           66685.3 |        68711.5 |    -2.94887 |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 19

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
| feature                         |   z_score |   current_mean |     ref_mean |
|:--------------------------------|----------:|---------------:|-------------:|
| oil_fut_close                   |  10.0791  |     97.012     |  61.0879     |
| log_oil_fut_close               |   8.25089 |      4.57258   |   4.11071    |
| oil_fut_close_ret30d            |   7.43148 |      0.467479  |   0.00822566 |
| geo_ovx_close                   |   6.38951 |     99.3273    |  39.1297     |
| oil_fut_roll_return_20d         |   5.55056 |      0.304102  |   0.0074241  |
| wheat_fut_close                 |   3.79013 |    601.625     | 526.539      |
| oil_fut_front_next_spread_proxy |   3.68423 |      0.133332  |   0.00497819 |
| log_wheat_fut_close             |   3.62504 |      6.39951   |   6.26564    |
| geo_vix_close                   |   3.56981 |     26.609     |  17.4369     |
| geo_ovx_close_ret30d            |   3.30701 |      0.880349  |   0.10527    |
| rate_fvx_close                  |   2.73028 |      3.9021    |   3.69019    |
| gold_fut_close_ret30d           |   2.70807 |     -0.053536  |   0.0684535  |
| corn_fut_close                  |   2.5441  |    455.633     | 429.017      |
| log_corn_fut_close              |   2.45344 |      6.12151   |   6.0612     |
| gold_fut_roll_return_20d        |   2.4415  |     -0.0630882 |   0.0456839  |