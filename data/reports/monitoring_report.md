# 📊 BTC Model Monitoring Report
**Generated:** 2026-04-03 14:35:56
**Run ID:** run_20260403_233556

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 18 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $2504.75 | $4110.76 |
| **RMSE** | $3072.96 | $4318.85 |
| **MAPE** | 3.6% | 5.7% |
| **Count** | 150 | 59 |

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
- Drifted features (30d vs prev 180d): 18

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
| oil_fut_close                   |  10.971   |     94.605     |  60.9105     |
| log_oil_fut_close               |   8.86589 |      4.54707   |   4.10817    |
| oil_fut_close_ret30d            |   7.90098 |      0.452408  |   0.00515296 |
| geo_ovx_close                   |   6.87007 |     98.091     |  38.6807     |
| oil_fut_roll_return_20d         |   6.45752 |      0.30969   |   0.0046139  |
| oil_fut_front_next_spread_proxy |   4.3537  |      0.134217  |   0.00281761 |
| wheat_fut_close                 |   3.85481 |    600.108     | 525.719      |
| log_wheat_fut_close             |   3.68433 |      6.39692   |   6.26411    |
| geo_vix_close                   |   3.62784 |     26.5083    |  17.3578     |
| geo_ovx_close_ret30d            |   3.41644 |      0.892688  |   0.0999018  |
| oil_fut_close_ret7d             |   2.85198 |      0.103758  |   0.00296028 |
| gold_fut_close_ret30d           |   2.47425 |     -0.0426284 |   0.0684034  |
| rate_fvx_close                  |   2.46828 |      3.8822    |   3.68984    |
| corn_fut_close                  |   2.39459 |    454.658     | 428.599      |
| log_corn_fut_close              |   2.30677 |      6.11931   |   6.0602     |