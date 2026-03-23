# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-23 14:48:40
**Run ID:** run_20260323_234839

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 14 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $2504.75 | $3352.55 |
| **RMSE** | $3072.96 | $3709.26 |
| **MAPE** | 3.6% | 4.8% |
| **Count** | 150 | 99 |

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
- Drifted features (30d vs prev 180d): 14

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
| oil_fut_close                   |   8.63199 |      83.492    |     60.7087      |
| geo_ovx_close                   |   7.51352 |      86.3407   |     36.9769      |
| log_oil_fut_close               |   7.06309 |       4.41132  |      4.10515     |
| oil_fut_roll_return_20d         |   6.4021  |       0.285616 |      0.00154871  |
| oil_fut_close_ret30d            |   5.8497  |       0.315456 |     -0.00220728  |
| oil_fut_front_next_spread_proxy |   5.08247 |       0.138783 |      0.000710362 |
| wheat_fut_close                 |   4.8034  |     590.975    |    521.708       |
| log_wheat_fut_close             |   4.53262 |       6.38139  |      6.25673     |
| oil_fut_close_ret7d             |   3.21628 |       0.106198 |      0.000997498 |
| geo_ovx_close_ret30d            |   2.97013 |       0.777761 |      0.0802363   |
| oil_fut_volume                  |   2.65067 |  540416        | 257224           |
| geo_vix_close                   |   2.55672 |      23.482    |     17.0937      |
| wheat_fut_close_ret30d          |   2.38778 |       0.100854 |      0.00809194  |
| gold_fut_oi_change_7d_proxy     |   2.28895 |      48.6319   |      5.49951     |