# 📊 BTC Model Monitoring Report
**Generated:** 2026-04-06 14:40:40
**Run ID:** run_20260406_234040

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
| oil_fut_close                   |   8.69549 |     97.7693    |  61.2492     |
| log_oil_fut_close               |   7.37061 |      4.58002   |   4.11285    |
| oil_fut_close_ret30d            |   6.63598 |      0.462735  |   0.0105974  |
| geo_ovx_close                   |   6.0331  |     99.6377    |  39.4103     |
| oil_fut_roll_return_20d         |   4.57618 |      0.296399  |   0.0100342  |
| wheat_fut_close                 |   3.57339 |    601.125     | 527.151      |
| geo_vix_close                   |   3.5023  |     26.613     |  17.4846     |
| log_wheat_fut_close             |   3.43583 |      6.39868   |   6.26674    |
| geo_ovx_close_ret30d            |   3.1863  |      0.863997  |   0.108879   |
| oil_fut_front_next_spread_proxy |   2.84099 |      0.127009  |   0.00694716 |
| rate_fvx_close                  |   2.81845 |      3.90893   |   3.69108    |
| gold_fut_close_ret30d           |   2.78215 |     -0.0569327 |   0.0684271  |
| corn_fut_close                  |   2.57119 |    455.775     | 429.283      |
| log_corn_fut_close              |   2.482   |      6.12183   |   6.06183    |
| gold_fut_roll_return_20d        |   2.47722 |     -0.0648429 |   0.0453518  |