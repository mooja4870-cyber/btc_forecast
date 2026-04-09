# 📊 BTC Model Monitoring Report
**Generated:** 2026-04-09 15:14:30
**Run ID:** run_20260410_001429

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 18 features

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
| feature                  |   z_score |   current_mean |     ref_mean |
|:-------------------------|----------:|---------------:|-------------:|
| oil_fut_close            |   6.4746  |     99.1847    |  61.7381     |
| log_oil_fut_close        |   5.75952 |      4.59406   |   4.11923    |
| oil_fut_close_ret30d     |   4.67703 |      0.432922  |   0.018416   |
| geo_ovx_close            |   4.51416 |     98.88      |  40.5714     |
| rate_fvx_close           |   3.16824 |      3.93427   |   3.69325    |
| wheat_fut_close          |   3.10797 |    599.808     | 528.925      |
| gold_fut_close_ret30d    |   3.07375 |     -0.0708819 |   0.0679959  |
| log_wheat_fut_close      |   3.01954 |      6.39649   |   6.26996    |
| oil_fut_roll_return_20d  |   2.80829 |      0.259879  |   0.0179202  |
| geo_vix_close            |   2.79137 |     26.1247    |  17.7242     |
| corn_fut_close           |   2.70554 |    456.45      | 430.006      |
| rate_fvx_close_ret30d    |   2.66034 |      0.0796404 |  -0.00358983 |
| gold_fut_roll_return_20d |   2.63539 |     -0.0718827 |   0.0443103  |
| log_corn_fut_close       |   2.61908 |      6.12334   |   6.06354    |
| rate_tnx_close           |   2.58349 |      4.3145    |   4.11772    |