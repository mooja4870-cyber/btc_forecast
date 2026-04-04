# 📊 BTC Model Monitoring Report
**Generated:** 2026-04-04 14:24:56
**Run ID:** run_20260404_232456

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 19 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $2504.75 | $4087.30 |
| **RMSE** | $3072.96 | $4317.91 |
| **MAPE** | 3.6% | 5.6% |
| **Count** | 150 | 53 |

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
| oil_fut_close                   |  10.8163  |     95.9943    |  60.9816     |
| log_oil_fut_close               |   8.74106 |      4.56192   |   4.10922    |
| oil_fut_close_ret30d            |   7.84309 |      0.464304  |   0.0064864  |
| geo_ovx_close                   |   6.62247 |     98.7293    |  38.8971     |
| oil_fut_roll_return_20d         |   6.17649 |      0.309373  |   0.00574115 |
| oil_fut_front_next_spread_proxy |   4.20446 |      0.136001  |   0.00367737 |
| wheat_fut_close                 |   3.85748 |    601.108     | 526.085      |
| log_wheat_fut_close             |   3.68614 |      6.39864   |   6.2648     |
| geo_vix_close                   |   3.56086 |     26.5183    |  17.4038     |
| geo_ovx_close_ret30d            |   3.3926  |      0.890536  |   0.101741   |
| oil_fut_close_ret7d             |   2.74569 |      0.104724  |   0.00393069 |
| gold_fut_close_ret30d           |   2.63135 |     -0.0497144 |   0.0686182  |
| rate_fvx_close                  |   2.60292 |      3.89277   |   3.68972    |
| corn_fut_close                  |   2.48621 |    455.275     | 428.781      |
| log_corn_fut_close              |   2.39631 |      6.12071   |   6.06063    |