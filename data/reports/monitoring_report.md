# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-18 14:57:31
**Run ID:** run_20260318_235730

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 15 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $2373.02 | $2384.66 |
| **RMSE** | $2991.27 | $3001.70 |
| **MAPE** | 3.4% | 3.4% |
| **Count** | 133 | 132 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-03-18 00:00:00 | 30d       |           68388.3 |        71405.8 |    -4.22586 |
| 2026-03-18 00:00:00 | 30d       |           67944.8 |        71405.8 |    -4.84702 |
| 2026-03-18 00:00:00 | 30d       |           68081.7 |        71405.8 |    -4.65525 |
| 2026-03-18 00:00:00 | 30d       |           68005.3 |        71405.8 |    -4.76233 |
| 2026-03-18 00:00:00 | 30d       |           67561.2 |        71405.8 |    -5.38416 |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 15

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
| oil_fut_close                   |   6.68717 |     77.9537    |     60.6781      |
| geo_ovx_close                   |   6.63721 |     78.5253    |     36.5473      |
| log_oil_fut_close               |   5.58552 |      4.34212   |      4.10468     |
| oil_fut_close_ret30d            |   4.71574 |      0.242954  |     -0.00544502  |
| wheat_fut_close                 |   4.66195 |    583.2       |    520.376       |
| oil_fut_roll_return_20d         |   4.6517  |      0.211326  |     -0.00022912  |
| oil_fut_front_next_spread_proxy |   4.54656 |      0.121965  |      0.000139258 |
| log_wheat_fut_close             |   4.39932 |      6.36786   |      6.25422     |
| oil_fut_close_ret7d             |   3.23165 |      0.104623  |      0.000830944 |
| oil_fut_volume                  |   2.53347 | 524714         | 257422           |
| wheat_fut_close_ret30d          |   2.46089 |      0.0993369 |      0.0040872   |
| geo_vix_close                   |   2.33084 |     22.6907    |     16.9546      |
| geo_ovx_close_ret30d            |   2.32082 |      0.620911  |      0.0772302   |
| geo_ovx_close_ret7d             |   2.19237 |      0.259463  |      0.0167522   |
| gold_fut_close                  |   2.06474 |   5109.18      |   4177.53        |