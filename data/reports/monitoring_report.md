# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-15 14:21:13
**Run ID:** run_20260315_232112

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 14 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $1787.96 | $1796.52 |
| **RMSE** | $2235.83 | $2244.47 |
| **MAPE** | 2.7% | 2.7% |
| **Count** | 112 | 111 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-03-15 00:00:00 | 30d       |           67388.4 |        71502.7 |    -5.75397 |
| 2026-03-15 00:00:00 | 30d       |           67251.6 |        71502.7 |    -5.94536 |
| 2026-03-15 00:00:00 | 30d       |           68067.5 |        71502.7 |    -4.80424 |
| 2026-03-15 00:00:00 | 30d       |           67996.7 |        71502.7 |    -4.9033  |
| 2026-03-15 00:00:00 | 30d       |           67415.9 |        71502.7 |    -5.71554 |

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
| geo_ovx_close                   |   5.55779 |     71.4037    |     36.4374      |
| oil_fut_close                   |   5.42878 |     74.703     |     60.6775      |
| log_oil_fut_close               |   4.60612 |      4.30049   |      4.10467     |
| wheat_fut_close                 |   4.41173 |    577.633     |    519.607       |
| log_wheat_fut_close             |   4.1762  |      6.35823   |      6.25276     |
| oil_fut_front_next_spread_proxy |   3.86545 |      0.1046    |     -0.000253335 |
| oil_fut_close_ret30d            |   3.84835 |      0.195752  |     -0.00730232  |
| oil_fut_roll_return_20d         |   3.68557 |      0.169561  |     -0.00216487  |
| gold_fut_oi_change_7d_proxy     |   3.53831 |     72.1727    |      5.50836     |
| oil_fut_close_ret7d             |   2.9926  |      0.0969615 |      0.000791089 |
| wheat_fut_close_ret30d          |   2.40494 |      0.0960225 |      0.00161588  |
| oil_fut_volume                  |   2.39653 | 511590         | 255336           |
| geo_vix_close                   |   2.18672 |     22.1617    |     16.861       |
| gold_fut_close                  |   2.13244 |   5109.98      |   4149.28        |