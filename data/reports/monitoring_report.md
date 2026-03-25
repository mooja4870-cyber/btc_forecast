# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-25 14:52:40
**Run ID:** run_20260325_235240

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 13 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $2504.75 | $3681.95 |
| **RMSE** | $3072.96 | $3984.75 |
| **MAPE** | 3.6% | 5.2% |
| **Count** | 150 | 78 |

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
| oil_fut_close                   |   9.07735 |     85.0867    |     60.735       |
| geo_ovx_close                   |   7.51676 |     88.6537    |     37.2583      |
| log_oil_fut_close               |   7.41506 |      4.43181   |      4.10556     |
| oil_fut_roll_return_20d         |   6.59991 |      0.296772  |      0.00215487  |
| oil_fut_close_ret30d            |   6.0201  |      0.332625  |     -0.000871832 |
| oil_fut_front_next_spread_proxy |   4.88892 |      0.134992  |      0.00103922  |
| wheat_fut_close                 |   4.53175 |    591.95      |    522.435       |
| log_wheat_fut_close             |   4.2927  |      6.38307   |      6.25808     |
| geo_ovx_close_ret30d            |   3.05977 |      0.802634  |      0.0836644   |
| oil_fut_close_ret7d             |   2.95128 |      0.0992826 |      0.00141478  |
| geo_vix_close                   |   2.74055 |     23.9737    |     17.1447      |
| oil_fut_volume                  |   2.65809 | 543575         | 258992           |
| wheat_fut_close_ret30d          |   2.18842 |      0.0960336 |      0.00980053  |