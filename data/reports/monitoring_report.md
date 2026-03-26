# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-26 14:57:06
**Run ID:** run_20260326_235706

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 13 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $2504.75 | $3690.77 |
| **RMSE** | $3072.96 | $4024.55 |
| **MAPE** | 3.6% | 5.1% |
| **Count** | 150 | 69 |

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
| oil_fut_close                   |   9.28259 |     85.836     |     60.747       |
| log_oil_fut_close               |   7.57993 |      4.44154   |      4.10574     |
| geo_ovx_close                   |   7.50926 |     89.761     |     37.3827      |
| oil_fut_roll_return_20d         |   6.60933 |      0.298343  |      0.00240385  |
| oil_fut_close_ret30d            |   6.12652 |      0.341065  |     -0.000181954 |
| oil_fut_front_next_spread_proxy |   4.772   |      0.132338  |      0.00116056  |
| wheat_fut_close                 |   4.46229 |    592.692     |    522.808       |
| log_wheat_fut_close             |   4.23105 |      6.38435   |      6.25877     |
| geo_ovx_close_ret30d            |   3.11404 |      0.816432  |      0.0845097   |
| oil_fut_volume                  |   2.84581 | 563788         | 259419           |
| geo_vix_close                   |   2.83918 |     24.2357    |     17.1695      |
| oil_fut_close_ret7d             |   2.79293 |      0.094864  |      0.00163428  |
| wheat_fut_close_ret30d          |   2.14412 |      0.0948172 |      0.0105942   |