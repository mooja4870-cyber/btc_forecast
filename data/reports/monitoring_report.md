# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-28 14:24:39
**Run ID:** run_20260328_232438

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 13 features

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
| feature                         |   z_score |   current_mean |        ref_mean |
|:--------------------------------|----------:|---------------:|----------------:|
| oil_fut_close                   |   9.97672 |     87.9327    |     60.7606     |
| log_oil_fut_close               |   8.10826 |      4.46771   |      4.10595    |
| geo_ovx_close                   |   7.41421 |     91.9163    |     37.6713     |
| oil_fut_roll_return_20d         |   6.66785 |      0.3015    |      0.00255868 |
| oil_fut_close_ret30d            |   6.58047 |      0.367934  |      0.00151723 |
| oil_fut_front_next_spread_proxy |   4.79426 |      0.133155  |      0.00122552 |
| wheat_fut_close                 |   4.44081 |    595.733     |    523.392      |
| log_wheat_fut_close             |   4.20858 |      6.38952   |      6.25985    |
| geo_ovx_close_ret30d            |   3.19049 |      0.837201  |      0.0882757  |
| geo_vix_close                   |   2.96841 |     24.6427    |     17.2322     |
| oil_fut_volume                  |   2.83534 | 560037         | 261697          |
| oil_fut_close_ret7d             |   2.70956 |      0.0928077 |      0.00182301 |
| wheat_fut_close_ret30d          |   2.09827 |      0.0950324 |      0.0117182  |