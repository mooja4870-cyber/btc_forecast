# 📊 BTC Model Monitoring Report
**Generated:** 2026-08-12 14:50:52
**Run ID:** run_20260812_235051

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Recent MAPE (22.7%) exceeds threshold (15.0%)
- ⚠️ Expansion feature drift detected: 4 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $4937.36 | $14487.83 |
| **RMSE** | $6262.94 | $14508.93 |
| **MAPE** | 6.8% | 22.7% |
| **Count** | 265 | 13 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-08-12 00:00:00 | 180d      |           49040.1 |        63520.7 |    -22.7967 |
| 2026-08-12 00:00:00 | 180d      |           48616.5 |        63520.7 |    -23.4635 |
| 2026-08-12 00:00:00 | 180d      |           48739.8 |        63520.7 |    -23.2694 |
| 2026-08-12 00:00:00 | 180d      |           50254.3 |        63520.7 |    -20.8852 |
| 2026-08-12 00:00:00 | 180d      |           50052.5 |        63520.7 |    -21.2028 |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 4

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
| feature                     |   z_score |   current_mean |   ref_mean |
|:----------------------------|----------:|---------------:|-----------:|
| rate_irx_close              |   3.37716 |        3.72317 |    3.60658 |
| expected_policy_rate_3m     |   3.37716 |        3.72317 |    3.60658 |
| expected_policy_rate_6m     |   2.29239 |        3.98049 |    3.7519  |
| gold_fut_oi_change_7d_proxy |   2.13199 |      101.031   |    9.95256 |