# 📊 BTC Model Monitoring Report
**Generated:** 2026-09-10 17:22:09
**Run ID:** run_20260911_022208

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Recent MAPE (35.4%) exceeds threshold (15.0%)
- ⚠️ Expansion feature drift detected: 4 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $7403.20 | $22591.77 |
| **RMSE** | $12689.72 | $29243.91 |
| **MAPE** | 10.8% | 35.4% |
| **Count** | 303 | 47 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-08-17 00:00:00 | 180d      |            121097 |        64506.3 |     87.7291 |
| 2026-08-17 00:00:00 | 180d      |            121661 |        64506.3 |     88.6034 |
| 2026-08-17 00:00:00 | 180d      |            120565 |        64506.3 |     86.9048 |
| 2026-08-18 00:00:00 | 180d      |            119377 |        64680.7 |     84.5637 |
| 2026-08-18 00:00:00 | 180d      |            118828 |        64680.7 |     83.7142 |

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
| feature             |   z_score |   current_mean |   ref_mean |
|:--------------------|----------:|---------------:|-----------:|
| wheat_fut_close     |   2.90331 |      706.2     |  609.818   |
| corn_fut_close      |   2.84906 |      492.083   |  443.526   |
| log_wheat_fut_close |   2.72718 |        6.5584  |    6.41171 |
| log_corn_fut_close  |   2.66496 |        6.19724 |    6.09402 |