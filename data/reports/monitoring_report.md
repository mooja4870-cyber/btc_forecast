# 📊 BTC Model Monitoring Report
**Generated:** 2026-05-18 16:44:37
**Run ID:** run_20260519_014436

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 4 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $4499.32 | $7660.24 |
| **RMSE** | $5568.72 | $8195.54 |
| **MAPE** | 6.0% | 9.7% |
| **Count** | 242 | 51 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-05-18 00:00:00 | 90d       |           67004.6 |          76392 |   -12.2884  |
| 2026-05-18 00:00:00 | 90d       |           66981.6 |          76392 |   -12.3185  |
| 2026-05-18 00:00:00 | 90d       |           68201.8 |          76392 |   -10.7213  |
| 2026-05-18 00:00:00 | 90d       |           82102.7 |          76392 |     7.47551 |
| 2026-05-18 00:00:00 | 90d       |           66299.9 |          76392 |   -13.2109  |

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
| gold_fut_oi_change_7d_proxy |   2.89975 |      135.163   |    9.58498 |
| wheat_fut_close             |   2.20176 |      620.4     |  547.094   |
| log_wheat_fut_close         |   2.12507 |        6.42988 |    6.30282 |
| rate_fvx_close              |   2.05767 |        4.0063  |    3.74935 |