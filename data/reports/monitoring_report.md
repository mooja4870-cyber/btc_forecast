# 📊 BTC Model Monitoring Report
**Generated:** 2026-05-19 16:44:01
**Run ID:** run_20260520_014400

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 5 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $4478.13 | $7973.42 |
| **RMSE** | $5535.69 | $8343.61 |
| **MAPE** | 6.0% | 10.1% |
| **Count** | 250 | 51 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-05-19 00:00:00 | 90d       |           80858.8 |          76683 |     5.44555 |
| 2026-05-19 00:00:00 | 90d       |           80606.4 |          76683 |     5.11643 |
| 2026-05-19 00:00:00 | 90d       |           79797.9 |          76683 |     4.06201 |
| 2026-05-19 00:00:00 | 90d       |           80192.8 |          76683 |     4.57698 |
| 2026-05-19 00:00:00 | 90d       |           80084.3 |          76683 |     4.43553 |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 5

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
| gold_fut_oi_change_7d_proxy |   2.55308 |      120.154   |    9.59247 |
| wheat_fut_close             |   2.25825 |      622.783   |  547.575   |
| log_wheat_fut_close         |   2.17408 |        6.43368 |    6.3037  |
| rate_fvx_close              |   2.16553 |        4.0203  |    3.75072 |
| rate_tnx_close              |   2.00173 |        4.38383 |    4.16583 |