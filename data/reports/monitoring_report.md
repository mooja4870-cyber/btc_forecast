# 📊 BTC Model Monitoring Report
**Generated:** 2026-05-13 16:13:55
**Run ID:** run_20260514_011354

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 1 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $3734.18 | $7261.72 |
| **RMSE** | $4740.12 | $7790.24 |
| **MAPE** | 5.1% | 9.6% |
| **Count** | 205 | 51 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-04-20 00:00:00 | 60d       |           70555   |        75872.5 |    -7.00851 |
| 2026-05-11 00:00:00 | 90d       |           71274.3 |        81728.3 |   -12.7912  |
| 2026-05-11 00:00:00 | 90d       |           71274.3 |        81728.3 |   -12.7912  |
| 2026-05-11 00:00:00 | 90d       |           71274.3 |        81728.3 |   -12.7912  |
| 2026-05-11 00:00:00 | 90d       |           71274.3 |        81728.3 |   -12.7912  |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 1

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
| feature         |   z_score |   current_mean |   ref_mean |
|:----------------|----------:|---------------:|-----------:|
| wheat_fut_close |   2.01866 |        611.633 |    544.621 |