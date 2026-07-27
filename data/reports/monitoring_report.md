# 📊 BTC Model Monitoring Report
**Generated:** 2026-07-27 16:07:52
**Run ID:** run_20260728_010751

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 5 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $4444.68 | $0.00 |
| **RMSE** | $5512.55 | $0.00 |
| **MAPE** | 6.0% | 0.0% |
| **Count** | 252 | 0 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-05-19 00:00:00 | 90d       |           79909.5 |        76750.9 |    4.11537  |
| 2026-05-19 00:00:00 | 90d       |           80084.3 |        76750.9 |    4.34312  |
| 2026-05-19 00:00:00 | 90d       |           80858.8 |        76750.9 |    5.35224  |
| 2026-05-20 00:00:00 | 90d       |           76766.3 |        77457.8 |   -0.89271  |
| 2026-05-20 00:00:00 | 90d       |           77838.2 |        77457.8 |    0.491192 |

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
| feature                 |   z_score |   current_mean |   ref_mean |
|:------------------------|----------:|---------------:|-----------:|
| rate_irx_close          |   3.70192 |        3.7064  |    3.59247 |
| expected_policy_rate_3m |   3.70192 |        3.7064  |    3.59247 |
| expected_policy_rate_6m |   2.19898 |        3.93543 |    3.72584 |
| log_gold_fut_close      |   2.17305 |        8.31173 |    8.45042 |
| gold_fut_close          |   2.05852 |     4071.63    | 4686.52    |