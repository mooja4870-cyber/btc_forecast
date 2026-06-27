# 📊 BTC Model Monitoring Report
**Generated:** 2026-06-27 15:08:08
**Run ID:** run_20260628_000808

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
| curve_2y10y_spread_proxy         |                       10 |            3 |
| expected_policy_rate_3m          |                       10 |            3 |
| expected_policy_rate_6m          |                       10 |            3 |
| rate_fvx_close                   |                       10 |            3 |
| rate_irx_close                   |                       10 |            3 |
| rate_tnx_close                   |                       10 |            3 |
| commodity_shock_score            |                        0 |            0 |
| corn_fut_close                   |                        0 |            0 |
| corn_fut_close_ret1d             |                        0 |            0 |
| corn_fut_close_ret30d            |                        0 |            0 |
| corn_fut_close_ret7d             |                        0 |            0 |
| corn_fut_days_to_expiry          |                        0 |            0 |
| corn_fut_expiry_week             |                        0 |            0 |
| corn_fut_front_next_spread_proxy |                        0 |            0 |
| corn_fut_oi_change_7d_proxy      |                        0 |            0 |

### Drift Snapshot (Top 15 by z-score)
| feature                          |   z_score |   current_mean |   ref_mean |
|:---------------------------------|----------:|---------------:|-----------:|
| corn_fut_close_ret30d            |   2.86908 |     -0.0854725 | 0.0125238  |
| corn_fut_roll_return_20d         |   2.39749 |     -0.0662905 | 0.00821122 |
| expected_policy_rate_6m          |   2.23924 |      3.85913   | 3.68679    |
| corn_fut_front_next_spread_proxy |   2.08545 |     -0.0347637 | 0.00407686 |
| rate_fvx_close                   |   2.07423 |      4.21144   | 3.84039    |