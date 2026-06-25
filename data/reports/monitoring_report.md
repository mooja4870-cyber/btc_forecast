# 📊 BTC Model Monitoring Report
**Generated:** 2026-06-25 16:14:25
**Run ID:** run_20260626_011425

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
| curve_2y10y_spread_proxy         |                     3.33 |            1 |
| expected_policy_rate_3m          |                     3.33 |            1 |
| expected_policy_rate_6m          |                     3.33 |            1 |
| rate_fvx_close                   |                     3.33 |            1 |
| rate_irx_close                   |                     3.33 |            1 |
| rate_tnx_close                   |                     3.33 |            1 |
| commodity_shock_score            |                     0    |            0 |
| corn_fut_close                   |                     0    |            0 |
| corn_fut_close_ret1d             |                     0    |            0 |
| corn_fut_close_ret30d            |                     0    |            0 |
| corn_fut_close_ret7d             |                     0    |            0 |
| corn_fut_days_to_expiry          |                     0    |            0 |
| corn_fut_expiry_week             |                     0    |            0 |
| corn_fut_front_next_spread_proxy |                     0    |            0 |
| corn_fut_oi_change_7d_proxy      |                     0    |            0 |

### Drift Snapshot (Top 15 by z-score)
| feature                          |   z_score |   current_mean |   ref_mean |
|:---------------------------------|----------:|---------------:|-----------:|
| corn_fut_close_ret30d            |   2.71948 |     -0.0801169 | 0.0126088  |
| corn_fut_roll_return_20d         |   2.3802  |     -0.065652  | 0.00832179 |
| expected_policy_rate_6m          |   2.29392 |      3.85767   | 3.68504    |
| corn_fut_front_next_spread_proxy |   2.15471 |     -0.0356358 | 0.00431171 |
| rate_fvx_close                   |   2.14686 |      4.212     | 3.83319    |