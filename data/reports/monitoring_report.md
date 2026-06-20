# 📊 BTC Model Monitoring Report
**Generated:** 2026-06-20 15:28:32
**Run ID:** run_20260621_002832

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 6 features

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
- Drifted features (30d vs prev 180d): 6

### Quality Snapshot (Top 15 by missing/staleness)
| feature                          |   missing_pct_recent_30d |   stale_days |
|:---------------------------------|-------------------------:|-------------:|
| curve_2y10y_spread_proxy         |                     6.67 |            2 |
| expected_policy_rate_3m          |                     6.67 |            2 |
| expected_policy_rate_6m          |                     6.67 |            2 |
| rate_fvx_close                   |                     6.67 |            2 |
| rate_irx_close                   |                     6.67 |            2 |
| rate_tnx_close                   |                     6.67 |            2 |
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
| rate_fvx_close                   |   2.43782 |      4.21896   | 3.81527    |
| expected_policy_rate_6m          |   2.43646 |      3.85116   | 3.6805     |
| corn_fut_roll_return_20d         |   2.15805 |     -0.0589965 | 0.00826153 |
| corn_fut_close_ret30d            |   2.10113 |     -0.0595819 | 0.0121444  |
| rate_tnx_close                   |   2.0942  |      4.50729   | 4.2237     |
| corn_fut_front_next_spread_proxy |   2.00532 |     -0.0335844 | 0.00396812 |