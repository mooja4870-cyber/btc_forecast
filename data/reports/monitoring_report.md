# 📊 BTC Model Monitoring Report
**Generated:** 2026-06-04 15:07:05
**Run ID:** run_20260604_150702

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 6 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $4444.68 | $7771.95 |
| **RMSE** | $5512.55 | $8271.02 |
| **MAPE** | 6.0% | 9.8% |
| **Count** | 252 | 51 |

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
| corn_fut_oi_change_7d_proxy      |                    13.33 |            0 |
| gold_fut_oi_change_7d_proxy      |                    13.33 |            0 |
| oil_fut_oi_change_7d_proxy       |                    13.33 |            0 |
| wheat_fut_oi_change_7d_proxy     |                    13.33 |            0 |
| commodity_shock_score            |                     0    |            0 |
| corn_fut_close                   |                     0    |            0 |
| corn_fut_close_ret1d             |                     0    |            0 |
| corn_fut_close_ret30d            |                     0    |            0 |
| corn_fut_close_ret7d             |                     0    |            0 |
| corn_fut_days_to_expiry          |                     0    |            0 |
| corn_fut_expiry_week             |                     0    |            0 |
| corn_fut_front_next_spread_proxy |                     0    |            0 |
| corn_fut_roll_return_20d         |                     0    |            0 |
| corn_fut_volume                  |                     0    |            0 |
| curve_2y10y_spread_proxy         |                     0    |            0 |

### Drift Snapshot (Top 15 by z-score)
| feature                     |   z_score |   current_mean |   ref_mean |
|:----------------------------|----------:|---------------:|-----------:|
| oil_fut_close               |   3.07671 |     97.923     | 64.3505    |
| log_oil_fut_close           |   2.93158 |      4.58048   |  4.15265   |
| rate_fvx_close              |   2.53074 |      3.94563   |  3.71218   |
| gold_fut_close_ret30d       |   2.13912 |     -0.0592174 |  0.0569702 |
| rate_tnx_close              |   2.11749 |      4.32037   |  4.13319   |
| gold_fut_oi_change_7d_proxy |   2.015   |     43.5295    |  5.67072   |