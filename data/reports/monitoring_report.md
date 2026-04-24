# 📊 BTC Model Monitoring Report
**Generated:** 2026-04-24 15:02:02
**Run ID:** run_20260425_000201

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 6 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $3600.45 | $6823.11 |
| **RMSE** | $4554.23 | $7346.44 |
| **MAPE** | 5.0% | 9.1% |
| **Count** | 201 | 51 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-04-19 00:00:00 | 60d       |           71313   |        73856.4 |    -3.44369 |
| 2026-04-19 00:00:00 | 60d       |           72725.8 |        73856.4 |    -1.53081 |
| 2026-04-19 00:00:00 | 60d       |           72459.5 |        73856.4 |    -1.89131 |
| 2026-04-20 00:00:00 | 60d       |           70386.1 |        75872.5 |    -7.23117 |
| 2026-04-20 00:00:00 | 60d       |           70555   |        75872.5 |    -7.00851 |

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