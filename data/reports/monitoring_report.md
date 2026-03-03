# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-03 14:36:41
**Run ID:** run_20260303_233640

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 6 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $1463.48 | $1470.44 |
| **RMSE** | $1863.38 | $1871.63 |
| **MAPE** | 2.2% | 2.2% |
| **Count** | 91 | 90 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-02-25 00:00:00 | 7d        |           67034   |        67960.1 |    -1.36278 |
| 2026-02-25 00:00:00 | 7d        |           67155   |        67960.1 |    -1.18466 |
| 2026-02-25 00:00:00 | 7d        |           66018.1 |        67960.1 |    -2.85754 |
| 2026-02-26 00:00:00 | 7d        |           66165.8 |        67453.8 |    -1.90939 |
| 2026-02-26 00:00:00 | 7d        |           66320   |        67453.8 |    -1.68076 |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 6

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
| feature              |   z_score |   current_mean |     ref_mean |
|:---------------------|----------:|---------------:|-------------:|
| geo_ovx_close        |   3.79841 |     52.872     |   35.2764    |
| wheat_fut_close      |   2.70156 |    552.633     |  518.106     |
| log_wheat_fut_close  |   2.60979 |      6.31398   |    6.24988   |
| gold_fut_close       |   2.23446 |   5027.55      | 4047.18      |
| oil_fut_close_ret30d |   2.10796 |      0.0792049 |   -0.0174398 |
| log_gold_fut_close   |   2.05298 |      8.52222   |    8.29995   |