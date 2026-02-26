# 📊 BTC Model Monitoring Report
**Generated:** 2026-02-26 16:02:22
**Run ID:** run_20260226_160222

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 7 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $1461.95 | $1468.89 |
| **RMSE** | $1862.41 | $1870.65 |
| **MAPE** | 2.2% | 2.2% |
| **Count** | 91 | 90 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-02-25 00:00:00 | 7d        |           67034   |        67960.1 |    -1.36278 |
| 2026-02-25 00:00:00 | 7d        |           67155   |        67960.1 |    -1.18466 |
| 2026-02-25 00:00:00 | 7d        |           66018.1 |        67960.1 |    -2.85754 |
| 2026-02-26 00:00:00 | 7d        |           66165.8 |        67383.7 |    -1.80744 |
| 2026-02-26 00:00:00 | 7d        |           66320   |        67383.7 |    -1.57857 |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 7

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
| geo_ovx_close        |   4.36658 |     51.0803    |   34.9074    |
| geo_ovx_close_ret30d |   2.85998 |      0.409659  |    0.0173995 |
| oil_fut_close_ret30d |   2.75276 |      0.0903248 |   -0.0203146 |
| gold_fut_close       |   2.36553 |   4991.31      | 4000.41      |
| wheat_fut_close      |   2.1557  |    544.367     |  517.625     |
| log_gold_fut_close   |   2.14631 |      8.51493   |    8.28866   |
| log_wheat_fut_close  |   2.10678 |      6.29922   |    6.24897   |