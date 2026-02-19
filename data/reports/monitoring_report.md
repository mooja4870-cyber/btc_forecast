# 📊 BTC Model Monitoring Report
**Generated:** 2026-02-19 21:00:04
**Run ID:** run_20260219_210003

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 6 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $925.40 | $930.58 |
| **RMSE** | $1053.17 | $1064.51 |
| **MAPE** | 1.4% | 1.4% |
| **Count** | 18 | 17 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-02-19 00:00:00 | 1d        |           67504.1 |        66495.6 |    1.51665  |
| 2026-02-19 00:00:00 | 1d        |           67918   |        66495.6 |    2.13899  |
| 2026-02-19 00:00:00 | 1d        |           67352.2 |        66495.6 |    1.28819  |
| 2026-02-19 00:00:00 | 1d        |           66918.8 |        66495.6 |    0.636444 |
| 2026-02-19 00:00:00 | 1d        |           66383.1 |        66495.6 |   -0.169263 |

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
| feature               |   z_score |   current_mean |      ref_mean |
|:----------------------|----------:|---------------:|--------------:|
| geo_ovx_close         |   4.64999 |     48.8433    |   34.5106     |
| geo_ovx_close_ret30d  |   3.68813 |      0.443559  |   -0.00309837 |
| oil_fut_close_ret30d  |   2.62492 |      0.0816841 |   -0.0211664  |
| gold_fut_close        |   2.55744 |   4959.16      | 3945.87       |
| log_gold_fut_close    |   2.2875  |      8.50855   |    8.27531    |
| rate_irx_close_ret30d |   2.06762 |      0.0134754 |   -0.0289096  |