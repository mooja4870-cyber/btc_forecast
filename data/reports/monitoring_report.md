# 📊 BTC Model Monitoring Report
**Generated:** 2026-02-25 16:13:53
**Run ID:** run_20260225_161353

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 8 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $1439.25 | $1446.09 |
| **RMSE** | $1858.63 | $1867.03 |
| **MAPE** | 2.2% | 2.2% |
| **Count** | 89 | 88 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-02-25 00:00:00 | 7d        |           67034   |        67627.3 |   -0.877379 |
| 2026-02-25 00:00:00 | 7d        |           66735.8 |        67627.3 |   -1.31824  |
| 2026-02-25 00:00:00 | 7d        |           67155   |        67627.3 |   -0.698384 |
| 2026-02-25 00:00:00 | 7d        |           66018.1 |        67627.3 |   -2.37949  |
| 2026-02-25 00:00:00 | 7d        |           66559.8 |        67627.3 |   -1.57849  |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 8

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
| feature               |   z_score |   current_mean |     ref_mean |
|:----------------------|----------:|---------------:|-------------:|
| geo_ovx_close         |   4.42011 |     50.7077    |   34.8541    |
| geo_ovx_close_ret30d  |   3.03583 |      0.41779   |    0.0143109 |
| oil_fut_close_ret30d  |   2.74169 |      0.0901462 |   -0.0202782 |
| gold_fut_close        |   2.4059  |   4987.24      | 3990.5       |
| log_gold_fut_close    |   2.17549 |      8.51414   |    8.28625   |
| wheat_fut_close       |   2.05789 |    543.167     |  517.632     |
| log_wheat_fut_close   |   2.01234 |      6.29699   |    6.24898   |
| rate_irx_close_ret30d |   2.00311 |      0.0120652 |   -0.0289296 |