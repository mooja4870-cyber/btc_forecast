# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-19 14:42:50
**Run ID:** run_20260319_234249

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 15 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $2361.43 | $2427.10 |
| **RMSE** | $2952.78 | $3004.74 |
| **MAPE** | 3.4% | 3.5% |
| **Count** | 140 | 135 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-03-19 00:00:00 | 30d       |           66452.5 |        69161.8 |    -3.91728 |
| 2026-03-19 00:00:00 | 30d       |           66824.1 |        69161.8 |    -3.37998 |
| 2026-03-19 00:00:00 | 30d       |           66902.5 |        69161.8 |    -3.26664 |
| 2026-03-19 00:00:00 | 30d       |           67040.1 |        69161.8 |    -3.06778 |
| 2026-03-19 00:00:00 | 30d       |           66941.6 |        69161.8 |    -3.21013 |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 15

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
| feature                         |   z_score |   current_mean |         ref_mean |
|:--------------------------------|----------:|---------------:|-----------------:|
| oil_fut_close                   |   7.17589 |      79.2027   |     60.6763      |
| geo_ovx_close                   |   6.90328 |      80.3167   |     36.5981      |
| log_oil_fut_close               |   5.95711 |       4.35773  |      4.10465     |
| oil_fut_roll_return_20d         |   5.08141 |       0.229023 |      0.000300692 |
| oil_fut_close_ret30d            |   5.03836 |       0.260879 |     -0.00479271  |
| oil_fut_front_next_spread_proxy |   4.78869 |       0.128386 |      0.000173476 |
| wheat_fut_close                 |   4.73733 |     585.017    |    520.617       |
| log_wheat_fut_close             |   4.4668  |       6.37102  |      6.25468     |
| oil_fut_close_ret7d             |   3.41349 |       0.110432 |      0.000654395 |
| oil_fut_volume                  |   2.63474 |  535222        | 257284           |
| geo_ovx_close_ret30d            |   2.50566 |       0.664159 |      0.07695     |
| wheat_fut_close_ret30d          |   2.48726 |       0.100669 |      0.00479351  |
| geo_vix_close                   |   2.33141 |      22.7497   |     16.9826      |
| geo_ovx_close_ret7d             |   2.21325 |       0.262578 |      0.0151027   |
| gold_fut_close                  |   2.03201 |    5103.76     |   4186.85        |