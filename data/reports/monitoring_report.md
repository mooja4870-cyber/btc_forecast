# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-06 14:29:08
**Run ID:** run_20260306_232907

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 8 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $1662.16 | $1670.58 |
| **RMSE** | $2116.61 | $2125.70 |
| **MAPE** | 2.5% | 2.5% |
| **Count** | 99 | 98 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-03-05 00:00:00 | 15d       |           66206.7 |        70841.1 |    -6.54196 |
| 2026-03-05 00:00:00 | 15d       |           66926.2 |        70841.1 |    -5.52635 |
| 2026-03-05 00:00:00 | 15d       |           66182.4 |        70841.1 |    -6.5763  |
| 2026-03-06 00:00:00 | 15d       |           65970.1 |        69061.6 |    -4.47644 |
| 2026-03-06 00:00:00 | 15d       |           66683.9 |        69061.6 |    -3.44274 |

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
| feature              |   z_score |   current_mean |     ref_mean |
|:---------------------|----------:|---------------:|-------------:|
| geo_ovx_close        |   3.69789 |     54.791     |   35.5529    |
| wheat_fut_close      |   2.95259 |    556.425     |  518.436     |
| log_wheat_fut_close  |   2.84569 |      6.32088   |    6.25051   |
| gold_fut_close       |   2.25897 |   5062.55      | 4069.83      |
| oil_fut_close_ret30d |   2.11384 |      0.0875654 |   -0.0148705 |
| oil_fut_close        |   2.1113  |     66.0537    |   60.6559    |
| log_gold_fut_close   |   2.07274 |      8.52938   |    8.30557   |
| log_oil_fut_close    |   2.01019 |      4.18895   |    4.10434   |