# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-07 14:15:11
**Run ID:** run_20260307_231510

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 8 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $1643.47 | $1651.70 |
| **RMSE** | $2096.46 | $2105.43 |
| **MAPE** | 2.5% | 2.5% |
| **Count** | 99 | 98 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-03-05 00:00:00 | 15d       |           66206.7 |        70841.1 |    -6.54196 |
| 2026-03-05 00:00:00 | 15d       |           66926.2 |        70841.1 |    -5.52635 |
| 2026-03-05 00:00:00 | 15d       |           66182.4 |        70841.1 |    -6.5763  |
| 2026-03-06 00:00:00 | 15d       |           65970.1 |        68136.5 |    -3.17955 |
| 2026-03-06 00:00:00 | 15d       |           66683.9 |        68136.5 |    -2.13181 |

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
| geo_ovx_close        |   3.77369 |     55.8407    |   35.6492    |
| wheat_fut_close      |   3.16275 |    559.233     |  518.504     |
| log_wheat_fut_close  |   3.03859 |      6.32585   |    6.25064   |
| oil_fut_close        |   2.43383 |     66.911     |   60.6629    |
| log_oil_fut_close    |   2.26578 |      4.20018   |    4.10444   |
| oil_fut_close_ret30d |   2.24636 |      0.0968761 |   -0.0138552 |
| gold_fut_close       |   2.23936 |   5066.61      | 4078.06      |
| log_gold_fut_close   |   2.05642 |      8.53019   |    8.30756   |