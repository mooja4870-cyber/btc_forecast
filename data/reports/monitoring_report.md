# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-05 14:37:55
**Run ID:** run_20260305_233754

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 6 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $1753.09 | $1762.63 |
| **RMSE** | $2366.09 | $2376.84 |
| **MAPE** | 2.6% | 2.6% |
| **Count** | 97 | 96 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-03-05 00:00:00 | 15d       |           66624.3 |        72668.6 |    -8.31761 |
| 2026-03-05 00:00:00 | 15d       |           66926.2 |        72668.6 |    -7.90212 |
| 2026-03-05 00:00:00 | 15d       |           66791.2 |        72668.6 |    -8.08785 |
| 2026-03-05 00:00:00 | 15d       |           66182.4 |        72668.6 |    -8.92567 |
| 2026-03-05 00:00:00 | 15d       |           66206.7 |        72668.6 |    -8.89219 |

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
| geo_ovx_close        |   3.60568 |     53.9077    |   35.4861    |
| wheat_fut_close      |   2.84697 |    554.942     |  518.378     |
| log_wheat_fut_close  |   2.74632 |      6.31818   |    6.2504    |
| gold_fut_close       |   2.27422 |   5057.14      | 4061.48      |
| log_gold_fut_close   |   2.0852  |      8.52829   |    8.30353   |
| oil_fut_close_ret30d |   2.06374 |      0.0828317 |   -0.0158032 |