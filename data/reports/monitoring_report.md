# 📊 BTC Model Monitoring Report
**Generated:** 2026-04-14 15:06:47
**Run ID:** run_20260415_000646

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 16 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $2936.88 | $5232.85 |
| **RMSE** | $3758.48 | $5681.12 |
| **MAPE** | 4.1% | 7.1% |
| **Count** | 163 | 51 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-04-14 00:00:00 | 60d       |           64847.8 |        75034.1 |   -13.5756  |
| 2026-04-14 00:00:00 | 60d       |           67257.5 |        75034.1 |   -10.3641  |
| 2026-04-14 00:00:00 | 60d       |           64109.1 |        75034.1 |   -14.5601  |
| 2026-04-14 00:00:00 | 60d       |           67747.2 |        75034.1 |    -9.71143 |
| 2026-04-14 00:00:00 | 60d       |           65302.7 |        75034.1 |   -12.9694  |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 16

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
| feature                     |   z_score |   current_mean |     ref_mean |
|:----------------------------|----------:|---------------:|-------------:|
| oil_fut_close               |   4.80292 |     99.997     |  62.5721     |
| log_oil_fut_close           |   4.4171  |      4.60293   |   4.13003    |
| gold_fut_close_ret30d       |   3.35263 |     -0.0868551 |   0.0661049  |
| rate_fvx_close              |   3.33297 |      3.9563    |   3.69856    |
| oil_fut_close_ret30d        |   2.92062 |      0.36699   |   0.0311344  |
| gold_fut_oi_change_7d_proxy |   2.85798 |     59.5972    |   5.77577    |
| geo_ovx_close               |   2.85679 |     93.2403    |  42.8283     |
| rate_fvx_close_ret30d       |   2.77508 |      0.0827979 |  -0.00161945 |
| corn_fut_close              |   2.71121 |    455.633     | 431.197      |
| rate_tnx_close              |   2.70444 |      4.33107   |   4.12223    |
| log_corn_fut_close          |   2.64788 |      6.12152   |   6.06635    |
| wheat_fut_close             |   2.50824 |    595.342     | 531.511      |
| log_wheat_fut_close         |   2.46469 |      6.38893   |   6.27463    |
| gold_fut_roll_return_20d    |   2.46009 |     -0.0655359 |   0.0425694  |
| rate_tnx_close_ret30d       |   2.40575 |      0.0604608 |  -0.00401626 |