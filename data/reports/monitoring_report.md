# 📊 BTC Model Monitoring Report
**Generated:** 2026-05-11 16:24:00
**Run ID:** run_20260512_012400

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 1 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $3736.25 | $7270.03 |
| **RMSE** | $4744.70 | $7801.44 |
| **MAPE** | 5.1% | 9.6% |
| **Count** | 205 | 51 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-04-20 00:00:00 | 60d       |           70555   |        75872.5 |    -7.00851 |
| 2026-05-11 00:00:00 | 90d       |           71274.3 |        81834.2 |   -12.9041  |
| 2026-05-11 00:00:00 | 90d       |           71274.3 |        81834.2 |   -12.9041  |
| 2026-05-11 00:00:00 | 90d       |           71274.3 |        81834.2 |   -12.9041  |
| 2026-05-11 00:00:00 | 90d       |           71274.3 |        81834.2 |   -12.9041  |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 1

### Quality Snapshot (Top 15 by missing/staleness)
| feature                          |   missing_pct_recent_30d |   stale_days |
|:---------------------------------|-------------------------:|-------------:|
| corn_fut_oi_change_7d_proxy      |                     6.67 |            0 |
| gold_fut_oi_change_7d_proxy      |                     6.67 |            0 |
| oil_fut_oi_change_7d_proxy       |                     6.67 |            0 |
| wheat_fut_oi_change_7d_proxy     |                     6.67 |            0 |
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
| gold_fut_oi_change_7d_proxy |   2.33094 |        109.891 |    9.50969 |