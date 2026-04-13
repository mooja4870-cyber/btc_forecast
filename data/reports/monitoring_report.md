# 📊 BTC Model Monitoring Report
**Generated:** 2026-04-13 15:08:46
**Run ID:** run_20260414_000845

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 15 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $2565.97 | $4444.29 |
| **RMSE** | $3132.37 | $4575.07 |
| **MAPE** | 3.7% | 6.1% |
| **Count** | 154 | 51 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-03-21 00:00:00 | 30d       |           66685.3 |        68711.5 |    -2.94887 |
| 2026-04-11 00:00:00 | 60d       |           68192.5 |        73054.3 |    -6.655   |
| 2026-04-11 00:00:00 | 60d       |           68192.5 |        73054.3 |    -6.655   |
| 2026-04-11 00:00:00 | 60d       |           68192.5 |        73054.3 |    -6.655   |
| 2026-04-11 00:00:00 | 60d       |           68192.5 |        73054.3 |    -6.655   |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 15

### Quality Snapshot (Top 15 by missing/staleness)
| feature                          |   missing_pct_recent_30d |   stale_days |
|:---------------------------------|-------------------------:|-------------:|
| corn_fut_oi_change_7d_proxy      |                    13.33 |            4 |
| gold_fut_oi_change_7d_proxy      |                    13.33 |            4 |
| oil_fut_oi_change_7d_proxy       |                    13.33 |            4 |
| wheat_fut_oi_change_7d_proxy     |                    13.33 |            4 |
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
| feature                  |   z_score |   current_mean |     ref_mean |
|:-------------------------|----------:|---------------:|-------------:|
| oil_fut_close            |   5.13412 |     99.888     |  62.3753     |
| log_oil_fut_close        |   4.68131 |      4.60185   |   4.12756    |
| rate_fvx_close           |   3.36349 |      3.95413   |   3.69718    |
| oil_fut_close_ret30d     |   3.31524 |      0.384913  |   0.0280074  |
| gold_fut_close_ret30d    |   3.28829 |     -0.0834464 |   0.0665361  |
| geo_ovx_close            |   3.12665 |     94.6073    |  42.3436     |
| corn_fut_close           |   2.80368 |    455.958     | 431.035      |
| rate_fvx_close_ret30d    |   2.79859 |      0.0834237 |  -0.00213484 |
| log_corn_fut_close       |   2.73659 |      6.12224   |   6.06598    |
| rate_tnx_close           |   2.73642 |      4.33      |   4.12099    |
| wheat_fut_close          |   2.65889 |    596.408     | 530.992      |
| log_wheat_fut_close      |   2.60262 |      6.3907    |   6.27372    |
| gold_fut_roll_return_20d |   2.51885 |     -0.0676245 |   0.0430582  |
| rate_tnx_close_ret30d    |   2.4268  |      0.0611147 |  -0.00450947 |
| geo_vix_close            |   2.31739 |     25.3593    |  17.9596     |