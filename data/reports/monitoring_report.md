# 📊 BTC Model Monitoring Report
**Generated:** 2026-04-12 14:32:32
**Run ID:** run_20260412_233232

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 16 features

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
- Drifted features (30d vs prev 180d): 16

### Quality Snapshot (Top 15 by missing/staleness)
| feature                          |   missing_pct_recent_30d |   stale_days |
|:---------------------------------|-------------------------:|-------------:|
| corn_fut_oi_change_7d_proxy      |                       10 |            3 |
| gold_fut_oi_change_7d_proxy      |                       10 |            3 |
| oil_fut_oi_change_7d_proxy       |                       10 |            3 |
| wheat_fut_oi_change_7d_proxy     |                       10 |            3 |
| commodity_shock_score            |                        0 |            0 |
| corn_fut_close                   |                        0 |            0 |
| corn_fut_close_ret1d             |                        0 |            0 |
| corn_fut_close_ret30d            |                        0 |            0 |
| corn_fut_close_ret7d             |                        0 |            0 |
| corn_fut_days_to_expiry          |                        0 |            0 |
| corn_fut_expiry_week             |                        0 |            0 |
| corn_fut_front_next_spread_proxy |                        0 |            0 |
| corn_fut_roll_return_20d         |                        0 |            0 |
| corn_fut_volume                  |                        0 |            0 |
| curve_2y10y_spread_proxy         |                        0 |            0 |

### Drift Snapshot (Top 15 by z-score)
| feature                  |   z_score |   current_mean |     ref_mean |
|:-------------------------|----------:|---------------:|-------------:|
| oil_fut_close            |   5.57276 |     99.9593    |  62.1752     |
| log_oil_fut_close        |   5.01896 |      4.60258   |   4.12504    |
| oil_fut_close_ret30d     |   3.74541 |      0.403213  |   0.0250682  |
| geo_ovx_close            |   3.45388 |     96.0143    |  41.8522     |
| rate_fvx_close           |   3.40772 |      3.9523    |   3.69574    |
| gold_fut_close_ret30d    |   3.29066 |     -0.0821536 |   0.0670789  |
| rate_fvx_close_ret30d    |   2.85917 |      0.0846188 |  -0.00269767 |
| wheat_fut_close          |   2.84196 |    597.992     | 530.35       |
| corn_fut_close           |   2.82587 |    456.342     | 430.738      |
| log_wheat_fut_close      |   2.76904 |      6.39336   |   6.27257    |
| rate_tnx_close           |   2.75886 |      4.32853   |   4.11981    |
| log_corn_fut_close       |   2.75013 |      6.1231    |   6.06528    |
| gold_fut_roll_return_20d |   2.60681 |     -0.0709424 |   0.0435195  |
| geo_vix_close            |   2.47662 |     25.628     |  17.89       |
| rate_tnx_close_ret30d    |   2.45769 |      0.0617842 |  -0.00497849 |