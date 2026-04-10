# 📊 BTC Model Monitoring Report
**Generated:** 2026-04-10 14:43:22
**Run ID:** run_20260410_234321

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 18 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $2504.75 | $4176.63 |
| **RMSE** | $3072.96 | $4386.58 |
| **MAPE** | 3.6% | 5.8% |
| **Count** | 150 | 51 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-03-20 00:00:00 | 30d       |           65625.2 |        70522.6 |    -6.94435 |
| 2026-03-20 00:00:00 | 30d       |           65697.1 |        70522.6 |    -6.84244 |
| 2026-03-20 00:00:00 | 30d       |           66811.2 |        70522.6 |    -5.26275 |
| 2026-03-21 00:00:00 | 30d       |           66381.1 |        68711.5 |    -3.39156 |
| 2026-03-21 00:00:00 | 30d       |           66685.3 |        68711.5 |    -2.94887 |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 18

### Quality Snapshot (Top 15 by missing/staleness)
| feature                          |   missing_pct_recent_30d |   stale_days |
|:---------------------------------|-------------------------:|-------------:|
| corn_fut_oi_change_7d_proxy      |                     3.33 |            1 |
| gold_fut_oi_change_7d_proxy      |                     3.33 |            1 |
| oil_fut_oi_change_7d_proxy       |                     3.33 |            1 |
| wheat_fut_oi_change_7d_proxy     |                     3.33 |            1 |
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
| oil_fut_close            |   6.3147  |     99.779     |  61.8552     |
| log_oil_fut_close        |   5.60878 |      4.60048   |   4.12085    |
| oil_fut_close_ret30d     |   4.48583 |      0.42967   |   0.020226   |
| geo_ovx_close            |   4.2018  |     98.326     |  40.9446     |
| rate_fvx_close           |   3.25444 |      3.9404    |   3.69411    |
| gold_fut_close_ret30d    |   3.17871 |     -0.075752  |   0.0678306  |
| wheat_fut_close          |   3.03519 |    599.492     | 529.379      |
| log_wheat_fut_close      |   2.95076 |      6.39595   |   6.27079    |
| corn_fut_close           |   2.79555 |    456.783     | 430.212      |
| rate_fvx_close_ret30d    |   2.73578 |      0.0814507 |  -0.00326772 |
| log_corn_fut_close       |   2.70989 |      6.1241    |   6.06404    |
| geo_vix_close            |   2.67878 |     25.976     |  17.7806     |
| rate_tnx_close           |   2.65278 |      4.31967   |   4.1183     |
| gold_fut_roll_return_20d |   2.64133 |     -0.0721207 |   0.0441491  |
| oil_fut_roll_return_20d  |   2.62688 |      0.251536  |   0.0195889  |