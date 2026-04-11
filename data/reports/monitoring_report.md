# 📊 BTC Model Monitoring Report
**Generated:** 2026-04-11 14:27:36
**Run ID:** run_20260411_232735

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 18 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $2558.56 | $4421.93 |
| **RMSE** | $3121.20 | $4551.95 |
| **MAPE** | 3.7% | 6.1% |
| **Count** | 154 | 51 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-03-21 00:00:00 | 30d       |           66685.3 |        68711.5 |    -2.94887 |
| 2026-04-11 00:00:00 | 60d       |           68192.5 |        72769.2 |    -6.28937 |
| 2026-04-11 00:00:00 | 60d       |           68192.5 |        72769.2 |    -6.28937 |
| 2026-04-11 00:00:00 | 60d       |           68192.5 |        72769.2 |    -6.28937 |
| 2026-04-11 00:00:00 | 60d       |           68192.5 |        72769.2 |    -6.28937 |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 18

### Quality Snapshot (Top 15 by missing/staleness)
| feature                          |   missing_pct_recent_30d |   stale_days |
|:---------------------------------|-------------------------:|-------------:|
| corn_fut_oi_change_7d_proxy      |                     6.67 |            2 |
| gold_fut_oi_change_7d_proxy      |                     6.67 |            2 |
| oil_fut_oi_change_7d_proxy       |                     6.67 |            2 |
| wheat_fut_oi_change_7d_proxy     |                     6.67 |            2 |
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
| oil_fut_close            |   6.0303  |     99.9623    |  61.9917     |
| log_oil_fut_close        |   5.37105 |      4.60261   |   4.12269    |
| oil_fut_close_ret30d     |   4.20228 |      0.419833  |   0.0221984  |
| geo_ovx_close            |   3.85833 |     97.4443    |  41.357      |
| rate_fvx_close           |   3.35405 |      3.94707   |   3.69487    |
| gold_fut_close_ret30d    |   3.22981 |     -0.0787384 |   0.0674779  |
| wheat_fut_close          |   2.93538 |    598.633     | 529.854      |
| log_wheat_fut_close      |   2.85668 |      6.39446   |   6.27167    |
| rate_fvx_close_ret30d    |   2.82569 |      0.0836004 |  -0.00297223 |
| corn_fut_close           |   2.80863 |    456.5       | 430.464      |
| rate_tnx_close           |   2.72754 |      4.3249    |   4.11899    |
| log_corn_fut_close       |   2.72723 |      6.12346   |   6.06463    |
| gold_fut_roll_return_20d |   2.62828 |     -0.0715562 |   0.0439102  |
| geo_vix_close            |   2.56962 |     25.7947    |  17.8374     |
| rate_tnx_close_ret30d    |   2.42606 |      0.0610002 |  -0.00522376 |