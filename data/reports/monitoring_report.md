# 📊 BTC Model Monitoring Report
**Generated:** 2026-04-08 15:02:40
**Run ID:** run_20260409_000239

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 19 features

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
- Drifted features (30d vs prev 180d): 19

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
| feature                  |   z_score |   current_mean |     ref_mean |
|:-------------------------|----------:|---------------:|-------------:|
| oil_fut_close            |   7.20711 |     99.2533    |  61.5653     |
| log_oil_fut_close        |   6.29551 |      4.5946    |   4.11702    |
| oil_fut_close_ret30d     |   5.34686 |      0.450067  |   0.0156616  |
| geo_ovx_close            |   4.89163 |     99.0403    |  40.1816     |
| oil_fut_roll_return_20d  |   3.39857 |      0.278415  |   0.0150419  |
| wheat_fut_close          |   3.20954 |    600.092     | 528.353      |
| log_wheat_fut_close      |   3.11204 |      6.39697   |   6.26891    |
| rate_fvx_close           |   3.05049 |      3.92557   |   3.69262    |
| gold_fut_close_ret30d    |   3.01743 |     -0.0677811 |   0.0683198  |
| geo_vix_close            |   2.98356 |     26.2483    |  17.6439     |
| geo_ovx_close_ret30d     |   2.7207  |      0.799436  |   0.118639   |
| gold_fut_roll_return_20d |   2.62744 |     -0.0716332 |   0.0445836  |
| corn_fut_close           |   2.62572 |    456.192     | 429.782      |
| rate_fvx_close_ret30d    |   2.56035 |      0.0769435 |  -0.00386282 |
| log_corn_fut_close       |   2.53814 |      6.12275   |   6.063      |