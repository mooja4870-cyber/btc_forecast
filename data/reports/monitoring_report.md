# 📊 BTC Model Monitoring Report
**Generated:** 2026-04-02 14:47:20
**Run ID:** run_20260402_234719

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 19 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $2504.75 | $4110.76 |
| **RMSE** | $3072.96 | $4318.85 |
| **MAPE** | 3.6% | 5.7% |
| **Count** | 150 | 59 |

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
| feature                         |   z_score |   current_mean |        ref_mean |
|:--------------------------------|----------:|---------------:|----------------:|
| oil_fut_close                   |  11.2467  |     93.4927    |     60.8489     |
| log_oil_fut_close               |   9.05391 |      4.53471   |      4.10728    |
| oil_fut_close_ret30d            |   7.86457 |      0.442244  |      0.00421315 |
| geo_ovx_close                   |   7.04649 |     97.3277    |     38.4936     |
| oil_fut_roll_return_20d         |   6.70627 |      0.311685  |      0.00382048 |
| oil_fut_front_next_spread_proxy |   4.66028 |      0.135133  |      0.00205751 |
| wheat_fut_close                 |   3.87198 |    599.125     |    525.331      |
| log_wheat_fut_close             |   3.70093 |      6.39524   |      6.26339    |
| geo_vix_close                   |   3.62329 |     26.405     |     17.3296     |
| geo_ovx_close_ret30d            |   3.40653 |      0.889353  |      0.0980998  |
| oil_fut_close_ret7d             |   2.9885  |      0.103529  |      0.00210804 |
| gold_fut_close_ret30d           |   2.35213 |     -0.0373048 |      0.0682417  |
| rate_fvx_close                  |   2.32594 |      3.87107   |      3.69019    |
| oil_fut_volume                  |   2.30967 | 533879         | 273208          |
| corn_fut_close                  |   2.29309 |    453.825     |    428.407      |