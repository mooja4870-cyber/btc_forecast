# 📊 BTC Model Monitoring Report
**Generated:** 2026-04-07 15:00:55
**Run ID:** run_20260408_000054

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
| oil_fut_close            |   7.78644 |     98.3587    |  61.4083     |
| log_oil_fut_close        |   6.73473 |      4.58595   |   4.11495    |
| oil_fut_close_ret30d     |   5.8773  |      0.454571  |   0.0131619  |
| geo_ovx_close            |   5.37887 |     99.289     |  39.8007     |
| oil_fut_roll_return_20d  |   3.87799 |      0.285848  |   0.0125166  |
| wheat_fut_close          |   3.37429 |    600.508     | 527.739      |
| log_wheat_fut_close      |   3.25904 |      6.39766   |   6.2678     |
| geo_vix_close            |   3.22239 |     26.4257    |  17.5641     |
| geo_ovx_close_ret30d     |   2.94472 |      0.831325  |   0.113882   |
| rate_fvx_close           |   2.9248  |      3.9167    |   3.69181    |
| gold_fut_close_ret30d    |   2.91724 |     -0.0630133 |   0.068431   |
| corn_fut_close           |   2.59183 |    455.958     | 429.528      |
| gold_fut_roll_return_20d |   2.56852 |     -0.068928  |   0.0449447  |
| log_corn_fut_close       |   2.50341 |      6.12224   |   6.0624     |
| rate_fvx_close_ret30d    |   2.45742 |      0.0742456 |  -0.00418811 |