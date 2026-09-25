# 📊 BTC Model Monitoring Report
**Generated:** 2026-09-25 18:09:04
**Run ID:** run_20260926_030903

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 11 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $7403.20 | $0.00 |
| **RMSE** | $12689.72 | $0.00 |
| **MAPE** | 10.8% | 0.0% |
| **Count** | 303 | 0 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-08-17 00:00:00 | 180d      |            121097 |        64506.3 |     87.7291 |
| 2026-08-17 00:00:00 | 180d      |            121661 |        64506.3 |     88.6034 |
| 2026-08-17 00:00:00 | 180d      |            120565 |        64506.3 |     86.9048 |
| 2026-08-18 00:00:00 | 180d      |            119377 |        64680.7 |     84.5637 |
| 2026-08-18 00:00:00 | 180d      |            118828 |        64680.7 |     83.7142 |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 11

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
| feature                 |   z_score |   current_mean |     ref_mean |
|:------------------------|----------:|---------------:|-------------:|
| corn_fut_close          |   3.90827 |    519.117     | 446.878      |
| log_corn_fut_close      |   3.61261 |      6.25195   |   6.10143    |
| rate_irx_close          |   3.53418 |      3.8409    |   3.63957    |
| expected_policy_rate_3m |   3.53418 |      3.8409    |   3.63957    |
| wheat_fut_close         |   3.2329  |    727.625     | 619.192      |
| log_wheat_fut_close     |   3.0428  |      6.58942   |   6.42698    |
| expected_policy_rate_6m |   2.89522 |      4.16718   |   3.83301    |
| rate_irx_close_ret30d   |   2.4739  |      0.0349307 |   0.00571308 |
| rate_fvx_close          |   2.40888 |      4.6566    |   4.12317    |
| rate_tnx_close          |   2.35999 |      4.8613    |   4.44735    |
| corn_fut_close_ret30d   |   2.1228  |      0.138567  |   0.0119173  |