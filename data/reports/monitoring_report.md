# 📊 BTC Model Monitoring Report
**Generated:** 2026-04-19 14:33:42
**Run ID:** run_20260419_233341

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 12 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $3662.58 | $7009.77 |
| **RMSE** | $4594.74 | $7402.01 |
| **MAPE** | 5.1% | 9.3% |
| **Count** | 199 | 51 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-04-19 00:00:00 | 60d       |           71170.7 |        75852.2 |    -6.1719  |
| 2026-04-19 00:00:00 | 60d       |           72459.5 |        75852.2 |    -4.47281 |
| 2026-04-19 00:00:00 | 60d       |           72689.1 |        75852.2 |    -4.17009 |
| 2026-04-19 00:00:00 | 60d       |           73001.6 |        75852.2 |    -3.75817 |
| 2026-04-19 00:00:00 | 60d       |           71009.3 |        75852.2 |    -6.38465 |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 12

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
| feature               |   z_score |   current_mean |     ref_mean |
|:----------------------|----------:|---------------:|-------------:|
| oil_fut_close         |   3.68294 |     98.703     |  63.4797     |
| log_oil_fut_close     |   3.46846 |      4.58885   |   4.14156    |
| rate_fvx_close        |   3.25058 |      3.96417   |   3.70418    |
| gold_fut_close_ret30d |   3.15173 |     -0.0865287 |   0.0627896  |
| rate_tnx_close        |   2.62465 |      4.3357    |   4.12708    |
| rate_fvx_close_ret30d |   2.37642 |      0.0747852 |   0.00149222 |
| corn_fut_close        |   2.14706 |    453.775     | 432.117      |
| wheat_fut_close       |   2.13007 |    593.5       | 533.657      |
| log_corn_fut_close    |   2.11365 |      6.11743   |   6.06843    |
| log_wheat_fut_close   |   2.108   |      6.38586   |   6.27843    |
| geo_ovx_close         |   2.09253 |     87.806     |  44.915      |
| rate_tnx_close_ret30d |   2.08738 |      0.0539973 |  -0.001192   |