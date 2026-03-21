# 📊 BTC Model Monitoring Report
**Generated:** 2026-03-21 14:17:31
**Run ID:** run_20260321_231730

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Expansion feature drift detected: 14 features

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $2534.88 | $2767.76 |
| **RMSE** | $3105.23 | $3290.44 |
| **MAPE** | 3.6% | 4.0% |
| **Count** | 150 | 132 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-03-20 00:00:00 | 30d       |           65625.2 |        70522.6 |    -6.94435 |
| 2026-03-20 00:00:00 | 30d       |           65697.1 |        70522.6 |    -6.84244 |
| 2026-03-20 00:00:00 | 30d       |           66811.2 |        70522.6 |    -5.26275 |
| 2026-03-21 00:00:00 | 30d       |           66381.1 |        70971.2 |    -6.46756 |
| 2026-03-21 00:00:00 | 30d       |           66685.3 |        70971.2 |    -6.03896 |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 14

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
| feature                         |   z_score |   current_mean |         ref_mean |
|:--------------------------------|----------:|---------------:|-----------------:|
| oil_fut_close                   |   7.96159 |      81.2857   |     60.6782      |
| geo_ovx_close                   |   7.4278  |      83.8247   |     36.7129      |
| log_oil_fut_close               |   6.56794 |       4.38435  |      4.10468     |
| oil_fut_roll_return_20d         |   5.74899 |       0.257417 |      0.000821009 |
| oil_fut_close_ret30d            |   5.48307 |       0.288746 |     -0.00362408  |
| oil_fut_front_next_spread_proxy |   4.99792 |       0.134487 |      0.00025847  |
| wheat_fut_close                 |   4.97788 |     589.225    |    521.022       |
| log_wheat_fut_close             |   4.68401 |       6.37836  |      6.25545     |
| oil_fut_close_ret7d             |   3.3814  |       0.109421 |      0.00052309  |
| geo_ovx_close_ret30d            |   2.86479 |       0.748272 |      0.0766604   |
| wheat_fut_close_ret30d          |   2.61454 |       0.105298 |      0.0061135   |
| oil_fut_volume                  |   2.49406 |  521801        | 257121           |
| geo_vix_close                   |   2.39964 |      23.025    |     17.0303      |
| geo_ovx_close_ret7d             |   2.16347 |       0.257882 |      0.014425    |