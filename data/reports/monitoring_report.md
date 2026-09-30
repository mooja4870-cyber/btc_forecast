# 📊 BTC Model Monitoring Report
**Generated:** 2026-09-30 18:42:58
**Run ID:** run_20261001_034257

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
| rate_irx_close          |   4.51815 |      3.90173   |   3.64302    |
| expected_policy_rate_3m |   4.51815 |      3.90173   |   3.64302    |
| rate_irx_close_ret30d   |   3.77869 |      0.0527366 |   0.00542582 |
| expected_policy_rate_6m |   3.65391 |      4.24411   |   3.84475    |
| corn_fut_close          |   3.40545 |    521.442     | 448.969      |
| log_corn_fut_close      |   3.19143 |      6.25643   |   6.10585    |
| rate_fvx_close          |   2.99027 |      4.75767   |   4.14733    |
| rate_tnx_close          |   2.96806 |      4.9461    |   4.46659    |
| wheat_fut_close         |   2.36856 |    718.275     | 623.935      |
| log_wheat_fut_close     |   2.30416 |      6.57657   |   6.43411    |
| rate_irx_close_ret7d    |   2.21827 |      0.0211296 |   0.00134786 |