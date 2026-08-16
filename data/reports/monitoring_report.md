# 📊 BTC Model Monitoring Report
**Generated:** 2026-08-16 14:12:02
**Run ID:** run_20260816_231202

## 🚨 Status Dashboard
❌ Alerts Active
- ⚠️ Recent MAPE (21.8%) exceeds threshold (15.0%)
- ⚠️ Expansion features quality issue count: 4

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | $5759.64 | $13841.82 |
| **RMSE** | $7817.20 | $15809.07 |
| **MAPE** | 8.2% | 21.8% |
| **Count** | 293 | 41 |

## 📉 Recent Error Trend
*(Last 5 predictions)*
| target_date         | horizon   |   predicted_price |   actual_price |   error_pct |
|:--------------------|:----------|------------------:|---------------:|------------:|
| 2026-08-16 00:00:00 | 180d      |           66597.7 |        62995.5 |     5.7182  |
| 2026-08-16 00:00:00 | 180d      |           66741.7 |        62995.5 |     5.94674 |
| 2026-08-16 00:00:00 | 180d      |           66622.9 |        62995.5 |     5.75813 |
| 2026-08-16 00:00:00 | 180d      |           66645.6 |        62995.5 |     5.79427 |
| 2026-08-16 00:00:00 | 180d      |           66541.6 |        62995.5 |     5.62915 |

## 🧩 Expansion Feature Health
- Tracked features: 72
- Drifted features (30d vs prev 180d): 0

### Quality Snapshot (Top 15 by missing/staleness)
| feature                  |   missing_pct_recent_30d |   stale_days |
|:-------------------------|-------------------------:|-------------:|
| rate_fvx_close_ret30d    |                   100    |   1000000000 |
| rate_irx_close_ret30d    |                   100    |   1000000000 |
| rate_fvx_close_ret7d     |                    46.67 |            0 |
| rate_irx_close_ret7d     |                    46.67 |            0 |
| rate_fvx_close_ret1d     |                    26.67 |            0 |
| rate_irx_close_ret1d     |                    26.67 |            0 |
| curve_2y10y_spread_proxy |                    23.33 |            0 |
| expected_policy_rate_3m  |                    23.33 |            0 |
| expected_policy_rate_6m  |                    23.33 |            0 |
| rate_fvx_close           |                    23.33 |            0 |
| rate_irx_close           |                    23.33 |            0 |
| commodity_shock_score    |                     0    |            0 |
| corn_fut_close           |                     0    |            0 |
| corn_fut_close_ret1d     |                     0    |            0 |
| corn_fut_close_ret30d    |                     0    |            0 |