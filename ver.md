# Version History

## v1.0.2 (2026-05-28)
- **Label Clean**: Modified yfinance fallback source label format from `Yahoo Finance/yfinance` to `Yahoo Finance` in `src/data_fetcher.py`.

## v1.0.1 (2026-05-28)
- **KOSDAQ Card Integration**: Added KOSDAQ (^KQ11) index card next to KOSPI.
- **Data Fetching Integration**: Created `fetch_naver_kosdaq` in `src/data_fetcher.py` and linked it via Naver Mobile JSON API.
- **Trend Charts**: Integrated KOSDAQ into individual 30-year trend view and the combined 6-asset comparison tool.
