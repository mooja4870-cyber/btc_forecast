"""
Scenario bands — honest "range, not point" forecasting.
================================================================
We proved (multi-seed hold-out) that directional/drift prediction has no
statistically significant skill at any horizon. Volatility, however, is
persistent and estimable. So instead of pretending to predict a single future
price, we present a *cone of plausible outcomes*:

- Central path = historical MEDIAN h-day return (a historical tendency, NOT a
  prediction).
- Bands = empirical percentiles of historical h-day log-returns (captures BTC's
  fat tails / skew), with the spread scaled by the CURRENT volatility regime
  (recent realized vol / long-run vol) so the cone widens in turbulent markets
  and narrows in calm ones.

This is deliberately model-free: no transformer, no overfitting, no false
confidence — just the realized distribution conditioned on today's volatility.
"""
import numpy as np
import pandas as pd

DEFAULT_HORIZONS = [0, 7, 15, 30, 60, 90, 120, 180, 270, 365]
DEFAULT_PERCENTILES = (5, 25, 50, 75, 95)


def compute_scenario_bands(
    price_series: pd.Series,
    horizons=None,
    current_price: float = None,
    current_date: pd.Timestamp = None,
    vol_window: int = 30,
    percentiles=DEFAULT_PERCENTILES,
    min_samples: int = 60,
):
    """Build a scenario-band DataFrame (one row per horizon, price per percentile).

    Returns (bands_df, stats_dict).
    bands_df columns: horizon_days, target_date, p5, p25, p50, p75, p95 (prices).
    stats_dict: realized volatility info used for the band scaling.
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    s = pd.to_numeric(price_series, errors="coerce").dropna()
    s = s[s > 0]
    if s.empty:
        return pd.DataFrame(), {}

    log_p = np.log(s)
    daily_ret = log_p.diff().dropna()
    sigma_hist = float(daily_ret.std()) if len(daily_ret) > 2 else 0.0
    sigma_recent = (
        float(daily_ret.tail(vol_window).std())
        if len(daily_ret) >= max(5, vol_window // 2)
        else sigma_hist
    )
    vol_ratio = (sigma_recent / sigma_hist) if sigma_hist > 0 else 1.0

    if current_price is None:
        current_price = float(s.iloc[-1])
    if current_date is None:
        current_date = s.index[-1]
    current_date = pd.Timestamp(current_date)

    rows = []
    for h in horizons:
        h = int(h)
        if h <= 0:
            row = {"horizon_days": 0, "target_date": current_date}
            for p in percentiles:
                row[f"p{p}"] = round(float(current_price), 2)
            rows.append(row)
            continue

        h_ret = (log_p.shift(-h) - log_p).dropna()
        if len(h_ret) < min_samples:
            continue
        med = float(h_ret.median())

        row = {"horizon_days": h, "target_date": current_date + pd.Timedelta(days=h)}
        for p in percentiles:
            q = float(np.percentile(h_ret, p))
            # Scale the spread around the median by the current vol regime.
            adj = med + (q - med) * vol_ratio
            row[f"p{p}"] = round(float(current_price) * float(np.exp(adj)), 2)
        rows.append(row)

    bands = pd.DataFrame(rows).sort_values("horizon_days").reset_index(drop=True)

    stats = {
        "sigma_daily_hist": sigma_hist,
        "sigma_daily_recent": sigma_recent,
        "vol_ratio": vol_ratio,
        "ann_vol_recent_pct": sigma_recent * np.sqrt(365) * 100.0,
        "ann_vol_hist_pct": sigma_hist * np.sqrt(365) * 100.0,
        "n_history_days": int(len(s)),
        "current_price": float(current_price),
        "current_date": str(current_date.date()),
    }
    return bands, stats


def band_coverage_note(percentiles=DEFAULT_PERCENTILES) -> str:
    lo, hi = min(percentiles), max(percentiles)
    return (
        f"밴드는 역사적 수익률의 p{lo}~p{hi} 분위수를 현재 변동성으로 조정한 범위입니다 "
        f"(변동성이 역사적 평균과 비슷할 때 실제 가격의 약 {hi - lo}%가 이 안에 분포)."
    )


if __name__ == "__main__":
    import os
    from config import PROCESSED_DIR
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "merged_dataset.csv"),
                     index_col=0, parse_dates=True).sort_index()
    bands, stats = compute_scenario_bands(df["btc_close"])
    print("Volatility stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("\nScenario bands (price):")
    print(bands.to_string(index=False))
