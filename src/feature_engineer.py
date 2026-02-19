"""
BTC Price Prediction — Feature Engineering Module
===================================================
Transforms raw merged data into model-ready features:
- Lag features, moving averages, rolling stats
- Return rates, RSI, MACD-like signals
- Cross-asset ratios and rolling correlations
- Multi-horizon target variables (1d, 7d, 30d, 60d, 90d, 180d, 365d)
"""

import numpy as np
import pandas as pd
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import cfg

# Default params from config
_RAW_HORIZONS = cfg.features_config.get("horizons", [1, 7, 30, 60, 90, 180, 365])
DEFAULT_HORIZONS = sorted(
    {
        int(h)
        for h in _RAW_HORIZONS
        if str(h).strip().isdigit() and int(h) > 0
    }
)
if not DEFAULT_HORIZONS:
    DEFAULT_HORIZONS = [1, 7, 30, 60, 90, 180, 365]
HORIZONS = DEFAULT_HORIZONS


def _third_friday(year: int, month: int) -> pd.Timestamp:
    """Return third Friday of year-month."""
    first = pd.Timestamp(year=year, month=month, day=1)
    first_friday = first + pd.offsets.Week(weekday=4)
    return first_friday + pd.Timedelta(days=14)


def _next_month(year: int, month: int):
    if month == 12:
        return year + 1, 1
    return year, month + 1


def _days_to_monthly_expiry(index: pd.DatetimeIndex) -> pd.Series:
    """
    Approximate futures expiry timing with monthly third-Friday schedule.
    Used as a fail-forward proxy when exact contract calendars are unavailable.
    """
    vals = []
    for d in index:
        expiry = _third_friday(d.year, d.month)
        if d > expiry:
            ny, nm = _next_month(d.year, d.month)
            expiry = _third_friday(ny, nm)
        vals.append((expiry - d).days)
    return pd.Series(vals, index=index)


def apply_asof_release_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply as-of lag policy to avoid using not-yet-available external values.
    """
    asof_cfg = cfg.features_asof_config
    if not asof_cfg.get("enforce_no_future_data", True):
        return df

    out = df.copy()
    default_lag = int(asof_cfg.get("default_release_lag_days", 1))
    futures_lag = int(cfg.futures_feature_config.get("release_lag_days", default_lag))
    rates_lag = int(cfg.rates_expectation_config.get("release_lag_days", default_lag))
    geo_lag = int(cfg.geopolitical_feature_config.get("release_lag_days", default_lag))

    # FRED/macroeconomic series are typically reported with lag.
    macro_cols = [
        "fed_rate", "cpi", "m2", "unemployment", "gdp", "treasury_10y",
        "hashrate", "fear_greed",
    ]
    for col in macro_cols:
        if col in out.columns:
            out[col] = out[col].shift(default_lag)

    for col in [c for c in out.columns if c.startswith("rate_") or c in ["days_to_fomc", "is_fomc_week"]]:
        out[col] = out[col].shift(rates_lag)

    for col in [c for c in out.columns if c.startswith("geo_")]:
        out[col] = out[col].shift(geo_lag)

    for col in [c for c in out.columns if "_fut_" in c]:
        out[col] = out[col].shift(futures_lag)

    return out


def add_lag_features(df: pd.DataFrame, col: str = "btc_close",
                     lags: list = None) -> pd.DataFrame:
    """Add lagged values of a given column."""
    if lags is None:
        lags = cfg.features_config.get("lags", [1, 7, 14, 30, 60, 90])
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_moving_averages(df: pd.DataFrame, col: str = "btc_close",
                        windows: list = None) -> pd.DataFrame:
    """Add simple moving averages."""
    if windows is None:
        windows = cfg.features_config.get("rolling_windows", [7, 14, 30, 50, 90, 200])
    for w in windows:
        df[f"{col}_ma{w}"] = df[col].rolling(w).mean()
    return df


def add_return_features(df: pd.DataFrame, cols: list = None,
                        periods: list = None) -> pd.DataFrame:
    """Add percentage return features for specified columns."""
    if cols is None:
        cols = [c for c in df.columns if c.endswith("_close")]
    if periods is None:
        periods = [1, 7, 30]

    for col in cols:
        for p in periods:
            df[f"{col}_ret{p}d"] = df[col].pct_change(p)
    return df


def add_volatility(df: pd.DataFrame, col: str = "btc_close",
                   windows: list = None) -> pd.DataFrame:
    """Add rolling standard deviation (volatility) of log returns."""
    if windows is None:
        windows = [7, 30, 60]
    log_ret = np.log(df[col] / df[col].shift(1))
    for w in windows:
        df[f"btc_vol_{w}d"] = log_ret.rolling(w).std()
    return df


def add_rsi(df: pd.DataFrame, col: str = "btc_close",
            period: int = 14) -> pd.DataFrame:
    """Add Relative Strength Index."""
    period = cfg.features_config.get("rsi_period", 14)
    delta = df[col].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, col: str = "btc_close") -> pd.DataFrame:
    """Add MACD features."""
    fast = cfg.features_config.get("macd_fast", 12)
    slow = cfg.features_config.get("macd_slow", 26)
    signal = cfg.features_config.get("macd_signal", 9)
    
    ema_fast = df[col].ewm(span=fast).mean()
    ema_slow = df[col].ewm(span=slow).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, col: str = "btc_close",
                        window: int = 20) -> pd.DataFrame:
    """Add Bollinger Bands."""
    window = cfg.features_config.get("bollinger_window", 20)
    ma = df[col].rolling(window).mean()
    std = df[col].rolling(window).std()
    df["bb_upper"] = ma + 2 * std
    df["bb_lower"] = ma - 2 * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / ma
    df["bb_position"] = (df[col] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    return df


def add_cross_asset_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratios and rolling correlations between BTC and other assets."""
    btc = df.get("btc_close")
    if btc is None:
        return df

    for other in ["gold_close", "sp500_close", "nasdaq_close", "oil_close"]:
        if other in df.columns:
            label = other.replace("_close", "")
            # Ratio
            df[f"btc_{label}_ratio"] = btc / df[other]
            # 30-day rolling correlation
            df[f"btc_{label}_corr30"] = (
                btc.rolling(30).corr(df[other])
            )
    return df


def add_futures_term_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add futures-expiry and curve proxy features.
    """
    cfg_fut = cfg.futures_feature_config
    if not cfg_fut.get("enabled", False):
        return df

    assets = cfg_fut.get("assets", ["oil", "gold", "corn", "wheat"])
    days_to_expiry = _days_to_monthly_expiry(df.index)

    for asset in assets:
        fut_close = f"{asset}_fut_close"
        spot_close = f"{asset}_close"
        fut_volume = f"{asset}_fut_volume"
        spot_volume = f"{asset}_volume"

        px_col = fut_close if fut_close in df.columns else spot_close
        vol_col = fut_volume if fut_volume in df.columns else spot_volume
        if px_col not in df.columns:
            continue

        base = f"{asset}_fut"
        px = df[px_col]

        if cfg_fut.get("include_days_to_expiry", True):
            df[f"{base}_days_to_expiry"] = days_to_expiry

        if cfg_fut.get("include_expiry_week_dummy", True):
            df[f"{base}_expiry_week"] = ((days_to_expiry >= 0) & (days_to_expiry <= 7)).astype(int)

        if cfg_fut.get("include_front_next_spread", True):
            ma_21 = px.rolling(21).mean().replace(0, np.nan)
            df[f"{base}_front_next_spread_proxy"] = (px - ma_21) / ma_21

        if cfg_fut.get("include_roll_return", True):
            df[f"{base}_roll_return_20d"] = px.pct_change(20)

        if cfg_fut.get("include_open_interest_change", True):
            if vol_col in df.columns:
                df[f"{base}_oi_change_7d_proxy"] = df[vol_col].pct_change(7)
            else:
                df[f"{base}_oi_change_7d_proxy"] = np.nan

    return df


def add_rates_expectation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add policy-rate expectation proxies.
    """
    cfg_rates = cfg.rates_expectation_config
    if not cfg_rates.get("enabled", False):
        return df

    short = None
    long = None
    mid = None

    if "rate_irx_close" in df.columns:
        short = df["rate_irx_close"]
    elif "fed_rate" in df.columns:
        short = df["fed_rate"]

    if "rate_fvx_close" in df.columns:
        mid = df["rate_fvx_close"]

    if "rate_tnx_close" in df.columns:
        long = df["rate_tnx_close"]
    elif "treasury_10y" in df.columns:
        long = df["treasury_10y"]

    if cfg_rates.get("include_days_to_fomc", True):
        if "days_to_fomc" not in df.columns:
            df["days_to_fomc"] = _days_to_monthly_expiry(df.index)
        if "is_fomc_week" not in df.columns:
            df["is_fomc_week"] = ((df["days_to_fomc"] >= 0) & (df["days_to_fomc"] <= 7)).astype(int)

    if cfg_rates.get("include_expected_policy_rate_3m", True):
        if short is not None:
            df["expected_policy_rate_3m"] = short
        else:
            df["expected_policy_rate_3m"] = np.nan

    if cfg_rates.get("include_expected_policy_rate_6m", True):
        if short is not None and mid is not None:
            df["expected_policy_rate_6m"] = 0.6 * short + 0.4 * mid
        elif short is not None:
            df["expected_policy_rate_6m"] = short.rolling(21).mean()
        else:
            df["expected_policy_rate_6m"] = np.nan

    if cfg_rates.get("include_curve_2y10y_spread", True):
        if short is not None and long is not None:
            df["curve_2y10y_spread_proxy"] = long - short
        else:
            df["curve_2y10y_spread_proxy"] = np.nan

    return df


def add_geopolitical_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add continuous geopolitical risk proxies.
    """
    cfg_geo = cfg.geopolitical_feature_config
    if not cfg_geo.get("enabled", False):
        return df

    components = []
    if "geo_vix_close" in df.columns:
        components.append(df["geo_vix_close"])
    if "geo_ovx_close" in df.columns:
        components.append(df["geo_ovx_close"])

    if components:
        geo = pd.concat(components, axis=1).mean(axis=1)
        geo_z = (geo - geo.rolling(60).mean()) / geo.rolling(60).std().replace(0, np.nan)
        geo_idx = geo_z
    else:
        geo_idx = pd.Series(np.nan, index=df.index)

    if cfg_geo.get("include_geo_risk_index", True):
        df["geo_risk_index"] = geo_idx

    event_cols = [c for c in df.columns if c.startswith("event_")]
    if event_cols:
        event_intensity = df[event_cols].rolling(7, min_periods=1).mean().mean(axis=1)
    else:
        event_intensity = pd.Series(0.0, index=df.index)

    if cfg_geo.get("include_war_intensity_score", True):
        df["war_intensity_score"] = 0.6 * geo_idx.fillna(0) + 0.4 * event_intensity.fillna(0)

    if cfg_geo.get("include_commodity_shock_score", True):
        oil_ret = df["oil_close"].pct_change(1).abs() if "oil_close" in df.columns else pd.Series(np.nan, index=df.index)
        gold_ret = df["gold_close"].pct_change(1).abs() if "gold_close" in df.columns else pd.Series(np.nan, index=df.index)
        df["commodity_shock_score"] = oil_ret.fillna(0) + 0.5 * gold_ret.fillna(0)

    return df


def add_multi_horizon_targets(df: pd.DataFrame,
                                horizons: list = None) -> pd.DataFrame:
    """
    Create target variables for multiple horizons.
    Each target: log(price[t+h] / price[t]) for h in horizons.
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS
    
    for h in horizons:
        future_price = df["btc_close"].shift(-h)
        df[f"target_log_return_{h}d"] = np.log(future_price / df["btc_close"])
        df[f"target_direction_{h}d"] = (df[f"target_log_return_{h}d"] > 0).astype(int)
        df[f"target_future_price_{h}d"] = future_price
    
    # Keep backward compatibility if 30d is in horizons
    if 30 in horizons:
        df["target_log_return"] = df.get("target_log_return_30d")
        df["target_direction"] = df.get("target_direction_30d")
        df["target_future_price"] = df.get("target_future_price_30d")
    
    return df


def engineer_features(df: pd.DataFrame, horizons: list = None, save_path: str = None) -> pd.DataFrame:
    """
    Master function: apply all feature engineering steps.
    Returns a copy with all features added.
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS
    
    df = df.copy()

    print("Feature Engineering:")
    print("  ✓ As-of lag policy")
    df = apply_asof_release_lags(df)

    print("  ✓ Lag features")
    df = add_lag_features(df)

    print("  ✓ Moving averages")
    df = add_moving_averages(df)

    print("  ✓ Return features")
    df = add_return_features(df)

    print("  ✓ Volatility")
    df = add_volatility(df)

    print("  ✓ RSI")
    df = add_rsi(df)

    print("  ✓ MACD")
    df = add_macd(df)

    print("  ✓ Bollinger Bands")
    df = add_bollinger_bands(df)

    print("  ✓ Cross-asset ratios & correlations")
    df = add_cross_asset_ratios(df)

    if cfg.feature_enabled("futures_term_structure") or cfg.futures_feature_config.get("enabled", False):
        print("  ✓ Futures expiry / curve proxies")
        df = add_futures_term_structure_features(df)

    if cfg.feature_enabled("rates_expectation") or cfg.rates_expectation_config.get("enabled", False):
        print("  ✓ Rates expectation proxies")
        df = add_rates_expectation_features(df)

    if cfg.feature_enabled("geopolitical_risk") or cfg.geopolitical_feature_config.get("enabled", False):
        print("  ✓ Geopolitical risk proxies")
        df = add_geopolitical_risk_features(df)

    print(f"  ✓ Multi-horizon targets ({', '.join(str(h)+'d' for h in horizons)})")
    df = add_multi_horizon_targets(df, horizons=horizons)

    # ── Relative price positions (BTC price vs MAs) — normalized ──
    if "btc_close" in df.columns:
        for ma in [30, 90, 200]:
            ma_col = f"btc_close_ma{ma}"
            if ma_col in df.columns:
                df[f"btc_above_ma{ma}"] = (df["btc_close"] > df[ma_col]).astype(int)
                df[f"btc_ma{ma}_pct"] = (df["btc_close"] - df[ma_col]) / df[ma_col]

    # ── Log-transform price columns for stationarity ──
    for col in [c for c in df.columns if c.endswith("_close")]:
        df[f"log_{col}"] = np.log(df[col].clip(lower=0.01))

    # Normalize pathological values from ratio/division features.
    df = df.replace([np.inf, -np.inf], np.nan)

    # Summary
    n_features = df.shape[1]
    print(f"\n✅ Total features: {n_features}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)
        print(f"\n💾 Saved to {save_path}")

    return df


# ================================================================
if __name__ == "__main__":
    # Test run
    input_path = os.path.join(cfg.processed_dir, "merged_dataset.csv")
    if os.path.exists(input_path):
        df = pd.read_csv(input_path, index_col=0, parse_dates=True)
        featured = engineer_features(df)
        print(f"Shape: {featured.shape}")
    else:
        print(f"✗ Input file not found: {input_path}")
