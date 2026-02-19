import os
import re
import html as html_lib
from datetime import datetime, timedelta

import requests
import yfinance as yf

try:
    import FinanceDataReader as fdr
except Exception:
    fdr = None

try:
    import ccxt
except Exception:
    ccxt = None


ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()

INVESTING_URLS = {
    "GC=F": "https://www.investing.com/commodities/gold",
    "^GSPC": "https://www.investing.com/indices/us-spx-500",
    "KRW=X": "https://www.investing.com/currencies/usd-krw",
}

SHINHAN_ENDPOINT_URL = "https://bank.shinhan.com/serviceEndpoint/httpDigital"
SHINHAN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/json; charset=UTF-8",
    "Origin": "https://bank.shinhan.com",
    "Referer": "https://bank.shinhan.com/rib/easy/index.jsp#020707010000",
}

WOORI_GOLDBANK_URL = "https://spot.wooribank.com/pot/jcc?withyou=POGLD0005&__ID=c007226"
WOORI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://spot.wooribank.com/pot/Dream?withyou=POGLD0005",
}

NAVER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.naver.com/",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}


def _to_float(value):
    try:
        return float(str(value).replace(",", "").replace("%", "").strip())
    except Exception:
        return None


def _fetch_shinhan_goldrush_rows_for_date(date_yyyymmdd):
    """Fetch Shinhan GoldRush daily rows for a specific date (YYYYMMDD)."""
    payload = {
        "dataHeader": {
            "trxCd": "RSRDE0500A01",
            "subChannel": "02",
            "channelGbn": "D0",
            "language": "ko",
        },
        "dataBody": {
            "ACTION_TYPE": "DAILY",
            "P_FROM_DATE": date_yyyymmdd,
            "P_TO_DATE": date_yyyymmdd,
            "ricInptRootInfo": {
                "serviceType": "TG",
                "serviceCode": "TGS1001",
                "isRule": "Y",
                "webUri": "/rib/gold/GS09/GS09010RM00.xml",
            },
        },
    }

    try:
        resp = requests.post(
            SHINHAN_ENDPOINT_URL,
            headers=SHINHAN_HEADERS,
            json=payload,
            timeout=8,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        body = data.get("dataBody", {})
        rows = body.get("OK_EXCHANGE_RATE", [])
        return rows if isinstance(rows, list) else []
    except Exception:
        return []


def fetch_shinhan_goldrush_krw(max_lookback_days=14):
    """Primary KRW gold source: Shinhan Bank GoldRush published KRW-per-gram prices."""
    today = datetime.now().date()

    latest_date = None
    latest_rows = []
    for day_offset in range(max_lookback_days + 1):
        d = (today - timedelta(days=day_offset)).strftime("%Y%m%d")
        rows = _fetch_shinhan_goldrush_rows_for_date(d)
        if rows:
            latest_date = d
            latest_rows = rows
            break

    if not latest_rows:
        return None, None, None

    latest_row = latest_rows[-1]
    current = _to_float(latest_row.get("XAU_RATE1"))
    if current is None:
        current = _to_float(latest_row.get("XAU_RATE"))
    if current is None:
        return None, None, None

    change = 0.0
    latest_day_offset = (today - datetime.strptime(latest_date, "%Y%m%d").date()).days
    for day_offset in range(latest_day_offset + 1, max_lookback_days + 1):
        d = (today - timedelta(days=day_offset)).strftime("%Y%m%d")
        prev_rows = _fetch_shinhan_goldrush_rows_for_date(d)
        if not prev_rows:
            continue
        prev_row = prev_rows[-1]
        prev = _to_float(prev_row.get("XAU_RATE1"))
        if prev is None:
            prev = _to_float(prev_row.get("XAU_RATE"))
        if prev not in (None, 0):
            change = (current - prev) / prev * 100
        break

    date_display = latest_row.get("D_DATE", latest_date)
    time_display = latest_row.get("D_TIME", "")
    source = f"실시간 (신한은행 골드리슈 고시가격 {date_display} {time_display})".strip()
    return current, change, source


def fetch_shinhan_silverrush_krw(max_lookback_days=14):
    """Primary KRW silver source: Shinhan Bank SilverRush (신한실버리슈실버테크)."""
    today = datetime.now().date()

    latest_date = None
    latest_rows = []
    for day_offset in range(max_lookback_days + 1):
        d = (today - timedelta(days=day_offset)).strftime("%Y%m%d")
        rows = _fetch_shinhan_goldrush_rows_for_date(d)
        if rows:
            latest_date = d
            latest_rows = rows
            break

    if not latest_rows:
        return None, None, None

    latest_row = latest_rows[-1]
    current = _to_float(latest_row.get("XAG_RATE1"))
    if current is None:
        current = _to_float(latest_row.get("XAG_RATE"))
    if current is None:
        return None, None, None

    change = 0.0
    latest_day_offset = (today - datetime.strptime(latest_date, "%Y%m%d").date()).days
    for day_offset in range(latest_day_offset + 1, max_lookback_days + 1):
        d = (today - timedelta(days=day_offset)).strftime("%Y%m%d")
        prev_rows = _fetch_shinhan_goldrush_rows_for_date(d)
        if not prev_rows:
            continue
        prev_row = prev_rows[-1]
        prev = _to_float(prev_row.get("XAG_RATE1"))
        if prev is None:
            prev = _to_float(prev_row.get("XAG_RATE"))
        if prev not in (None, 0):
            change = (current - prev) / prev * 100
        break

    date_display = latest_row.get("D_DATE", latest_date)
    time_display = latest_row.get("D_TIME", "")
    source = f"실시간 (신한은행 신한실버리슈실버테크 {date_display} {time_display})".strip()
    return current, change, source


def _clean_html_text(value):
    value = re.sub(r"<br\\s*/?>", " ", str(value), flags=re.I)
    value = re.sub(r"<[^>]+>", "", value)
    return html_lib.unescape(value).replace("\xa0", " ").strip()


def _fetch_woori_goldbank_rows_for_date(date_yyyymmdd):
    payload = {
        "BAS_DT": date_yyyymmdd,
        "NAT_CODE": "XAU",
        "BAS_SDT": "",
        "BAS_EDT": "",
        "START_DATE1": date_yyyymmdd,
        "START_DATE1Y": date_yyyymmdd[:4],
        "START_DATE1M": date_yyyymmdd[4:6],
        "START_DATE1D": date_yyyymmdd[6:8],
    }

    try:
        resp = requests.post(
            WOORI_GOLDBANK_URL,
            headers=WOORI_HEADERS,
            data=payload,
            timeout=8,
        )
        if resp.status_code != 200:
            return [], None
        page = resp.text
        if "당일의 금가격이 고시되지 않았습니다" in page:
            return [], None

        date_display = None
        m_date = re.search(r"조회기준일\\s*:</dt>\\s*<dd[^>]*>(.*?)</dd>", page, re.S)
        if m_date:
            date_display = _clean_html_text(m_date.group(1))

        rows = []
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S | re.I):
            cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S | re.I)
            if len(cells) < 6:
                continue
            parsed = [_clean_html_text(c) for c in cells[:6]]
            rows.append(
                {
                    "apply_time": parsed[0],
                    "base_price": _to_float(parsed[1]),
                    "buy_price": _to_float(parsed[2]),
                    "sell_price": _to_float(parsed[3]),
                    "xau_usd_oz": _to_float(parsed[4]),
                    "usd_krw": _to_float(parsed[5]),
                }
            )

        return rows, date_display
    except Exception:
        return [], None


def fetch_woori_goldbank_krw(max_lookback_days=14):
    """Primary KRW gold source: Woori Bank Gold Banking (기준가격, KRW)."""
    today = datetime.now().date()

    latest_date = None
    latest_rows = []
    latest_date_display = None
    for day_offset in range(max_lookback_days + 1):
        d = (today - timedelta(days=day_offset)).strftime("%Y%m%d")
        rows, date_display = _fetch_woori_goldbank_rows_for_date(d)
        if rows:
            latest_date = d
            latest_rows = rows
            latest_date_display = date_display
            break

    if not latest_rows:
        return None, None, None

    latest_row = latest_rows[0]
    current = latest_row.get("base_price")
    if current is None:
        return None, None, None

    change = 0.0
    latest_day_offset = (today - datetime.strptime(latest_date, "%Y%m%d").date()).days
    for day_offset in range(latest_day_offset + 1, max_lookback_days + 1):
        d = (today - timedelta(days=day_offset)).strftime("%Y%m%d")
        prev_rows, _ = _fetch_woori_goldbank_rows_for_date(d)
        if not prev_rows:
            continue
        prev = prev_rows[0].get("base_price")
        if prev not in (None, 0):
            change = (current - prev) / prev * 100
        break

    date_display = latest_date_display or f"{latest_date[:4]}.{latest_date[4:6]}.{latest_date[6:]}"
    time_display = latest_row.get("apply_time", "")
    source = f"실시간 (우리은행 골드뱅크가격조회 기준가격 {date_display} {time_display})".strip()
    return current, change, source


def fetch_exchangerate_api_usd_krw():
    """Primary KRW/USD source: ExchangeRate-API (USD->KRW)."""
    endpoints = [
        "https://open.er-api.com/v6/latest/USD",
        "https://api.exchangerate-api.com/v4/latest/USD",
    ]
    for url in endpoints:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code != 200:
                continue
            data = resp.json()
            rates = data.get("rates", {})
            krw = _to_float(rates.get("KRW"))
            if krw is not None:
                return krw, 0.0, "실시간 (ExchangeRate-API)"
        except Exception:
            continue
    return None, None, None


def fetch_naver_marketindex_usd_krw():
    """Primary KRW/USD source: Naver Finance marketindex '매매기준율' (USD/KRW)."""
    try:
        url = "https://finance.naver.com/marketindex/?tabSel=exchange#tab_section"
        resp = requests.get(url, headers=NAVER_HEADERS, timeout=6)
        if resp.status_code != 200:
            return None, None, None

        page = resp.text

        # USD/KRW card block on marketindex page.
        block_match = re.search(
            r'<a href="/marketindex/exchangeDetail\.naver\?marketindexCd=FX_USDKRW".*?</a>',
            page,
            re.S | re.I,
        )
        block = block_match.group(0) if block_match else ""

        value_match = re.search(r'<span class="value">\s*([0-9,\.]+)\s*</span>', block, re.I)
        current = _to_float(value_match.group(1)) if value_match else None
        if current is None:
            return None, None, None

        change_pct = 0.0
        change_match = re.search(r'<span class="change">\s*([0-9,\.]+)\s*</span>', block, re.I)
        if change_match:
            delta = _to_float(change_match.group(1))
            if delta is not None:
                is_up = "point_up" in block
                is_down = "point_dn" in block or "point_down" in block
                signed_delta = delta if is_up else (-delta if is_down else 0.0)
                prev = current - signed_delta
                if prev not in (None, 0):
                    change_pct = (signed_delta / prev) * 100.0

        return current, change_pct, "실시간 (네이버 금융 매매기준율)"
    except Exception:
        return None, None, None


def fetch_google_finance_usd_krw():
    """Secondary KRW/USD source: Google Finance page parsing."""
    try:
        url = "https://www.google.com/finance/quote/USD-KRW"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code != 200:
            return None, None, None
        html = r.text

        # Common Google Finance price node
        m = re.search(r'class="YMlKec fxKbKc">\s*([0-9,\.]+)\s*<', html)
        if not m:
            m = re.search(r'USD\s*/\s*KRW.*?([0-9,\.]{3,})', html, re.S)
        if not m:
            return None, None, None

        price = _to_float(m.group(1))
        if price is None:
            return None, None, None

        # Change percent if present
        change = 0.0
        mc = re.search(r'([+\-]?[0-9]+\.?[0-9]*)%\s*<', html)
        if mc:
            parsed = _to_float(mc.group(1))
            if parsed is not None:
                change = parsed

        return price, change, "실시간 (Google Finance)"
    except Exception:
        return None, None, None


def fetch_financedatareader_usd_krw():
    """Tertiary KRW/USD source: FinanceDataReader."""
    if fdr is None:
        return None, None, None

    symbols = ["USD/KRW", "USDKRW"]
    for sym in symbols:
        try:
            df = fdr.DataReader(sym)
            if df is None or df.empty or "Close" not in df.columns:
                continue
            close = df["Close"].dropna()
            if close.empty:
                continue
            current = _to_float(close.iloc[-1])
            prev = _to_float(close.iloc[-2]) if len(close) > 1 else current
            if current is None:
                continue
            if prev in (None, 0):
                change = 0.0
            else:
                change = (current - prev) / prev * 100
            return current, change, "실시간 (FinanceDataReader)"
        except Exception:
            continue

    return None, None, None


def fetch_btc_binance():
    """Fetch BTC price from Binance using CCXT."""
    if ccxt is None:
        return None, None, None
    try:
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker("BTC/USDT")
        current = ticker["last"]
        change = ticker["percentage"]
        return current, change, "실시간 (Binance)"
    except Exception as e:
        print(f"Binance fetch failed: {e}")
        return None, None, None


def fetch_btc_upbit_krw():
    """Fetch BTC KRW price directly from Upbit public API."""
    try:
        url = "https://api.upbit.com/v1/ticker"
        resp = requests.get(url, params={"markets": "KRW-BTC"}, timeout=5)
        if resp.status_code != 200:
            return None, None, None

        data = resp.json()
        if not isinstance(data, list) or not data:
            return None, None, None

        row = data[0]
        current = _to_float(row.get("trade_price"))
        change_rate = _to_float(row.get("signed_change_rate"))
        if current is None:
            return None, None, None
        change = (change_rate * 100.0) if change_rate is not None else 0.0
        return current, change, "실시간 (Upbit KRW)"
    except Exception as e:
        print(f"Upbit fetch failed: {e}")
        return None, None, None


def fetch_btc_coinmarketcap():
    """Fetch BTC price from CoinMarketCap (Scraping)."""
    try:
        url = "https://coinmarketcap.com/currencies/bitcoin/"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code != 200:
            return None, None, None

        match = re.search(
            r'property="og:description" content="The live Bitcoin price today is \\$([0-9,.]+)',
            r.text,
        )
        if match:
            price = _to_float(match.group(1))
            if price is None:
                return None, None, None
            change = 0.0
            match_c = re.search(r'"priceChangePercent24h":([-0-9.]+)', r.text)
            if match_c:
                parsed = _to_float(match_c.group(1))
                if parsed is not None:
                    change = parsed
            return price, change, "실시간 (CoinMarketCap)"

        return None, None, None
    except Exception as e:
        print(f"CMC fetch failed: {e}")
        return None, None, None


def fetch_yahoo_quote_api(symbol):
    """Primary Yahoo Finance quote endpoint."""
    try:
        quote_url = "https://query1.finance.yahoo.com/v7/finance/quote"
        resp = requests.get(quote_url, params={"symbols": symbol}, timeout=5)
        if resp.status_code != 200:
            return None, None, None

        data = resp.json().get("quoteResponse", {}).get("result", [])
        if not data:
            return None, None, None

        q = data[0]
        current = _to_float(q.get("regularMarketPrice"))
        change = _to_float(q.get("regularMarketChangePercent"))
        if current is None:
            return None, None, None
        if change is None:
            change = 0.0
        return current, change, "실시간 (Yahoo Finance Quote API)"
    except Exception:
        return None, None, None


def fetch_yfinance_quote(symbol):
    """yfinance fallback from Yahoo Finance market data."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        if hist.empty:
            return None, None, None
        current = _to_float(hist["Close"].iloc[-1])
        prev = _to_float(hist["Close"].iloc[-2]) if len(hist) > 1 else current
        if current is None:
            return None, None, None
        if prev in (None, 0):
            change = 0.0
        else:
            change = (current - prev) / prev * 100
        time_str = datetime.now().strftime("%H:%M")
        return current, change, f"실시간 (Yahoo Finance/yfinance {time_str})"
    except Exception:
        return None, None, None


def _fetch_alpha_global_quote(alpha_symbol):
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": alpha_symbol,
            "apikey": ALPHA_VANTAGE_API_KEY,
        }
        resp = requests.get(url, params=params, timeout=6)
        if resp.status_code != 200:
            return None, None
        gq = resp.json().get("Global Quote", {})
        if not gq:
            return None, None
        price = _to_float(gq.get("05. price"))
        change = _to_float(gq.get("10. change percent"))
        if price is None:
            return None, None
        if change is None:
            change = 0.0
        return price, change
    except Exception:
        return None, None


def fetch_alpha_vantage(symbol):
    """Official Alpha Vantage API fallback (requires ALPHA_VANTAGE_API_KEY)."""
    if not ALPHA_VANTAGE_API_KEY:
        return None, None, None

    try:
        if symbol == "KRW=X":
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": "USD",
                "to_currency": "KRW",
                "apikey": ALPHA_VANTAGE_API_KEY,
            }
            resp = requests.get(url, params=params, timeout=6)
            if resp.status_code == 200:
                fx = resp.json().get("Realtime Currency Exchange Rate", {})
                price = _to_float(fx.get("5. Exchange Rate"))
                if price is not None:
                    return price, 0.0, "실시간 (Alpha Vantage FX)"

        if symbol == "^GSPC":
            for alpha_symbol in ["^GSPC", "SPX"]:
                price, change = _fetch_alpha_global_quote(alpha_symbol)
                if price is not None:
                    return price, change, "실시간 (Alpha Vantage Index)"

        if symbol == "GC=F":
            for alpha_symbol in ["GC=F", "XAUUSD"]:
                price, change = _fetch_alpha_global_quote(alpha_symbol)
                if price is not None:
                    return price, change, "실시간 (Alpha Vantage Gold)"

            # Fallback: XAU/USD exchange rate endpoint
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": "XAU",
                "to_currency": "USD",
                "apikey": ALPHA_VANTAGE_API_KEY,
            }
            resp = requests.get(url, params=params, timeout=6)
            if resp.status_code == 200:
                fx = resp.json().get("Realtime Currency Exchange Rate", {})
                price = _to_float(fx.get("5. Exchange Rate"))
                if price is not None:
                    return price, 0.0, "실시간 (Alpha Vantage XAU/USD)"

        # Generic attempt
        price, change = _fetch_alpha_global_quote(symbol)
        if price is not None:
            return price, change, "실시간 (Alpha Vantage)"

        return None, None, None
    except Exception:
        return None, None, None


def fetch_investing_html(symbol):
    """Investing.com HTML fallback for supported symbols."""
    url = INVESTING_URLS.get(symbol)
    if not url:
        return None, None, None

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = requests.get(url, headers=headers, timeout=6)
        if r.status_code != 200:
            return None, None, None
        html = r.text

        price = None
        change = 0.0

        m = re.search(r'data-test="instrument-price-last">\s*([0-9,\.]+)\s*<', html)
        if m:
            price = _to_float(m.group(1))

        if price is None:
            m = re.search(r'"last"\s*:\s*"([0-9,\.]+)"', html)
            if m:
                price = _to_float(m.group(1))

        m = re.search(r'data-test="instrument-price-change-percent">\s*([+\-0-9\.,]+)%\s*<', html)
        if m:
            parsed = _to_float(m.group(1))
            if parsed is not None:
                change = parsed
        else:
            m = re.search(r'"chg_percent"\s*:\s*"?([+\-0-9\.]+)"?', html)
            if m:
                parsed = _to_float(m.group(1))
                if parsed is not None:
                    change = parsed

        if price is None:
            return None, None, None

        return price, change, "실시간 (Investing.com HTML)"
    except Exception:
        return None, None, None


def fetch_yahoo_requests(symbol):
    """Fallback: scraping Yahoo Finance HTML if API is blocked."""
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        r = requests.get(url, headers=headers, timeout=5)

        if r.status_code != 200:
            return None, None, None

        html = r.text

        match = re.search(r'property="og:description" content=".*?price of ([0-9,.]+)"', html)
        if match:
            price = _to_float(match.group(1))
            if price is not None:
                return price, 0.0, "실시간 (Yahoo Finance HTML-Meta)"

        match_json = re.search(r'"regularMarketPrice":\s*\{.*?"raw":\s*([0-9.,]+)', html)
        if match_json:
            price = _to_float(match_json.group(1))
            if price is not None:
                return price, 0.0, "실시간 (Yahoo Finance HTML-JSON)"

        idx = html.find('data-field="regularMarketPrice"')
        if idx != -1:
            val_idx = html.find('value="', idx)
            if val_idx != -1:
                start = val_idx + 7
                end = html.find('"', start)
                price = _to_float(html[start:end])
                if price is not None:
                    return price, 0.0, "실시간 (Yahoo Finance HTML)"

        return None, None, None

    except Exception as e:
        print(f"Yahoo Scrape failed for {symbol}: {e}")
        return None, None, None


def fetch_data_robust(symbol, asset_type="crypto"):
    """
    Source priority:
    - BTC: Upbit KRW -> CoinMarketCap -> Binance(CCXT) -> Yahoo Finance chain
    - WOORI_GOLDBANK_KRW: Woori Gold Banking KRW
    - SHINHAN_SILVER_KRW: Shinhan SilverRush KRW
    - GC=F, ^GSPC: Yahoo Finance -> Alpha Vantage -> Investing.com
    - KRW=X: Naver Finance(매매기준율) -> ExchangeRate-API -> Google Finance -> FinanceDataReader
    - Others: Yahoo Finance chain
    """
    # Woori Gold Banking KRW dedicated chain
    if symbol == "WOORI_GOLDBANK_KRW":
        p, c, s = fetch_woori_goldbank_krw()
        if p is not None:
            return p, c, s
        return None, None, None

    # Shinhan SilverRush KRW dedicated chain
    if symbol == "SHINHAN_SILVER_KRW":
        p, c, s = fetch_shinhan_silverrush_krw()
        if p is not None:
            return p, c, s
        return None, None, None

    # BTC special path
    if symbol == "BTC-USD":
        p, c, s = fetch_btc_upbit_krw()
        if p is not None:
            return p, c, s

        p, c, s = fetch_btc_coinmarketcap()
        if p is not None:
            return p, c, s

        p, c, s = fetch_btc_binance()
        if p is not None:
            return p, c, s

    # KRW/USD dedicated trusted chain
    if symbol == "KRW=X":
        for fn in [
            fetch_naver_marketindex_usd_krw,
            fetch_exchangerate_api_usd_krw,
            fetch_google_finance_usd_krw,
            fetch_financedatareader_usd_krw,
        ]:
            p, c, s = fn()
            if p is not None:
                return p, c, s
        return None, None, None

    # Requested trusted source chain for key metrics
    if symbol in {"GC=F", "^GSPC"}:
        for fn in [
            fetch_yahoo_quote_api,
            fetch_yfinance_quote,
            fetch_alpha_vantage,
            fetch_investing_html,
            fetch_yahoo_requests,
        ]:
            p, c, s = fn(symbol)
            if p is not None:
                return p, c, s
        return None, None, None

    # Generic chain
    for fn in [fetch_yahoo_quote_api, fetch_yfinance_quote, fetch_yahoo_requests]:
        p, c, s = fn(symbol)
        if p is not None:
            return p, c, s

    return None, None, None
