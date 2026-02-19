
import yfinance as yf
import pandas as pd

tickers = ["BTC-USD", "GC=F", "^GSPC", "KRW=X"]

print("Testing yfinance connectivity...")

for t in tickers:
    print(f"\n--- Testing {t} ---")
    try:
        ticker = yf.Ticker(t)
        
        # Test 1: Fast Info
        # print("Attempting fast_info...")
        # try:
        #     price = ticker.fast_info.get('last_price')
        #     print(f"  fast_info price: {price}")
        # except Exception as e:
        #     print(f"  fast_info failed: {e}")
            
        # Test 2: History
        print("Attempting history(period='5d')...")
        hist = ticker.history(period="5d")
        if not hist.empty:
            print(f"  history success! Last close: {hist['Close'].iloc[-1]}")
            print(f"  Last date: {hist.index[-1]}")
        else:
            print("  history returned empty DataFrame.")
            
    except Exception as e:
        print(f"  Error initializing/fetching {t}: {e}")
