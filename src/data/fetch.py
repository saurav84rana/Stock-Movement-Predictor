import time
import os
import yfinance as yf
from pathlib import Path

# 24 tickers
TICKERS: list[str] = [
    # tech
    "NVDA", "AAPL", "MSFT", "AMD", "GOOGL", "META",
    # automotive / industrial
    "F", "GM", "CAT", "BA", "TSLA", "RACE",
    # finance
    "JPM", "GS", "V", "MA", "BAC", "WFC",
    # consumer
    "NKE", "MCD", "SBUX", "KO", "PEP", "DIS",
]

# OUT_DIR = .../stock-predictor/data/raw
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
os.makedirs(OUT_DIR, exist_ok=True)

failed = []
for ticker in TICKERS:
    out_path = OUT_DIR / f"{ticker}.csv"

    if out_path.exists():
        print(f"{ticker} already exists, skipping")
        continue

    try:
        df = yf.download(
            ticker,
            start="2018-01-01",
            end="2026-01-01",
            auto_adjust=True,
            multi_level_index=False,
            progress=True
        )
        if df.empty:
            print(f"No data for {ticker}, skipping...")
            continue
        df.to_csv(out_path)
    except Exception as e:
        print(f"Failed to download {ticker}: {e}")
        failed.append(ticker)

    time.sleep(0.5) # to avoid rate limiting

if failed:
    print(f"\n{len(failed)} tickers failed: {failed}")