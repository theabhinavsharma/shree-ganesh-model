"""Fetch global rates / risk indices via FRED — free, no auth.

Pulls daily history for:
  • US 10y treasury yield (DGS10)
  • US 2y treasury yield  (DGS2)
  • US 3m T-bill          (DGS3MO)
  • DXY broad dollar      (DTWEXBGS) — trade-weighted USD
  • US VIX                (VIXCLS)
  • S&P 500               (SP500)
  • USDJPY                (DEXJPUS)
  • Effective Fed Funds   (DFF)
  • Yield curve 10y-2y    (T10Y2Y) — recession proxy
  • TED spread            (TEDRATE) — discontinued; use BAMLH0A0HYM2 (HY OAS) instead
  • HY OAS                (BAMLH0A0HYM2) — credit risk indicator
  • EM bond spread        (BAMLEMHBHYCRPIOAS) — EM credit risk

Why these matter for India 2x-in-180d:
  • US 10y → FII flows
  • DXY → INR pressure
  • VIX → global risk-off detector
  • Yield curve inversion → recession/sell-off proxy
  • HY OAS rising → credit risk-off, EM equities crack

Output: data/derived/global_rates.parquet
"""
from __future__ import annotations
import io
import time
import urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
OUT = ROOT / "data/derived/global_rates.parquet"

UA = "Mozilla/5.0 (compatible; ZoomFetcher/1.0)"

SERIES = {
    "DGS10":             "us_10y",
    "DGS2":              "us_2y",
    "DGS3MO":            "us_3m",
    "DTWEXBGS":          "dxy",         # trade-weighted broad dollar (DXY-equivalent)
    "VIXCLS":            "us_vix",
    "SP500":             "spx",
    "DEXJPUS":           "usdjpy",      # JPY/USD inverse
    "DFF":               "fed_funds",
    "T10Y2Y":            "yc_10y2y",    # yield-curve spread (recession proxy)
    "BAMLH0A0HYM2":      "hy_oas",      # US high-yield OAS
    "BAMLEMHBHYCRPIOAS": "em_hy_oas",   # EM corporate HY OAS
    "DEXCHUS":           "usdcny",      # USD/CNY (China FX)
    "DEXUSEU":           "eurusd",
    "BAMLC0A0CM":        "ig_oas",      # US investment-grade OAS
    "WPU101":            "gold_ppi",    # PPI gold proxy (monthly), used as gold trend signal
}


def fred_csv(series_id: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
    except Exception:
        return pd.DataFrame()
    text = raw.decode("utf-8", errors="replace")
    if "observation_date" not in text and "DATE" not in text:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text))
    date_col = "observation_date" if "observation_date" in df.columns else "DATE"
    df = df.rename(columns={date_col: "trade_date"})
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    val_col = series_id
    if val_col not in df.columns:
        for c in df.columns:
            if c != "trade_date":
                val_col = c
                break
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[val_col])
    return df.rename(columns={val_col: "Close"})


def main() -> None:
    print("== fetch_global_rates (FRED) ==")
    merged: pd.DataFrame | None = None
    ok, fail = 0, 0
    for series_id, col in SERIES.items():
        try:
            df = fred_csv(series_id)
            if df.empty:
                print(f"  {series_id:<22} → {col:<10}  EMPTY")
                fail += 1
                continue
            df = df.rename(columns={"Close": col})
            df = df[["trade_date", col]]
            print(f"  {series_id:<22} → {col:<10}  {len(df):>5} rows  latest={df.iloc[-1][col]:.3f}")
            merged = df if merged is None else merged.merge(df, on="trade_date", how="outer")
            ok += 1
            time.sleep(0.3)
        except Exception as e:
            print(f"  {series_id:<22} → {col:<10}  FAIL: {type(e).__name__}: {str(e)[:80]}")
            fail += 1

    if merged is None or merged.empty:
        print("no data fetched")
        return

    merged = merged.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    daily_idx = pd.bdate_range(merged["trade_date"].min(), pd.Timestamp.today())
    merged = merged.set_index("trade_date").reindex(daily_idx).ffill().reset_index().rename(
        columns={"index": "trade_date"})
    merged = merged.tail(1500)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(merged)} days × {len([c for c in merged.columns if c!='trade_date'])} indicators  ({ok} ok, {fail} fail)")
    last = merged.iloc[-1]
    print(f"  latest: us_10y={last.get('us_10y',float('nan')):.2f}  dxy={last.get('dxy',float('nan')):.2f}  "
          f"vix={last.get('us_vix',float('nan')):.2f}  spx={last.get('spx',float('nan')):.2f}  "
          f"hy_oas={last.get('hy_oas',float('nan')):.2f}")


if __name__ == "__main__":
    main()
