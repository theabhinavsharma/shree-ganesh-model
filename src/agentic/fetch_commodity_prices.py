"""Fetch commodity prices via FRED (Federal Reserve Economic Data) — free, no auth.

Pulls daily/business-day series for global commodities:
  • Brent crude (DCOILBRENTEU)
  • WTI crude (DCOILWTICO)
  • Gold London PM fix (GOLDAMGBD228NLBM, GOLDPMGBD228NLBM)
  • Silver London PM fix (SLVPRUSD)
  • Henry Hub natural gas (DHHNGSP)
  • Wheat (PWHEAMTUSDM, monthly)
  • Sugar (PSUGAUSDM, monthly)
  • Copper (PCOPPUSDM, monthly)

Why these matter for India 2x-in-180d:
  • Crude → OMC margins, aviation costs, CPI, INR pressure
  • Gold → safe-haven flow signal (risk-off proxy)
  • Copper → "Dr. Copper" — global growth bellwether
  • Natural Gas → ONGC/GAIL/IGL/MGL margins
  • Wheat/Sugar → agri inflation → rural demand

Output: data/derived/commodity_prices.parquet
"""
from __future__ import annotations
import io
import time
import urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
OUT = ROOT / "data/derived/commodity_prices.parquet"

UA = "Mozilla/5.0 (compatible; ZoomFetcher/1.0)"

# FRED series → output column
SERIES = {
    "DCOILBRENTEU":      "brent",       # Brent crude, daily
    "DCOILWTICO":        "wti",         # WTI crude, daily
    # gold — no reliable free FRED daily series; using GLD ETF in fetch_global_rates instead
    "DHHNGSP":           "natgas",      # Henry Hub natural gas, daily
    "PCOPPUSDM":         "copper",      # Copper monthly (forward-fill)
    "PWHEAMTUSDM":       "wheat",       # Wheat monthly
    "PSUGAISAUSDM":      "sugar",       # Sugar (ISA daily price), monthly
    "PALUMUSDM":         "aluminum",    # Aluminum monthly
    "PZINCUSDM":         "zinc",        # Zinc monthly
    "PNICKUSDM":         "nickel",      # Nickel monthly
    "PMAIZMTUSDM":       "corn",        # Maize/corn monthly
    "PCOTTINDUSDM":      "cotton",      # Cotton monthly
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
    # date column is either 'observation_date' or 'DATE'
    date_col = "observation_date" if "observation_date" in df.columns else "DATE"
    df = df.rename(columns={date_col: "trade_date"})
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    val_col = series_id
    if val_col not in df.columns:
        # FRED sometimes returns lowercase
        for c in df.columns:
            if c != "trade_date":
                val_col = c
                break
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[val_col])
    return df.rename(columns={val_col: "Close"})


def main() -> None:
    print("== fetch_commodity_prices (FRED) ==")
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
    # forward-fill monthly series across business days
    daily_idx = pd.bdate_range(merged["trade_date"].min(), pd.Timestamp.today())
    merged = merged.set_index("trade_date").reindex(daily_idx).ffill().reset_index().rename(
        columns={"index": "trade_date"})
    # only last 1500 business days
    merged = merged.tail(1500)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(merged)} days × {len([c for c in merged.columns if c!='trade_date'])} commodities  ({ok} ok, {fail} fail)")
    if "brent" in merged.columns:
        last = merged.iloc[-1]
        print(f"  latest: brent=${last.get('brent', float('nan')):.2f}  gold=${last.get('gold', float('nan')):.2f}  "
              f"copper=${last.get('copper', float('nan')):.2f}  natgas=${last.get('natgas', float('nan')):.2f}")


if __name__ == "__main__":
    main()
