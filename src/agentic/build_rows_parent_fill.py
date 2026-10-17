"""Model rows with parent-group industry inputs for small industries (EXP-2026-10-18-parent-in-model; A/B only).

Plain English: in the model's training / scoring rows, a stock whose industry has fewer than 5 names gets no industry
inputs (ind_heat_pct, ind_heat, ind_ret20, ind_breadth are blank). This copy fills them from the stock's nearest parent
group in NSE's hierarchy that has >= 5 names on the same date (broad_industry, then sector, then broad_sector), computed
the way anatomy_1p5x computes them (mean 60- and 20-session return, share above the 50-day average); ind_heat_pct = the
parent heat's place among that day's industry heats (share at or below). ind_n is left as it is. Reads the live rows,
writes logs/leader_sleeve/parent_in_model/rows.parquet (+ manifest); the live rows and model are not touched.
"""
import html, json, sys
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "logs/leader_sleeve/anatomy_1p5x/rows.parquet"; OUT = ROOT / "logs/leader_sleeve/parent_in_model/rows.parquet"
LEVELS = ("broad_industry", "sector", "broad_sector")


def main():
    R = pd.read_parquet(SRC); R["trade_date"] = pd.to_datetime(R["trade_date"])
    H = pd.read_parquet(ROOT / "data/derived/screener_industry.parquet"); H = H[H["status"].str.startswith("OK")].dropna(subset=["industry"])
    H["industry"] = H["industry"].map(html.unescape)
    par = {lv: H.dropna(subset=[lv]).groupby("industry")[lv].agg(lambda x: x.mode().iat[0]) for lv in LEVELS}
    small = (R["ind_n"] < 5) & R["ind"].notna()
    hp = R.dropna(subset=["ind_heat"]).drop_duplicates(["trade_date", "ind"]).groupby("trade_date")["ind_heat"].apply(lambda s: np.sort(s.to_numpy())).to_dict()
    filled = pd.Series(False, index=R.index); level = pd.Series(None, index=R.index, dtype=object)
    for lv in LEVELS:
        R["_p"] = R["ind"].map(par[lv]); g = R.dropna(subset=["_p"]).groupby(["trade_date", "_p"])
        A = pd.DataFrame(dict(n=g["ret60"].count(), heat=g["ret60"].mean(), r20=g["ret20"].mean(), br=g["px_sma50"].apply(lambda s: (s > 0).mean())))
        A = A[A["n"] >= 5]
        need = small & ~filled & R["_p"].notna()
        key = pd.MultiIndex.from_arrays([R.loc[need, "trade_date"], R.loc[need, "_p"]])
        hit = key.isin(A.index); idx = R.index[need][hit]; k = key[hit]
        R.loc[idx, "ind_heat"] = A["heat"].reindex(k).to_numpy(); R.loc[idx, "ind_ret20"] = A["r20"].reindex(k).to_numpy(); R.loc[idx, "ind_breadth"] = A["br"].reindex(k).to_numpy()
        R.loc[idx, "ind_heat_pct"] = [np.searchsorted(hp[d], h, side="right") / len(hp[d]) if d in hp and len(hp[d]) else np.nan for d, h in zip(R.loc[idx, "trade_date"], R.loc[idx, "ind_heat"])]
        filled.loc[idx] = True; level.loc[idx] = lv
    R = R.drop(columns="_p")
    OUT.parent.mkdir(parents=True, exist_ok=True); R.to_parquet(OUT, index=False)
    info = dict(rows=len(R), small_rows=int(small.sum()), filled=int(filled.sum()), by_level=level[filled].value_counts().to_dict())
    (OUT.parent / "rows.parquet.manifest.json").write_text(json.dumps(dict(dataset="rows.parquet", experiment="EXP-2026-10-18-parent-in-model", producer="src/agentic/build_rows_parent_fill.py",
        source=str(SRC.relative_to(ROOT)), definitions=__doc__, info=info, units=dict(ind_heat="fraction", ind_ret20="fraction", ind_breadth="share 0-1", ind_heat_pct="percentile 0-1"),
        updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
