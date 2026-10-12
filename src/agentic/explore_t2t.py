"""Exploration (not a registered test; changes no rule): what trade-for-trade (BE / BZ series) looks like in our data.

Source: the series NSE prints for each stock each day in its bhavcopy (price panel `series`; BE/BZ rows backfilled 2015-2024
on 2026-09-27). EQ = normal rolling settlement; BE / BZ = trade-for-trade (every trade settles by delivery).
Liquid = 20-day average traded value >= Rs 1 cr and price > Rs 50 the day before (V3.1's bar).
  A  how many liquid stocks sit in BE / BZ, by year
  B  spells: a stock's run of consecutive BE/BZ sessions (start = first BE/BZ session after an EQ session, 2016+);
     how long they last, how many return to EQ within 3 / 6 / 12 months
  C  around the switch into BE/BZ (liquid at the switch, 2019+): run-up over the 60 sessions before; return over the next
     20 / 60 / 126 sessions from the first BE close vs the equal-weight liquid market over the same days; share of
     'frozen' sessions (high == low: the price never moved, usually stuck at its price band) before and after
  D  frozen sessions: BE/BZ vs EQ liquid stock-days, by year
  E  the 5 trade-for-trade names on the 2026-10-09 V3.1 list: since when, run-up before the switch, frozen days lately
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp  # noqa: E402

OUT = ROOT / "logs/leader_sleeve/t2t_explore"
TODAY5 = ["KANORICHEM", "TIRUPATIFL", "PANACHE", "STLTECH", "ACSTECH"]


def main() -> None:
    L = rp.load_panel(["close", "high", "low", "price_adjustment_factor_to_present", "avg_traded_value_20d", "series"])
    L["trade_date"] = pd.to_datetime(L["trade_date"])
    L["raw"] = rp.raw_price(L, "close"); L["adv"] = L["avg_traded_value_20d"] / 1e7
    L["t2t"] = L["series"].isin(["BE", "BZ"]).astype(float)
    L["bz"] = (L["series"] == "BZ").astype(float)
    L["frozen"] = ((L["high"] - L["low"]).abs() < 1e-9).astype(float)
    cal = pd.DatetimeIndex(sorted(L["trade_date"].unique()))
    W = {c: rp.wide(L, c, cal) for c in ("t2t", "bz", "close", "raw", "adv", "frozen")}
    del L
    C = W["close"].ffill(limit=5)
    liq = ((W["adv"] >= 1.0) & (W["raw"] > 50)).shift(1, fill_value=False).astype(bool)
    T2 = W["t2t"]; traded = T2.notna()
    res = {}

    print("=== A. liquid stocks in trade-for-trade, average per session by year ===")
    a = pd.DataFrame({"liquid": liq.sum(axis=1), "t2t": ((T2 == 1) & liq).sum(axis=1), "bz": ((W["bz"] == 1) & liq).sum(axis=1)})
    A = a.groupby(a.index.year).mean().round(0); A["t2t_share%"] = (100 * A["t2t"] / A["liquid"]).round(1)
    print(A.to_string()); res["A"] = A.reset_index().to_dict("records")

    print("\n=== B. trade-for-trade spells (start after an EQ session, 2016+) ===")
    Tn = T2.to_numpy(); n_cal = len(cal); spells = []
    for j, s in enumerate(T2.columns):
        col = Tn[:, j]; idx = np.where(~np.isnan(col))[0]
        if len(idx) < 2:
            continue
        v = col[idx]
        starts = np.where((v[1:] == 1) & (v[:-1] == 0))[0] + 1
        for st in starts:
            rest = np.where(v[st:] == 0)[0]
            end_k = st + rest[0] if len(rest) else None
            spells.append(dict(symbol=s, start=cal[idx[st]], sessions=(idx[end_k] - idx[st]) if end_k is not None else None,
                               back_to_eq=end_k is not None, i0=idx[st], j=j))
    Sp = pd.DataFrame(spells); Sp = Sp[Sp["start"] >= "2016-01-01"]
    done = Sp[Sp["back_to_eq"]]
    B = dict(spells=len(Sp), stocks=int(Sp["symbol"].nunique()), still_open=int((~Sp["back_to_eq"]).sum()),
             median_sessions=float(done["sessions"].median()), p25=float(done["sessions"].quantile(0.25)), p75=float(done["sessions"].quantile(0.75)),
             back_within_63=round(100 * float((Sp["sessions"] <= 63).mean()), 1), back_within_126=round(100 * float((Sp["sessions"] <= 126).mean()), 1),
             back_within_252=round(100 * float((Sp["sessions"] <= 252).mean()), 1))
    print(B); res["B"] = B
    by_y = Sp.groupby(Sp["start"].dt.year).agg(spells=("symbol", "size"), median_sessions=("sessions", "median")); print("spells started per year:", by_y.to_dict("index"))
    res["B_by_year"] = {int(k): v for k, v in by_y.to_dict("index").items()}

    print("\n=== C. around the switch into trade-for-trade (liquid at the switch, 2019+) ===")
    Cn, Fz, LQ = C.to_numpy(), W["frozen"].to_numpy(), liq.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        R1 = np.vstack([np.full((1, Cn.shape[1]), np.nan), Cn[1:] / Cn[:-1] - 1])
    R1 = np.clip(R1, -0.5, 1.0)
    ewr = np.nanmean(np.where(LQ, R1, np.nan), axis=1); ewr = np.nan_to_num(ewr); EW = np.cumprod(1 + ewr)
    ev = []
    for r in Sp[Sp["start"] >= "2019-01-01"].itertuples():
        i, j = r.i0, r.j
        if not LQ[i, j] or i < 61:
            continue
        row = dict(symbol=r.symbol, start=r.start, pre60=Cn[i - 1, j] / Cn[i - 61, j] - 1, mkt_pre60=EW[i - 1] / EW[i - 61] - 1,
                   frozen_before=np.nanmean(Fz[i - 20:i, j]), frozen_after=np.nanmean(Fz[i:i + 20, j]) if i + 20 <= n_cal else np.nan)
        for k in (20, 60, 126):
            if i + k < n_cal:
                row[f"post{k}"] = Cn[i + k, j] / Cn[i, j] - 1; row[f"mkt{k}"] = EW[i + k] / EW[i] - 1
        ev.append(row)
    E = pd.DataFrame(ev)
    summ = dict(switches=len(E), pre60_median=round(100 * E["pre60"].median(), 1), pre60_avg=round(100 * E["pre60"].mean(), 1), mkt_pre60_avg=round(100 * E["mkt_pre60"].mean(), 1),
                frozen_before=round(100 * E["frozen_before"].mean(), 1), frozen_after=round(100 * E["frozen_after"].mean(), 1))
    for k in (20, 60, 126):
        x = E.dropna(subset=[f"post{k}"])
        summ[f"after{k}_median"] = round(100 * x[f"post{k}"].median(), 1); summ[f"after{k}_avg"] = round(100 * x[f"post{k}"].mean(), 1)
        summ[f"after{k}_vs_mkt_avg"] = round(100 * (x[f"post{k}"] - x[f"mkt{k}"]).mean(), 1); summ[f"after{k}_lost30"] = round(100 * (x[f"post{k}"] <= -0.3).mean(), 1)
        summ[f"after{k}_up50"] = round(100 * (x[f"post{k}"] >= 0.5).mean(), 1)
    print(summ); res["C"] = summ
    E["era"] = np.where(E["start"].dt.year <= 2022, "2019-22", "2023+")
    ce = E.groupby("era").agg(n=("pre60", "size"), pre60_med=("pre60", "median"), post126_med=("post126", "median"), post126_avg=("post126", "mean"),
                              lost30_126=("post126", lambda x: (x <= -0.3).mean()), up50_126=("post126", lambda x: (x >= 0.5).mean())).round(3)
    print(ce.to_string()); res["C_by_era"] = ce.reset_index().to_dict("records")

    print("\n=== D. frozen sessions (high == low), liquid stock-days, by year ===")
    fz = W["frozen"].where(liq)
    D = pd.DataFrame({"t2t_frozen%": (100 * fz.where(T2 == 1).stack().groupby(level=0).mean()).groupby(lambda d: d.year).mean().round(1),
                      "eq_frozen%": (100 * fz.where(T2 == 0).stack().groupby(level=0).mean()).groupby(lambda d: d.year).mean().round(1)})
    print(D.to_string()); res["D"] = D.reset_index().to_dict("records")

    print("\n=== E. the 5 trade-for-trade names on the 2026-10-09 V3.1 list ===")
    rows = []
    d9 = cal.get_loc(pd.Timestamp("2026-10-09"))
    for s in TODAY5:
        sp = Sp[Sp["symbol"] == s].sort_values("start")
        cur = sp[~sp["back_to_eq"]].tail(1)
        if cur.empty:
            rows.append(dict(symbol=s, note="no open spell found")); continue
        i, j = int(cur["i0"].iat[0]), int(cur["j"].iat[0])
        rows.append(dict(symbol=s, in_t2t_since=str(cur["start"].iat[0].date()), sessions_so_far=int(d9 - i + 1), earlier_spells=int(len(sp) - 1),
                         runup_60_before=round(100 * (Cn[i - 1, j] / Cn[i - 61, j] - 1), 0) if i >= 61 else None,
                         since_switch=round(100 * (Cn[d9, j] / Cn[i, j] - 1), 0), frozen_last20=int(np.nansum(Fz[d9 - 19:d9 + 1, j]))))
    print(pd.DataFrame(rows).to_string(index=False)); res["E"] = rows

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    (OUT / "results.json.manifest.json").write_text(json.dumps(dict(dataset="results.json", producer="src/agentic/explore_t2t.py", kind="exploration", definitions=__doc__,
        units=dict(pre60="percent return over the 60 sessions before the first BE/BZ session", post="percent from the first BE/BZ close",
                   frozen="percent of sessions with high == low", sessions="trading sessions"), updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    (OUT / "README.md").write_text("# Trade-for-trade (BE/BZ) in our data\n\nExploration. results.json: A = how many liquid stocks are in it, B = how long spells last, "
                                   "C = before/after the switch, D = frozen sessions, E = today's five.\n")


if __name__ == "__main__":
    main()
