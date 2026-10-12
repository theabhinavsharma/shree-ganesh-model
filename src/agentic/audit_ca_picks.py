"""Audit (2026-10-12): could corporate actions, renames, suspensions or delistings have skewed the V3.1 / V3.2 picks, their
backtest returns, or the 1.5x model's labels? Read-only; changes no data.

Big move = one-day change in the ADJUSTED close of >= +45% or <= -35% (consecutive trading rows of one symbol). Each is put
in the first class that fits:
  split_bonus      NSE split / bonus record (stock_corporate_actions.parquet) within 5 days
  other_ca         demerger / rights / scheme / special dividend / consolidation (price_only_ca_factors.parquet) within 5
                   days; 'other_ca_not_applied' if that record was not applied to prices and the raw price moved the same way
  rename           NSE symbol change (symbolchange.csv) within 5 days
  after_gap        the previous trading row is more than 10 calendar days earlier (suspension / relisting)
  real_move        raw and adjusted moved alike (within 2 points) and no event: a genuine price move (e.g. no price band)
  SUSPECT          adjusted and raw differ with no record, or anything else
Then: (A) every big move since 2016 among stocks trading >= Rs 1 cr a day; (B) the V3.1 picks (phase 0 since 2019, from
logs/leader_sleeve/v31_hits_by_year/v31_picks.csv): big moves from 260 sessions before entry to the exit, picks with no trade
near their exit date, and the 20 best picks' raw vs adjusted return; (C) model labels: positive 1.5x labels whose 95-session
window holds a SUSPECT or other_ca_not_applied up-move.
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
import nse_symbols  # noqa: E402
import research_panel as rp  # noqa: E402

OUT = ROOT / "logs/audits/ca_picks_20261012"
NEAR = pd.Timedelta(days=5)


def main() -> None:
    L = rp.load_panel(["close", "price_adjustment_factor_to_present", "avg_traded_value_20d", "series"])
    L["trade_date"] = pd.to_datetime(L["trade_date"]); L["raw"] = rp.raw_price(L, "close"); L["adv"] = L["avg_traded_value_20d"] / 1e7
    L = L.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    g = L.groupby("symbol")
    L["adj_ret"] = L["close"] / g["close"].shift(1) - 1
    L["raw_ret"] = L["raw"] / g["raw"].shift(1) - 1
    L["gap_days"] = (L["trade_date"] - g["trade_date"].shift(1)).dt.days
    L["adv_prev"] = g["adv"].shift(1)
    big = L[(L["adj_ret"] >= 0.45) | (L["adj_ret"] <= -0.35)].copy()

    ca = pd.read_parquet(ROOT / "data/corporate_actions_full_history/normalized/stock_corporate_actions.parquet")
    ca = ca[(ca["is_bonus"] | ca["is_split"]) & (ca["series"] == "EQ")]
    ca["ex_date"] = pd.to_datetime(ca["ex_date"])
    ca_sym = pd.concat([ca[["symbol", "ex_date"]], ca[["mapped_from_symbol", "ex_date"]].rename(columns={"mapped_from_symbol": "symbol"}).dropna()])
    po = pd.read_parquet(ROOT / "data/corporate_actions_full_history/normalized/price_only_ca_factors.parquet")
    po = po[po["kind"] != "large_dividend"].copy(); po["ex_date"] = pd.to_datetime(po["ex_date"])
    ren = []
    for line in open(nse_symbols.FILE, encoding="utf-8", errors="replace"):
        p = line.strip().rsplit(",", 3)
        if len(p) == 4:
            d = pd.to_datetime(p[3], format="%d-%b-%Y", errors="coerce")
            if pd.notna(d):
                ren += [(p[1].strip(), d), (p[2].strip(), d)]
    rn = pd.DataFrame(ren, columns=["symbol", "date"])

    def near(tab: pd.DataFrame, col: str) -> set:
        m = big[["symbol", "trade_date"]].reset_index().merge(tab.rename(columns={col: "ev"}), on="symbol")
        return set(m.loc[(m["ev"] - m["trade_date"]).abs() <= NEAR, "index"])
    s_split, s_rn = near(ca_sym, "ex_date"), near(rn, "date")
    m = big[["symbol", "trade_date"]].reset_index().merge(po[["symbol", "ex_date", "applied", "kind"]], on="symbol")
    m = m[(m["ex_date"] - m["trade_date"]).abs() <= NEAR]
    s_po_applied, s_po_not = set(m.loc[m["applied"], "index"]), set(m.loc[~m["applied"], "index"])

    def cls(i, r):
        if i in s_split:
            return "split_bonus"
        if i in s_po_applied:
            return "other_ca"
        if i in s_po_not:
            return "other_ca_not_applied" if abs(r["adj_ret"] - r["raw_ret"]) < 0.02 else "other_ca"
        if i in s_rn:
            return "rename"
        if r["gap_days"] > 10:
            return "after_gap"
        if abs(r["adj_ret"] - r["raw_ret"]) < 0.02:
            return "real_move"
        return "SUSPECT"
    big["cls"] = [cls(i, r) for i, r in big.iterrows()]
    big["dir"] = np.where(big["adj_ret"] > 0, "up", "down")

    # (A) universe since 2016, liquid the day before
    A = big[(big["trade_date"] >= "2016-01-01") & (big["adv_prev"] >= 1.0)]
    tabA = A.groupby(["cls", "dir"]).size().unstack(fill_value=0)
    print(f"(A) big one-day moves since 2016, stocks trading >= Rs 1 cr a day: {len(A)}")
    print(tabA.to_string())
    print("   examples of SUSPECT / other_ca_not_applied:")
    ex = A[A["cls"].isin(["SUSPECT", "other_ca_not_applied"])].sort_values("trade_date")
    print(ex[["symbol", "trade_date", "series", "adj_ret", "raw_ret", "gap_days", "cls"]].tail(15).round(3).to_string(index=False) if len(ex) else "   none")

    # (B) picks
    pk = pd.read_parquet if False else None
    pk = pd.read_csv(ROOT / "logs/leader_sleeve/v31_hits_by_year/v31_picks.csv", parse_dates=["trade_date"])
    cal = pd.DatetimeIndex(sorted(L["trade_date"].unique())); pos = {d: i for i, d in enumerate(cal)}
    pk["win_lo"] = [cal[max(0, pos[d] - 260)] for d in pk["trade_date"]]
    pk["exit_d"] = [cal[min(len(cal) - 1, pos[d] + 126)] for d in pk["trade_date"]]
    bm = big[["symbol", "trade_date", "adj_ret", "raw_ret", "cls"]]
    j = pk[["symbol", "trade_date", "win_lo", "exit_d", "ret", "full"]].reset_index().merge(bm.rename(columns={"trade_date": "move_d"}), on="symbol")
    j = j[(j["move_d"] >= j["win_lo"]) & (j["move_d"] <= j["exit_d"])]
    j["where"] = np.where(j["move_d"] <= j["trade_date"], "before entry (selection)", "while held (return)")
    tabB = j.groupby(["where", "cls"])["index"].nunique().unstack(fill_value=0)
    print(f"\n(B) V3.1 picks ({len(pk)} pick-weeks, {pk['symbol'].nunique()} stocks): pick-weeks with a big move in the window")
    print(tabB.to_string())
    bad = j[j["cls"].isin(["SUSPECT", "other_ca_not_applied"])].drop_duplicates(["symbol", "move_d"])
    print("   distinct SUSPECT / not-applied moves touching picks:")
    print(bad[["symbol", "move_d", "adj_ret", "raw_ret", "cls", "where"]].round(3).to_string(index=False) if len(bad) else "   none")
    # exits: a trade on or within 5 sessions before the exit date
    has = L.groupby("symbol")["trade_date"].apply(lambda s: np.array(sorted(s.values)))
    def traded_near_exit(s, d):
        a = has.get(s)
        if a is None:
            return False
        k = np.searchsorted(a, np.datetime64(d), side="right")
        return k > 0 and (pd.Timestamp(d) - pd.Timestamp(a[k - 1])).days <= 8
    full = pk[pk["full"]].copy()
    full["exit_ok"] = [traded_near_exit(s, d) for s, d in zip(full["symbol"], full["exit_d"])]
    ne = full[~full["exit_ok"]]
    print(f"   picks with no trade within ~5 sessions of their exit: {len(ne)} of {len(full)} ({ne['symbol'].nunique()} stocks) "
          f"{sorted(ne['symbol'].unique())[:15]}; their avg 6m return {100 * ne['ret'].mean():+.0f}%" if len(ne) else "   all picks traded near their exit")
    # top 20 returns: raw vs adjusted
    top = full.sort_values("ret", ascending=False).drop_duplicates("symbol").head(20)
    rows = []
    for s, d, ex_d, r in zip(top["symbol"], top["trade_date"], top["exit_d"], top["ret"]):
        a = L[(L["symbol"] == s) & (L["trade_date"] > d)].head(1); b = L[(L["symbol"] == s) & (L["trade_date"] <= ex_d)].tail(1)
        if a.empty or b.empty:
            continue
        evs = [f"{x.date()} split/bonus" for x in ca_sym[(ca_sym["symbol"] == s) & (ca_sym["ex_date"] > d) & (ca_sym["ex_date"] <= ex_d)]["ex_date"]] + \
              [f"{x.ex_date.date()} {x.kind}{'' if x.applied else ' (not applied)'}" for x in po[(po["symbol"] == s) & (po["ex_date"] > d) & (po["ex_date"] <= ex_d)].itertuples()]
        rows.append(dict(symbol=s, entry=str(d.date()), adj_ret=round(100 * float(b["close"].iat[0] / a["close"].iat[0] - 1)), raw_ret=round(100 * float(b["raw"].iat[0] / a["raw"].iat[0] - 1)),
                         events=", ".join(evs) or "none"))
    T20 = pd.DataFrame(rows)
    T20["explained"] = (T20["adj_ret"] - T20["raw_ret"]).abs().lt(2) | T20["events"].ne("none")
    print("\n   20 best picks: adjusted vs raw return over the hold (close to close) and the events in between")
    print(T20.to_string(index=False))

    # (C) model labels
    pr = pd.read_parquet(ROOT / "logs/leader_sleeve/anatomy_1p5x/bakeoff_preds.parquet", columns=["symbol", "trade_date", "y95", "adv", "raw_close"])
    pr["trade_date"] = pd.to_datetime(pr["trade_date"]); advc = pr["adv"] / 1e7 if pr["adv"].median() > 1e5 else pr["adv"]
    pos_l = pr[(pr["y95"] == 1) & (advc >= 1.0) & (pr["raw_close"] > 50)][["symbol", "trade_date"]]
    sus_up = big[big["cls"].isin(["SUSPECT", "other_ca_not_applied", "after_gap"]) & (big["dir"] == "up")][["symbol", "trade_date"]].rename(columns={"trade_date": "move_d"})
    jj = pos_l.reset_index().merge(sus_up, on="symbol")
    jj = jj[(jj["move_d"] > jj["trade_date"]) & (jj["move_d"] <= jj["trade_date"] + pd.Timedelta(days=140))]
    nbad = jj["index"].nunique()
    print(f"\n(C) model labels: {len(pos_l):,} positive 1.5x labels (liquid, 2019+ scored rows); {nbad:,} ({100 * nbad / max(1, len(pos_l)):.2f}%) have a SUSPECT / "
          f"not-applied / after-gap up-move in their window (could be fake winners)")
    OUT.mkdir(parents=True, exist_ok=True)
    A.to_csv(OUT / "big_moves_liquid_2016.csv", index=False); j.to_csv(OUT / "pick_window_moves.csv", index=False); T20.to_csv(OUT / "top20_raw_vs_adjusted.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps(dict(producer="src/agentic/audit_ca_picks.py", definitions=__doc__, universe_table=tabA.to_dict(), picks_table=tabB.to_dict(),
        picks_no_exit_trade=int(len(ne)), label_suspect=int(nbad), label_positive=int(len(pos_l)), updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))


if __name__ == "__main__":
    main()
