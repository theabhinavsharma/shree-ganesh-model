"""Weekly regime explainer (numbers part). Runs Friday 3 PM IST before the 7:30 PM list (skill sgm-regime-explainer adds the
news / government context and sends the phone note). Exploration only: it changes no rule and proposes tests, never adopts them.

What it does:
  1. V3.1 weekly lists since 2019 (v3_rule.frame_v31 / picks_v31, phase-0 weekly dates), each list's 6-month outcome after
     liquidity-scaled costs (bought at the next open, sold at the close of session 126) and the equal-weight market's 6-month
     return over the same days (stocks trading >= Rs 1 cr a day, price > Rs 50). Grouped by the quarter bought; quarters
     ranked great / ok / bad (thirds of the list return).
  2. Macro picture at each quarter's first list date, only values known by then: FPI 60-day net flow (% of FPI turnover,
     NSDL), India VIX, rupee / US 10y / dollar / Brent 3-month change, S&P 500 vs its 200-day, share of liquid stocks above
     their 200-day, Fed target and India call-money rate 6-month change (FRED).
  3. Events (configs/event_calendar.yaml) and policy-rate moves (derived: Fed target changes; India call rate moves of
     >= 0.25 pts month on month) that fell while each quarter's lists were held (quarter start .. quarter end + 6 months).
  4. Repeat check: for every event type, the lists bought in the 90 days before and the 90 days after each occurrence vs
     the average list. A type is a 'pattern' only with >= 5 occurrences and >= 80% of them on the same side.
  5. Today: the 3 past quarters whose macro picture is closest (z-scored distance) and what their lists did; upcoming events.
  6. Registered tests logged in the last 7 days, with the calendar years each arm won or lost.
Output: reports/regime/regime_<date>.md, logs/regime/regime_<date>.json (+ manifest; telegram_lines for the phone note).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import v3_rule  # noqa: E402

HOLD, START = 126, pd.Timestamp("2019-01-01")
REP, LOG = ROOT / "reports/regime", ROOT / "logs/regime"
FEATS = {"fpi_60d_pct": "FPI 60-day net flow, % of their turnover", "india_vix": "India VIX", "rupee_3m": "rupee weakening, 3m",
         "us10y_3m": "US 10y yield change, 3m (pts)", "dollar_3m": "dollar index change, 3m", "brent_3m": "Brent change, 3m",
         "spx_vs_200": "S&P 500 vs 200-day", "above200": "share of liquid stocks above 200-day", "fed_6m": "Fed target change, 6m (pts)",
         "india_rate_6m": "India call rate change, 6m (pts)"}


def asof(dates: pd.DatetimeIndex, s: pd.Series) -> np.ndarray:
    s = s.dropna().sort_index(); s = s[~s.index.duplicated(keep="last")]
    return s.reindex(s.index.union(dates)).ffill().reindex(dates).to_numpy()


def main() -> None:
    today = pd.Timestamp.now().normalize()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
    ctx = v3_rule.context(imap); F1 = v3_rule.frame_v31(X["F"], D)
    wk = [d for d in sp.weekly_grid(cal, 0) if d >= START]
    lists = v3_rule.picks_v31(F1, wk, imap, P, ctx)
    L = rp.load_panel(["open", "close", "price_adjustment_factor_to_present", "avg_traded_value_20d"])
    L["raw"] = rp.raw_price(L, "close"); L["adv"] = L["avg_traded_value_20d"] / 1e7
    O, C, RAW, ADV = (rp.wide(L, c, cal) for c in ("open", "close", "raw", "adv")); del L
    On, Cn, An = O.to_numpy(), C.ffill(limit=300).to_numpy(), ADV.to_numpy()
    col = {s: i for i, s in enumerate(O.columns)}; pos = {d: i for i, d in enumerate(cal)}; T = len(cal)
    liq = ((ADV >= 1.0) & (RAW > 50)).to_numpy()
    rows = []
    for d in wk:
        i = pos[d]; e = i + 1
        if e + HOLD - 1 > T - 1:
            continue
        cols = [col[s] for s in lists[d] if s in col]
        if cols:
            ent = On[e, cols]; ex = Cn[e + HOLD - 1, cols]; ok = np.isfinite(ent) & (ent > 0) & np.isfinite(ex)
            adv = An[i, cols]; cost = np.where(adv >= 5, 0.005, np.where(adv >= 1, 0.010, 0.020))
            r = ex[ok] / ent[ok] - 1 - cost[ok]
            lr, hit = (float(r.mean()), float((r >= 0.5).mean())) if ok.any() else (np.nan, np.nan)
        else:
            lr, hit = np.nan, np.nan
        u = np.where(liq[i])[0]; me, mx = On[e, u], Cn[e + HOLD - 1, u]; mk = np.isfinite(me) & (me > 0) & np.isfinite(mx)
        mret = float(np.clip(mx[mk] / me[mk] - 1, -0.95, 10).mean()) if mk.any() else np.nan
        rows.append(dict(date=d, names=len(cols), list_ret=lr, hit_end50=hit, mkt_ret=mret))
    W = pd.DataFrame(rows).set_index("date"); W["excess"] = W["list_ret"] - W["mkt_ret"]

    # ---- macro features at each list date (known by then)
    M = pd.DataFrame(index=W.index.union(pd.DatetimeIndex([today])))
    fr = pd.read_parquet(ROOT / "data/derived/macro_drivers_fred.parquet"); fr["known_from"] = pd.to_datetime(fr["known_from"])
    ser = lambda n: fr[fr["series"] == n].groupby("known_from")["value"].last()  # noqa: E731
    idx = M.index
    for name, s, kind, back in (("rupee_3m", "usdinr", "pct", 3), ("us10y_3m", "us_10y", "diff", 3), ("dollar_3m", "broad_dollar", "pct", 3),
                                ("brent_3m", "brent", "pct", 3), ("fed_6m", "fed_target_upper", "diff", 6), ("india_rate_6m", "india_call_rate", "diff", 6)):
        x = ser(s); now, then = asof(idx, x), asof(idx - pd.DateOffset(months=back), x)
        M[name] = (now / then - 1) if kind == "pct" else (now - then)
    v = pd.read_parquet(ROOT / "data/derived/india_vix.parquet"); v["date"] = pd.to_datetime(v["date"])
    M["india_vix"] = asof(idx - pd.Timedelta(days=1), v.set_index("date")["vix_close"])
    um = pd.read_parquet(ROOT / "data/derived/us_macro_fred.parquet"); um["trade_date"] = pd.to_datetime(um["trade_date"])
    spx = um.set_index("trade_date")["spx"].dropna().sort_index(); M["spx_vs_200"] = asof(idx - pd.Timedelta(days=1), spx / spx.rolling(200).mean() - 1)
    Fp = pd.read_parquet(ROOT / "data/derived/nsdl_fpi_daily.parquet")
    se = Fp[(Fp["asset"] == "Equity") & (Fp["route"] == "Stock Exchange")].copy(); se["known_from"] = pd.to_datetime(se["known_from"])
    se = se.sort_values("known_from"); g = se["gross_purchases_cr"] + se["gross_sales_cr"]
    se["sig"] = se["net_cr"].rolling(60, min_periods=48).sum() / g.rolling(60, min_periods=48).sum()
    M["fpi_60d_pct"] = asof(idx, se.set_index("known_from")["sig"])
    Cf = C.ffill(limit=5); sma = Cf.rolling(200, min_periods=200).mean(); kn = pd.DataFrame(liq, index=cal, columns=C.columns) & sma.notna()
    ab = ((Cf > sma) & kn).sum(axis=1) / kn.sum(axis=1).replace(0, np.nan)
    M["above200"] = asof(idx, ab)
    W = W.join(M)

    # ---- quarters
    W["quarter"] = W.index.to_period("Q")
    Q = W.groupby("quarter").agg(lists=("list_ret", "size"), list_ret=("list_ret", "mean"), mkt_ret=("mkt_ret", "mean"), excess=("excess", "mean"),
                                 names=("names", "mean"), first=("list_ret", lambda x: x.index.min()))
    full = Q["lists"] >= 4                                               # a quarter needs >= 4 finished lists to be graded
    Q["grade"] = "partial"
    Q.loc[full, "grade"] = pd.qcut(Q.loc[full, "list_ret"].rank(method="first"), 3, labels=["bad", "ok", "great"]).astype(str)
    for f in FEATS:
        Q[f] = [W.loc[d, f] for d in Q["first"]]

    # ---- events: calendar + derived rate moves
    cal_ev = pd.DataFrame(yaml.safe_load((ROOT / "configs/event_calendar.yaml").read_text())["events"]); cal_ev["date"] = pd.to_datetime(cal_ev["date"])
    def turns(s: pd.Series, step: float, region: str, label: str) -> list:
        """Cycle turns only: the first move up (or down) after moves the other way or after 180+ days with no move."""
        out, last_dir, last_d = [], 0, None
        for d, x in s.diff().items():
            if not np.isfinite(x) or abs(x) < step:
                continue
            dr = 1 if x > 0 else -1
            if dr != last_dir or (last_d is not None and (d - last_d).days > 180):
                out.append(dict(date=d, type=f"{'global' if region == 'US' else 'in'}_rate_turn_{'up' if dr > 0 else 'down'}", region=region,
                                title=f"{label} turns {'up' if dr > 0 else 'down'} ({s[d]:.2f}%)", source="FRED", verified=True))
            last_dir, last_d = dr, d
        return out
    rate_ev = turns(ser("fed_target_upper").groupby(level=0).last(), 0.01, "US", "Fed target") + \
        turns(fr[fr["series"] == "india_call_rate"].set_index("date")["value"].sort_index(), 0.25, "India", "India call rate")
    EV = pd.concat([cal_ev, pd.DataFrame(rate_ev)], ignore_index=True).sort_values("date")
    EV = EV[EV["date"] >= "2018-06-01"]
    hold_events = {}
    for q in Q.index:
        a, b = q.start_time, q.end_time + pd.DateOffset(months=6)
        hold_events[str(q)] = EV[(EV["date"] >= a) & (EV["date"] <= b)][["date", "type", "title", "verified"]].assign(date=lambda x: x["date"].dt.date.astype(str)).to_dict("records")

    # ---- repeat check per event type
    avg = float(W["list_ret"].mean())
    rep = []
    for t, g in EV.groupby("type"):
        occ, kept = [], []
        for d in sorted(g["date"]):                                     # episodes: an occurrence within 120 days of the last one is the same episode
            if kept and (d - kept[-1]).days <= 120:
                continue
            kept.append(d)
        for d in kept:
            bef = W[(W.index >= d - pd.Timedelta(days=90)) & (W.index < d)]["list_ret"]
            aft = W[(W.index >= d) & (W.index < d + pd.Timedelta(days=90))]["list_ret"]
            if len(bef) or len(aft):
                occ.append(dict(date=str(d.date()), before=float(bef.mean()) if len(bef) else np.nan, after=float(aft.mean()) if len(aft) else np.nan))
        if not occ:
            continue
        O_ = pd.DataFrame(occ); r = dict(type=t, occurrences=len(O_), dates=", ".join(O_["date"]))
        for side in ("before", "after"):
            x = O_[side].dropna() - avg
            same = max((x > 0).mean(), (x < 0).mean()) if len(x) else np.nan
            r[f"{side}_vs_avg"] = round(100 * float(x.mean()), 1) if len(x) else None
            r[f"{side}_same_side"] = round(100 * float(same), 0) if len(x) else None
            r[f"{side}_pattern"] = bool(len(x) >= 5 and same >= 0.8)
        rep.append(r)
    R = pd.DataFrame(rep).sort_values("occurrences", ascending=False)

    # ---- today's look-alikes
    feats = [f for f in FEATS if Q[f].notna().mean() > 0.8]
    Z = Q.loc[Q["grade"] != "partial", feats].astype(float); mu, sd = Z.mean(), Z.std().replace(0, np.nan)
    now = M.loc[today, feats].astype(float)
    dist = (((Z - mu) / sd - (now - mu) / sd) ** 2).sum(axis=1, min_count=len(feats) - 2) ** 0.5
    near = dist.dropna().sort_values().head(3)
    analogs = [dict(quarter=str(q), distance=round(float(dist[q]), 2), list_ret=round(100 * float(Q.loc[q, "list_ret"]), 1), grade=str(Q.loc[q, "grade"])) for q in near.index]
    upcoming = EV[(EV["date"] > today) & (EV["date"] <= today + pd.Timedelta(days=150))][["date", "title", "verified"]].assign(date=lambda x: x["date"].dt.date.astype(str)).to_dict("records")

    # ---- registered tests in the last 7 days
    exps = []
    for l in (ROOT / "logs/experiments.jsonl").read_text().splitlines():
        if not l.strip().startswith("{"):
            continue
        e = json.loads(l)
        if str(e.get("id", "")).endswith("-RESULT") and e.get("ts", "") >= (today - pd.Timedelta(days=7)).isoformat():
            exps.append(dict(id=e["id"].replace("-RESULT", ""), verdict=e.get("verdict"), calendar_years=e.get("calendar_years")))

    # ---- write
    stamp = today.strftime("%Y%m%d"); REP.mkdir(parents=True, exist_ok=True); LOG.mkdir(parents=True, exist_ok=True)
    pct = lambda x: "n/a" if x is None or not np.isfinite(x) else f"{100 * x:+.0f}%"  # noqa: E731
    md = [f"# Regime note — {today.date()}", "", "Exploration, changes no rule. Numbers: V3.1 weekly lists (6-month outcome after costs) by the quarter bought; "
          "market = equal-weight Rs 1 cr universe over the same days. Events are listed while those lists were held; 'unverified' events still need their source checked.", "",
          "## Quarters", "", "| Quarter | Grade | Lists | List 6m | Market 6m | Edge | FPI 60d | India VIX | Rupee 3m | Fed 6m | India rate 6m |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for q, r in Q.iterrows():
        md.append(f"| {q} | {r['grade']} | {int(r['lists'])} | {pct(r['list_ret'])} | {pct(r['mkt_ret'])} | {pct(r['excess'])} | {pct(r['fpi_60d_pct'])} | "
                  f"{r['india_vix']:.0f} | {pct(r['rupee_3m'])} | {r['fed_6m']:+.2f} | {r['india_rate_6m']:+.2f} |")
    md += ["", "## Events while each great / bad quarter's lists were held", ""]
    for q, r in Q[Q["grade"].isin(["great", "bad"])].iterrows():
        ev = hold_events[str(q)]
        md.append(f"- **{q} ({r['grade']}, lists {pct(r['list_ret'])})**: " + ("; ".join(f"{x['date']} {x['title']}" + ("" if x["verified"] else " (unverified)") for x in ev) or "no listed events"))
    md += ["", "## Did it repeat? (lists bought 90 days before / after each event vs the average list)", "",
           "| Event type | Episodes | Before vs avg | Same side | After vs avg | Same side | Pattern? | Dates |", "|---|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        md.append(f"| {r['type']} | {r['occurrences']} | {r['before_vs_avg']}% | {r['before_same_side']}% | {r['after_vs_avg']}% | {r['after_same_side']}% | "
                  f"{'yes' if r['before_pattern'] or r['after_pattern'] else 'no'} | {r['dates']} |")
    md += ["", f"Average list: {100 * avg:+.1f}%. Occurrences within 120 days of each other count as one episode; policy rates count only cycle turns. "
           "A pattern needs >= 5 episodes and >= 80% on the same side; even then it is a hypothesis to register, not a rule.", "",
           "## Today", "", f"Closest past quarters by macro picture: " + "; ".join(f"{a['quarter']} (lists {a['list_ret']:+.0f}%, {a['grade']})" for a in analogs),
           "", "Today's readings: " + ", ".join(f"{FEATS[f]} {now[f]:+.3g}" for f in feats if np.isfinite(now[f])), "",
           "Upcoming events (150 days): " + ("; ".join(f"{u['date']} {u['title']}" + ("" if u["verified"] else " (unverified)") for u in upcoming) or "none listed"), "",
           "## Registered tests in the last 7 days", ""]
    md += [f"- {e['id']}: {e['verdict']}" for e in exps] or ["- none"]
    md += ["", "## Context (added by the weekly skill; not evidence)", "", "_pending_"]
    (REP / f"regime_{stamp}.md").write_text("\n".join(md) + "\n")
    pat = [r["type"] for _, r in R.iterrows() if r["before_pattern"] or r["after_pattern"]]
    tele = [f"🧭 Regime note · {today:%b %d}",
            "Closest past quarters: " + ", ".join(f"{a['quarter']} {a['list_ret']:+.0f}%" for a in analogs),
            "Repeating event patterns: " + (", ".join(pat) if pat else "none yet (too few cases)"),
            "Coming up: " + (", ".join(f"{pd.Timestamp(u['date']):%b %d} {u['title']}" for u in upcoming[:3]) or "nothing listed")]
    out = dict(date=str(today.date()), quarters=Q.reset_index().assign(quarter=lambda x: x["quarter"].astype(str), first=lambda x: x["first"].astype(str)).to_dict("records"),
               hold_events=hold_events, repeat_check=R.to_dict("records"), average_list=avg, analogs=analogs, today=now.to_dict(), upcoming=upcoming, recent_tests=exps,
               telegram_lines=tele)
    (LOG / f"regime_{stamp}.json").write_text(json.dumps(out, indent=1, default=str))
    (LOG / f"regime_{stamp}.json.manifest.json").write_text(json.dumps(dict(dataset=f"regime_{stamp}.json", producer="src/agentic/regime_explainer.py", kind="exploration",
        definitions=__doc__, inputs=["v3_rule (V3.1)", "price panel", "data/derived/macro_drivers_fred.parquet", "data/derived/india_vix.parquet", "data/derived/us_macro_fred.parquet",
                                     "data/derived/nsdl_fpi_daily.parquet", "configs/event_calendar.yaml", "logs/experiments.jsonl"],
        units=dict(list_ret="fraction, 6-month list return after costs", mkt_ret="fraction", excess="fraction", fpi_60d_pct="fraction of FPI turnover",
                   fed_6m="percentage points", india_rate_6m="percentage points", before_vs_avg="percentage points vs the average list"),
        updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    (REP / "README.md").write_text("# Weekly regime notes\n\nOne note per Friday (src/agentic/regime_explainer.py + skill sgm-regime-explainer): which quarters' V3.1 lists did "
                                   "well or badly, the macro picture and events while they were held, whether event types repeat, and today's closest past quarters. "
                                   "Exploration only; data in logs/regime/.\n")
    print("\n".join(md[:8])); print(f"... wrote reports/regime/regime_{stamp}.md and logs/regime/regime_{stamp}.json"); print("\n".join(tele))


if __name__ == "__main__":
    main()
