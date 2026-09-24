"""LEADER SLEEVE report — reports/leader_sleeve_<session>.md, every session.

Sibling of render_basket_report.py (which sits at the 400-LOC line): same QUANT /
QUAL / MACRO columns, reusing its _qual / _macro / _sector_map helpers so filings,
insider, block-deal and holder data come from exactly the same sources. Adds the
sleeve's own facts: hot-industry group return, own 60d/252d run (fresh vs extended),
cheap-vs-industry PE, days held / 126, peak, trough, and the P(2x) prior of the sim
cell the name sits in. Derived from logs/leader_sleeve/{screen_*.json,outcomes.jsonl}
only — never a second source of truth. PAPER sizing until >= 13 weekly cohorts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
from render_basket_report import (ANN, BLOCK, CA, INDUSTRY, MACRO, NEWS, PIT, PRICES,  # noqa: E402
                                  SUPERSTAR, _latest_basket, _macro, _qual, _sector_map)

DIR = ROOT / "logs/leader_sleeve"
NEWS_FEED = ROOT / "data/derived/news_feed.parquet"          # RSS headlines, Apr-2026+ only
FULLTEXT = ROOT / "data/derived/filing_fulltext.parquet"     # fetch_filing_text.py
# Only a change-of-control offer — NOT routine SAST Reg 29/31 holding/pledge disclosures.
TAKEOVER_RE = r"open offer|detailed public statement|draft letter of offer"
NEWS_DAYS = 30
HOLD_TD, PAPER_MIN_COHORTS = 126, 13
# P(touched 2x within 126td) by sim cell and era. EXTENDED = production cell on the full nse4
# universe (sim_leader_cell_v2 --map nse4, survivorship-clean). Cheap/fresh cells: v1 sim on the
# old smIndustry map (logs/leader_sleeve/exp_extended_20260923.log) — not yet re-run on nse4.
PRIOR_2X = {"LEADER & cheap": (6.4, 12.3), "EXTENDED": (5.4, 7.5), "FRESH": (2.1, 6.9)}
# Theme drivers read from macro_panel (industry -> column, label). Gold = build_gold_feed.py.
DRIVERS = {"Gems Jewellery And Watches": ("gold_inr_idx", "gold INR (NSE gold ETFs)"),
           "Oil Exploration/Production": ("brent", "Brent")}


def _driver(industry: str, macp: pd.DataFrame, since: str | None, ret: pd.Series | None = None) -> str | None:
    col, lab = DRIVERS.get(industry, (None, None))
    if col is None or macp.empty or col not in macp:
        return None
    s = macp.set_index("trade_date")[col].dropna()
    if len(s) < 61:
        return None
    ch = lambda k: (s.iloc[-1] / s.iloc[-1 - k] - 1) * 100
    out = f"driver {lab}: 20d {ch(20):+.1f}%, 60d {ch(60):+.1f}%"
    if since:
        base = s[s.index <= pd.Timestamp(since)]
        if len(base):
            out += f", since screen {(s.iloc[-1] / base.iloc[-1] - 1) * 100:+.1f}%"
    if ret is not None and since:
        d = s.pct_change()
        t0 = pd.Timestamp(since)
        out += (f"; stock-vs-driver daily corr 60d pre-screen {_corr(ret, d, t0 - pd.Timedelta(days=90), t0)}, "
                f"since {_corr(ret, d, t0 + pd.Timedelta(days=1), s.index[-1])}")
    return out + f" (thru {s.index[-1].date()})"


def _name_hit(txt: pd.Series, sym: str, company: str | None) -> pd.Series:
    """Headline mentions a name: the symbol as an UPPER-CASE word (so OIL != 'oil'), or the first
    two words of the registered name, any case (the RSS tagger matches symbols only and missed
    'GRT Jewellers to acquire ... TBZ (Tribhovandas Bhimji Zaveri)')."""
    hit = txt.str.contains(rf"\b{sym}\b", case=True, regex=True)
    if company:
        w = [x for x in company.replace("(India)", "").split() if x.lower() not in {"limited", "ltd", "ltd.", "the"}]
        if len(w) >= 2 and len(" ".join(w[:2])) >= 8:
            hit |= txt.str.contains(" ".join(w[:2]), case=False, regex=False)
    return hit


def _news(sym: str, company: str | None, news: pd.DataFrame, thru: pd.Timestamp) -> str | None:
    if news.empty:
        return None
    k = news[(news["d"] > thru - pd.Timedelta(days=NEWS_DAYS)) & (news["d"] <= thru + pd.Timedelta(days=1))]
    k = k[_name_hit(k["txt"], sym, company)]
    if k.empty:
        return f"news {NEWS_DAYS}d: none in RSS store"
    top = "; ".join(f"{r.d:%d-%b} {r.title[:95]}" for r in k.sort_values("d").tail(2).itertuples())
    return f"news {NEWS_DAYS}d ({len(k)}): {top}"


def _takeover(sym: str, ann: pd.DataFrame | None, ft: pd.DataFrame, thru: pd.Timestamp) -> str | None:
    if ann is None or ann.empty:
        return None
    a = ann[(ann["event_date"] >= thru - pd.Timedelta(days=120))]
    txt = a["description"].fillna("") + " " + a["attachment_text"].fillna("")
    if not txt.str.contains(TAKEOVER_RE, case=False, regex=True).any():
        return None
    px = None
    if not ft.empty:
        t = ft[ft["symbol"] == sym]["text"].str.extract(r"offer price o\S* INR ([\d,]+\.\d+)", expand=False).dropna()
        px = t.iloc[0] if len(t) else None
    return "⚠ TAKEOVER TARGET (SAST open offer" + (f" at ₹{px}" if px else "") + ") — event-driven, not a theme leader"


def _corr(r: pd.Series, drv: pd.Series, a, b) -> str:
    j = pd.concat([r, drv], axis=1).loc[a:b].dropna()
    return f"{j.corr().iloc[0, 1]:+.2f} (n={len(j)})" if len(j) >= 8 else "n/a"


def _cell(n: dict) -> str:
    if n.get("pe_ind") is not None and n["pe_ind"] < 1:
        return "LEADER & cheap"
    return "EXTENDED" if n.get("run") == "extended" else "FRESH"


def _pct(v, nd=1):
    return "n/a" if v is None else f"{v*100:+.{nd}f}%"


def _quant(n: dict, o: dict | None) -> str:
    cell = _cell(n); disc, conf = PRIOR_2X[cell]
    bits = [f"hot industry grp60 {_pct(n['grp60'])} · own 60d {_pct(n['own60'])} / 252d {_pct(n.get('own252'))} "
            f"({n.get('run', '?').upper()})",
            f"PE {n['pe']:.1f} → {n['pe_ind']:.2f}× industry median ({'cheap' if n['pe_ind'] < 1 else 'not cheap'})"
            if n.get("pe") is not None and n.get("pe_ind") is not None else "PE n/a (loss or unmapped)",
            f"RTW {n['rtw']} · band {n['band']} · {'above' if n.get('above200') else 'BELOW'} 200DMA",
            f"P(2x/126td) prior {conf:.1f}% conf / {disc:.1f}% disc ({cell} cell)"]
    if o and o.get("status") != "PENDING":
        bits.append(f"day {o['days']}/{HOLD_TD} · peak {o['peak']:+.1f}% · trough {o['trough']:+.1f}%")
    return " · ".join(bits)


def main() -> None:
    screens = [json.loads(p.read_text()) for p in sorted(DIR.glob("screen_*.json"))]
    if not screens:
        raise SystemExit("no leader screens")
    outs = {}
    if (DIR / "outcomes.jsonl").exists():
        for l in (DIR / "outcomes.jsonl").read_text().splitlines():
            if l.strip():
                d = json.loads(l); outs[d["screen_id"]] = d          # last line per screen wins
    live = [s for s in screens if outs.get(s["screen_id"], {}).get("status") != "CLOSED"]
    names = sorted({n["symbol"] for s in live for n in s["names"]})

    def _opt(path, **kw):
        try:
            return pd.read_parquet(path, **kw) if path.exists() else pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    last = pd.to_datetime(_opt(PRICES, columns=["trade_date"], filters=[("symbol", "in", names)])["trade_date"]).max()
    ann = _opt(ANN, filters=[("symbol", "in", names)])
    if not ann.empty:
        ann["event_date"] = pd.to_datetime(ann["event_date"], errors="coerce")
    pit = _opt(PIT, columns=["symbol", "intimDt", "personCategory", "tdpTransactionType"])
    if not pit.empty:
        pit = pit[pit["symbol"].isin(names)].copy(); pit["d"] = pd.to_datetime(pit["intimDt"], errors="coerce")
    blk = _opt(BLOCK)
    if not blk.empty:
        blk = blk[blk["BD_SYMBOL"].isin(names)].copy()
        blk["dt"] = pd.to_datetime(blk["BD_DT_DATE"], format="mixed", errors="coerce")
    st = _opt(SUPERSTAR, columns=["symbol", "investor_name"])
    stars = {s: sorted(set(g["investor_name"])) for s, g in st.groupby("symbol")} if not st.empty else {}
    ca = _opt(CA, columns=["symbol", "company_name"])
    company = dict(ca.drop_duplicates("symbol").values) if not ca.empty else {}
    sec_map = _sector_map()
    ind = _opt(INDUSTRY)
    macp = _opt(MACRO)
    if not macp.empty:
        macp["trade_date"] = pd.to_datetime(macp["trade_date"]); macp = macp.sort_values("trade_date")
    mac = macp.iloc[-1] if not macp.empty else None
    news = _opt(NEWS_FEED, columns=["title", "desc", "pub_ts"])
    if not news.empty:
        news["d"] = pd.to_datetime(news["pub_ts"], errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        news = news.drop_duplicates("title"); news["txt"] = news["title"].fillna("") + " " + news["desc"].fillna("")
    ft = _opt(FULLTEXT, columns=["symbol", "text"])
    rets = _opt(PRICES, columns=["symbol", "trade_date", "close"], filters=[("symbol", "in", names)])
    if not rets.empty:
        rets["trade_date"] = pd.to_datetime(rets["trade_date"])
        rets = rets.pivot(index="trade_date", columns="symbol", values="close").sort_index().pct_change(fill_method=None)
    try:
        regime = json.loads(_latest_basket().read_text())["regime_gate"] + " (15d gate — sleeve is not regime-gated)"
    except SystemExit:
        regime = "n/a"

    sections = []
    for s in sorted(live, key=lambda s: s["screen_id"], reverse=True):
        o = outs.get(s["screen_id"], {})
        by = {r["symbol"]: r for r in o.get("names", [])}
        rows = []
        for n in s["names"]:
            sym, r = n["symbol"], by.get(n["symbol"])
            sub = lambda df, col="symbol": df[df[col] == sym] if not df.empty else None
            qual = _qual(sym, company.get(sym), n["industry"], sub(ann), sub(pit), sub(blk, "BD_SYMBOL"),
                         stars.get(sym, []), last)
            extra = [x for x in (_takeover(sym, sub(ann), ft, last), _news(sym, company.get(sym), news, last)) if x]
            if extra:
                qual = " · ".join(extra) + " · " + qual
            macro = _macro(sym, regime, sec_map, ind, mac)
            drv = _driver(n["industry"], macp, s["data_through"],
                          rets[sym] if not rets.empty and sym in rets else None)
            if drv:
                macro = drv + " · " + macro
            if r and r["status"] != "PENDING":
                pos = f"{r['status']} d{r['days']} | {r['entry']:.2f} ({r['entry_date']}) | {r['last']:.2f} | {r['ret_net']:+.1f}%"
            else:
                pos = "PENDING | next open | — | —"
            rows.append(f"| {sym} (#{n['rank']}) | {pos} | {_quant(n, r)} | {qual} | {macro} |")
        summ = (f"EW-{o['n_entered']} {o['ew_net']:+.2f}% net ({o['ew_gross']:+.2f}% gross) · top-4 RTW {o['top4_net']:+.2f}% net · "
                f"winners {o['winners']}/{o['n_entered']} · P(touched 2x) {o['p_2x']:.0%} · P(+50%) {o['p_50']:.0%} · "
                f"worst trough {o['worst_trough']:+.1f}%" if o.get("n_entered") else "not yet entered")
        sections.append(
            f"## Cohort {s['screen_id']} — data {s['data_through']} · day {o.get('days_held', 0)}/{HOLD_TD}"
            f"{' · EXTENDED-only' if s.get('extended_only') else ''}\n\n{summ}\n\n"
            "| Stock | Status | Entry | Last | Net | Rationale — QUANT | Rationale — QUAL (filings, insiders, holders) | Rationale — MACRO |\n"
            "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows))
    n_coh = len(screens)
    md = (f"# Leader sleeve — session {last.date()}\n\n"
          f"_PAPER ONLY · {n_coh} weekly cohort(s) on record, sizing needs >= {PAPER_MIN_COHORTS} · entry next open after "
          f"screen · hold {HOLD_TD}td, exit at close (TIME exit, no stops) · 0.5% round-trip in Net_\n\n"
          + "\n\n".join(sections) +
          "\n\n_Cell = hot industry (grp ret60 top decile) × top-3 own ret60 leaders. Backtest: LEADER & cheap +11.1/+19.3%/trade "
          "net by era, ~maxDD -10%, median troughs -14..-19% — drawdowns inside the hold are the base case, not a signal. "
          "Company one-liners marked [analyst note] are general knowledge, not pipeline data._\n")
    out = ROOT / f"reports/leader_sleeve_{last.date()}.md"
    out.write_text(md)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
