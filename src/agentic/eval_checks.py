"""Coded checks for evals/registry.yaml (2026-09-29). One function per `check:` name.

Each takes a context dict and returns dict(status, value, detail):
  status  PASS | FAIL | WARN (soft miss) | PENDING (waiting on a person) | SKIP (input not present yet)
The plain-English statement, threshold and action live in the registry, not here; this file only measures.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Code/Zoom")
DER = ROOT / "data/derived"
PANEL = DER / "stock_daily_facts_adjusted_2015plus.parquet"
PY = "/usr/bin/python3"
EVAL_DIR = ROOT / "logs/evals"


def _res(status: str, value=None, detail: str = "") -> dict:
    return dict(status=status, value=value, detail=detail)


def last_session() -> pd.Timestamp:
    d = pd.read_parquet(PANEL, columns=["trade_date"])["trade_date"]
    return pd.to_datetime(d).max()


# ---------------------------------------------------------------- data
def coverage(ctx: dict) -> dict:
    r = subprocess.run([PY, str(ROOT / "src/agentic/eval_panel_coverage.py"), "--sessions", "5"], capture_output=True, text=True, cwd=ROOT)
    f = EVAL_DIR / f"panel_coverage_{ctx['last_session'].date()}.json"
    if not f.exists():
        return _res("FAIL", None, f"no coverage file for {ctx['last_session'].date()} (eval exit {r.returncode})")
    v = json.loads(f.read_text())
    rows = [x for x in v.get("sessions", v.get("results", [])) if isinstance(x, dict) and x.get("coverage") is not None]
    last = rows[-1] if rows else {}
    val = f"{last.get('panel')}/{last.get('raw')} ({last.get('coverage', 0):.2%})" if last else None
    return _res("PASS" if v.get("status") == "PASS" else "FAIL", val, "; ".join(v.get("fails", []))[:300])


def freshness(ctx: dict) -> dict:
    r = subprocess.run([PY, str(ROOT / "src/agentic/verify_freshness.py")], capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "SGM_FRESH_CORE": "1"})   # 15D engine outputs: checked by the 15D step itself
    out = r.stdout + r.stderr
    n = re.search(r"(\d+) inputs checked", out)
    if r.returncode == 0 and "ALL FRESH" in out:
        return _res("PASS", n.group(1) + " inputs fresh" if n else "all fresh")
    bad = [l.strip() for l in out.splitlines() if "FAIL" in l or "STALE" in l][:4]
    return _res("FAIL", None, " | ".join(bad)[:300])


def no_duplicates(ctx: dict) -> dict:
    p = pd.read_parquet(PANEL, columns=["symbol", "trade_date"])
    n = int(p.duplicated(["symbol", "trade_date"]).sum())
    return _res("PASS" if n == 0 else "FAIL", n, f"{len(p):,} rows checked")


def no_holiday_copies(ctx: dict) -> dict:
    lo = ctx["last_session"] - pd.Timedelta(days=45)
    p = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "open", "high", "low", "close"], filters=[("trade_date", ">=", lo)])
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    days = sorted(p["trade_date"].unique())[-21:]
    worst, flagged = 0.0, []
    for a, b in zip(days[:-1], days[1:]):
        m = p[p["trade_date"] == b].merge(p[p["trade_date"] == a], on="symbol", suffixes=("", "_p"))
        if m.empty:
            continue
        same = ((m["open"] == m["open_p"]) & (m["high"] == m["high_p"]) & (m["low"] == m["low_p"]) & (m["close"] == m["close_p"])).mean()
        worst = max(worst, float(same))
        if same >= 0.90:
            flagged.append(str(pd.Timestamp(b).date()))
    return _res("PASS" if not flagged else "FAIL", f"max repeat share {worst:.1%}", ", ".join(flagged))


def ca_before_prices(ctx: dict) -> dict:
    s = json.loads((ROOT / "logs/daily_data_layer_status.json").read_text())
    passed = s.get("passed", [])
    if "corp_actions" not in passed or "prices" not in passed:
        return _res("FAIL", s.get("date"), f"passed={passed[:6]}")
    ok = passed.index("corp_actions") < passed.index("prices")
    return _res("PASS" if ok else "FAIL", s.get("date"), "corp_actions ran before prices" if ok else "prices ran first")


MANIFEST_TABLES = ["stock_daily_facts_adjusted_2015plus.parquet", "pnl_quarterly.parquet", "order_fulltext.parquet", "order_daily.parquet",
                   "order_book_filings.parquet", "industry_scores.parquet", "industry_scores_policy.parquet", "budget_capex.parquet",
                   "iip_monthly.parquet", "core_sector_monthly.parquet", "pib_releases.parquet", "macro_drivers_fred.parquet",
                   "mcap_pit.parquet", "security_master.parquet", "usdinr_history.parquet"]


def manifests(ctx: dict) -> dict:
    missing = []
    for t in MANIFEST_TABLES:
        m = DER / (t + ".manifest.json")
        if not m.exists():
            missing.append(t)
            continue
        txt = m.read_text()
        if not any(k in txt for k in ('"units"', '"columns"', '"unit"')):
            missing.append(t + " (no units/columns)")
    return _res("PASS" if not missing else "WARN", f"{len(MANIFEST_TABLES) - len(missing)}/{len(MANIFEST_TABLES)}", ", ".join(missing))


# ---------------------------------------------------------------- point in time
def filings_after_period(ctx: dict) -> dict:
    q = pd.read_parquet(DER / "pnl_quarterly.parquet", columns=["quarter_end", "filing_dt"])
    early = int((pd.to_datetime(q["filing_dt"]) < pd.to_datetime(q["quarter_end"])).sum())
    late = 0
    if (DER / "order_daily.parquet").exists():
        o = pd.read_parquet(DER / "order_daily.parquet", columns=["ts", "fetched"])
        late = int((pd.to_datetime(o["ts"]) > pd.to_datetime(o["fetched"])).sum())
    return _res("PASS" if early + late == 0 else "FAIL", early + late, f"P&L filed before quarter end: {early}; orders dated after fetch: {late}")


def model_scores_current(ctx: dict) -> dict:
    d = pd.to_datetime(pd.read_parquet(ROOT / "logs/leader_sleeve/anatomy_1p5x/bakeoff_preds.parquet", columns=["trade_date"])["trade_date"]).max()
    age = (ctx["last_session"] - d).days
    return _res("PASS" if age <= 7 else "WARN", f"{age} days", f"latest score {d.date()} vs last session {ctx['last_session'].date()}")


# ---------------------------------------------------------------- provenance / hallucination guards
def external_rows_sourced(ctx: dict) -> dict:
    spec = {"budget_capex.parquet": ("source_url", "pub_date"), "iip_monthly.parquet": ("source_url", "release_date"),
            "core_sector_monthly.parquet": ("source_url", "release_date"), "pib_releases.parquet": ("pib_id", "pub_date")}
    bad, total, notes = 0, 0, []
    for f, (src, dt) in spec.items():
        p = DER / f
        if not p.exists():
            notes.append(f"{f} missing")
            continue
        x = pd.read_parquet(p, columns=[src, dt])
        n = int((x[src].isna() | (x[src].astype(str).str.strip() == "") | x[dt].isna()).sum())
        bad += n; total += len(x)
        if n:
            notes.append(f"{f}: {n} unsourced")
    return _res("PASS" if bad == 0 and not any("missing" in s for s in notes) else "FAIL", f"{bad} of {total:,} rows unsourced", "; ".join(notes))


NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
UNIT_MULT = (1.0, 0.01, 0.1, 100.0, 1e3, 1e5, 1e-7, 1e-5)   # crore, lakh, million, billion, thousand crore, lakh crore, rupees, thousand


def _grounded(amount: float, text: str) -> bool:
    for s in NUM.findall(text or ""):
        try:
            v = float(s.replace(",", ""))
        except ValueError:
            continue
        if v <= 0:
            continue
        if any(abs(v * m - amount) <= 0.02 * amount for m in UNIT_MULT):
            return True
        if any(60 <= amount / (v * m) <= 100 for m in (0.1, 100.0, 1e-7)):   # USD million / billion / plain, at a historical rupee rate
            return True
    return False


def order_amount_grounded(ctx: dict) -> dict:
    p = DER / "order_daily.parquet"
    if not p.exists():
        return _res("SKIP", None, "order_daily not built yet")
    o = pd.read_parquet(p, columns=["symbol", "seq_id", "amount_cr", "amount_snippet", "headline"]).dropna(subset=["amount_cr"])
    ok = o.apply(lambda r: _grounded(float(r["amount_cr"]), f"{r['amount_snippet']} {r['headline']}"), axis=1)
    share = float(ok.mean()) if len(o) else 1.0
    miss = o.loc[~ok, "symbol"].head(5).tolist()
    return _res("PASS" if share >= 0.98 else "WARN", f"{int(ok.sum())}/{len(o)} grounded ({share:.0%})", f"ungrounded e.g. {miss}" if miss else "")


# ---------------------------------------------------------------- model and strategy
def ranking_lift(ctx: dict) -> dict:
    P = pd.read_parquet(ROOT / "logs/leader_sleeve/anatomy_1p5x/bakeoff_preds.parquet", columns=["trade_date", "year", "y95", "tradable", "ensemble"])
    P = P[P["tradable"].astype(bool) & P["ensemble"].notna()]
    lab = P.groupby("year")["y95"].apply(lambda s: s.notna().mean())
    yrs = [y for y, v in lab.items() if v >= 0.95]
    if not yrs:
        return _res("SKIP", None, "no fully labelled year")
    y = max(yrs); Y = P[P["year"] == y]
    top = Y.sort_values(["trade_date", "ensemble"], ascending=[True, False]).groupby("trade_date").head(10)
    hit, base = top["y95"].mean(), Y["y95"].mean()
    lift = hit / base if base else np.nan
    return _res("PASS" if lift >= 1.5 else "WARN", f"{lift:.2f}x in {y}", f"top-10 hit {hit:.1%} vs base {base:.1%}")


def preregistered(ctx: dict) -> dict:
    L = [json.loads(l) for l in (ROOT / "logs/experiments.jsonl").read_text().splitlines() if l.strip().startswith("{")]
    reg, reg_text = {}, {}
    for e in L:
        if str(e.get("status", "")).startswith("REGISTERED"):
            reg.setdefault(e["id"], e.get("ts", "")); reg_text.setdefault(e["id"], json.dumps(e))
    viol, n = [], 0
    for e in L:
        i = str(e.get("id", ""))
        if "-RESULT" not in i or str(e.get("ts", "")) < "2026-09-27":
            continue
        n += 1
        base = i.split("-RESULT")[0]
        if base not in reg:   # a planned sub-arm (e.g. ...-industry-policy-H_policy) counts only if the parent's earlier
            par = next((p for p in reg if base.startswith(p + "-") and base[len(p) + 1:] in reg_text[p]), None)  # registration names it
            base = par or base
        if base not in reg or reg[base] > e.get("ts", ""):
            viol.append(i)
    return _res("PASS" if not viol else "FAIL", f"{n - len(viol)}/{n} results pre-registered", ", ".join(viol[:5]))


# ---------------------------------------------------------------- output
SCREEN_DIRS = {"sleeve": ROOT / "logs/leader_sleeve", "model": ROOT / "logs/model_screen", "sri_lakshmi": ROOT / "logs/sri_lakshmi"}


def screens_immutable(ctx: dict) -> dict:
    store = EVAL_DIR / "screen_hashes.json"
    seen = json.loads(store.read_text()) if store.exists() else {}
    changed, new = [], 0
    for tag, d in SCREEN_DIRS.items():
        for f in sorted(d.glob("screen_*.json")):
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            k = f"{tag}/{f.name}"
            if k in seen and seen[k] != h:
                changed.append(k)
            elif k not in seen:
                seen[k] = h; new += 1
    store.write_text(json.dumps(seen, indent=1))
    return _res("PASS" if not changed else "FAIL", f"{len(seen)} screens tracked ({new} new)", ", ".join(changed))


def one_screen_per_week(ctx: dict) -> dict:
    dup = []
    for tag, d in SCREEN_DIRS.items():
        wk = {}
        for f in d.glob("screen_*.json"):
            t = pd.Timestamp(json.loads(f.read_text())["data_through"]).isocalendar()[:2]
            wk.setdefault(tuple(t), []).append(f.name)
        dup += [f"{tag} {k}: {v}" for k, v in wk.items() if len(v) > 1]
    return _res("PASS" if not dup else "FAIL", len(dup), "; ".join(dup))


LIVE_TRACKS = ("sleeve", "sri_lakshmi")   # still scored daily (2026-09-30: model screen and SL-old stopped); immutability
                                          # (screens_immutable) still covers every folder in SCREEN_DIRS


def paper_scored(ctx: dict) -> dict:
    behind = []
    for tag, d in SCREEN_DIRS.items():
        if tag not in LIVE_TRACKS:
            continue
        outs = {}
        of = d / "outcomes.jsonl"
        if of.exists():
            for l in of.read_text().splitlines():
                if l.strip():
                    x = json.loads(l); outs[x["screen_id"]] = x
        for f in d.glob("screen_*.json"):
            sc = json.loads(f.read_text()); o = outs.get(sc["screen_id"])
            if pd.Timestamp(sc["data_through"]) >= ctx["last_session"]:
                continue                                                    # entry pending
            if o is None or o.get("status") == "CLOSED":
                if o is None:
                    behind.append(f"{tag} {sc['screen_id']}: never scored")
                continue
            if pd.Timestamp(o["scored_through"]) < ctx["last_session"]:
                behind.append(f"{tag} {sc['screen_id']}: through {o['scored_through']}")
    return _res("PASS" if not behind else "WARN", len(behind), "; ".join(behind))


def daily_run_happened(ctx: dict) -> dict:
    s = json.loads((ROOT / "logs/daily_data_layer_status.json").read_text())
    now = datetime.now()                                                   # laptop clock (US Eastern)
    exp = now.date() if now.hour * 60 + now.minute >= 19 * 60 + 30 else (now - timedelta(days=1)).date()
    while exp.weekday() >= 5:
        exp -= timedelta(days=1)
    got = pd.Timestamp(s.get("date")).date()
    return _res("PASS" if got >= exp else "FAIL", str(got), f"expected a run for {exp} (NSE holidays not yet excluded)")


def drive_checksums(ctx: dict) -> dict:
    base = Path.home() / "Library/CloudStorage/GoogleDrive-abhiengg.98@gmail.com/My Drive/SGM backups/backup"
    if not base.exists():
        return _res("FAIL", None, "Drive backup folder not mounted")
    latest = sorted(p for p in base.iterdir() if p.is_dir())[-1]
    m = json.loads((latest / "full_manifest.json").read_text())
    bad = []
    for v in m["volumes"]:
        h = hashlib.sha256()
        with open(latest / v["name"], "rb") as fh:
            for ch in iter(lambda: fh.read(1 << 23), b""):
                h.update(ch)
        if h.hexdigest() != v["sha256"]:
            bad.append(v["name"])
    return _res("PASS" if not bad else "FAIL", f"{len(m['volumes']) - len(bad)}/{len(m['volumes'])} volumes match ({latest.name})", ", ".join(bad))


# ---------------------------------------------------------------- human judgement
GOLD = ROOT / "evals/golden"


def order_book_golden(ctx: dict) -> dict:
    g = GOLD / "order_book_to_grade.csv"
    if not g.exists():
        return _res("SKIP", None, "sample not generated (src/agentic/make_golden_set.py)")
    x = pd.read_csv(g)
    done = x["human_verdict"].notna() & (x["human_verdict"].astype(str).str.strip() != "")
    if done.sum() < len(x):
        return _res("PENDING", f"{int(done.sum())}/{len(x)} graded", f"grade {g.relative_to(ROOT)}")
    acc = float((x["human_verdict"].str.strip().str.lower() == "correct").mean())
    return _res("PASS" if acc >= 0.80 else "WARN", f"{acc:.0%} correct (n={len(x)})")


def weekly_pick_review(ctx: dict) -> dict:
    need, missing = [], []
    for tag, d in SCREEN_DIRS.items():
        fs = sorted(d.glob("screen_*.json"))
        if fs:
            sid = json.loads(fs[-1].read_text())["screen_id"]
            need.append(f"{tag}_{sid}")
    for k in need:
        f = ROOT / "evals/human_review" / f"{k}.md"
        if not f.exists():
            missing.append(k); continue
        t = f.read_text()
        if "- [ ]" in t or "- [x]" not in t or "Signed:" not in t:
            missing.append(k + " (unchecked items or unsigned)")
    return _res("PASS" if not missing else "PENDING", f"{len(need) - len(missing)}/{len(need)} signed", ", ".join(missing))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def lessons_intact(ctx: dict) -> dict:
    sys.path.insert(0, str(ROOT / "src/agentic/trust"))
    import incident
    ok, probs = incident.check()
    L = incident._ledger()
    return _res("PASS" if ok else "FAIL", f"{len(L['evals'])} evals · {len(L['incidents'])} incidents · {sum(len(v) for v in L['lessons'].values())} lessons",
                "; ".join(probs[:5]) or "nothing that was ever recorded is missing")


def no_unexplained_cliffs(ctx: dict) -> dict:
    px = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "close", "avg_traded_value_20d", "price_adjustment_factor_to_present"])
    px = px.sort_values(["symbol", "trade_date"])
    g = px.groupby("symbol")
    r = g["close"].pct_change(fill_method=None)
    core = (g["avg_traded_value_20d"].shift(1) / 1e7 >= 5) & ((px["close"] / px["price_adjustment_factor_to_present"]).groupby(px["symbol"]).shift(1) > 50)
    import sim_leader_portfolio_7x as _sp                    # stocks only: ETFs have no industry and are never picked
    stocks = set(_sp.industry_maps()["analogs"].index)
    c = px.loc[(r <= -0.45) & core & px["symbol"].isin(stocks), ["symbol", "trade_date"]]
    ok = pd.read_csv(ROOT / "evals/verified_real_crashes.csv", parse_dates=["date"])
    c = c[~c.set_index(["symbol", "trade_date"]).index.isin(list(zip(ok["symbol"], ok["date"])))]
    return _res("PASS" if c.empty else "FAIL", len(c), "; ".join(f"{a} {b.date()}" for a, b in c.head(5).itertuples(index=False)) or "none in core-band stocks")


def renames_mapped(ctx: dict) -> dict:
    ca = pd.read_parquet(ROOT / "data/corporate_actions_full_history/normalized/stock_corporate_actions.parquet")
    ca["ex_date"] = pd.to_datetime(ca["ex_date"], errors="coerce")
    f = ROOT / "data/raw/nse_symbol_change/symbolchange.csv"
    if not f.exists():
        return _res("FAIL", None, "NSE symbolchange.csv missing")
    sc = pd.read_csv(f, header=None, names=["company", "old", "new", "date"], dtype=str)
    sc["date"] = pd.to_datetime(sc["date"].str.strip(), format="%d-%b-%Y", errors="coerce")
    sc["old"], sc["new"] = sc["old"].str.strip(), sc["new"].str.strip()
    orig = ca[ca["mapped_from_symbol"].isna() & ca["adjustment_factor"].notna()] if "mapped_from_symbol" in ca.columns else ca[ca["adjustment_factor"].notna()]
    back = {}
    for r in sc.dropna(subset=["date"]).itertuples():
        back.setdefault(r.new, []).append((r.date, r.old))
    have = set(zip(ca["symbol"], ca["ex_date"], ca["subject"]))
    need, miss = 0, []
    for r in orig.dropna(subset=["ex_date"]).itertuples():
        sym, seen = r.symbol, set()
        while sym in back and sym not in seen:          # full rename chain: the symbol traded on the ex-date
            seen.add(sym)
            prev = [o for d, o in sorted(back[sym], reverse=True) if d > r.ex_date]
            if not prev:
                break
            sym = prev[-1]
        if sym != r.symbol:
            need += 1
            if (sym, r.ex_date, r.subject) not in have:
                miss.append(f"{sym} (now {r.symbol}) {r.ex_date.date()}")
    return _res("PASS" if not miss else "FAIL", len(miss), "; ".join(miss[:5]) or f"{need} pre-rename splits/bonuses all mapped")


def insider_fresh(ctx: dict) -> dict:
    p = pd.read_parquet(ROOT / "data/derived/pit_history.parquet", columns=["date"])
    d = pd.to_datetime(p["date"], format="%d-%b-%Y %H:%M", errors="coerce")
    d = d[d <= pd.Timestamp.now()]                       # NSE rows with typo dates in the future are ignored
    newest = d.max()
    age = (pd.Timestamp.now() - newest).days
    return _res("PASS" if age <= 5 else "FAIL", str(newest.date()), f"newest disclosure {age} days old")


def message_grounded(ctx: dict) -> dict:
    rows = [r for r in _jsonl(ROOT / "logs/outbox/checks.jsonl") if r["kind"] in ("daily", "weekly")]
    if not rows:
        return _res("PENDING", None, "no message checked yet (run_sgm.py writes one per send)")
    r = rows[-1]
    why = ", ".join(r["unsourced"][:5] + r["problems"][:2])
    return _res("PASS" if r["ok"] else "FAIL", r["ts"][:16], f"{r['kind']} message " + ("ok" if r["ok"] else "held: " + why))


def claims_sourced(ctx: dict) -> dict:
    rows = [r for r in _jsonl(ROOT / "logs/trust/claims.jsonl") if not str(r.get("session", "")).startswith("test")
            and datetime.fromisoformat(r["ts"]) >= datetime.now() - timedelta(days=7)]
    n = sum(r["n_numbers"] for r in rows)
    if n == 0:
        return _res("PENDING", None, "no replies logged by the Stop hook in the last 7 days (hook loads on a new session)")
    bad = sum(r["n_unsourced"] for r in rows)
    return _res("PASS" if bad / n <= 0.05 else "FAIL", round(100 * bad / n, 1),
                f"{bad} of {n} numbers without a receipt in {len(rows)} replies")


def results_reproduce(ctx: dict) -> dict:
    last = {}
    for r in _jsonl(ROOT / "logs/trust/reproductions.jsonl"):
        last[r["exp"]] = r
    if not last:
        return _res("PENDING", None, "no reproduction run yet")
    bad = [e for e, r in last.items() if r["status"] == "NOT REPRODUCED"]
    old = [e for e, r in last.items() if datetime.fromisoformat(r["ts"]) < datetime.now() - timedelta(days=35)]
    worst = max(r.get("worst_diff_pts") or 0 for r in last.values())
    if bad:
        return _res("FAIL", worst, "not reproduced: " + ", ".join(bad))
    return _res("WARN" if old else "PASS", worst, f"{len(last)} tests, worst gap {worst} pts" + (f"; stale: {', '.join(old)}" if old else ""))


def claude_golden(ctx: dict) -> dict:
    auth = re.compile(r"Failed to authenticate|authentication_error|/login|Invalid API key")
    runs = [r for r in _jsonl(ROOT / "logs/trust/golden_runs.jsonl")      # a run where the CLI was logged out graded nothing
            if not all(auth.search(a.get("answer", "")) for a in r.get("answers", []))]
    if not runs:
        return _res("PENDING", None, "no golden run yet (needs the claude CLI logged in)")
    r = runs[-1]
    return _res("PASS" if r["accuracy"] >= 0.9 else "FAIL", round(100 * r["accuracy"]),
                f"{r['n']} questions on {r['ts'][:10]}; wrong: {', '.join(r['wrong']) or 'none'}")


def every_feed_guarded(ctx: dict) -> dict:
    """Every feed in daily_data_layer.sh has a freshness guard (configs/feed_guards.json) and its newest row is recent.
    FAIL = a feed with no guard, or its file/column is missing (fix the guard). WARN = a guarded feed is stale."""
    feeds = re.findall(r"^\s*run (\w+) ", (ROOT / "src/agentic/daily_data_layer.sh").read_text(), re.M)
    G = json.loads((ROOT / "configs/feed_guards.json").read_text())["feeds"]
    today, last = pd.Timestamp.now().normalize(), ctx["last_session"]
    broken, stale = [f"{f}: no guard" for f in feeds if f not in G], []
    for f in [f for f in feeds if f in G]:
        g, p = G[f], ROOT / G[f]["file"]
        if not p.exists():
            broken.append(f"{f}: {g['file']} missing"); continue
        if g["date_col"]:
            try:
                s = pd.read_parquet(p, columns=[g["date_col"]])[g["date_col"]]
            except Exception:
                broken.append(f"{f}: column {g['date_col']} missing"); continue
            d = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=True, utc=True).dt.tz_localize(None)
            newest = d[d <= pd.Timestamp.now()].max()
            ref = last if g["date_col"] in ("trade_date", "date", "event_date", "broadcast_date", "pub_date") else today
        else:
            newest, ref = pd.Timestamp(datetime.fromtimestamp(p.stat().st_mtime)), today
        if pd.isna(newest):
            broken.append(f"{f}: no readable dates"); continue
        age = int(np.busday_count(newest.date(), max(ref, newest).date()))
        if age > g["max_stale_bd"]:
            stale.append(f"{f} {age}bd old (limit {g['max_stale_bd']})")
    st = "FAIL" if broken else "WARN" if stale else "PASS"
    return _res(st, f"{len(feeds) - len(broken) - len(stale)}/{len(feeds)} feeds fresh", "; ".join(broken + stale)[:400])


def holidays_match_panel(ctx: dict) -> dict:
    """NSE's holiday list covers this year and agrees with the price panel: no rows on a listed weekday holiday (Muhurat
    sessions excepted) and no weekday missing from the panel unless NSE lists it. Buy/sell days in messages use this list."""
    f = ROOT / "data/derived/nse_holidays.parquet"
    if not f.exists():
        return _res("FAIL", None, "no NSE holiday list — run src/agentic/fetch_nse_holidays.py")
    H = pd.read_parquet(f, columns=["date", "description"]); H["date"] = pd.to_datetime(H["date"]).dt.normalize()
    last = ctx["last_session"]; y = last.year
    if y not in set(H["date"].dt.year):
        return _res("FAIL", None, f"holiday list has no {y} rows")
    panel = set(pd.to_datetime(pd.read_parquet(PANEL, columns=["trade_date"])["trade_date"].unique()))
    Hy = H[H["date"].dt.year == y]
    special = set(Hy.loc[Hy["description"].str.contains("Laxmi Pujan|Muhurat", case=False, na=False), "date"])
    traded = sorted(str(x.date()) for x in Hy["date"] if x.weekday() < 5 and x <= last and x in panel and x not in special)
    missing = sorted(str(x.date()) for x in pd.bdate_range(f"{y}-01-01", last) if x not in panel and x not in set(Hy["date"]))
    detail = "; ".join(p for p in (("panel has rows on listed holidays: " + ", ".join(traded)) if traded else "",
                                   ("weekdays missing from the panel that NSE does not list: " + ", ".join(missing)) if missing else "") if p)
    return _res("PASS" if not (traded or missing) else "FAIL", f"{len(Hy)} NSE holidays in {y}", detail)


def results_data_ready(ctx: dict) -> dict:
    """Every RESULT / RERUN logged from 2026-10-02 on has a DATA-READY line with status READY for the same experiment and
    run, logged earlier the same day (trust/data_ready.gate). Catches a test that computes a verdict without the GIGO gate."""
    L = [json.loads(l) for l in (ROOT / "logs/experiments.jsonl").read_text().splitlines() if l.strip().startswith("{")]
    ready = {}
    for e in L:
        if str(e.get("id", "")).endswith("-DATA-READY") and e.get("status") == "READY":
            ready.setdefault((e["id"][: -len("-DATA-READY")], e.get("run", "RESULT")), []).append(e.get("ts", ""))
    viol, n = [], 0
    for e in L:
        i, ts = str(e.get("id", "")), str(e.get("ts", ""))
        if ts < "2026-10-02T12:00" or ("-RESULT" not in i and "-RERUN-" not in i):
            continue
        n += 1
        base, run = (i.split("-RESULT")[0], "RESULT") if "-RESULT" in i else (i.split("-RERUN-")[0], i.split("-RERUN-")[1])
        if not any(r <= ts and r[:10] == ts[:10] for r in ready.get((base, run), [])):
            viol.append(i)
    return _res("PASS" if not viol else "FAIL", f"{n - len(viol)}/{n} results had their data checked", ", ".join(viol[:5]))


REGISTRY = {n: f for n, f in globals().items() if callable(f) and not n.startswith("_") and n not in ("last_session",)}
