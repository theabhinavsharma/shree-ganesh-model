"""EXP-2026-10-14-v32-stock-cap (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: V3.2 buys the same names week after week, so one stock can grow to ~20% of the money. Arms: no cap, 15% cap, 10%
cap, checked at purchase only (winners are never trimmed): a name already near the cap gets only the room left, the rest of that
week's lot goes equally to the list's other names; money that fits nowhere stays as cash in the lot. Chronological 26-slice ladder
(126-session hold, liquidity-scaled entry costs, dividends via Rv, first big payout / demerger in the hold as cash, list-day close).
PASS per cap vs no cap: CAGR higher in both eras, max drawdown not worse by > 2 points, at schedule 0 and >= 4 of 5.
Diagnostic: do repeat picks do better? Each pick's 6-month return after costs by how many of the previous 25 lists had it.
"""
import json, os, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic")); sys.path.insert(0, str(ROOT / "src/agentic/trust"))
import data_ready as dr, research_panel as rp, sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre, v3_rule  # noqa: E402
from test_v3_l1_newlist_cost import run_cost  # noqa: E402
from test_v32_hold_after_tax import boot  # noqa: E402
EXP_ID = "EXP-2026-10-14-v32-stock-cap"; OUTDIR = ROOT / "logs/leader_sleeve/v32_stock_cap"
START, HOLD, SLOTS, YEARS = pd.Timestamp("2019-04-01"), 126, 26, (2023, 2024, 2025, 2026)
CAPS = {"C_none": None, "C15": 0.15, "C10": 0.10}


def main():
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]; T = len(cal)
    ctx = v3_rule.context(imap); F1 = v3_rule.frame_v31(X["F"], D)
    L = rp.load_panel(["avg_traded_value_20d"]); L["adv"] = L["avg_traded_value_20d"] / 1e7
    ADV = rp.wide(L, "adv", cal).reindex(columns=D["syms"]).to_numpy(); del L
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
    TW = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill(); del S
    tcol = {s: i for i, s in enumerate(TW.columns)}; tpos = {d: i for i, d in enumerate(cal)}; TWn = TW.to_numpy(); del TW
    t2t = lambda s, d: (s in tcol) and TWn[tpos[d], tcol[s]] == 1.0  # noqa: E731
    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731
    lg = np.vstack([np.zeros((1, D["Rv"].shape[1])), np.cumsum(np.log1p(np.nan_to_num(D["Rv"])), axis=0)]); vr = D.get("vr", {})

    def path(q, i0, end):
        """Growth of Rs 1 in column q from the close of i0 to each close i0..end (first big payout / demerger -> cash)."""
        g = np.exp(lg[i0 + 1:end + 2, q] - lg[i0 + 1, q])
        for k, f in vr.get(int(q), ()):
            if i0 + 1 <= k <= i0 + HOLD - 1 and k <= end:
                r = k - i0; g[r + 1:] = f * g[r + 1:] + (1 - f) * g[r]; break
        return g

    def simulate(picks, weeks, cap):
        """Chronological ladder; returns (NAV series, max single-stock weight at purchase dates, per-list fill share)."""
        lots, slot_last = [], {}                     # lot: dict(i0, end, cols, amt (after cost), G (sessions x names), cash)
        maxw, placed = [], []
        for k, d in enumerate(weeks):
            i0 = tpos[d]; s = k % SLOTS; end = min(i0 + HOLD, T - 1)
            prev = slot_last.get(s)
            C = 1.0 if prev is None else float(lots[prev]["cash"] + lots[prev]["amt"] @ lots[prev]["G"][-1])
            cols = [D["cidx"][n] for n in picks.get(d, []) if n in D["cidx"]]
            # holdings and portfolio value at this close (other slices)
            expo, V = {}, C
            for s2 in range(SLOTS):
                if s2 == s:
                    continue
                j = slot_last.get(s2)
                if j is None:
                    V += 1.0; continue
                lt = lots[j]
                if lt["end"] >= i0:
                    r = i0 - lt["i0"]; vals = lt["amt"] * lt["G"][r]; V += lt["cash"] + vals.sum()
                    for q, v in zip(lt["cols"], vals):
                        expo[q] = expo.get(q, 0.0) + v
                else:
                    V += lt["cash"] + float(lt["amt"] @ lt["G"][-1])
            if expo:
                maxw.append(max(expo.values()) / V)
            amt = np.zeros(len(cols)); left = C
            if cols:
                c = cost(i0, np.array(cols))
                room = np.full(len(cols), np.inf) if cap is None else np.array([max(0.0, cap * V - expo.get(q, 0.0)) for q in cols]) / (1 - c)
                active = room > 1e-12
                while left > 1e-12 and active.any():
                    share = left / active.sum(); add = np.where(active, np.minimum(share, room - amt), 0.0)
                    amt += add; left -= add.sum(); active &= (room - amt) > 1e-12
                placed.append(1 - left / C)
                inv = amt * (1 - c)
                G = np.column_stack([path(q, i0, end) for q in cols])
            else:
                inv = np.zeros(0); G = np.ones((end - i0 + 1, 0)); placed.append(0.0)
            lots.append(dict(i0=i0, end=end, cols=cols, amt=inv, G=G, cash=left)); slot_last[s] = len(lots) - 1
        # NAV: each slice's value every session
        start = tpos[weeks[0]]; Vt = np.zeros(T)
        by_slot = {}
        for k, lt in enumerate(lots):
            by_slot.setdefault(k % SLOTS, []).append(lt)
        for s in range(SLOTS):
            v = np.ones(T)                                   # cash 1 until the slice's first lot
            for j, lt in enumerate(by_slot.get(s, [])):
                seg = lt["cash"] + lt["G"] @ lt["amt"] if len(lt["cols"]) else np.full(lt["end"] - lt["i0"] + 1, lt["cash"])
                v[lt["i0"]:lt["end"] + 1] = seg
                nxt = by_slot[s][j + 1]["i0"] if j + 1 < len(by_slot[s]) else T
                v[lt["end"] + 1:nxt] = seg[-1]
            Vt += v
        nav = pd.Series(Vt[start:] / SLOTS, index=cal[start:])
        return nav, (max(maxw) if maxw else 0.0), float(np.mean(placed))

    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    checks = [dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1)),
              dr.span("model scores", P["trade_date"], pd.Timestamp("2019-01-01"), cal[-1] - pd.Timedelta(days=10))]
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run="RESULT")
    del Px

    def met(nav):
        m = sp.metrics(nav); n = nav / nav.iloc[0]; ye = n.groupby(n.index.year).last(); r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
        mo = n.resample("ME").last().pct_change().min() * 100
        return {**{k: float(m[k]) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, "worst_month": float(mo), **{y: float(r.get(y, np.nan)) for y in YEARS}}

    lists = {o: {d: [s for s in l if not t2t(s, d)] for d, l in v3_rule.picks_v31(F1, [d for d in sp.weekly_grid(cal, o) if d >= START], imap, P, ctx).items()} for o in range(5)}
    w0 = list(lists[0])
    ref = sp.metrics(run_cost(D, X, lists[0], w0, "E0", cost)[0])["cagr"]; mine = sp.metrics(simulate(lists[0], w0, None)[0])["cagr"]
    print(f"wiring check: no-cap engine {mine:.2f} vs run_cost {ref:.2f} %/yr (tolerance 0.5)", flush=True)
    if abs(mine - ref) > 0.5:
        raise SystemExit("wiring check FAILED — no result reported")

    res, navs, info, ph = {a: [] for a in CAPS}, {}, {a: [] for a in CAPS}, {a: [] for a in ("C15", "C10")}
    for o in range(5):
        wk = list(lists[o])
        for a, cap in CAPS.items():
            nav, mw, pl = simulate(lists[o], wk, cap); navs[(a, o)] = nav; res[a].append(met(nav)); info[a].append((mw, pl))
        for a in ("C15", "C10"):
            ph[a].append(bool(sp.beats(res[a][o], res["C_none"][o])))
        print(f"schedule {o}: " + " · ".join(f"{a} {res[a][o]['cagr_disc']:.1f}/{res[a][o]['cagr_conf']:.1f}/{res[a][o]['maxdd']:.1f} (max stock {info[a][o][0]:.0%})" for a in CAPS), flush=True)
    passed = {a: bool(v[0] and sum(v) >= 4) for a, v in ph.items()}
    bs = {a: [boot(navs[(a, o)], navs[("C_none", o)]) for o in range(5)] for a in ("C15", "C10")}
    yrs = {a: {y: round(float(np.mean([x[y] for x in res[a]])), 1) for y in YEARS} for a in CAPS}

    # ---- diagnostic: repeat picks vs first-time picks (schedule 0)
    rows = []
    for k, d in enumerate(w0):
        i0 = tpos[d]
        if i0 + HOLD > T - 1:
            continue
        prev = set().union(*[set(lists[0][x]) for x in w0[max(0, k - 25):k]]) if k else set()
        cnt = {n: sum(n in lists[0][x] for x in w0[max(0, k - 25):k]) for n in lists[0][d]}
        for n in lists[0][d]:
            q = D["cidx"].get(n)
            if q is None:
                continue
            c = float(cost(i0, np.array([q]))[0]); rows.append(dict(n=cnt[n], r=((1 - c) * path(q, i0, i0 + HOLD)[-1] - 1) * 100))
    Rr = pd.DataFrame(rows); Rr["grp"] = pd.cut(Rr["n"], [-1, 0, 3, 9, 100], labels=["first time", "1-3 earlier lists", "4-9", "10+"])
    rep = Rr.groupby("grp", observed=True)["r"].agg(["size", "mean", "median", lambda s: (s < 0).mean() * 100]).round(1)
    rep.columns = ["picks", "avg 6m %", "median %", "lost %"]

    print(f"\n=== {EXP_ID} · entries {START.date()}..{cal[-1].date()} · schedule 0 (years = avg of 5) ===")
    for a in CAPS:
        r = res[a][0]
        print(f"{a:6s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · worst month {r['worst_month']:+.1f}% · "
              f"biggest stock {max(x[0] for x in info[a]):.0%} · money placed {np.mean([x[1] for x in info[a]]):.1%} · range {min(x['cagr'] for x in res[a]):.1f}..{max(x['cagr'] for x in res[a]):.1f}"
              + " · years " + " ".join(f"{y}:{yrs[a][y]:+.0f}" for y in YEARS))
    for a in ("C15", "C10"):
        print(f"{a} vs no cap: beats in {sum(ph[a])}/5 · gap {bs[a][0][0]:+.1f} pts/yr (95% {bs[a][0][1]:+.1f}..{bs[a][0][2]:+.1f}) · {'PASS' if passed[a] else 'no pass'}")
    print("\nrepeat picks (schedule 0, 6-month return after costs):"); print(rep.to_string())
    if os.environ.get("SGM_REPRO") == "1":
        return
    OUTDIR.mkdir(parents=True, exist_ok=True); now = datetime.now().isoformat(timespec="seconds")
    pd.DataFrame([dict(arm=a, **{c: round(res[a][0][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd", "worst_month")}, **{str(y): yrs[a][y] for y in YEARS},
                       biggest_stock=round(max(x[0] for x in info[a]), 3)) for a in CAPS]).to_csv(OUTDIR / "results.csv", index=False)
    rep.to_csv(OUTDIR / "repeat_picks.csv")
    for f, what in (("results.csv", "one row per arm, schedule 0; years = average of 5 schedules; biggest_stock = largest single-stock share of the portfolio at any purchase date"),
                    ("repeat_picks.csv", "V3.2 picks (schedule 0) grouped by how many of the previous 25 lists had the name; 6-month return after costs, percent")):
        (OUTDIR / f"{f}.manifest.json").write_text(json.dumps(dict(dataset=f, experiment=EXP_ID, producer="src/agentic/test_v32_stock_cap.py", definitions=__doc__, contents=what, updated=now), indent=1))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nPer-stock cap at purchase (10% / 15%) vs none on V3.2. C15: {'PASS' if passed['C15'] else 'no pass'}; C10: {'PASS' if passed['C10'] else 'no pass'}. results.csv, repeat_picks.csv.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=now, id=EXP_ID + "-RESULT", verdict=[a for a, v in passed.items() if v] or "no pass",
                                 arms={a: [round(res[a][0][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] for a in CAPS},
                                 phases={a: f"{sum(v)}/5" for a, v in ph.items()}, bootstrap=bs, calendar_years=yrs,
                                 biggest_stock={a: round(max(x[0] for x in info[a]), 3) for a in CAPS}, repeat_picks=rep.reset_index().astype(str).to_dict("records"),
                                 wiring=dict(no_cap=round(mine, 2), run_cost=round(ref, 2)), cols="CAGR, 2019-22, 2023+, maxDD")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
