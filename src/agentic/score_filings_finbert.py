"""UNIVERSE-WIDE filing scorer (law 2026-09-08: ALL stocks, ALL variables).

Two scores per filing, 2016-2026, every symbol:
  1. tone  : FinBERT (ProsusAI/finbert) pos/neg/neutral on desc + attchmntText
             — scored once per UNIQUE text (exchange summaries are templated,
             dedup cuts 1.29M rows to a fraction), then broadcast back.
  2. catdir: deterministic category direction from desc/attchmntText keywords —
             landmines (pledge creation, auditor resignation, rating downgrade,
             litigation/raid, promoter sell, dilution) vs boosters (order win,
             capacity expansion, buyback, promoter buy, rating upgrade).

Output: data/derived/filing_scores.parquet
  [symbol, event_dt, tone_pos, tone_neg, cat, catdir]
Checkpointed by unique-text batch; resume-safe. CPU-only, single process.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/Users/abhinavs./Documents/Zoom")
OUT = ROOT / "data/derived/filing_scores.parquet"
CKPT = ROOT / "data/derived/pnl_history/finbert_tone_ckpt.jsonl"

print("load filings…", flush=True)
ah = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet",
                     columns=["symbol", "sort_date", "desc", "attchmntText"])
ah["event_dt"] = pd.to_datetime(ah["sort_date"], errors="coerce")
ah = ah.dropna(subset=["event_dt"])
ah["text"] = (ah["desc"].fillna("") + ". " + ah["attchmntText"].fillna("")).str.strip().str.slice(0, 480)
print(f"  filings {len(ah):,} · unique texts {ah['text'].nunique():,}", flush=True)

# ---------- categorical direction (vectorized regex) ----------
NEG = {
    "pledge_up": r"pledge|encumbr",
    "auditor": r"resignation of (statutory )?auditor|auditor.{0,30}resign",
    "rating_down": r"downgrad",
    "litigation": r"insolvenc|NCLT|liquidation|search and seizure|raid|show cause|penal",
    "promoter_sell": r"disposal.{0,40}promoter|promoter.{0,40}(disposal|sale of)",
    "dilution": r"preferential (issue|allotment)|QIP|qualified institution|warrant",
    "default": r"default|delay in payment|one[- ]time settlement",
}
POS = {
    "order_win": r"receipt of order|order (received|win|worth)|bagg(ed|ing)|letter of intent|work order|contract (award|received)",
    "capacity": r"capacity (expansion|addition)|commercial production|new (plant|facility)|commission",
    "buyback": r"buy[- ]?back",
    "promoter_buy": r"acquisition.{0,40}promoter|promoter.{0,40}(acquisition|purchase)",
    "rating_up": r"upgrad",
    "dividend": r"bonus issue|interim dividend|special dividend",
}
ah["cat"] = ""
ah["catdir"] = 0
low = ah["text"].str.lower()
for name, pat in NEG.items():
    m = low.str.contains(pat, regex=True, na=False)
    ah.loc[m & (ah["cat"] == ""), "cat"] = name
    ah.loc[m, "catdir"] = -1
for name, pat in POS.items():
    m = low.str.contains(pat, regex=True, na=False) & (ah["catdir"] == 0)
    ah.loc[m & (ah["cat"] == ""), "cat"] = name
    ah.loc[m, "catdir"] = 1
print(f"  catdir: -1 {(ah['catdir']==-1).sum():,} · +1 {(ah['catdir']==1).sum():,} "
      f"· 0 {(ah['catdir']==0).sum():,}", flush=True)

# ---------- FinBERT tone on unique texts ----------
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
mdl.eval()
torch.set_num_threads(6)

uniq = ah["text"].drop_duplicates().reset_index(drop=True)
done = {}
if CKPT.exists():
    with open(CKPT) as f:
        for line in f:
            d = json.loads(line)
            done[d["t"]] = (d["p"], d["n"])
todo = [t for t in uniq if t not in done]
print(f"  tone: unique {len(uniq):,} · done {len(done):,} · todo {len(todo):,}", flush=True)

B = 96
with open(CKPT, "a") as ck, torch.no_grad():
    for i in range(0, len(todo), B):
        batch = todo[i:i + B]
        enc = tok(batch, return_tensors="pt", truncation=True, max_length=64, padding=True)
        probs = torch.softmax(mdl(**enc).logits, dim=1).numpy()  # [pos, neg, neu]
        for t, pr in zip(batch, probs):
            done[t] = (round(float(pr[0]), 4), round(float(pr[1]), 4))
            ck.write(json.dumps({"t": t, "p": done[t][0], "n": done[t][1]}) + "\n")
        ck.flush()
        if (i // B) % 50 == 0:
            print(f"  [{i:,}/{len(todo):,}]", flush=True)

ah["tone_pos"] = ah["text"].map(lambda t: done.get(t, (np.nan, np.nan))[0])
ah["tone_neg"] = ah["text"].map(lambda t: done.get(t, (np.nan, np.nan))[1])
out = ah[["symbol", "event_dt", "tone_pos", "tone_neg", "cat", "catdir"]]
out.to_parquet(OUT, index=False)
print(f"filing_scores: {len(out):,} rows -> {OUT.name}", flush=True)
print("FINBERT SCORING COMPLETE", flush=True)
