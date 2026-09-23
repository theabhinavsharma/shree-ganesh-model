# Basket report — 2026-09-08

_basket 2026-09-08_15d5pct.json · data through 2026-09-07 · regime WAIT · exit contract C2_

| Stock | Buy range | Target | Confidence | Rationale — micro | Rationale — macro | ETA (vol-implied) |
|---|---|---|---|---|---|---|
| INDOTECH (#1) | 3629.93–3703.27 | 3849.93 (+5%) | 6.05 | 1 engine confirms · band_fit 3.0/3 (RSI 52.5, 20d -3.6% vs earned bands) · ML 0.67 (honest zone 0.5-0.7) · cs 0.57 · 5d -1.0% · ADV ₹9.2cr · SL -7.8% (vol-scaled) · filings: quiet tape, no material filings 15d | regime WAIT · industry: Electrical Equipment (no sector panel) | day 1-3 |
| KRISHANA (#2) | 181.01–184.67 | 191.98 (+5%) | 4.62 | ML-only (Tier-2, no consensus) · band_fit 3.0/3 (RSI 47.9, 20d -0.4% vs earned bands) · ML 0.63 (honest zone 0.5-0.7) · cs 0.57 · 5d -4.3% · ADV ₹14.8cr · SL -5.9% (vol-scaled) · filings: quiet tape, no material filings 15d | regime WAIT · industry: None (no sector panel) | day 1-5 |
| SPARC (#3) | 203.81–207.93 | 216.16 (+5%) | 4.54 | ML-only (Tier-2, no consensus) · band_fit 3.0/3 (RSI 51.5, 20d -0.7% vs earned bands) · ML 0.53 (honest zone 0.5-0.7) · cs 0.55 · 5d +3.3% · ADV ₹18.4cr · SL -7.2% (vol-scaled) · filings: quiet tape, no material filings 15d | regime WAIT · industry: Pharmaceuticals (no sector panel) | day 1-4 |
| KAYNES (#4) | 3567.96–3640.04 | 3784.20 (+5%) | 4.44 | ML-only (Tier-2, no consensus) · band_fit 3.0/3 (RSI 45.9, 20d -3.5% vs earned bands) · ML 0.51 (honest zone 0.5-0.7) · cs 0.45 · 5d -2.2% · ADV ₹449.7cr · SL -7.9% (vol-scaled) · filings: quiet tape, no material filings 15d | regime WAIT · industry: None (no sector panel) | day 1-3 |
| MPSLTD (#5) | 2609.34–2662.06 | 2767.49 (+5%) | 4.11 | ML-only (Tier-2, no consensus) · band_fit 2.5/3 (RSI 48.0, 20d -8.0% vs earned bands) · ML 0.64 (honest zone 0.5-0.7) · cs 0.57 · 5d -1.8% · ADV ₹6.6cr · SL -6.0% (vol-scaled) · filings: quiet tape, no material filings 15d | regime WAIT · industry: Printing And Publishing (no sector panel) | day 1-5 |
| JGCHEM (#6) | 575.54–587.16 | 610.42 (+5%) | 4.06 | ML-only (Tier-2, no consensus) · band_fit 2.5/3 (RSI 49.4, 20d -6.6% vs earned bands) · ML 0.67 (honest zone 0.5-0.7) · cs 0.60 · 5d -8.2% · ADV ₹45.1cr · SL -11.6% (vol-scaled) · filings: quiet tape, no material filings 15d | regime WAIT · industry: None (no sector panel) | day 1-2 |
| MAPMYINDIA (#7) | 966.24–985.76 | 1024.80 (+5%) | 4.02 | ML-only (Tier-2, no consensus) · band_fit 2.5/3 (RSI 44.5, 20d -0.9% vs earned bands) · ML 0.65 (honest zone 0.5-0.7) · cs 0.44 · 5d -3.7% · ADV ₹19.7cr · SL -6.5% (vol-scaled) · filings: quiet tape, no material filings 15d | regime WAIT · industry: None (no sector panel) | day 1-4 |
| MBAPL (#8) | 161.38–164.64 | 171.16 (+5%) | 3.91 | ML-only (Tier-2, no consensus) · band_fit 2.5/3 (RSI 54.5, 20d -0.9% vs earned bands) · ML 0.73 (above honest zone) · cs 0.54 · 5d +1.0% · ADV ₹14.4cr · SL -4.7% (vol-scaled) · filings: quiet tape, no material filings 15d | regime WAIT · industry: None (no sector panel) | day 2-8 |
| AEGISLOG (RESERVE) | 1241.16–1266.24 | 1316.38 (+5%) | 4.12 | ML-only (Tier-2, no consensus) · band_fit 2.5/3 (RSI 45.6, 20d -6.0% vs earned bands) · ML 0.62 (honest zone 0.5-0.7) · cs 0.54 · 5d -0.9% · ADV ₹105.7cr · SL -9.1% (vol-scaled) · filings: quiet tape, no material filings 15d | regime WAIT · industry: None (no sector panel) | day 1-2 |
| AVANTEL (RESERVE) | 155.17–158.31 | 164.58 (+5%) | 4.10 | ML-only (Tier-2, no consensus) · band_fit 2.5/3 (RSI 42.5, 20d -3.4% vs earned bands) · ML 0.61 (honest zone 0.5-0.7) · cs 0.44 · 5d -2.1% · ADV ₹12.0cr · SL -4.0% (vol-scaled) · filings: quiet tape, no material filings 15d | regime WAIT · industry: None (no sector panel) | day 3-11 |

**Entry:** Place AMO limit orders over the weekend at buy_high — they enter Monday's 9:00-9:07 pre-open auction; you get the auction price if it opens inside your limit If a pick opens ABOVE buy_high: DO NOT CHASE — its +5% is already spent in the gap

**Exit (C2):** +5% target touch: sell HALF, trail remainder at +2.5% · VOL-SCALED SL per pick (sl_pct field = 3x stock's own 20d daily vol, floor -3%, cap -12%): sell 100% at sl_3pct price, no exceptions · Day 15: exit whatever remains at market

_ETA = median first-passage of a driftless walk to +5% in the stock's own 20d vol; not a calibrated forecast. Confidence = min(engines,3)×1.5 + band_fit + honest-zone-ML×2 + cs×0.5._
