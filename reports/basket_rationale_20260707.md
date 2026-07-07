# Basket Rationale — 2026-07-07 (15D/+5%, 100% deploy, 8 × 12.5%)

**Data through**: 2026-07-06 close · **Regime gate**: `WAIT` · **Freshness**: 24/24 checks pass · **Exit rules**: book half at +5%, trail rest at +2.5%, hard SL -3%, timeout day 15

Every rationale below was independently fact-checked against the data by an adversarial verifier. Corrections applied where the checker caught errors.

---

## Master table

| # | Symbol | Sector | LTP | Buy zone | Target | SL | Band-fit | ML | RSI | 5d% | 20d% | 60d% | 120d% | 252d% | vs50DMA | vs200DMA | off 52wH | Vol/20d | Dlv/20d | ADV₹cr | Conviction |
|--:|:--|:--|--:|:-:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| 1 | **VIJAYA** | Diagnostics | 1347.40 | 1334–1361 | 1414.77 | 1306.98 | 3.0 | 0.61 | 54.4 | -1.5 | -1.8 | +37.2 | +36.4 | +33.5 | +5.6% | +26.9% | -3.6% | 0.69 | 0.99 | 49.2 | **4/5** |
| 2 | **DELTACORP** | Gaming/casinos | 65.22 | 64.57–65.87 | 68.48 | 63.26 | 3.0 | 0.67 | 46.8 | +1.2 | -1.5 | +11.8 | -6.7 | -28.3 | -6.0% | -4.2% | -31.5% | 0.38 | **1.36** | 18.7 | 3/5 |
| 3 | **MANALIPETC** | Petrochemicals | 63.31 | 62.68–63.94 | 66.48 | 61.41 | 3.0 | 0.57 | 50.8 | -0.3 | -2.0 | +15.0 | +8.4 | -9.9 | +4.8% | +6.0% | -20.1% | 0.32 | 0.72 | 5.8 | 3/5 |
| 4 | **IRCON** | Rail infra PSU | 134.26 | 132.92–135.60 | 140.97 | 130.23 | 3.0 | 0.55 | 45.1 | +0.1 | -1.7 | -1.6 | -17.4 | -30.6 | -5.1% | -11.5% | -31.5% | 0.70 | 0.92 | 23.3 | 3/5 |
| 5 | **PNBGILTS** | G-sec dealer | 89.63 | 88.73–90.53 | 94.11 | 86.94 | 2.5 | 0.67 | 53.8 | -1.6 | -5.1 | +27.4 | +11.0 | -6.8 | +9.3% | +10.2% | -22.3% | 0.31 | 0.92 | 13.7 | 3/5 |
| 6 | **CHAMBLFERT** | Urea/fertilizer | 476.05 | 471.29–480.81 | 499.85 | 461.77 | 2.5 | 0.61 | 52.7 | +1.5 | +1.8 | +4.2 | +8.2 | -13.5 | +3.0% | +4.0% | -16.4% | 0.28 | **1.21** | 49.0 | 3/5 |
| 7 | **CHENNPETRO** | Refining PSU | 1120.40 | 1109.20–1131.60 | 1176.42 | 1086.79 | 2.5 | 0.56 | 49.5 | +1.7 | -5.2 | +15.0 | +30.3 | +52.7 | +2.0% | +17.2% | -8.6% | 0.37 | 0.97 | 102.7 | 3/5 |
| 8 | **KIOCL** | Iron ore PSU | 387.05 | 383.18–390.92 | 406.40 | 375.44 | 2.5 | 0.54 | 44.1 | -1.2 | -4.5 | +8.1 | +3.5 | +28.8 | -2.7% | +0.8% | -38.3% | 0.33 | **1.24** | 6.8 | **2/5** |

All 8 are Tier-2 (ML-discovered, 0 engine consensus, regime WAIT → historical hit ≈ 24% vs 27.5% for the band-fit optimum).

## Macro context (all fresh, 2026-07-06)

| Signal | Value | Δ20d | Basket implication |
|---|--:|--:|---|
| **Brent** | $71.59 | **-26.4%** | Feedstock/fuel tailwind: MANALIPETC, CHAMBLFERT, CHENNPETRO (GRM, but near-term inventory-loss risk), IRCON (diesel/bitumen), DELTACORP (travel) |
| **WTI** | $71.87 | -23.8% | Confirms crude collapse |
| **USDINR** | 95.40 | +0.47% | Mild: helps KIOCL exports, hurts VIJAYA's imported reagents |
| **US 10Y** | 4.49% | -6 bps | Global duration rally → PNBGILTS trading-book markup |
| **DXY** | 120.69 | +0.51% | Neutral-slight EM headwind |
| **VIX** | 15.81 | **-26.5%** | Risk-on — supports mean-reversion trades |
| **SPX** | 7,537 | +2.08% | US firm |

## Sector RS (20d, vs NIFTY 500)

**Leaders**: REALTY +8.6% · PHARMA +2.9% (VIJAYA ✓) · MICROCAP +2.4% · INFRA +2.0% (IRCON ✓) · FMCG +1.4%
**Laggards**: SMALLCAP 100 -9.4% (MANALIPETC ✗) · OIL & GAS -9.0% (CHENNPETRO ✗) · METAL -8.4% (KIOCL ✗) · IT -7.7% · PSE -2.7% (IRCON/CHENNPETRO/KIOCL ✗)

---

## 1. VIJAYA — Diagnostics · conviction 4/5 · ✅ verified clean

**Setup**: consolidation-in-uptrend
**Why here**: All three optimal backtest bands align: 20d -1.8% in the [-5%,0%] sweet spot (27.5% hit), RSI 54.4 inside [45,55], ML 0.608 in the honest [0.5,0.7] zone — band_fit 3.0. Quiet low-volume digestion (-1.5% 5d, volume 0.69x) of a strong trend (+37.2% 60d, +33.5% 252d), just -3.6% off the 52-week high, in the second-strongest sector (PHARMA RS +2.93%).
**Bull**: Only a 3.6% move retakes the 52w high, so the +5% target at 1414.77 is a breakout continuation. Sector rotation at its back; delivery 0.99x through the pullback = holders not distributing.
**Bear**: Stretched +26.9% above the 200-SMA after a 37% two-month run; 50-SMA is 5.6% below price, so a routine mean-reversion to the 50-SMA blows through the -3% stop. Volume 0.69x can be read as buyer exhaustion near highs.
**Catalysts 15d**: None scheduled — a no-catalyst bands-and-sector trade (52w-high breakout attempt + pharma rotation).
**Macro**: Cheap energy is disinflationary for the domestic consumer whose discretionary spend drives diagnostic volumes; VIX collapse favors quality mid-cap healthcare. INR 95.4 a mild negative (imported reagents, no export offset).
**Flags**: extended vs 200-SMA · SL above the 50-SMA · no hard catalyst · RSI at top edge of optimal band

## 2. DELTACORP — Gaming/casinos · conviction 3/5 · ✔ verified (1 correction applied)

**Setup**: oversold-base
**Why here**: Perfect 3/3 band-fit: 20d -1.5% (best band), RSI 46.8 (sweet spot), ML 0.665 (honest zone). 60d recovery +11.8% consolidating; +32.2% off the 52w low with delivery 1.36x — quiet accumulation into a mild pullback. The two 2026-07-02 "Spurt in Volume" notices are exchange surveillance flags (the tagger mislabels them as order wins) — someone traded it heavily, not a business win.
**Bull**: Price 6% below 50-DMA and 4.2% below 200-DMA — a simple 50-DMA reclaim more than covers the +5% target. Delivery 1.36x + volume-spurt flags suggest accumulation in a stock -31.5% off its 52w high — cheap beta for a risk-on tape.
**Bear**: A bounce inside a genuine downtrend: -28.3% over 252d, below both DMAs, 20d drift already negative — continuation alone tags the SL. Volume just 0.38x = accumulation unconfirmed. Sector bucket (OTHER) lags NIFTY 500 by ~2.3 points of 20d return *(corrected from "over 6 points" — checker caught it)*.
**Catalysts 15d**: No dated business catalyst; possible Q1 results in window (unconfirmed). Technical trigger = 50-DMA reclaim.
**Macro**: Brent -26.4% cuts travel costs → Goa leisure demand; VIX risk-on favors beaten-down high-beta consumption; weak INR nudges tourists domestic.
**Flags**: structural downtrend · thin tape 0.38x · no real catalyst · surveillance attention raises whipsaw risk

## 3. MANALIPETC — Petrochemicals · conviction 3/5 · ✅ verified clean

**Setup**: consolidation-in-uptrend
**Why here**: Perfect band_fit 3.0 (20d -2.0%, RSI 50.8, ML 0.573). Quiet pullback in a medium-term uptrend: +15.0% 60d, +4.8% above 50-SMA, +6.0% above 200-SMA. June 12/18 volume-spurt notices = someone traded it heavily last month.
**Bull**: Brent -26.4% is a direct feedstock tailwind — polyol/PG spreads widen if product prices lag the crude crash. Stock digested its +15% move with only -2% drift; +5% to 66.48 needs no new high (-20.1% below 52w high = room under resistance).
**Bear**: The stop is tighter than the structure: 50-SMA is ~4.8% below price, so ordinary mean-reversion stops it out (whipsaw 44.9%). Weak current demand: volume 0.32x, delivery 0.72x, size cohort is the worst tape on the board (SMALLCAP 100 RS -9.4%). Falling crude drags polyol selling prices too.
**Catalysts 15d**: Spread expansion filtering into channel checks; possible Q1 board-meeting notice near window end. No dated trigger.
**Macro**: Crude-linked feedstock cheapens directly; INR 95.4 makes competing imported polyols costlier (supports realizations); smallcap RS shows risk-on hasn't reached this cohort.
**Flags**: size-cohort headwind · volume dry-up · ADV 5.8cr exit slippage · spread story not price story

## 4. IRCON — Rail infra PSU · conviction 3/5 · ✅ verified clean

**Setup**: basing-after-decline
**Why here**: Band_fit 3.0 (20d -1.7%, RSI 45.1, ML 0.551). Price stopped falling and is basing (5d +0.1%, 60d -1.6%) after a deep markdown (-17.4% 120d, -31.5% off 52w high). **Real catalyst: genuine order-win disclosure 2026-06-23 ("Bagging/Receiving of orders/contracts") + approval/results updates 2026-06-26.** Note: target 140.97 sits almost exactly at the 50-DMA (price -5.1% below it) — the trade is a mean-reversion to the 50-DMA.
**Bull**: Reversion to the 50-DMA delivers the full +5% without a trend change. Fresh order news into an active disclosure cadence (22 announcements/60d); INFRA sector +2.04% RS; Brent -26.4% cuts diesel/bitumen/logistics costs.
**Bear**: A base inside a structural downtrend, not a confirmed reversal (-30.6% 252d, -11.5% below 200-DMA). Flat 20d can be a pause before continuation. Zero accumulation evidence (volume 0.7x, delivery 0.92x). The falling 50-DMA sits right at the target — sellers likely exactly where profit must be booked. PSE RS -2.71% headwind.
**Catalysts 15d**: Contract-value details on the 06-23 order win; possible fresh orders; railway-capex news flow. FY results already out — no earnings event in window.
**Macro**: Brent collapse = direct margin tailwind for a construction contractor and protects the govt capex budget funding IRCON's order book; VIX risk-on supports beaten-down PSUs.
**Flags**: sector split INFRA↑/PSE↓ · structural downtrend · no volume confirmation · target = falling 50-DMA resistance

## 5. PNBGILTS — G-sec primary dealer · conviction 3/5 · ✔ verified (verifier false-positives: backtest numbers are correct, from the 10-yr backtest table)

**Setup**: pullback-in-uptrend
**Why here**: 20d -5.1% at the junction of the two best pullback bands (27.5% / 26.3% hit); RSI 53.8 optimal; ML 0.666 sweet spot — band_fit 2.5. Pullback inside an intact uptrend: +27.4% 60d, +9.3% above 50-SMA, +10.2% above 200-SMA.
**Bull**: **The one name where macro IS the thesis**: Brent -26.4% is a disinflation impulse for India, US 10Y -6bps gives global duration a tailwind — falling G-sec yields mark up a primary dealer's trading book almost mechanically. June 3-4 volume spurts consistent with positional accumulation before the current quiet pullback.
**Bear**: Pullback on dead volume (0.31x, ADV 13.7cr) — no buyer stepping in yet; thin names gap through stops. Bigger risk: rate view inverting — INR 95.4 weakening + DXY 120.7 constrain RBI easing room; a hot CPI print pushes yields up = direct mark-to-market hit → SL.
**Catalysts 15d**: June CPI print (mid-July) — Brent's slide should feed a soft reading, the key yield trigger in the window; weekly G-sec auctions; possible Q1 results. *(These are inferred macro events, not from the announcement file.)*
**Macro**: Most macro-aligned pick in the basket — disinflation + global bond rally flow straight into a primary dealer's P&L. Currency weakness is the one line that caps the rally.
**Flags**: volume 0.31x no follow-through · gap risk through SL · smallcap tape headwind · overhead supply -22.3% off high · RBI easing room limited

## 6. CHAMBLFERT — Urea/fertilizer · conviction 3/5 · ✅ verified clean

**Setup**: consolidation-in-uptrend
**Why here**: RSI 52.7 optimal; ML 0.611 sweet spot; 20d +1.8% narrowly misses the top band → band_fit 2.5. Constructive placement: +3% above 50-DMA, +4% above 200-DMA, grinding up (+4.2% 60d, +8.2% 120d), -16.4% off 52w high = room to target. **Delivery 1.21x on very quiet volume (0.28x) = low-noise accumulation.**
**Bull**: **Peak kharif urea demand coincides with a collapsing energy complex** — Brent -26.4% lowers gas-linked feedstock costs and eases subsidy math. Coiled just above both major MAs with neutral RSI; 476→500 is exactly the +5% move. ADV 49cr = clean entry.
**Bear**: Nobody is actually trading it — volume 0.28x means the +5% move needs buyers who aren't showing up yet. 252d -13.5% = a year of trapped supply overhead. Sector bucket lagging (money rotating to realty/pharma, not agri). An ordinary 3% market dip takes it through the SL.
**Catalysts 15d**: No scheduled event. Seasonal drivers: July kharif/monsoon urea offtake + gas-cost tailwind. Delivery uptick suggests positioning ahead of results season, but no Q1 date on record.
**Macro**: Brent collapse is the single most favorable macro input for a urea maker (gas feedstock + subsidy burden both improve). INR 95.4 partially offsets via dollar-priced gas imports — small next to the energy move.
**Flags**: thin participation 0.28x · sector lagging · overhead supply · no dated catalyst

## 7. CHENNPETRO — Refining PSU · conviction 3/5 · ✅ verified clean

**Setup**: pullback-to-support (50-DMA) in intact uptrend
**Why here**: 20d -5.2% in the second-best [-15%,-5%] band (26.3% hit); RSI 49.5 dead-center optimal; ML 0.561 sweet spot — band_fit 2.5. Trend intact: +52.7% over 252d, +17.2% above 200-DMA; the pullback landed it just 2% above the 50-DMA, -8.6% off the 52w high — a textbook reset, not a breakdown.
**Bull**: Mean reversion off the 50-DMA inside a +52.7% 1-yr uptrend; the +5% target sits below the 52w high (no fresh-high resistance needed). Brent -26.4% is a cheap-feedstock GRM tailwind once the fall stabilizes. Deep liquidity (ADV ₹103cr).
**Bear**: **A good chart in one of the worst sectors on the board** (OIL & GAS RS -9.0%, ENERGY -4.6%, PSE -2.7%) — the pullback may be sector distribution, not consolidation. The speed of the crude crash means a pure refiner faces inventory/adventitious losses into Q1 results. Volume 0.37x = zero accumulation footprint; one more sector-led red week takes the stop before any GRM narrative forms.
**Catalysts 15d**: Catalyst-light. Q1 earnings season kicks off mid-July (CPCL's print likely lands near window edge, but peer GRM commentary reprices it earlier); crude trajectory / OPEC+ headlines; rotation back into beaten-down O&G if -9.0% sector RS mean-reverts.
**Macro**: Brent -26.4% cuts feedstock and is medium-term GRM-supportive, **but a fall this fast first shows up as inventory losses — the macro tailwind and the near-term earnings risk are the same number.**
**Flags**: sector headwind O&G -9.0% + PSE -2.7% · inventory-loss risk into Q1 · no accumulation signature · no scheduled catalyst

## 8. KIOCL — Iron ore pellets PSU · conviction 2/5 (weakest) · ✔ verified (2 corrections applied)

**Setup**: pullback-to-support
**Why here**: 20d -4.5% in the best band; ML 0.542 sweet spot; RSI 44.1 just below optimal → band_fit 2.5. Pullback within an uptrend: -2.7% below 50-DMA but +0.8% above 200-DMA, +28.8% over 252d.
**Bull**: Mean-reversion off the 200-DMA back toward the 50-DMA covers most of the +5% path. Delivery 1.24x while volume dries to 0.33x hints supply is being absorbed by positional hands. Double macro assist: Brent -26.4% cuts pellet fuel/freight; INR 95.4 lifts export realizations.
**Bear**: METAL is the **third-worst sector** (RS -8.4%; SMALLCAP 100 and OIL & GAS are weaker) *(corrected from "second-worst" — checker caught it)*, and its second index home PSE is also negative. A stock 0.8% above its 200-DMA in a sector this weak can lose that support in one bad session; the SL at 375.44 sits ~2.2% below the 200-DMA *(corrected — SL is -3% from close, not from the DMA)*. -38.3% off the 52w high; ADV 6.8cr = exit slippage. No catalyst — the 60d flow is results clarifications (an exchange query trail, not good news).
**Catalysts 15d**: Nothing scheduled. Pure statistical mean-reversion bet plus any METAL sector RS recovery.
**Macro**: Brent cuts pellet-plant energy and freight; INR helps a pellet exporter — both real tailwinds, currently overwhelmed by sector-level selling.
**Flags**: METAL RS -8.4% · PSE RS -2.7% · volume 0.33x · ADV 6.8cr thin · -38.3% off high · results-clarification query trail · RSI below optimal band

---

## Basket-level summary

**Conviction ranking**: VIJAYA (4) > DELTACORP = MANALIPETC = IRCON = PNBGILTS = CHAMBLFERT = CHENNPETRO (3) > KIOCL (2)

**Strengths**
- 4 of 8 at perfect band_fit 3.0; the rest at 2.5 — the whole basket sits in the backtest's proven zones
- 3 names with delivery ratio >1.2 (DELTACORP 1.36, KIOCL 1.24, CHAMBLFERT 1.21) = accumulation signatures
- IRCON carries the only genuine order win (2026-06-23); PNBGILTS is the cleanest macro-thesis trade (disinflation + bond rally)
- Brent -26% is a tailwind for 5 of 8 names' input costs

**Weaknesses**
- 0 engine consensus, regime WAIT → Tier-2 historical hit ~24%, below the 27.5% band-fit optimum
- 3 of 8 fight sector headwinds (CHENNPETRO O&G -9.0%, KIOCL METAL -8.4%, MANALIPETC smallcap -9.4%)
- Universal volume dry-up: every pick trades at 0.28–0.70x its 20d average — mean-reversion bets with unconfirmed buying
- Only IRCON has a dated catalyst; the rest are bands-and-seasonality trades

**Expected P&L at 100% deploy** (10-yr backtest evidence): mean 15d +1.76% (~₹44K on ₹25L), median +1.20%, 65% weeks positive, worst -3.0%, best +14.6%, ~34.6% CAGR compounded.

**Fact-check ledger**: 16 agents (8 analysts + 8 adversarial verifiers). 5/8 clean on first pass. 3 corrections applied: DELTACORP sector-gap 2.3pts not 6, KIOCL sector rank 3rd-worst not 2nd, KIOCL SL-to-200DMA distance ~2.2% not 3%. PNBGILTS verifier flags were false positives (backtest table wasn't in the checker's context; the cited hit rates are correct per reports/backtest_10yr_findings.md).
