---
name: sgm-telegram
description: How SGM / Sri Lakshmi phone messages (Telegram) look and how to change them safely. Use when adding, editing or reviewing any Telegram message, alert or report format, or when the user says a message is too long or hard to read.
---

# Telegram messages: format rules

Abhinav reads these on his phone. **Short, scannable, one line per topic.** (His words: "condensed, not long unreadable ones".)

## Rules
1. **First line = what + date.** e.g. `SGM · Sep 30 · ran 6:45 PM ✅`. The output gate checks the data date is on line 1.
2. **Problems spelled out, everything fine collapses to one ✅ line.** Never list 20 green items.
3. **Max ~12 lines.** Lists: top 5 (orders), top 3 each side (industries), then `+N`.
4. **Plain words, no codes.** Not `data.freshness`, not `G1H`, not `1/26 batch`. Use `QC_LABEL` / registration text.
5. **Numbers only from files.** Every number must be a file value, a stated formula (limit = close × 1.05, rupees = sleeve ÷ 26 ÷ stocks, shares = rupees ÷ close), or a template constant. `src/agentic/trust/check_message.py` holds anything else back.
6. **Dates month-first** (`Sep 29`), times in **US Eastern** (IST only where the market needs it: `3:30 PM IST = 6:00 AM ET`).
7. **Links as `📄 <url>`** → shown as a tappable "📄 read" (HTML mode, previews off).
8. **Spacing (texting hygiene):** one blank line between sections, none inside a section, none at the start/end;
   single spaces only; ` · ` is the only separator; one item per line in lists. `notify.tidy()` enforces it.
9. **Bold title:** the first line is sent bold (Telegram HTML, always on; plain-text fallback if Telegram refuses).
10. **Fixed section order (daily):** title → data/QC → orders → paper + sell → industries → queue.
11. **Emoji legend:** ✅ ok · ❌ failed/blocked · ⚠️ warning · 🔥 big order (15%+ of a year's revenue) · ⭐ a stock we hold · 🛒 buy list · 🔴 sell now · 🟡 sell soon · 💤 nothing due · 🧪 research/queue · 📥 data · 📦 orders · 📈 paper returns · 🌡 warming · ❄️ cooling.

## The 6 messages (code → example)
1. **Daily** (`notify.daily_text`, after every weekday run)
```
SGM · Sep 30 · ran 6:45 PM ✅ · 0:16

📥 Data ✅ 20/20 · QC ✅ 11/11 · checks OK

📦 Orders: 3 new · 1 big
🔥 HILINFRA ₹221cr (28% rev) 📄read
• TRANSRAILL ₹574cr (8% rev) 📄read

📈 SL Sep 28 +1.2% · SL-old +1.2% · Model +2.5% · Prod Sep 08 -6.4%
💤 Sell: none due · next ~Mar 23 2027

🌡 Warming: Aerospace & Defense, … +1
❄️ Cooling: 2/3 Wheelers, … +2

🧪 Queue 4/6 · now: Promoter pledges (needs code: ask Claude)
```
2. **Friday buy list** (`notify.weekly_text`)
```
🛒 Sri Lakshmi v2 · buy Mon Oct 5 at open · data Oct 2
8 stocks · ₹14,423 each
1. DEEDEV · 19 sh · limit ₹688.27
2. TBZ · 20 sh · limit ₹662.29 · ⚑ takeover
Dropped (financials): XYZFIN
Sell all at close ~Mar 29 2027 · no stop-loss
Sign the Saturday review before real money
```
3. **Sell** (`notify.sell_lines`, inside the daily when a batch reaches session 126)
```
🔴 SELL SL batch Sep 28 · sell at today's close (3:30 PM IST = 6:00 AM ET)
• DEEDEV 19 sh · ₹653.90 → ₹676.90 (+3.5%)
Batch after costs +1.2% · money rolls into next week's batch
```
4. **Failure** (`run_sgm.Run.stop` / `notify.py fail`)
```
❌ SGM daily run stopped at the data check (7:02 PM ET)
• All inputs fresh: prices 2 sessions old
Picks on hold until this passes. Record: logs/runs/20260930_daily.json
```
5. **A/B result** (`research_queue.report`)
```
🧪 Shareholding detail · A/B done (3/6)
✅ PASS: small shareholders fell 5%+ → 37.0% vs 36.8%/yr
❌ 4 rules failed
Next: Promoter pledges
```
6. **Queue needs code** (`research_queue.run`): `🧪 Queue 5/6: Promoter pledges needs code · ask Claude`

## Changing a format safely
1. Edit the function above, keep these rules.
2. Preview without sending:
   `/usr/bin/python3 -c "import sys; sys.path.insert(0,'src/agentic'); import notify; print(notify.daily_text())"`
3. Run the output gate on it (`check_message.check("daily", text, log=False)` must be ok). If a new number is flagged, add its source or formula to `check_message.py`, never loosen the matching.
4. Only send a real preview if Abhinav asks (`notify.send(text, "preview")`).
5. Commit with the before/after in the message.

## Lessons from incidents
(appended by src/agentic/trust/incident.py; never edit or delete these lines)
- 2026-09-30 [INC-2026-09-30-telegram-letters] Build messages as a list of lines and check the outbox line count before sending anything by hand.
