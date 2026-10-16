/* SGM Data World content: every source the daily run fetches (36 guarded feeds in configs/feed_guards.json), grouped by
   publisher. Station coordinates are the publisher's city; `building` is set only where the exact building was checked on
   OpenStreetMap. No amounts or personal details. */
window.SGM_DATA = (function () {
"use strict";
const DAILY = "Weekdays 6:45 PM IST (if missed: 9:45 PM, then 7:15 AM)";
const STATIONS = [
  { id: "blr", name: "Bengaluru", sub: "your laptop", real: [12.97, 77.59], at: [10.6, 83.2], alt: 1.7, r: 1.15, kind: "home",
    about: "Your laptop. Every crate on this globe ends up here. Two production lines run inside it: the daily line at 6:45 PM (fetch the feeds, run the data checks, send the daily note) and the Friday line at 7:30 PM (model scores, industry heat, themes, the V3.2 list, re-running the registered tests, then Telegram). The inside view comes next." },
  { id: "bom", name: "Mumbai", sub: "NSE · NSDL · AMFI · news", real: [19.07, 72.88], at: [17.2, 65.2], alt: 1.8, r: 1.2, kind: "source",
    about: "Most of the data starts here. NSE sits in Bandra Kurla Complex, NSDL in Lower Parel and BSE on Dalal Street. The files themselves are served from cloud and CDN servers, so the pin marks who publishes the data, not a server room." },
  { id: "del", name: "New Delhi", sub: "PIB · news", real: [28.61, 77.21], at: [31.2, 76.4], alt: 2.0, r: 0.85, kind: "source",
    about: "The Press Information Bureau publishes every government press release. We read them to score which industries the government is pushing. Business Standard and Mint headlines also come from Delhi." },
  { id: "web", name: "On the web", sub: "screener.in · tickertape.in", real: null, at: [24.5, 89.5], alt: 3.0, r: 0.8, kind: "web",
    about: "Two websites we read. We have not confirmed which city each company works from, so this island has no pin on the map." },
  { id: "stl", name: "St. Louis", sub: "FRED, St. Louis Fed", real: [38.63, -90.2], at: [38.63, -90.2], alt: 1.6, r: 0.9, kind: "source",
    about: "FRED is the Federal Reserve Bank of St. Louis's free data library. It gives us world prices: crude oil, gas, metals, crops, the dollar, the rupee and US interest rates." },
  { id: "fra", name: "Frankfurt", sub: "ECB exchange rates", real: [50.11, 8.68], at: [50.11, 8.68], alt: 1.6, r: 0.8, kind: "source",
    about: "The European Central Bank publishes daily reference exchange rates. We read them through the free Frankfurter API." },
  { id: "mtv", name: "California", sub: "Google News · Google Drive", real: [37.39, -122.08], at: [37.39, -122.08], alt: 1.6, r: 0.85, kind: "source",
    about: "Google News gives us world headlines for a mood score. Google Drive keeps the backups. The pin marks Google's head office in Mountain View." },
  { id: "dxb", name: "Dubai", sub: "Telegram", real: [25.2, 55.27], at: [26.4, 53.0], alt: 1.6, r: 0.75, kind: "out",
    about: "Telegram, the company headquartered in Dubai, carries your phone messages: the 6:45 PM daily note, the Friday buy list and any failure alert." }
];
const BOTS = [
  // Mumbai
  { id: "bhav", st: "bom", name: "Bhav", hat: "hard", pants: 0x35588a, use: "live", role: "Price fetcher, NSE",
    says: "Every weekday evening I collect NSE's official bhavcopy: one row per stock with its open, high, low and close.",
    carries: ["Daily prices for every NSE stock since 2015", "Each stock's series, so trade-for-trade (BE/BZ) names can be dropped"],
    from: ["nsearchives.nseindia.com"], when: DAILY, writes: "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
    used: "Everything in V3.2: the trend test, the ₹1 cr liquidity bar, the model's inputs and every return." },
  { id: "split", st: "bom", name: "Splitty", hat: "cap", hatColor: 0x4f8a4f, pants: 0x5a4a7a, use: "live", role: "Corporate actions, NSE",
    says: "When a stock splits 1:10 I fix its old prices, so a split never looks like a 90% crash.",
    carries: ["Splits, bonuses and dividends", "Price adjustment factors"], from: ["www.nseindia.com", "nsearchives.nseindia.com"], when: DAILY,
    writes: "data/corporate_actions_full_history/normalized/stock_corporate_actions.parquet", used: "Keeps every price in V3.2 honest after splits and bonuses." },
  { id: "filer", st: "bom", name: "Filer", hat: "cap", hatColor: 0x7a4f9a, pants: 0x2f6b6b, use: "live", role: "Company filings, NSE",
    says: "I read what companies file with NSE: order wins, results, letters of intent. I note the rupee amounts.",
    carries: ["Corporate announcements and their archive", "Order amounts and terms", "Quarterly profit and loss", "Event ledger"],
    from: ["www.nseindia.com"], when: DAILY, writes: "data/derived/announcements_historical.parquet, order_amounts.parquet, pnl_quarterly.parquet",
    used: "Which themes a company is exposed to (V3.2's fading-theme drop), and the order lines in your daily note." },
  { id: "promo", st: "bom", name: "Promo", hat: "peak", hatColor: 0x2c3e66, pants: 0x6b4e2e, use: "ctx", role: "Promoter watcher, NSE",
    says: "I watch the owners: which promoters bought shares, and who pledged theirs.", carries: ["Insider trades", "Promoter pledges"],
    from: ["www.nseindia.com"], when: DAILY, writes: "data/derived/pit_history.parquet, pledge_events.parquet", used: "Research tests and pledge alerts." },
  { id: "tally", st: "bom", name: "Tally", hat: "hard", hatColor: 0xf0a640, pants: 0x3c3c3c, use: "ctx", role: "Market counter, NSE",
    says: "I count who bought and sold today, the big block deals, India VIX, and which days the market is shut.",
    carries: ["FII and DII flows", "Block deals", "India VIX", "NSE trading holidays"], from: ["www.nseindia.com"], when: DAILY,
    writes: "data/derived/fii_dii_flows.parquet, block_deals.parquet, india_vix.parquet, nse_holidays.parquet",
    used: "Market context. The holiday list sets your buy and sell days." },
  { id: "nsdl", st: "bom", name: "Videshi", hat: "beret", hatColor: 0x1f4f6f, pants: 0x7a6a3a, use: "ctx", role: "Foreign money tracker, NSDL",
    says: "I track how much foreign investors put into or pull out of Indian shares, every day and by sector.",
    carries: ["FPI daily flows", "FPI flows by sector (fortnightly)"], from: ["www.fpi.nsdl.co.in"], when: DAILY,
    writes: "data/derived/nsdl_fpi_daily.parquet, nsdl_fpi_sector.parquet", used: "The weekly regime note." },
  { id: "amfi", st: "bom", name: "Mutu", hat: "cap", hatColor: 0xb0623a, pants: 0x35588a, use: "ctx", role: "Mutual fund reader, AMFI",
    says: "I read how much money mutual funds manage and where.", carries: ["Mutual fund assets"], from: ["www.amfiindia.com"], when: DAILY,
    writes: "data/derived/amfi_mf_aum.parquet", used: "Market context." },
  { id: "khabri", st: "bom", name: "Khabri", hat: "cap", hatColor: 0xb43b3b, pants: 0x444444, use: "ctx", role: "News reader, Mumbai papers",
    says: "I skim Economic Times and Moneycontrol headlines and broker recommendations.", carries: ["News headlines", "Broker recommendations"],
    from: ["economictimes.indiatimes.com", "www.moneycontrol.com"], when: DAILY, writes: "data/derived/news_feed.parquet, broker_recos.parquet",
    used: "A news mood score for context." },
  { id: "dalal", st: "bom", name: "Dalal", hat: "night", pants: 0x5d5d5d, use: "off", sleeping: true, role: "BSE, asleep",
    says: "Zzz. Only the CIAN watcher used me, and BSE has blocked our requests since Sep 30.", carries: ["Nothing at the moment"],
    from: ["bseindia.com"], when: "Not running", writes: "—", used: "Not used by V3.2." },
  // Delhi
  { id: "sarkari", st: "del", name: "Sarkari", hat: "peak", hatColor: 0xf3efe6, pants: 0x2f4f7f, use: "live", role: "Government press releases, PIB",
    says: "I read every government press release and count which industries the government keeps talking about.",
    carries: ["PIB press releases"], from: ["www.pib.gov.in"], when: DAILY, writes: "data/derived/pib_releases.parquet",
    used: "The policy score behind V3.2's policy veto, and how strong each government theme is." },
  { id: "khabri2", st: "del", name: "Khabri Jr", hat: "cap", hatColor: 0x3a6fa8, pants: 0x6b4e2e, use: "ctx", role: "News reader, Delhi papers",
    says: "I bring Business Standard and Mint headlines to the same news pile.", carries: ["News headlines"],
    from: ["www.business-standard.com", "www.livemint.com"], when: DAILY, writes: "data/derived/news_feed.parquet", used: "A news mood score for context." },
  // web
  { id: "screeny", st: "web", name: "Screeny", hat: "cap", hatColor: 0x1d6b73, pants: 0x4a4a6a, use: "live", role: "Industry labels, screener.in",
    says: "I tell everyone which industry a company belongs to. Saturdays I also bring company fundamentals.",
    carries: ["NSE four-level industry for each stock", "Company fundamentals (Saturdays)"], from: ["www.screener.in"], when: "Industry labels with the daily run; fundamentals Saturdays",
    writes: "data/derived/screener_industry.parquet, screener_fundamentals.parquet", used: "The industry map V3.2 uses to find hot industries." },
  { id: "tikki", st: "web", name: "Tikki", hat: "cap", hatColor: 0x8a5a9a, pants: 0x2f6b6b, use: "ctx", role: "Big investors' holdings, tickertape.in",
    says: "I note what well-known investors hold.", carries: ["Superstar investor holdings"], from: ["www.tickertape.in"], when: DAILY,
    writes: "data/derived/superstar_holdings.parquet", used: "Research only." },
  // abroad
  { id: "fed", st: "stl", name: "Fed", hat: "top", pants: 0x3b3b55, use: "live", role: "World prices, FRED",
    says: "I send the world's prices: crude, gas, copper, aluminium, zinc, nickel, iron ore, sugar, cotton, wheat, the dollar and US rates.",
    carries: ["Commodity prices", "USD/INR history", "Dollar index, US yields, S&P 500, US VIX"], from: ["fred.stlouisfed.org"], when: DAILY,
    writes: "data/derived/macro_drivers_fred.parquet, commodity_prices.parquet, global_rates.parquet, usdinr_history.parquet",
    used: "Commodity themes (rising or fading) in V3.2, and the weekly regime note." },
  { id: "euro", st: "fra", name: "Euro", hat: "beret", hatColor: 0x2c3e66, pants: 0x6b5a3a, use: "ctx", role: "Exchange rates, ECB",
    says: "I bring the official euro, pound, yen and dollar rates against the rupee.", carries: ["Daily reference exchange rates"],
    from: ["api.frankfurter.app"], when: DAILY, writes: "data/derived/macro_timeseries.parquet", used: "Market context." },
  { id: "gnews", st: "mtv", name: "Gnews", hat: "cap", hatColor: 0x3a8a5a, pants: 0x35588a, use: "ctx", role: "World headlines, Google News",
    says: "I gather world headlines so we can tell if the global mood is calm or scared.", carries: ["Global headlines"],
    from: ["news.google.com"], when: DAILY, writes: "data/derived/global_macro_sentiment.parquet", used: "A global mood score for context." },
  { id: "drive", st: "mtv", name: "Backup", hat: "hard", hatColor: 0x8fb7d8, pants: 0x4a4a4a, use: "msg", role: "Backups, Google Drive",
    says: "I keep copies of your work in Google Drive, so one broken laptop can't lose it.", carries: ["Backups from the laptop"],
    from: ["Google Drive (My Drive / SGM backups)"], when: "After the runs", writes: "Google Drive", used: "Safety copy, not an input." },
  { id: "pigeon", st: "dxb", name: "Pigeon", hat: "headset", pants: 0x2f6fae, use: "msg", role: "Your messages, Telegram",
    says: "I carry the 6:45 PM daily note, the Friday buy list and any failure alert to your phone.", carries: ["Messages from the laptop"],
    from: ["api.telegram.org"], when: "After each run", writes: "Your phone", used: "How results reach you." },
  // Bengaluru
  { id: "clock", st: "blr", name: "Clock", hat: "peak", hatColor: 0x1d6b73, pants: 0x3b3b55, use: "msg", role: "Scheduler, your laptop",
    says: "I start the runs: weekdays 6:45 PM, Fridays 7:30 PM for the buy list, weekends 11:30 AM and 7:30 PM.",
    carries: ["The timetable"], from: ["macOS launchd"], when: "Always on", writes: "logs/runs/", used: "Starts every job, so nothing needs a manual click." },
  { id: "watch", st: "blr", name: "Watchdog", hat: "peak", hatColor: 0x222a3a, pants: 0x5a4a3a, use: "msg", role: "Watchdog, your laptop",
    says: "Every 30 minutes I check that no run is stuck and that your messages went out. If not, I raise the alarm.",
    carries: ["Run health"], from: ["the laptop's own logs"], when: "Every 30 minutes", writes: "logs/runs/watchdog_state.json", used: "Catches stuck or silent runs." },
  { id: "paisa", st: "blr", name: "Paisa", hat: "cap", hatColor: 0xd4a43a, pants: 0x2f4f7f, use: "msg", role: "Your SBI account",
    says: "Your salary lands with me. The money lane to the broker and back comes in a later version.", carries: ["Salary (no amounts shown here)"],
    from: ["SBI"], when: "Monthly", writes: "—", used: "Funds the weekly batches." }
];
const USE_TEXT = { live: "Feeds the live rule (V3.2)", msg: "Messages and running the system", ctx: "Context and research", off: "Not in use" };
const USE_COLOR = { live: 0x3a8752, msg: 0x2f6fae, ctx: 0xc08a3a, off: 0x978b82 };
// country of origin for the bots' flag T-shirts
const FLAG = { bom: "IN", del: "IN", web: "IN", blr: "IN", stl: "US", mtv: "US", fra: "DE", dxb: "AE" };
// exact buildings checked on OpenStreetMap (Nominatim), 2026-10-16
const BUILDINGS = {
  nse: { name: "National Stock Exchange · Exchange Plaza", st: "bom", lon: 72.8598629, lat: 19.0601927, note: "OpenStreetMap office node, G Block, Bandra Kurla Complex" }
};
// Bengaluru is shown at a public landmark (Vidhana Soudha), never a home address
const HOME = { name: "Your laptop · Bengaluru", lon: 77.5907, lat: 12.9796, note: "Shown at Vidhana Soudha, a public landmark" };
return { DAILY, STATIONS, BOTS, USE_TEXT, USE_COLOR, FLAG, BUILDINGS, HOME };
})();
