# SGM Data World — plan and progress

## End goal
A 3D world you can fly through that explains the whole SGM automation. Every input starts at its real place on a real Earth. Little orange bots carry it to the laptop in Bengaluru, where worker bots turn it into the Friday list. A money lane shows salary (SBI) → broker → stocks → back. Scrolling tells the story; dragging lets you explore; clicking any bot tells you its job.

## Decisions (user, 2026-10-16)
- Style: toy diorama (reference 3) for the globe and cities; isometric workshop (closer to reference 2) inside the laptop.
- The user is the boss bot: a giant behind the laptop, toy-styled from a photo the user will send.
- Scroll story comes from the user's own drawings (one frame per scroll step). Build it after the drawings arrive.
- Fixed explainer first. Live green/red status from real runs is a later option.

## Quality bar (check every milestone against this)
1. Truthful: only feeds that exist in the repo (configs/feed_guards.json, daily_data_layer.sh). Pins sit at the publisher's city; anything unverified gets no city. Ships and tunnels are labelled as a cartoon of undersea cables.
2. Private: no rupee amounts, holdings or personal data on the page.
3. Clear before clever: a first-time viewer understands "data comes from here, goes there" within 10 seconds.
4. Smooth on a phone: no stutter, readable cards, tap targets big enough.
5. One look: toy diorama outside, isometric workshop inside. No neon, no mixed styles.

## Milestones
- [x] M1 Base world: real-coastline globe, floating city islands tethered to their real spots, bots with outfits, couriers and ships moving data, click a bot or island for its card, fly-to buttons.
- [ ] M2 Inside the laptop: isometric workshop with the 6:45 PM daily line and the Friday 7:30 PM line (checks, model scores, heat, themes, V3.2 rails, test re-runs, Telegram).
- [ ] M3 Boss giant: the user's toy figure behind the laptop.
- [ ] M4 Scroll story: chapters from the user's drawings.
- [ ] M5 Money lane: SBI → broker → Monday pre-open order → 26 weekly slices → sale → roll-over.
- [ ] M6 (optional) Live status colours from the daily run.

## Not doing (scope guard)
- No extra decoration that doesn't explain a step (no particle storms, no random effects).
- No made-up sources, cities or numbers.
- No new data fetching for the page itself.

## Files
- `world.html` is the page source (published as the claude.ai artifact https://claude.ai/artifact/4AqJRp1Dc1hMtAn2giYB4P; republish this file to keep the same link).
- Local preview: `.claude/launch.json` entry `sgm-world` (python http.server on port 8793 serving this folder).

## Direction change (user, 2026-10-16 night): realistic world on the website
- v1 (toy globe artifact) stays as a backup link. The main build is now a **realistic Earth** on shreeganeshmodel.vercel.app (home page). The old 15D/5% showcase moves to an **Older Models** tab (`/older-models/`).
- Engine: CesiumJS. Google Photorealistic 3D Tiles near the ground (key on the user's Google Cloud, locked to the site, daily cap). NASA imagery for the globe.
- Real time of day: real day, evening and night. From space, the real sun line (terminator) and city lights; near the ground, a colour grade by the local sun.
- Detail by zoom:
  - space: data streaks on routes;
  - country: bot swarms;
  - street: bots running real roads (OpenStreetMap routes);
  - building: tap for the card.
- Bots: little orange Claude-style bots in T-shirts with their country's flag (IN, US, DE, AE), plus astrobots for any real satellite feed later (none today). Built by a parallel agent: `docs/sgm-world/bots/`, models in `shreeganeshmodel-deploy/assets/bots/`.
- Journey: a "Start the journey" button plays the user's drawn frames (placeholder 4 steps until the drawings arrive), then free exploring.

## Realistic-world milestones
- [x] R0 Site restructure: world is `index.html`, showcase at `/older-models/` (weekly sync script), top-right "Older Models" link.
- [x] R1 Test run: real-time globe, Google 3D city at NSE (key live 2026-10-17), bot routes. Phone check still to do on a real phone.
- [x] R2 Journey from the user's 5 drawings (crew line-up, zoom-out angles, ships, India, Bengaluru) + free explore. Cards in plain words with 2 buttons (next bot; company list or second bot). The user will define the buttons' final actions later.
- [ ] R3 Every source at its real building, its route and the zoom layers.
- [ ] R4 Laptop workshop, boss giant, money lane.
- [ ] Deploy to production only after the user says yes.

## Progress log
- 2026-10-16: plan written; M1 started.
- 2026-10-16: M1 done and published (version 1).
  - What it has: real coastlines (the ECharts 4.9 world map, land only, no borders), 8 floating islands (7 tethered to their real cities; the web island has no pin), 21 station bots, 8 couriers, 2 ships on the simplified undersea-cable routes, the Telegram plane, click cards, fly-to buttons, and the About card with the honesty notes.
  - Quality-bar check:
    1. Truthful: every bot maps to a feed in the repo. screener.in and tickertape.in have no city pin (unverified). BSE is shown asleep.
    2. Private: no amounts.
    3. Clear: labels never overlap.
    4. Phone: layout checked at 375px; tilted island views work.
    5. One look: toy diorama throughout.
  - Known gaps: the laptop interior is not built (M2); boss giant and scroll story are waiting for the user's photo and drawings.
- 2026-10-16 late: R0 done. R1 code works locally: the night side over India from space; BKC street map with the NSE label at OpenStreetMap's node; night grade. Google billing is waiting on the user (no billing account existed). Bot models are being built by a parallel agent.
- 2026-10-17: key live (Google Cloud project "My First Project", key locked to shreeganeshmodel.vercel.app and localhost:8795, Map Tiles API only; free trial ₹28,796 credit, so the card isn't charged without an upgrade; Google won't allow lowering the daily cap on a trial). Key backed up to Drive (SGM backups/keys). Bots in full-flag T-shirts; props built by the agent: oil tanker, container ship, submarine, ML bot, "ME" figure (from the user's sketch; photo later), giant laptop, ₹ note, data crate. Journey wired: lineup.js (three.js) then journey.js (Cesium). Paths glow (amber data, cyan messages, green money); bots and ships leave trails; at night the toys are drawn unlit so they stay visible. Company lists: src/agentic/build_world_companies.py → world/companies.json. Not deployed yet; waiting for the user's yes.
- 2026-10-17 night: game-design pass (rules in GAME_DESIGN.md). The crew page now has a slim strip of tiny characters plus a liquid-glass roster sheet, and a shorter card. Every hand-off is visible: the crate is thrown and caught, sinks into the laptop while 21 checks tick off, gets hammered on the anvil into the Friday list, then the phone buzzes. Stops are now toy set pieces (build_sets.py: Fed, ECB, NSE, PIB, SBI, port crane, cable hut, Telegram tower, desk, anvil, phone, calendar, trees). Bots gained crouch/throw/catch/jump/cheer. Preview artifact only; the live site is untouched until the user says go. The Cesium world doesn't have these upgrades yet (next).
- 2026-10-17 late night: follow-cam journeys on the real Earth (world/chase.js), from the user's brief. One character's whole run in about 30 s:
  - **The run:** from space down to the source building, along real roads (OSRM), a jump into the submarine or tanker at a real beach, across the sea, then a courier from Versova to the laptop.
  - **Size and camera:** bots are human-sized. The camera runs about 22 m behind at street level for the first and last 35 m of each road. In between, a drone climbs over a time-lapse, and a bright streak marks every km covered.
  - **Ground heights:** measured and saved per journey (routes/heights_*.json).
  - **Status:** Euro is recorded as a preview video (docs/sgm-world/follow, artifact AKNmYPrzkjRM7yF3LuN6SC). Fed is wired and measured.
  - **Looks:** every character now has its own model and quirk (bot_<id>.glb).
  - **Not deployed.**
