# SGM Data World (shreeganeshmodel.vercel.app home page)

What it is: a real-time 3D Earth (CesiumJS) showing where every input of the SGM daily run comes from, with cartoon bots carrying the data to a laptop in Bengaluru. The old 15D/5% showcase lives at `/older-models/` (synced weekly by `src/agentic/sync_showcase.py`).

## Files
- `../index.html`: page shell (Cesium 1.121.0 from jsDelivr).
- `app.js`: the world. It covers the globe and the day/night/evening grade, the lazy Google 3D tiles, the streams, the bots, the journey and the cards.
- `data.js`: content. Stations and bots come from `configs/feed_guards.json`; exact buildings are checked on OpenStreetMap.
- `chase.js`: the follow-cam journeys (one character's whole run in about 30 seconds). See "Follow-cam journeys" below.
- `routes/*.json`: real road routes the bots run along (OSRM on OpenStreetMap data; source inside each file). Built by `src/agentic/build_world_routes.py`, including the short walks from the road to the water's edge.
- `routes/heights_<journey>.json`: measured ground and sea heights for the street-level stretches of each journey (Google's 3D map is measured from the ellipsoid, so Frankfurt's street is at about +145 m and Mumbai's at about -45 m).
- `config.js`: **not in git**. Holds `window.SGM_CONFIG = { googleKey: "..." }`. It is written from `~/.config/sgm/google_maps.env` by `src/agentic/world_config.py` at deploy time. Without it the page still works: it shows the satellite globe, with OpenStreetMap at street level.
- `../assets/bots/*.glb`: the bots. `bot_<id>.glb` is each character's own look (its country's full-flag tee plus its own quirk); `bot_<flag>.glb` is the fallback. Generators in `docs/sgm-world/bots/`.

## Data sources shown on the page
| Layer | Source |
|---|---|
| Globe, day | NASA EOSDIS GIBS Blue Marble Next Generation |
| Globe, night | NASA GIBS VIIRS city lights 2012 |
| Street level | Google Photorealistic 3D Tiles (with key), OpenStreetMap tiles as stand-in |
| Roads for bots | OSRM demo server, © OpenStreetMap contributors (ODbL) |
| Sun position | Real time, computed in the browser (Cesium) |

## Honesty rules
- Pins mark the publisher's city or building, not server rooms.
- Undersea-cable routes are simplified.
- "On the web" (screener.in, tickertape.in) has no pin.
- No amounts or personal data. Bengaluru is shown at a public landmark (Vidhana Soudha), never a home.

## Follow-cam journeys (`chase.js`)
- One film per character: Euro (Frankfurt to Marseille, then the submarine to Mumbai) and Fed (St. Louis to the New Jersey shore, then Ship 1). A courier runs Mumbai to Bengaluru in both.
- The film opens on the whole route from space, then dives to the start building.
- Bots are human-sized (about 1.8 m). For the first and last 35 m of each road the camera runs about 22 m behind, at running pace. In between it climbs like a drone, and the run turns into a time-lapse. The camera height follows the speed.
- A bright streak stays on every stretch already covered: amber for roads, cyan for the submarine, white for the tanker. A dashed white line shows the route still ahead, seen from far away.
- Before a film, the page loads saved heights (`routes/heights_<id>.json`). If the file is missing, it measures the four street-level spots behind a curtain.
- Entry points: the "Watch <name>'s journey" button on a bot's card, and `/#follow-euro` or `/#follow-fed`.
- Preview videos: `docs/sgm-world/follow/` (recorder, frame server, video maker).

## Test links
- `/#nse` jumps to NSE.
- `/#explore` skips the intro.
- `?debug` writes state to `<html data-debug>` for headless checks.
- `/#follow-euro` plays Euro's follow-cam journey.
- `?record&at=<ISO time>` is preview recording only. The page renders frame by frame and the sky is fixed at that time.
