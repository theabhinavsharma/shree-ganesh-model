# SGM Data World (shreeganeshmodel.vercel.app home page)

What it is: a real-time 3D Earth (CesiumJS) showing where every input of the SGM daily run comes from, with cartoon bots carrying the data to a laptop in Bengaluru. The old 15D/5% showcase lives at `/older-models/` (synced weekly by `src/agentic/sync_showcase.py`).

## Files
- `../index.html`: page shell (Cesium 1.121.0 from jsDelivr).
- `app.js`: the world. It covers the globe and the day/night/evening grade, the lazy Google 3D tiles, the streams, the bots, the journey and the cards.
- `data.js`: content. Stations and bots come from `configs/feed_guards.json`; exact buildings are checked on OpenStreetMap.
- `routes/*.json`: real road routes the bots run along (OSRM on OpenStreetMap data; source inside each file).
- `config.js`: **not in git**. Holds `window.SGM_CONFIG = { googleKey: "..." }`. It is written from `~/.config/sgm/google_maps.env` by `src/agentic/world_config.py` at deploy time. Without it the page still works: it shows the satellite globe, with OpenStreetMap at street level.
- `../assets/bots/*.glb`: the bots (generator in `docs/sgm-world/bots/`).

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

## Test links
- `/#nse` jumps to NSE.
- `/#explore` skips the intro.
- `?debug` writes state to `<html data-debug>` for headless checks.
