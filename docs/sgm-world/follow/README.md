# Follow-cam preview videos

These files make a short video of a follow-cam journey (`shreeganeshmodel-deploy/world/chase.js`), so it can be reviewed in a claude.ai artifact before anything goes live.

1. Serve the site locally on port 8795. The Google key allows `http://localhost:8795/*`.
2. Start the frame server: `/usr/bin/python3 docs/sgm-world/follow/frame_server.py <frames dir>`. It saves each POSTed frame as `<frames dir>/<journey>/<n>.jpg`.
3. Open `http://localhost:8795/?record&at=2026-10-17T07:00:00Z` with the viewport at 1280x720, then run `recorder.js` in the page with `window.JOURNEY = "euro"`. Near the ground a frame takes a few seconds, because each frame waits for Google's 3D tiles.
4. Copy `JSON.stringify(__rec.caps)` to a file and run `/usr/bin/python3 docs/sgm-world/follow/make_video.py <frames dir>/<journey> caps.json docs/sgm-world/follow/<journey>`.
5. `follow.html` is the preview page. Publish it with the `.mp4`, `.vtt` and poster `.jpg` beside it.

Recording-mode details live in `world/app.js`, where `RECORD` is true under `?record`:
- **Clock:** the clock stands still between frames, so the tileset's "load after the camera stops" options (`foveatedScreenSpaceError`, `cullRequestsWhileMoving`) are switched off. Otherwise street-level tiles never load.
- **Tile cache:** the tile cache is raised to 3 GB.
