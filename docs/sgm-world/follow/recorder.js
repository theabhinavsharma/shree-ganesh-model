/* Records a follow-cam film frame by frame from the local Data World page, for preview videos.
   1. Run the frame server:   /usr/bin/python3 docs/sgm-world/follow/frame_server.py <out dir>
   2. Open http://localhost:8795/?record&at=2026-10-17T07:00:00Z at 1280x720 (?at= fixes the sky; 07:00 UTC is daylight from Frankfurt to Bengaluru)
   3. Paste this file into the page's console with JOURNEY set (euro, fed). It posts every frame as a JPEG and keeps
      the captions in window.__rec.caps (copy them out with JSON.stringify(__rec.caps) for make_video.py).
   Why it waits on each frame: time stands still between recorded frames, so each frame renders until Google's 3D
   tiles have loaded (12 s cap near the ground, 4 s higher up). Yields use setTimeout, not MessageChannel: a tight
   MessageChannel loop starves the tile decoders and nothing ever finishes loading. */
const JOURNEY = window.JOURNEY || "euro", FPS = 20;
const FROM = window.REC_FROM || 0, TO = window.REC_TO === undefined ? Infinity : window.REC_TO;   // re-record just frames FROM..TO (others are stepped, not saved)
window.__rec = { k: 0, t: 0, done: false, err: null, caps: [], started: Date.now(), fps: FPS };
(async () => {
  try {
    const W = SGM_WORLD, v = W.viewer, sleep = ms => new Promise(r => setTimeout(r, ms));
    await SGM_CHASE.prepareRecord(JOURNEY);
    for (let i = 0; i < 20; i++) { v.render(); await sleep(30); }
    const settle = async () => {
      // "loaded" must hold for 8 renders in a row: between detail levels the map briefly reports loaded with only coarse tiles
      const ts0 = W.chase.tileset(), h = v.camera.positionCartographic.height;
      const ms = h < 4000 ? 12000 : (ts0 && ts0.show) ? 10000 : 4000, t0 = performance.now(); let n = 0, calm = 0;
      while (performance.now() - t0 < ms) {
        v.render(); n++; const ts = W.chase.tileset(), g = v.scene.globe;
        calm = ((!ts || !ts.show || ts.tilesLoaded) && (!g.show || g.tilesLoaded)) ? calm + 1 : 0;   // the 3D map near the ground, the globe imagery from space
        if (calm >= 8) break; await sleep(8);
      }
    };
    for (let k = 0; ; k++) {
      const st = SGM_CHASE.frame(k === 0 ? 0 : 1 / FPS);
      if (k < FROM) { if (k % 4 === 0) { v.render(); await sleep(1); } continue; }
      if (k > TO) break;
      await settle(); SGM_CHASE.frame(0); v.render();
      const blob = await new Promise(r => v.scene.canvas.toBlob(r, "image/jpeg", 0.9));
      await fetch(`http://127.0.0.1:8797/frame?dir=${JOURNEY}&n=${k}`, { method: "POST", body: blob, headers: { "Content-Type": "image/jpeg" } });
      __rec.caps.push([k, Math.round(st.t * 1000) / 1000, st.caption]); __rec.k = k; __rec.t = st.t;
      if (st.done) break;
    }
    __rec.done = true;
  } catch (e) { __rec.err = String(e && (e.stack || e)); }
})();
