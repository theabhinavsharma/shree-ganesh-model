// Records a follow-cam film in headless Chrome. Headless pages count as visible, so Chrome doesn't throttle the
// recorder's timers the way it does in a hidden tab.
// node docs/sgm-world/follow/record_headless.mjs <journey> <out caps.json> [gpu|swiftshader] [measure]
// With "measure" it only measures the journey's ground heights and writes them to <out> (for world/routes/heights_<journey>.json).
// Needs: the site on http://localhost:8795 and frame_server.py on 127.0.0.1:8797.
import { spawn } from "node:child_process";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const [journey = "euro", capsOut = "caps.json", mode = "gpu", task = "record"] = process.argv.slice(2);
const HERE = dirname(fileURLToPath(import.meta.url));
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
// SGM_AT sets the sky: 07:00 UTC is daylight from Frankfurt to Bengaluru; 12:30 UTC is dawn in St. Louis and sunset in India
const AT = process.env.SGM_AT || "2026-10-17T07:00:00Z";
const PORT = Number(process.env.SGM_PORT) || (task === "measure" ? 9334 : 9333), URL = `http://localhost:8795/?record&at=${AT}`;
const gpu = mode === "gpu" ? ["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"] : ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"];
const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), "sgm-rec-"))}`,
  "--window-size=1280,720", "--hide-scrollbars", "--no-first-run", "--disable-background-timer-throttling", "--disable-renderer-backgrounding", ...gpu, "about:blank"],
  { stdio: ["ignore", "ignore", "pipe"] });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let target;
for (let i = 0; i < 50 && !target; i++) { await sleep(300); try { target = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()).find(t => t.type === "page"); } catch {} }
if (!target) { console.error("Chrome did not start"); chrome.kill(); process.exit(1); }
const ws = new WebSocket(target.webSocketDebuggerUrl); let id = 0; const waiting = new Map();
ws.onmessage = e => { const m = JSON.parse(e.data); if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); } };
await new Promise(r => { ws.onopen = r; });
const send = (method, params = {}) => new Promise(r => { const i = ++id; waiting.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const evaluate = async expr => (await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;
await send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 720, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url: URL });
for (let i = 0; i < 60; i++) { await sleep(1000); if (await evaluate("!!(window.SGM_CHASE && window.SGM_WORLD)")) break; }
console.log("renderer:", await evaluate(`(() => { const g = document.createElement('canvas').getContext('webgl2'); const d = g && g.getExtension('WEBGL_debug_renderer_info'); return d ? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : 'unknown'; })()`));
if (task === "measure") {
  await evaluate(`SGM_WORLD.viewer.useDefaultRenderLoop = false; window.__H = null; SGM_CHASE.prepareRecord(${JSON.stringify(journey)}).then(H => { window.__H = H; }); 'ok'`);
  for (let i = 0; i < 180; i++) { await sleep(2000); const h = await evaluate("window.__H ? JSON.stringify(window.__H) : ''"); if (h) { writeFileSync(capsOut, h); console.log("heights ->", capsOut); break; } }
  ws.close(); chrome.kill(); process.exit(0);
}
// SGM_FROM / SGM_TO re-record a slice of frames into the same folder (the rest of the film is stepped through unsaved)
const range = `window.REC_FROM = ${Number(process.env.SGM_FROM || 0)}; ${process.env.SGM_TO ? `window.REC_TO = ${Number(process.env.SGM_TO)};` : ""}`;
await evaluate(`window.JOURNEY = ${JSON.stringify(journey)}; ${range} ${readFileSync(join(HERE, "recorder.js"), "utf8")}; 'started'`);
const t0 = Date.now();
while (true) {
  await sleep(10000);
  const st = JSON.parse(await evaluate("JSON.stringify({ k: __rec.k, t: __rec.t, done: __rec.done, err: __rec.err })") || "{}");
  console.log(`${Math.round((Date.now() - t0) / 1000)}s frame ${st.k} t=${(st.t || 0).toFixed(2)}`);
  if (st.err) { console.error(st.err); break; }
  if (st.done) { if (!process.env.SGM_FROM) writeFileSync(capsOut, await evaluate("JSON.stringify(__rec.caps)")); console.log("done ->", capsOut); break; }
}
ws.close(); chrome.kill();
