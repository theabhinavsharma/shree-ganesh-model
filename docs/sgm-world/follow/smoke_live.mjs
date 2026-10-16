// Smoke test of the live page as a visitor sees it: real time of day, normal render loop, no recording tricks.
// Clicks "Start the journey", then "Go!", and screenshots the film every few seconds. Console errors are printed.
// node docs/sgm-world/follow/smoke_live.mjs <out dir> [base url] [seconds]
import { spawn } from "node:child_process";
import { writeFileSync, mkdirSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [out = "smoke", base = "http://localhost:8795/", secs = "60"] = process.argv.slice(2);
mkdirSync(out, { recursive: true });
// a fresh port each run: a Chrome left over from an interrupted run would otherwise answer with its old, cached page
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", PORT = 9400 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), "sgm-smoke-"))}`,
  "--window-size=1280,720", "--hide-scrollbars", "--no-first-run", "--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist", "about:blank"], { stdio: "ignore" });
process.on("exit", () => chrome.kill()); process.on("SIGINT", () => process.exit(1)); process.on("SIGPIPE", () => process.exit(1));
const sleep = ms => new Promise(r => setTimeout(r, ms));
let target;
for (let i = 0; i < 50 && !target; i++) { await sleep(300); try { target = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()).find(t => t.type === "page"); } catch {} }
const ws = new WebSocket(target.webSocketDebuggerUrl); let id = 0; const waiting = new Map(), errors = [];
ws.onmessage = e => { const m = JSON.parse(e.data);
  if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); }
  if (m.method === "Runtime.exceptionThrown") errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
  if (m.method === "Log.entryAdded" && m.params.entry.level === "error" && !/404/.test(m.params.entry.text)) errors.push(m.params.entry.text); };
await new Promise(r => { ws.onopen = r; });
const send = (method, params = {}) => new Promise(r => { const i = ++id; waiting.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const evaluate = async expr => (await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;
await send("Runtime.enable"); await send("Log.enable");
await send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 720, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url: base });
for (let i = 0; i < 60; i++) { await sleep(1000); if (await evaluate("!!(window.SGM_CHASE && document.getElementById('startBtn'))")) break; }
const shot = async name => { const r = await send("Page.captureScreenshot", { format: "jpeg", quality: 70 }); writeFileSync(join(out, name + ".jpg"), Buffer.from(r.result.data, "base64")); };
await sleep(3000); await shot("00-home");
await evaluate("document.getElementById('startBtn').click()"); await sleep(10000); await shot("01-lineup");   // about as long as a visitor reads the lineup
await evaluate("window.__fr = 0; (function f() { window.__fr++; requestAnimationFrame(f); })(); 'ok'");
await evaluate("document.getElementById('nextBtn').click()");
const t0 = Date.now(); let k = 2;
while ((Date.now() - t0) / 1000 < Number(secs)) {
  await sleep(3000);
  const cap = await evaluate("document.getElementById('chaseText') ? document.getElementById('chaseText').textContent : ''");
  const curtain = await evaluate("!document.getElementById('chaseCurtain').hidden");
  const fps = Math.round((await evaluate("(() => { const n = window.__fr; window.__fr = 0; return n; })()")) / 3);
  await shot(String(k).padStart(2, "0") + "-" + Math.round((Date.now() - t0) / 1000) + "s");
  const chips = await evaluate("!document.getElementById('chips').hidden");
  console.log(`${Math.round((Date.now() - t0) / 1000)}s ${fps} fps ${curtain ? "[loading curtain] " : ""}${chips ? "[exploring] " : ""}${cap.slice(0, 70)}`);
  k++;
  if (await evaluate("!document.getElementById('chaseEnd').hidden")) { console.log("film ended, end panel shown"); await shot(String(k).padStart(2, "0") + "-end"); break; }
}
console.log(errors.length ? "ERRORS:\n" + errors.slice(0, 10).join("\n") : "no console errors");
ws.close(); chrome.kill();
