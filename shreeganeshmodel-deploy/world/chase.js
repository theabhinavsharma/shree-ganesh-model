/* Follow-cam journeys on the real Earth (from the user's brief, 2026-10-17): one character's whole run in about 30 seconds.
   The film opens on the whole route from space and dives to the real building where the data starts. The camera follows
   the bot down real roads (OSRM routes on OpenStreetMap, world/routes/*.json, built by src/agentic/build_world_routes.py).
   At the coast the bot jumps into its ship or the submarine, which crosses the sea. In Mumbai a courier catches the crate
   and runs the highway to the laptop in Bengaluru.
   The camera height follows the speed, so the run reads at street level and from space alike. The drone angle keeps swinging.
   Sea routes are simplified cable routes (the same waypoints as the ambient ships in app.js). */
(function () {
"use strict";
const W = window.SGM_WORLD; if (!W || !W.chase) return;
const C = W.C, X = W.chase, viewer = W.viewer, scene = viewer.scene, camera = viewer.camera;
const $ = id => document.getElementById(id);
const BOTS = "assets/bots/", PROPS = "assets/props/";
const ease = t => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v)), lerp = (a, b, t) => a + (b - a) * t, RAD = Math.PI / 180;
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

/* ---------------------------------------------------------------- the journeys. Sea points are [lat, lon] like SEA in app.js */
const TO_VERSOVA = [[19.12, 72.78], [19.1365, 72.800], [19.13568, 72.80797]];   // in to the water's edge at Versova beach, Mumbai
const PRADO_COVE = [[43.26149, 5.37175], [43.2608, 5.3709], [43.26045, 5.3692], [43.2595, 5.3640]];   // out of the cove at Prado beach, Marseille
const J = {
  // skyUTC: the film plays in daylight (07:00 UTC is day from Frankfurt to Bengaluru); exploring keeps the real sky
  euro: { name: "Euro", what: "exchange rates", flag: "DE", road1: "euro_fra_mrs", ship: "sub", skyUTC: 7, skyLine: "Shown in daylight", startLabel: "European Central Bank", portLabel: "Marseille",
    sea: [...PRADO_COVE, ...X.SEA.med, ...TO_VERSOVA],
    say: {
      intro: "Euro's journey: today's exchange rates, from Frankfurt to your laptop in Bengaluru.",
      start: "Euro picks up the day's rates at the European Central Bank in Frankfurt and sets off.",
      road: km => `Up the Rhine, past Lyon and down the Rhône to the Mediterranean: ${km} km of real road.`,
      board: "Marseille, where Europe's undersea cables meet the sea. Euro jumps into the submarine.",
      sea: "Under the Mediterranean, through Suez and the Red Sea, across the Arabian Sea. In real life the data takes a fraction of a second through the cable.",
      handoff: "Mumbai. The submarine surfaces off Versova and throws the crate to a courier.",
      home: "Into your laptop. 21 data checks tick off. Exchange rates are kept as market context; they don't pick stocks."
    } },
  fed: { name: "Fed", what: "world prices", flag: "US", road1: "fed_stl_nj", ship: "tanker", skyUTC: 12.5, skyLine: "Shown at dawn in St. Louis, sunset in India", startLabel: "St. Louis Fed (FRED)", portLabel: "New Jersey shore",
    sea: [[40.29965, -73.97619], [40.29, -73.95], ...X.SEA.atlantic, ...TO_VERSOVA],   // starts just past the surf at Long Branch
    say: {
      intro: "Fed's journey: oil, metal and crop prices, the dollar and US rates, from St. Louis to your laptop in Bengaluru.",
      start: "Fed picks up the prices at the St. Louis Fed, the home of FRED, and sets off.",
      road: km => `East through Indianapolis, Ohio and Pennsylvania to the Atlantic: ${km} km of real road.`,
      board: "The New Jersey shore, where Atlantic cables land. Fed jumps aboard Ship 1, the oil tanker.",
      sea: "Across the Atlantic, through Gibraltar, the Mediterranean, Suez and the Red Sea. In real life the data crosses in a fraction of a second, through undersea cable.",
      handoff: "Mumbai. Ship 1 slows off Versova and throws the crate to a courier.",
      home: "Into your laptop. 21 data checks tick off. On Friday the ML bots use these prices to mark which commodity themes are rising or fading."
    } }
};
const ROAD2 = "courier_mum_blr";
// films the user has checked go on the live site; the others still play from their deep link (#follow-<id>) for review
const LIVE_FILMS = [];   // none yet: in real time the street-level map loads too slowly (2026-10-17); films play from #follow-<id> for review
const road2Say = km => `Over the Western Ghats and across the Deccan: the courier runs ${km} km of highway to Bengaluru.`;
// seconds of film per phase (about 30 s in all)
const PHASES = [["intro", 2.2], ["dive", 2.8], ["road1", 7.0], ["board", 2.0], ["sea", 6.0], ["handoff", 1.8], ["road2", 6.4], ["home", 3.6]];
const TOTAL = PHASES.reduce((a, p) => a + p[1], 0);

/* ---------------------------------------------------------------- tracks: points, distances, and a speed profile slow at both ends */
function trackFromLonLat(ll) {
  const cart = ll.map(([lo, la]) => C.Cartesian3.fromDegrees(lo, la, 0)), cum = [0];
  for (let i = 1; i < cart.length; i++) cum.push(cum[i - 1] + C.Cartesian3.distance(cart[i - 1], cart[i]));
  return { ll, cart, cum, len: cum[cum.length - 1] };
}
function trackFromSea(wps) {
  return trackFromLonLat(X.seaPath(wps).map(p => { const c = C.Cartographic.fromCartesian(p); return [C.Math.toDegrees(c.longitude), C.Math.toDegrees(c.latitude)]; }));
}
function pointAt(tr, s) {                                // [lon, lat] at distance s along the track
  s = clamp(s, 0, tr.len); let lo = 0, hi = tr.cum.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (tr.cum[m] <= s) lo = m; else hi = m; }
  const f = (s - tr.cum[lo]) / Math.max(1e-6, tr.cum[hi] - tr.cum[lo]), a = tr.ll[lo], b = tr.ll[hi];
  return [lerp(a[0], b[0], f), lerp(a[1], b[1], f)];
}
function bearingAt(tr, s, ahead) {
  const a = pointAt(tr, s - ahead * 0.3), b = pointAt(tr, s + ahead);
  return X.headingAt(C.Cartesian3.fromDegrees(a[0], a[1]), C.Cartesian3.fromDegrees(b[0], b[1]));
}
// speed shape: flat (slow) for the first and last `edgeT` of the time, rising smoothly to a peak in the middle; the peak is
// chosen so exactly `edgeM` metres are covered in each slow stretch (the street-level or harbour part of the shot)
function profile(len, edgeT, edgeM) {
  const N = 600, smooth = x => x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x);
  const g = t => t < 0.5 ? smooth((t - edgeT) / (0.5 - edgeT)) : smooth((1 - edgeT - t) / (0.5 - edgeT));
  const build = lnR => { const W = [0]; for (let i = 1; i <= N; i++) W.push(W[i - 1] + Math.exp(lnR * g((i - 0.5) / N))); return W; };
  let lo = 0, hi = 20, Wt = build(0);
  for (let k = 0; k < 40; k++) { const mid = (lo + hi) / 2, Wm = build(mid); if (len * Wm[Math.round(edgeT * N)] / Wm[N] > edgeM) lo = mid; else hi = mid; Wt = Wm; }
  return { N, W: Wt, len, s: t => { const x = clamp(t, 0, 1) * N, i = Math.floor(x), f = x - i; return len * lerp(Wt[i], Wt[Math.min(N, i + 1)], f) / Wt[N]; } };
}

/* ---------------------------------------------------------------- models and scene pieces for one film */
const F = { id: null, ready: false, playing: false, t: 0, phase: -1, models: [], ents: [] };
function clip(m, name, once, speed) {
  if (!m) return;
  const go = () => { try { m.activeAnimations.removeAll(); m.activeAnimations.add({ name, loop: once ? C.ModelAnimationLoop.NONE : C.ModelAnimationLoop.REPEAT, multiplier: speed || 1 }); } catch (e) {} };
  if (m.ready) go(); else m.readyEvent.addEventListener(go);
}
const HPR = new C.HeadingPitchRoll();
function place(m, lon, lat, h, heading, pitch) {
  if (!m) return;
  HPR.heading = heading - Math.PI / 2; HPR.pitch = pitch || 0; HPR.roll = 0;   // models face +Z, which Cesium turns to +X
  C.Transforms.headingPitchRollToFixedFrame(C.Cartesian3.fromDegrees(lon, lat, h), HPR, C.Ellipsoid.WGS84, undefined, m.modelMatrix);
}
async function model(urls, scale, minPx) {
  const m = await X.loadModel(urls, { scale, minimumPixelSize: minPx, maximumScale: scale * 400 });
  if (m) { m.show = false; F.models.push(m); }
  return m;
}
function label(text, posFn, maxDist) {
  const e = viewer.entities.add({ position: new C.CallbackProperty(posFn, false), show: false,
    label: { text, font: "600 15px Fredoka, sans-serif", fillColor: C.Color.WHITE, showBackground: true, backgroundColor: C.Color.fromCssColorString("rgba(24,21,28,0.85)"),
             backgroundPadding: new C.Cartesian2(10, 6), pixelOffset: new C.Cartesian2(0, -54), disableDepthTestDistance: Number.POSITIVE_INFINITY,
             distanceDisplayCondition: new C.DistanceDisplayCondition(0, maxDist || 4e7) } });
  F.ents.push(e); return e;
}
// the covered path: every point already passed, plus where the runner is now. Drawn even through hills and buildings,
// so the bright line always shows how far the crate has come
function streak(color) {
  const st = { pts: [], idx: 0, head: null };
  const mat = () => new C.PolylineGlowMaterialProperty({ glowPower: 0.28, taperPower: 1, color });
  st.e = viewer.entities.add({ show: false, polyline: { positions: new C.CallbackProperty(() => st.head ? st.pts.slice(0, st.idx + 1).concat([st.head]) : [], false),
    width: 11, arcType: C.ArcType.NONE, material: mat(), depthFailMaterial: mat() } });
  F.ents.push(st.e); return st;
}
function streakTo(st, tr, s, h) {                       // the streak now reaches distance s on track tr
  if (!st.pts.length) st.pts = tr.ll.map((q, i) => C.Cartesian3.fromDegrees(q[0], q[1], st.hAt ? st.hAt(tr.cum[i]) : 2));
  let i = st.idx; while (i < tr.cum.length - 1 && tr.cum[i + 1] <= s) i++; st.idx = i;
  const q = pointAt(tr, s); st.head = C.Cartesian3.fromDegrees(q[0], q[1], h);
}
const routes = {};
async function loadRoute(id) { if (!routes[id]) routes[id] = await (await fetch(`world/routes/${id}.json`)).json(); return routes[id]; }

async function prepare(id) {
  if (F.id === id && F.ready) return;
  teardown(); F.id = id; F.ready = false; F.H = null; F.warm = false;
  const j = J[id], r1 = await loadRoute(j.road1), r2 = await loadRoute(ROAD2);
  F.j = j; F.km1 = Math.round(r1.distance_km).toLocaleString("en-IN"); F.km2 = Math.round(r2.distance_km).toLocaleString("en-IN");
  F.road1 = trackFromLonLat(r1.coordinates); F.sea = trackFromSea(j.sea); F.road2 = trackFromLonLat(r2.coordinates);
  // the first and last 35 m of each road at running pace, with the camera just behind; time-lapse in between
  // the courier stops 160 m short of the giant laptop (about 185 m wide, it sits over the last stretch of road) and throws the crate in
  F.home = [W.D.HOME.lon, W.D.HOME.lat];
  F.r2Stop = F.road2.len; while (F.r2Stop > 0 && metres(pointAt(F.road2, F.r2Stop), F.home) < 160) F.r2Stop -= 5;
  F.p1 = profile(F.road1.len, 0.3, 35); F.ps = profile(F.sea.len, 0.16, 350); F.p2 = profile(F.r2Stop, 0.3, 35);
  // each character's own model when it exists (bot_<id>), else its country's bot
  F.runner = await model([BOTS + `bot_${id}_carry.glb`, BOTS + `bot_${j.flag}_carry.glb`], 1.8, 40);
  F.ship = await model(j.ship === "sub" ? [PROPS + "submarine.glb"] : [PROPS + "tanker_oil.glb"], j.ship === "sub" ? 9 : 14, 70);
  F.courier = await model([BOTS + "bot_courier_carry.glb", BOTS + "bot_IN_carry.glb"], 1.8, 40);
  F.courierFree = await model([BOTS + "bot_courier.glb", BOTS + "bot_IN.glb"], 1.8, 40);   // empty hands before the catch and after the throw
  F.crate = await model([PROPS + "data_crate.glb"], 2.2, 18);
  // the whole route, faint, for the shots from space (hidden close up, where the trail tells the story)
  const all = F.road1.cart.concat(F.sea.cart, F.road2.cart);
  F.ents.push(viewer.entities.add({ show: false, polyline: { positions: all, width: 2.5, arcType: C.ArcType.NONE,
    material: new C.PolylineDashMaterialProperty({ color: C.Color.WHITE.withAlpha(0.55), dashLength: 14 }), distanceDisplayCondition: new C.DistanceDisplayCondition(4e5, 6e7) } }));
  F.cov = { r1: streak(X.AMBER), sea: streak(j.ship === "sub" ? X.CYAN : X.WAKE), r2: streak(X.AMBER) };
  F.cov.r1.hAt = s => baseH("r1", F.road1, s) + 1.5; F.cov.r2.hAt = s => baseH("r2", F.road2, s) + 1.5;
  F.cov.sea.hAt = s => (s < 30000 ? F.H.sea0 : s > F.sea.len - 30000 ? F.H.sea1 : 0) + 2;
  F.lblRunner = label(`${j.name} · ${j.what}`, () => F.runnerPos, 4e7);
  F.lblShip = label(j.ship === "sub" ? "The submarine · undersea cable" : "Ship 1 · oil tanker", () => F.shipPos, 4e7);
  F.lblCourier = label(`Courier · ${j.name}'s crate`, () => F.courierPos, 4e7);
  const s0 = F.road1.ll[0], e0 = F.road1.ll[F.road1.ll.length - 1];
  F.lblStart = label(j.startLabel, () => C.Cartesian3.fromDegrees(s0[0], s0[1], (F.g1 || 0) + 70), 2.5e5);
  F.lblPort = label(j.portLabel, () => C.Cartesian3.fromDegrees(e0[0], e0[1], (F.gPort || 0) + 70), 2.5e5);
  F.lblChecks = label("", () => C.Cartesian3.fromDegrees(F.home[0], F.home[1], (F.gHome || 0) + 230), 3e5);
  // the centre and size of the whole route, for the opening shot
  const sum = all.reduce((a, p) => C.Cartesian3.add(a, p, a), new C.Cartesian3());
  const cn = C.Cartesian3.normalize(sum, new C.Cartesian3()); C.Cartesian3.multiplyByScalar(cn, 6378137, cn);
  const cc = C.Cartographic.fromCartesian(cn);
  F.center = [C.Math.toDegrees(cc.longitude), C.Math.toDegrees(cc.latitude)];
  const cpt = C.Cartesian3.fromDegrees(F.center[0], F.center[1]);
  F.R0 = clamp(Math.max(...all.filter((_, i) => i % 7 === 0).map(p => C.Cartesian3.distance(p, cpt))) * 2.4, 6e6, 2.2e7);
  F.ready = true;
}
function teardown() {
  F.models.forEach(m => { scene.primitives.remove(m); }); F.ents.forEach(e => viewer.entities.remove(e));
  F.models = []; F.ents = []; F.ready = false; F.id = null;
}

/* ---------------------------------------------------------------- ground and sea heights
   Google's 3D map is measured from the ellipsoid, so the ground is not at 0: Frankfurt is about +150 m, Mumbai about -60 m.
   Before the film, the ground is measured properly (most detailed tiles) along the first and last 1.6 km of each road,
   where the camera is at street level, and on the water at both harbours. In the long middle the camera is high: the
   height is blended between the two ends, plus a lift that grows with the camera distance, so hills never swallow the bot. */
const EDGE = 1600, STEP = 50;
const tilesOn = () => { const ts = X.tileset(); return !!(ts && ts.show); };
function edgePoints(tr, fromEnd) { const out = []; for (let dd = 0; dd <= EDGE; dd += STEP) out.push(pointAt(tr, fromEnd ? tr.len - dd : dd)); return out; }
function profH(arr, dd) { const x = clamp(dd / STEP, 0, arr.length - 1), i = Math.floor(x); return lerp(arr[i], arr[Math.min(arr.length - 1, i + 1)], x - i); }
function roadH(key, tr, s, r) {
  if (!tilesOn()) return 0;                             // the plain globe sits on the ellipsoid
  const a = F.H[key + "a"], b = F.H[key + "b"];
  if (s <= EDGE) return profH(a, s);
  if (s >= tr.len - EDGE) return profH(b, tr.len - s);
  const base = lerp(a[a.length - 1], b[b.length - 1], (s - EDGE) / Math.max(1, tr.len - 2 * EDGE));
  return base + clamp((r - 250) * 0.03, 0, 3000);
}
function baseH(key, tr, s) {                           // ground without the lift, for the streak's points
  const a = F.H[key + "a"], b = F.H[key + "b"];
  if (s <= EDGE) return profH(a, s);
  if (s >= tr.len - EDGE) return profH(b, tr.len - s);
  return lerp(a[a.length - 1], b[b.length - 1], (s - EDGE) / Math.max(1, tr.len - 2 * EDGE));
}
function smooth(arr) {                                  // median of three: a car, a tree or a doorstep in one sample doesn't lift the runner
  if (arr.length < 3) return arr;
  return arr.map((v, i) => { const j = clamp(i, 1, arr.length - 2); return [arr[j - 1], arr[j], arr[j + 1]].sort((x, y) => x - y)[1]; });
}
function seaH(s) { return tilesOn() ? (s < F.sea.len / 2 ? F.H.sea0 : F.H.sea1) : 0; }
function fill(arr) {                                    // a failed sample takes its neighbour's value
  let lastOk = arr.find(v => v !== undefined && isFinite(v)); if (lastOk === undefined) return arr.map(() => 0);
  return arr.map(v => (v !== undefined && isFinite(v)) ? (lastOk = v) : lastOk);
}

/* ---------------------------------------------------------------- the director: where everything is at film time t, and the camera */
const cam = { lon: 0, lat: 0, h: 0, heading: 0, pitch: -Math.PI / 2, range: 1e7, init: false };
function phaseAt(t) { let a = 0; for (let i = 0; i < PHASES.length; i++) { if (t < a + PHASES[i][1]) return { i, name: PHASES[i][0], u: (t - a) / PHASES[i][1], dur: PHASES[i][1], t0: a }; a += PHASES[i][1]; } return { i: PHASES.length, name: "end", u: 1, dur: 1, t0: TOTAL }; }
function rangeFor(v, rmin, rmax) { return clamp(v * 0.5, rmin, rmax); }
function pitchFor(r, rmin, rmax, p0, p1) { return lerp(p0, p1, clamp(Math.log(r / rmin) / Math.log(rmax / rmin), 0, 1)) * RAD; }
function say(text) { if (F.caption !== text) { F.caption = text; $("chaseText").textContent = text; } }
function arc(a, b, ha, hb, k, peak) { return [lerp(a[0], b[0], k), lerp(a[1], b[1], k), lerp(ha, hb, k) + Math.sin(Math.PI * k) * peak]; }
function metres(a, b) { return C.Cartesian3.distance(C.Cartesian3.fromDegrees(a[0], a[1]), C.Cartesian3.fromDegrees(b[0], b[1])); }
// the street-level camera stands on the route itself, so it is always in the street, never inside a building:
// on the first half of a road it waits 18 m ahead and the bot runs towards it (you see its face); on the second half it
// follows 18 m behind along the road just covered. It sits 8 m up and looks at the bot's chest.
const CAM_BACK = 18, CAM_UP = 8;
function streetCam(key, tr, s, g, p) {
  const lead = s < tr.len / 2, sc = clamp(lead ? s + CAM_BACK : s - CAM_BACK, 0, tr.len), cp = pointAt(tr, sc);
  const ch = roadH(key, tr, sc, 0) + CAM_UP, horiz = Math.max(4, metres(cp, p)), vert = ch - (g + 1.1);
  return { heading: X.headingAt(C.Cartesian3.fromDegrees(cp[0], cp[1]), C.Cartesian3.fromDegrees(p[0], p[1])), pitch: -Math.atan2(vert, horiz), range: Math.hypot(horiz, vert) };
}
function speedAt(pr, u, dur) { const a = Math.max(0, u - 0.004), b = Math.min(1, u + 0.004); return (pr.s(b) - pr.s(a)) / ((b - a) * dur); }

function step(dt) {
  if (!F.ready) return;
  F.t = Math.min(TOTAL + 0.01, F.t + dt);
  const ph = phaseAt(F.t), j = F.j, d = { target: null, h: 0, heading: 0, pitch: -89 * RAD, range: 1e7, tau: 0.35 };
  if (ph.i !== F.phase) enterPhase(ph);
  const s0 = F.road1.ll[0], b0 = bearingAt(F.road1, 0, 60);
  const roadEnd = F.road1.ll[F.road1.ll.length - 1], seaStart = F.sea.ll[0], seaEnd = F.sea.ll[F.sea.ll.length - 1];
  const r2Start = F.road2.ll[0], r2End = pointAt(F.road2, F.r2Stop);
  const hStart = roadH("r1", F.road1, 0, 0), hPort = roadH("r1", F.road1, F.road1.len, 0);
  const hMum = roadH("r2", F.road2, 0, 0), hHome = roadH("r2", F.road2, F.r2Stop, 0);
  F.g1 = hStart; F.gPort = hPort; F.gHome = hHome;
  // where each piece is
  let runner = null, ship = null, courier = null, crate = null;
  if (ph.name === "intro" || ph.name === "dive") {
    runner = { lon: s0[0], lat: s0[1], h: hStart + 0.2, hd: b0 };
    const k = ph.name === "intro" ? 0 : ease(ph.u);
    if (ph.name === "intro") { d.target = F.center; d.range = F.R0 * (1 - 0.08 * ph.u); d.heading = 0; d.pitch = -89 * RAD; d.tau = 0.05; }
    else {                                              // dive from space to the front door, ending on the street camera
      const e = Math.pow(k, 0.85), sc = streetCam("r1", F.road1, 0, hStart, s0);
      d.target = [lerp(F.center[0], s0[0], ease(Math.min(1, ph.u * 1.4))), lerp(F.center[1], s0[1], ease(Math.min(1, ph.u * 1.4)))];
      d.range = Math.exp(lerp(Math.log(F.R0), Math.log(sc.range), e)); d.heading = wrap(sc.heading) * k; d.tau = 0.05; d.h = hStart; d.look = 1.1;
      // look steeply down while above the rooftops, then swoop onto the street: a shallow angle from far back goes through buildings
      d.pitch = d.range > 90 ? lerp(-89, -58, k) * RAD : lerp(-58 * RAD, sc.pitch, clamp(Math.log(90 / d.range) / Math.log(90 / sc.range), 0, 1));
    }
  } else if (ph.name === "road1" || ph.name === "road2") {
    const key = ph.name === "road1" ? "r1" : "r2", tr = ph.name === "road1" ? F.road1 : F.road2, pr = ph.name === "road1" ? F.p1 : F.p2;
    const s = pr.s(ph.u), v = speedAt(pr, ph.u, ph.dur), p = pointAt(tr, s), hd = bearingAt(tr, s, clamp(v * 0.25, 25, 40000));
    const droneR = rangeFor(v, 22, 2.2e5);
    const g = roadH(key, tr, s, droneR), who = { lon: p[0], lat: p[1], h: g + 0.2, hd };
    if (ph.name === "road1") runner = who; else courier = who;
    const up = clamp(Math.log(droneR / 22) / Math.log(3000 / 22), 0, 1);   // 0 on the street camera, 1 once the drone is up high
    const st = streetCam(key, tr, s, g, p), droneH = hd + 0.45 + 0.85 * Math.sin(ph.u * Math.PI * 1.6);
    d.target = p; d.h = g; d.look = 1.1;
    d.range = Math.exp(lerp(Math.log(st.range), Math.log(droneR), up)); d.heading = st.heading + wrap(droneH - st.heading) * up;
    d.pitch = lerp(st.pitch, pitchFor(droneR, 22, 2.2e5, -18, -58), up);
    streakTo(F.cov[key], tr, s, g + 1.5 + (d.range > 3000 ? d.range * 0.002 : 0));
    if (ph.name === "road1") ship = { lon: seaStart[0], lat: seaStart[1], h: seaH(0), hd: bearingAt(F.sea, 0, 3000) };
  } else if (ph.name === "board") {                     // the runner crouches, then jumps from the shore into the ship
    const gs = seaH(0), k = clamp((ph.u - 0.25) / 0.6, 0, 1), hop = metres(roadEnd, seaStart);
    const a = arc(roadEnd, seaStart, hPort + 0.2, gs + (j.ship === "sub" ? 9 : 16), ease(k), Math.max(12, hop * 0.35));
    runner = k < 1 ? { lon: a[0], lat: a[1], h: a[2], hd: X.headingAt(C.Cartesian3.fromDegrees(roadEnd[0], roadEnd[1]), C.Cartesian3.fromDegrees(seaStart[0], seaStart[1])) } : null;
    d.target = [(roadEnd[0] + seaStart[0]) / 2, (roadEnd[1] + seaStart[1]) / 2]; d.h = (hPort + gs) / 2; d.range = Math.max(70, hop * 1.7); d.look = 6;
    d.heading = X.headingAt(C.Cartesian3.fromDegrees(roadEnd[0], roadEnd[1]), C.Cartesian3.fromDegrees(seaStart[0], seaStart[1])) + Math.PI / 2 * 0.9; d.pitch = -15 * RAD; d.tau = 0.45;
    ship = { lon: seaStart[0], lat: seaStart[1], h: gs, hd: bearingAt(F.sea, 0, 3000) };
  } else if (ph.name === "sea") {
    const s = F.ps.s(ph.u), v = speedAt(F.ps, ph.u, ph.dur), p = pointAt(F.sea, s), hd = bearingAt(F.sea, s, clamp(v * 0.25, 200, 300000));
    d.range = rangeFor(v, 160, 3.2e6);
    const surface = seaH(s), deep = j.ship === "sub" && !tilesOn();   // out at sea, seen from high up, the submarine runs under the surface
    ship = { lon: p[0], lat: p[1], h: deep ? -45 : surface + 1 + Math.sin(F.t * 2.4) * 1.2, hd, roll: Math.sin(F.t * 1.9) * 0.05 };
    d.target = p; d.h = surface; d.heading = hd + Math.PI / 2 * 0.85 + 0.6 * Math.sin(ph.u * Math.PI * 1.3); d.pitch = pitchFor(d.range, 160, 3.2e6, -12, -62);
    streakTo(F.cov.sea, F.sea, s, deep ? -40 : surface + 2);
    courier = { lon: r2Start[0], lat: r2Start[1], h: hMum + 0.5, hd: bearingAt(F.road2, 0, 60), quiet: true };
  } else if (ph.name === "handoff") {                   // the ship throws the crate to the courier on the beach
    const gs = seaH(F.sea.len), k = clamp((ph.u - 0.2) / 0.55, 0, 1);
    ship = { lon: seaEnd[0], lat: seaEnd[1], h: gs + Math.sin(F.t * 2.4) * 1.2, hd: bearingAt(F.sea, F.sea.len, 3000) };
    const hd = X.headingAt(C.Cartesian3.fromDegrees(seaEnd[0], seaEnd[1]), C.Cartesian3.fromDegrees(r2Start[0], r2Start[1]));
    courier = { lon: r2Start[0], lat: r2Start[1], h: hMum + 0.5, hd: hd + Math.PI };
    if (k > 0 && k < 1) { const a = arc(seaEnd, r2Start, gs + 12, hMum + 1.3, ease(k), Math.max(10, metres(seaEnd, r2Start) * 0.3)); crate = { lon: a[0], lat: a[1], h: a[2], hd }; }
    d.target = [(seaEnd[0] + r2Start[0]) / 2, (seaEnd[1] + r2Start[1]) / 2]; d.h = (gs + hMum) / 2; d.range = Math.max(70, metres(seaEnd, r2Start) * 1.7); d.look = 5;
    d.heading = hd + Math.PI / 2 * 0.9; d.pitch = -15 * RAD; d.tau = 0.45;
  } else {                                              // home: the courier throws the crate into the laptop; checks count up
    const k = clamp((ph.u - 0.12) / 0.3, 0, 1);
    const hd = X.headingAt(C.Cartesian3.fromDegrees(r2End[0], r2End[1]), C.Cartesian3.fromDegrees(F.home[0], F.home[1]));
    courier = { lon: r2End[0], lat: r2End[1], h: hHome + 0.5, hd };
    if (k > 0 && k < 1) { const a = arc(r2End, F.home, hHome + 1.5, hHome + 95, ease(k), 160); crate = { lon: a[0], lat: a[1], h: a[2], hd }; }
    const n = Math.min(21, Math.floor(clamp((ph.u - 0.45) / 0.4, 0, 1) * 21));
    F.lblChecks.show = ph.u > 0.42; F.lblChecks.label.text = `✓ ${n} of 21 checks`;
    const pull = ease(clamp(ph.u / 0.55, 0, 1));
    d.target = [lerp(r2End[0], F.home[0], pull), lerp(r2End[1], F.home[1], pull)]; d.h = hHome; d.look = lerp(1.1, 60, pull);
    d.range = Math.exp(lerp(Math.log(22), Math.log(1000), pull)); d.heading = hd + lerp(0, 0.8, pull) - ph.u * 0.3; d.pitch = lerp(-18, -22, pull) * RAD; d.tau = 0.12;
  }
  // put the models where they are
  setPiece(F.runner, runner, "runnerPos", F.lblRunner, 30);
  setPiece(F.ship, ship, "shipPos", F.lblShip, 60);
  const free = courier && ((ph.name === "handoff" && ph.u < 0.75) || (ph.name === "home" && ph.u > 0.2) || ph.name === "sea");
  setPiece(F.courier, free ? null : courier, "courierPos", null, 30); setPiece(F.courierFree, free ? courier : null, "courierPos", null, 30);
  F.lblCourier.show = !!courier && !courier.quiet;
  setPiece(F.crate, crate, null, null, 0);
  // the camera: eased towards the shot, range in log space
  if (!cam.init || ph.name === "intro") { Object.assign(cam, { lon: d.target[0], lat: d.target[1], h: d.h, range: d.range, heading: d.heading, pitch: d.pitch, init: true }); }
  const a = 1 - Math.exp(-dt / Math.max(0.01, d.tau)), follow = ph.name.startsWith("road") || ph.name === "sea";
  cam.lon = lerp(cam.lon, d.target[0], follow ? 1 : a); cam.lat = lerp(cam.lat, d.target[1], follow ? 1 : a); cam.h = lerp(cam.h, d.h, follow ? 1 : a);
  cam.range = Math.exp(lerp(Math.log(cam.range), Math.log(d.range), a)); cam.heading = cam.heading + wrap(d.heading - cam.heading) * a; cam.pitch = lerp(cam.pitch, d.pitch, a);
  cam.look = lerp(cam.look === undefined ? 6 : cam.look, d.look || 6, follow ? 1 : a);
  camera.lookAt(C.Cartesian3.fromDegrees(cam.lon, cam.lat, cam.h + cam.look), new C.HeadingPitchRange(cam.heading, cam.pitch, cam.range));
  const bar = $("chaseProgress"); if (bar) bar.style.transform = `scaleX(${clamp(F.t / TOTAL, 0, 1)})`;
  if (F.t >= TOTAL && !F.ended && !F.warming) end();
}
function setPiece(m, at, key, lbl, up) {
  if (!m) return;
  m.show = !!at;
  if (lbl) lbl.show = !!at;
  if (!at) return;
  place(m, at.lon, at.lat, at.h, at.hd, at.pitch);
  if (at.roll) { const r = C.Matrix4.fromRotation(C.Matrix3.fromRotationX(at.roll)); C.Matrix4.multiply(m.modelMatrix, r, m.modelMatrix); }
  if (key) F[key] = C.Cartesian3.fromDegrees(at.lon, at.lat, at.h + up);
}
function enterPhase(ph) {
  F.phase = ph.i; const j = F.j;
  F.ents.forEach(e => { if (e.polyline && !e.label) e.show = true; });   // the route overview and the trails
  if (ph.name === "intro") { say(j.say.intro); F.lblStart.show = true; clip(F.runner, "idle"); clip(F.ship, "idle"); clip(F.courier, "idle"); }
  if (ph.name === "dive") say(j.say.start);
  if (ph.name === "road1") { say(j.say.road(F.km1)); clip(F.runner, "run"); F.lblStart.show = false; }
  if (ph.name === "board") { say(j.say.board); F.lblPort.show = true; clip(F.runner, "jump", true); }
  if (ph.name === "sea") { say(j.say.sea); F.lblPort.show = false; }
  if (ph.name === "handoff") { say(j.say.handoff); clip(F.courierFree, "idle"); clip(F.courier, "catch", true); }
  if (ph.name === "road2") { say(road2Say(F.km2)); clip(F.courier, "run"); }
  if (ph.name === "home") { say(j.say.home); clip(F.courierFree, "throw", true); X.playStatic("me", "cheer", false); }
}

/* ---------------------------------------------------------------- play, skip, end */
const ctl = scene.screenSpaceCameraController;
let last = 0;
async function play(id) {
  if (!J[id]) return;
  W.closeCard(); ["intro", "caption", "chips", "lineup"].forEach(k => { const e = $(k); if (e) e.hidden = true; });
  $("chase").hidden = false; $("chaseEnd").hidden = true; F.caption = null; say("Getting the map ready…");
  ctl.enableInputs = false; X.setAmbient(false);
  if (!X.RECORD) {                                       // the film's own sky, not the visitor's time of day
    const d = new Date(); d.setUTCHours(0, 0, 0, 0); d.setTime(d.getTime() + J[id].skyUTC * 3600e3);
    viewer.clock.currentTime = C.JulianDate.fromDate(d); X.setSunline(J[id].skyLine);
  }
  if (F.loading && F.loadingId === id) { $("chaseCurtain").hidden = false; await F.loading; }
  else await load(id);
  $("chaseCurtain").hidden = true;
  Object.values(F.cov).forEach(st => { st.pts = []; st.idx = 0; st.head = null; });
  F.t = 0; F.phase = -1; F.ended = false; F.caption = null; F.slowUsed = 0; cam.init = false; cam.look = undefined; X.playStatic("me", "wave", false);
  F.playing = !X.RECORD; last = performance.now();
}
// models, routes and heights; then (live only) a visit to each street-level spot so Google's tiles are cached before the
// film needs them. "Start the journey" runs this behind the crew lineup, which covers the whole screen.
async function load(id) {
  await prepare(id);
  if (!F.H) {                                          // heights measured once and saved with the routes; measured live if the file is missing
    try { const r = await fetch(`world/routes/heights_${id}.json`); if (r.ok) F.H = (await r.json()).heights; } catch (e) {}
    if (F.H) ["r1a", "r1b", "r2a", "r2b"].forEach(k => { F.H[k] = smooth(F.H[k]); });
    if (!F.H) await measure();
  }
  if (!X.RECORD && !F.warm) { await warm(); F.warm = true; }
}
function preload(id) {
  if (!J[id] || X.RECORD) return;
  F.loadingId = id; F.loading = load(id).catch(() => {}).then(() => { F.loading = null; camera.lookAtTransform(C.Matrix4.IDENTITY); });
  return F.loading;
}
// a dry run of the film behind the lineup: fast-forward, and at every street or harbour moment wait until that exact
// view's tiles are in, so the real film finds them in the cache (capped at 14 s; the curtain covers any overrun)
async function warm() {
  X.ensureTileset();
  const t0 = performance.now();
  F.t = 0; F.phase = -1; cam.init = false; F.warming = true;
  while (F.t < TOTAL - 0.6 && performance.now() - t0 < 14000) {
    step(0.25);
    if (cam.range < 3000) {
      const ts0 = performance.now(); let calm = 0;
      while (performance.now() - ts0 < 1500) { await sleep(100); const ts = X.tileset(); calm = ts && ts.show && ts.tilesLoaded ? calm + 1 : 0; if (calm >= 2) break; }
    } else await sleep(30);
  }
  F.warming = false; F.models.forEach(m => { m.show = false; }); camera.lookAtTransform(C.Matrix4.IDENTITY);
}
// behind a curtain, visit the four street-level places: Google's 3D tiles load and stay cached, and the ground is measured there
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function pump(ms, done) {                         // wait, rendering by hand in record mode
  const t0 = performance.now();
  while (performance.now() - t0 < ms && !(done && done())) { if (X.RECORD) viewer.render(); await sleep(X.RECORD ? 20 : 100); }
}
async function measure() {
  const cur = $("chaseCurtain"); cur.hidden = false;
  X.ensureTileset();
  const spots = [
    { key: "r1a", at: F.road1.ll[0], pts: edgePoints(F.road1, false) },
    { key: "r1b", at: F.road1.ll[F.road1.ll.length - 1], pts: edgePoints(F.road1, true), sea: ["sea0", F.sea.ll[0]] },
    { key: "r2a", at: F.road2.ll[0], pts: edgePoints(F.road2, false), sea: ["sea1", F.sea.ll[F.sea.ll.length - 1]] },
    { key: "r2b", at: F.road2.ll[F.road2.ll.length - 1], pts: edgePoints(F.road2, true) }
  ];
  const H = { log: [] }, exclude = F.models.concat(X.staticModels());
  camera.lookAt(C.Cartesian3.fromDegrees(spots[0].at[0], spots[0].at[1], 0), new C.HeadingPitchRange(0.5, -35 * RAD, 1400));
  await pump(40000, () => { const ts = X.tileset(); return ts && ts.show; });   // Google's 3D map is created on the first close-up view
  for (const sp of spots) {
    camera.lookAt(C.Cartesian3.fromDegrees(sp.at[0], sp.at[1], 0), new C.HeadingPitchRange(0.5, -35 * RAD, 1400));
    await pump(8000, () => { const ts = X.tileset(); return ts && ts.show && ts.tilesLoaded; });
    if (!tilesOn()) { H[sp.key] = sp.pts.map(() => 0); H.log.push(`${sp.key}: 3D map not on`); continue; }
    const list = sp.pts.map(p => C.Cartographic.fromDegrees(p[0], p[1])); if (sp.sea) list.push(C.Cartographic.fromDegrees(sp.sea[1][0], sp.sea[1][1]));
    let out = null;
    for (let attempt = 0; attempt < 3 && !out; attempt++)   // a slow tile server sometimes needs a second go
    try { if (scene.sampleHeightSupported) { const pr = scene.sampleHeightMostDetailed(list, exclude); let ok = false; pr.then(v => { out = v; ok = true; }, e => { ok = true; H.log.push(`${sp.key}: ${e}`); }); await pump(30000, () => ok); } }
    catch (e) { out = null; H.log.push(`${sp.key}: ${e}`); }
    H.log.push(`${sp.key}: ${out ? out.filter(c => c && isFinite(c.height)).length : "no result"} of ${list.length} heights`);
    const hs = out ? out.map(c => c && c.height) : list.map(() => undefined);
    if (sp.sea) { const v = hs.pop(); H[sp.sea[0]] = v !== undefined && isFinite(v) ? v : undefined; }
    H[sp.key] = smooth(fill(hs));
  }
  H.sea0 = H.sea0 !== undefined ? H.sea0 : H.r1b[0]; H.sea1 = H.sea1 !== undefined ? H.sea1 : H.r2a[0];
  F.H = H;
  camera.lookAtTransform(C.Matrix4.IDENTITY); cur.hidden = true;
}
function end() {
  F.ended = true; F.playing = false;
  const others = LIVE_FILMS.filter(k => k !== F.id);
  $("chaseEnd").hidden = false; $("chaseOther").hidden = !others.length;
  if (others.length) { $("chaseOther").textContent = `Watch ${J[others[0]].name}'s journey`; $("chaseOther").dataset.id = others[0]; }
}
function stop() {
  F.playing = false; if (!X.RECORD) { viewer.clock.currentTime = X.AT ? C.JulianDate.fromIso8601(X.AT) : C.JulianDate.now(); X.setSunline(null); } camera.lookAtTransform(C.Matrix4.IDENTITY); ctl.enableInputs = true; X.setAmbient(true); X.playStatic("me", "wave", false);
  F.models.forEach(m => { m.show = false; }); F.ents.forEach(e => { e.show = false; });
  $("chase").hidden = true; W.startExplore();
}
$("chaseSkip").addEventListener("click", stop);
$("chaseExplore").addEventListener("click", stop);
$("chaseAgain").addEventListener("click", () => play(F.id));
$("chaseOther").addEventListener("click", e => play(e.currentTarget.dataset.id));
scene.preUpdate.addEventListener(() => {
  const now = performance.now(), dt = Math.min(0.1, (now - last) / 1000); last = now;
  const ts = X.tileset();
  // the map normally holds back loading while the camera moves (fine when someone drags it); a film never stops moving,
  // so during a film (and its dry run) tiles load as they're needed
  if (ts && !X.RECORD) { const film = F.playing || !!F.warming; ts.foveatedScreenSpaceError = !film; ts.cullRequestsWhileMoving = !film; }
  if (!F.playing || X.RECORD) return;
  let k = W.REDUCED ? 0.6 : 1;   // close to the ground with tiles still coming in: run at 30% speed, at most 6 extra seconds per film
  if (cam.range < 2500 && ts && ts.show && !ts.tilesLoaded && (F.slowUsed || 0) < 6) { k *= 0.3; F.slowUsed = (F.slowUsed || 0) + dt * 0.7; }
  step(dt * k);
});

/* record mode (?record): the page is driven one frame at a time from outside, so a preview video can be made from it */
window.SGM_CHASE = { has: id => LIVE_FILMS.includes(id), play, stop, preload, ids: Object.keys(J), TOTAL,
  measured: () => F.H, remeasure: async () => { F.H = null; await measure(); return F.H; },
  async prepareRecord(id) { viewer.useDefaultRenderLoop = false; viewer.clock.shouldAnimate = false; F.clock0 = C.JulianDate.clone(viewer.clock.currentTime); await play(id); return F.H; },
  frame(dt) { step(dt); viewer.clock.currentTime = C.JulianDate.addSeconds(F.clock0, F.t, new C.JulianDate()); return { t: F.t, phase: PHASES[Math.min(PHASES.length - 1, F.phase)][0], caption: F.caption, done: F.t >= TOTAL }; } };
const m = location.hash.match(/^#follow-(\w+)$/);
if (m && J[m[1]] && !X.RECORD) setTimeout(() => play(m[1]), 600);
})();
