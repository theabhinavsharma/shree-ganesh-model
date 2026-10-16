/* SGM Data World (realistic version). CesiumJS globe in real time (day, evening, night), Google Photorealistic 3D Tiles
   near the ground when a key is configured, cartoon bots running real OpenStreetMap road routes, and data packets
   streaming home to Bengaluru. Content lives in world/data.js; the Google key in world/config.js (not in git). */
(function () {
"use strict";
const C = window.Cesium, D = window.SGM_DATA, CFG = window.SGM_CONFIG || {};
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const PHONE = () => window.innerWidth <= 640;
const $ = id => document.getElementById(id);
const LOW = 250000;                                   // below this height (m) we show the street-level city
const TZ = { bom: "Asia/Kolkata", del: "Asia/Kolkata", web: "Asia/Kolkata", blr: "Asia/Kolkata", stl: "America/Chicago",
             fra: "Europe/Berlin", mtv: "America/Los_Angeles", dxb: "Asia/Dubai" };

/* ---------------------------------------------------------------- viewer: NASA imagery for the globe, city lights at night */
const gibs = (layer, ext) => new C.UrlTemplateImageryProvider({
  url: `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/${layer}/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.${ext}`,
  maximumLevel: 8, credit: new C.Credit("Globe imagery: NASA EOSDIS GIBS (Blue Marble, VIIRS city lights)")
});
const viewer = new C.Viewer("cesium", {
  baseLayer: new C.ImageryLayer(gibs("BlueMarble_NextGeneration", "jpeg")),
  animation: false, timeline: false, baseLayerPicker: false, geocoder: false, homeButton: false, sceneModePicker: false,
  navigationHelpButton: false, fullscreenButton: false, infoBox: false, selectionIndicator: false, shouldAnimate: true
});
const scene = viewer.scene, camera = viewer.camera, globe = scene.globe;
const dayLayer = viewer.imageryLayers.get(0); dayLayer.dayAlpha = 1; dayLayer.nightAlpha = 0.04;
const nightLayer = viewer.imageryLayers.addImageryProvider(gibs("VIIRS_CityLights_2012", "jpeg"));
nightLayer.dayAlpha = 0; nightLayer.nightAlpha = 1; nightLayer.brightness = 2.4; nightLayer.contrast = 1.3;
const streetLayer = viewer.imageryLayers.addImageryProvider(new C.OpenStreetMapImageryProvider({ url: "https://tile.openstreetmap.org/" }));
streetLayer.show = false;                              // only a stand-in until the Google key is set
globe.enableLighting = true; globe.showGroundAtmosphere = true; globe.baseColor = C.Color.fromCssColorString("#0b1a2a");
scene.skyAtmosphere.show = true; scene.fog.enabled = true;
viewer.clock.currentTime = C.JulianDate.now(); viewer.clock.multiplier = 1; viewer.clock.shouldAnimate = true;
camera.setView({ destination: C.Cartesian3.fromDegrees(72, 16, PHONE() ? 26e6 : 19e6) });

/* ---------------------------------------------------------------- Google Photorealistic 3D Tiles (lazy: only near the ground) */
let tileset = null, tilesetTried = false;
function ensureTileset() {
  if (tileset || tilesetTried || !CFG.googleKey) return;
  tilesetTried = true;
  C.GoogleMaps.defaultApiKey = CFG.googleKey;
  C.createGooglePhotorealistic3DTileset(undefined, { showCreditsOnScreen: true, maximumScreenSpaceError: PHONE() ? 24 : 16 })
    .then(ts => { tileset = ts; scene.primitives.add(ts); sampleRouteHeights(); })
    .catch(err => showNote("Google's 3D city didn't load (" + (err && err.message || err) + "). Showing the satellite globe instead."));
}

/* ---------------------------------------------------------------- real-time sun: evening and night colour near the ground */
const tint = { color: new C.Cartesian3(1, 1, 1), amount: 0 };
scene.postProcessStages.add(new C.PostProcessStage({
  fragmentShader: `uniform sampler2D colorTexture; in vec2 v_textureCoordinates; uniform vec3 tintColor; uniform float amount;
    void main() { vec4 c = texture(colorTexture, v_textureCoordinates); float l = dot(c.rgb, vec3(0.299, 0.587, 0.114));
      vec3 graded = mix(vec3(l), c.rgb, 0.45) * tintColor; out_FragColor = vec4(mix(c.rgb, graded, amount), c.a); }`,
  uniforms: { tintColor: () => tint.color, amount: () => tint.amount }
}));
function sunElevation(carto, time) {
  const m = C.Transforms.computeIcrfToFixedMatrix(time) || C.Transforms.computeTemeToPseudoFixedMatrix(time);
  const sun = C.Matrix3.multiplyByVector(m, C.Simon1994PlanetaryPositions.computeSunPositionInEarthInertialFrame(time), new C.Cartesian3());
  const p = C.Cartographic.toCartesian(new C.Cartographic(carto.longitude, carto.latitude, 0));
  const up = C.Ellipsoid.WGS84.geodeticSurfaceNormalCartographic(carto);
  const to = C.Cartesian3.normalize(C.Cartesian3.subtract(sun, p, new C.Cartesian3()), new C.Cartesian3());
  return C.Math.toDegrees(Math.asin(C.Cartesian3.dot(up, to)));
}
const WARM = new C.Cartesian3(1.0, 0.8, 0.62), NIGHT = new C.Cartesian3(0.5, 0.6, 0.95);
function updateTint(elev, near) {
  if (!near || elev >= 8) { tint.amount = 0; return; }
  if (elev >= -4) { C.Cartesian3.clone(WARM, tint.color); tint.amount = (8 - elev) / 12 * 0.42; return; }
  const k = Math.min(1, (-4 - elev) / 6);
  C.Cartesian3.lerp(WARM, NIGHT, k, tint.color); tint.amount = 0.42 + k * 0.23;
}

/* ---------------------------------------------------------------- stations (city level) and the NSE building */
const ST = Object.fromEntries(D.STATIONS.map(s => [s.id, s]));
const byStation = {}; D.BOTS.forEach(b => (byStation[b.st] = byStation[b.st] || []).push(b));
const ORANGE = C.Color.fromCssColorString("#e8875c");
D.STATIONS.filter(s => s.real).forEach(s => {          // "On the web" has no confirmed city, so it gets no pin
  const ll = s.real;
  viewer.entities.add({
    id: "st-" + s.id, position: C.Cartesian3.fromDegrees(ll[1], ll[0], 0),
    point: { pixelSize: 9, color: ORANGE, outlineColor: C.Color.WHITE, outlineWidth: 2, disableDepthTestDistance: 5e6,
             distanceDisplayCondition: new C.DistanceDisplayCondition(30000, 4e7) },
    label: { text: s.name, font: "600 14px Fredoka, sans-serif", fillColor: C.Color.WHITE, outlineColor: C.Color.BLACK,
             outlineWidth: 3, style: C.LabelStyle.FILL_AND_OUTLINE, pixelOffset: new C.Cartesian2(0, -16), disableDepthTestDistance: 5e6,
             distanceDisplayCondition: new C.DistanceDisplayCondition(30000, 2.6e7), showBackground: false }
  });
});
const NSE = D.BUILDINGS.nse, NSE_POS = C.Cartesian3.fromDegrees(NSE.lon, NSE.lat, 0);
viewer.entities.add({
  id: "bld-nse", position: C.Cartesian3.fromDegrees(NSE.lon, NSE.lat, 95),
  label: { text: "NSE · Exchange Plaza", font: "600 15px Fredoka, sans-serif", fillColor: C.Color.WHITE, showBackground: true,
           backgroundColor: C.Color.fromCssColorString("rgba(24,21,28,0.8)"), backgroundPadding: new C.Cartesian2(10, 6),
           disableDepthTestDistance: Number.POSITIVE_INFINITY, distanceDisplayCondition: new C.DistanceDisplayCondition(0, 30000) }
});
const HOME_POS = C.Cartesian3.fromDegrees(D.HOME.lon, D.HOME.lat, 0);

/* ---------------------------------------------------------------- data streams (globe and country level) */
function arc(fromLL, toLL, peak, n) {
  const a = C.Cartographic.fromDegrees(fromLL[0], fromLL[1]), b = C.Cartographic.fromDegrees(toLL[0], toLL[1]);
  const g = new C.EllipsoidGeodesic(a, b), pts = [];
  for (let i = 0; i <= n; i++) { const t = i / n, c = g.interpolateUsingFraction(t); pts.push(C.Cartesian3.fromRadians(c.longitude, c.latitude, peak * Math.sin(Math.PI * t))); }
  return pts;
}
function sea(wps, h) {                                  // simplified undersea-cable route along sea waypoints [lat, lon]
  const pts = [];
  for (let i = 0; i < wps.length - 1; i++) {
    const g = new C.EllipsoidGeodesic(C.Cartographic.fromDegrees(wps[i][1], wps[i][0]), C.Cartographic.fromDegrees(wps[i + 1][1], wps[i + 1][0]));
    for (let k = 0; k < 12; k++) { const c = g.interpolateUsingFraction(k / 12); pts.push(C.Cartesian3.fromRadians(c.longitude, c.latitude, h)); }
  }
  const l = wps[wps.length - 1]; pts.push(C.Cartesian3.fromDegrees(l[1], l[0], h)); return pts;
}
const MUM_LAND = [19.1, 72.8];
const ROUTE_WPS = {
  us: [[40.2, -73.6], [39.5, -60], [38, -40], [37, -20], [36, -9], [35.95, -5.6], [37.5, 2], [37.3, 11], [35, 20], [33.5, 28], [31.4, 32.4], [29.8, 32.6], [27.5, 34], [22, 37.8], [16, 41.5], [12.6, 43.3], [12.2, 48], [13.5, 53], [16, 60], [18.5, 67], [19.0, 72.6]],
  eu: [[43.1, 5.4], [40.5, 6.5], [37.6, 10.5], [35, 20], [33.5, 28], [31.4, 32.4], [29.8, 32.6], [27.5, 34], [22, 37.8], [16, 41.5], [12.6, 43.3], [12.2, 48], [13.5, 53], [16, 60], [18.5, 67], [19.0, 72.6]]
};
const STREAMS = [];
function stream(id, pts, color, secs, label) {
  viewer.entities.add({ id: "route-" + id, polyline: { positions: pts, width: 2, arcType: C.ArcType.NONE,
    material: new C.PolylineDashMaterialProperty({ color: color.withAlpha(0.75), dashLength: 18 }),
    distanceDisplayCondition: new C.DistanceDisplayCondition(40000, 5e7) } });
  const spline = new C.CatmullRomSpline({ times: pts.map((_, i) => i / (pts.length - 1)), points: pts });
  for (let k = 0; k < 3; k++) {
    const s = { spline, secs, phase: k / 3 };
    viewer.entities.add({ position: new C.CallbackProperty(time => {
        const t = ((C.JulianDate.secondsDifference(time, T0) / s.secs + s.phase) % 1 + 1) % 1; return s.spline.evaluate(t); }, false),
      point: { pixelSize: 7, color: color, outlineColor: C.Color.WHITE.withAlpha(0.8), outlineWidth: 1.5,
               distanceDisplayCondition: new C.DistanceDisplayCondition(40000, 5e7) }, description: label });
    STREAMS.push(s);
  }
}
const T0 = C.JulianDate.now();
const DATA = C.Color.fromCssColorString("#f0a35e"), MSG = C.Color.fromCssColorString("#8fbdf0");
const ll = id => { const s = ST[id]; const p = s.real || s.at; return [p[1], p[0]]; };
stream("stl", arc(ll("stl"), [-73.9, 40.3], 300000, 40).concat(sea(ROUTE_WPS.us, 20000)), DATA, 9, "FRED prices by undersea cable");
stream("fra", arc(ll("fra"), [5.37, 43.3], 150000, 20).concat(sea(ROUTE_WPS.eu, 20000)), DATA, 7, "ECB rates by undersea cable");
stream("land", arc([MUM_LAND[1], MUM_LAND[0]], [D.HOME.lon, D.HOME.lat], 120000, 30), DATA, 3.5, "Mumbai cable landing to Bengaluru");
stream("bom", arc(ll("bom"), [D.HOME.lon, D.HOME.lat], 160000, 30), DATA, 3, "Mumbai to Bengaluru");
stream("del", arc(ll("del"), [D.HOME.lon, D.HOME.lat], 260000, 40), DATA, 4, "New Delhi to Bengaluru");
stream("mtv", arc(ll("mtv"), [D.HOME.lon, D.HOME.lat], 2500000, 90), DATA, 12, "California to Bengaluru");
stream("tg", arc([D.HOME.lon, D.HOME.lat], ll("dxb"), 400000, 40), MSG, 4, "Messages to Telegram");

/* ---------------------------------------------------------------- bots on a real road out of NSE (street level) */
const route = { pts: [], cum: [], len: 0, heights: null };
const bots = [];
function bearing(a, b) {                                // degrees clockwise from north, a/b = [lon, lat]
  const y = Math.sin((b[0] - a[0]) * C.Math.RADIANS_PER_DEGREE) * Math.cos(b[1] * C.Math.RADIANS_PER_DEGREE);
  const x = Math.cos(a[1] * C.Math.RADIANS_PER_DEGREE) * Math.sin(b[1] * C.Math.RADIANS_PER_DEGREE) -
            Math.sin(a[1] * C.Math.RADIANS_PER_DEGREE) * Math.cos(b[1] * C.Math.RADIANS_PER_DEGREE) * Math.cos((b[0] - a[0]) * C.Math.RADIANS_PER_DEGREE);
  return Math.atan2(y, x) * C.Math.DEGREES_PER_RADIAN;
}
function metres(a, b) { const R = 6371000, la = (a[1] + b[1]) / 2 * C.Math.RADIANS_PER_DEGREE;
  const dx = (b[0] - a[0]) * C.Math.RADIANS_PER_DEGREE * Math.cos(la) * R, dy = (b[1] - a[1]) * C.Math.RADIANS_PER_DEGREE * R; return Math.hypot(dx, dy); }
function routeAt(s) {
  let i = 1; while (i < route.cum.length - 1 && route.cum[i] < s) i++;
  const a = route.pts[i - 1], b = route.pts[i], t = (s - route.cum[i - 1]) / Math.max(1e-6, route.cum[i] - route.cum[i - 1]);
  const h = route.heights ? route.heights[i - 1] + (route.heights[i] - route.heights[i - 1]) * t : 0;
  return { lon: a[0] + (b[0] - a[0]) * t, lat: a[1] + (b[1] - a[1]) * t, h, heading: bearing(a, b) };
}
function sampleRouteHeights() {
  if (!tileset || !route.pts.length || route.heights) return;
  const cartos = route.pts.map(p => C.Cartographic.fromDegrees(p[0], p[1]));
  scene.sampleHeightMostDetailed(cartos, bots.map(b => b.model).filter(Boolean)).then(res => {
    const hs = res.map(c => c && isFinite(c.height) ? c.height : NaN), ok = hs.filter(isFinite);
    if (ok.length < hs.length * 0.6) return;                          // tiles not loaded well enough yet; try again later
    const med = ok.slice().sort((a, b) => a - b)[Math.floor(ok.length / 2)];
    route.heights = hs.map(h => isFinite(h) ? h + 0.3 : med);
  }).catch(() => {});
}
const BOT_SCALE = 6, RUN_SPEED = 55;                  // cartoon-sized bots, running fast along the road (m/s)
async function addBots() {
  try {
    const r = await fetch("world/routes/nse_bkc_exit.json"); const j = await r.json();
    route.pts = j.coordinates; route.cum = [0];
    for (let i = 1; i < route.pts.length; i++) route.cum.push(route.cum[i - 1] + metres(route.pts[i - 1], route.pts[i]));
    route.len = route.cum[route.cum.length - 1]; route.source = j.source;
  } catch (e) { return; }
  const info = D.BOTS.find(b => b.id === "bhav");
  for (let k = 0; k < 3; k++) {
    const bot = { offset: k * route.len / 3, info, model: null, fallback: null };
    try {
      bot.model = await C.Model.fromGltfAsync({ url: "assets/bots/bot_IN_carry.glb", scale: BOT_SCALE, id: { bot: "bhav" } });
      scene.primitives.add(bot.model);
      bot.model.readyEvent.addEventListener(() => { try { bot.model.activeAnimations.add({ name: "run", loop: C.ModelAnimationLoop.REPEAT, multiplier: 1.4 }); } catch (e) {} });
    } catch (e) {                                       // bot model not built yet: a plain orange stand-in
      bot.fallback = viewer.entities.add({ id: "bot-fallback-" + k, box: { dimensions: new C.Cartesian3(4, 4, 6), material: ORANGE }, position: NSE_POS, properties: { bot: "bhav" } });
    }
    bots.push(bot);
  }
}
const HPR = new C.HeadingPitchRoll(), POS = new C.Cartesian3();
function updateBots(secs, visible) {
  if (!route.len) return;
  bots.forEach(b => {
    const s = (secs * RUN_SPEED + b.offset) % route.len, p = routeAt(s);
    C.Cartesian3.fromDegrees(p.lon, p.lat, p.h, C.Ellipsoid.WGS84, POS);
    HPR.heading = C.Math.toRadians(p.heading - 90);
    if (b.model) { b.model.show = visible; C.Transforms.headingPitchRollToFixedFrame(POS, HPR, C.Ellipsoid.WGS84, undefined, b.model.modelMatrix); }
    if (b.fallback) { b.fallback.show = visible; b.fallback.position = C.Cartesian3.fromDegrees(p.lon, p.lat, p.h + 3); }
  });
}
addBots();

/* ---------------------------------------------------------------- the journey (placeholder until the user's drawings) */
function flyBuilding(lon, lat, range, heading, pitch, dur) {
  return new Promise(res => camera.flyToBoundingSphere(new C.BoundingSphere(C.Cartesian3.fromDegrees(lon, lat, 40), 120), {
    offset: new C.HeadingPitchRange(C.Math.toRadians(heading), C.Math.toRadians(pitch), range), duration: REDUCED ? 0 : dur, complete: res, cancel: res }));
}
function flyHeight(lon, lat, h, dur) {
  return new Promise(res => camera.flyTo({ destination: C.Cartesian3.fromDegrees(lon, lat, h), duration: REDUCED ? 0 : dur, complete: res, cancel: res }));
}
const STEPS = [
  { text: "Every weekday at 6:45 PM, little bots fetch the day's data from around the world.", go: () => flyHeight(55, 22, PHONE() ? 24e6 : 17e6, 3) },
  { text: "World prices come from the US and Europe through undersea cables that land in Mumbai.", go: () => flyHeight(40, 26, PHONE() ? 13e6 : 9e6, 3) },
  { text: "Most of the data starts in Mumbai: stock prices, company filings and foreign money flows.", go: () => flyHeight(72.86, 19.0, 90000, 3.5) },
  { text: "This is Bandra Kurla Complex. NSE's Exchange Plaza is here. The price bots grab today's bhavcopy and run.", go: () => flyBuilding(NSE.lon, NSE.lat, PHONE() ? 1300 : 900, 25, -28, 4) }
];
let step = -1, mode = "intro";
function showCaption(text) { $("captionText").textContent = text; $("caption").hidden = false; }
async function runStep(i) {
  step = i; const s = STEPS[i]; showCaption(s.text); $("nextBtn").disabled = true;
  await s.go(); $("nextBtn").disabled = false; $("nextBtn").textContent = i === STEPS.length - 1 ? "Explore on my own" : "Next";
}
$("startBtn").addEventListener("click", () => { $("intro").hidden = true; mode = "journey"; runStep(0); });
$("nextBtn").addEventListener("click", () => { if (step < STEPS.length - 1) runStep(step + 1); else startExplore(); });
$("skipBtn").addEventListener("click", startExplore);
$("exploreBtn").addEventListener("click", () => { $("intro").hidden = true; startExplore(); });

/* ---------------------------------------------------------------- explore mode: chips, cards, clicks */
function route_hash() {                                // deep links: #nse jumps to NSE, #explore skips the intro
  if (location.hash === "#nse") { $("intro").hidden = true; startExplore(); flyBuilding(NSE.lon, NSE.lat, PHONE() ? 1300 : 900, 25, -28, 0); }
  else if (location.hash === "#explore") { $("intro").hidden = true; startExplore(); }
}
window.addEventListener("hashchange", route_hash);
setTimeout(route_hash, 0);
const CHIPS = [["earth", "Whole Earth"], ["nse", "NSE, Mumbai"], ["bom", "Mumbai"], ["blr", "Bengaluru"], ["del", "New Delhi"],
               ["stl", "St. Louis"], ["fra", "Frankfurt"], ["mtv", "California"], ["dxb", "Dubai"], ["web", "On the web"]];
function startExplore() {
  mode = "explore"; $("caption").hidden = true; const chips = $("chips"); chips.hidden = false;
  if (!chips.children.length) CHIPS.forEach(([id, label]) => {
    const b = document.createElement("button"); b.type = "button"; b.className = "pill"; b.textContent = label;
    b.addEventListener("click", () => {
      chips.querySelectorAll(".pill").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
      if (id === "earth") { closeCard(); flyHeight(65, 18, PHONE() ? 26e6 : 19e6, 2.5); }
      else if (id === "nse") { flyBuilding(NSE.lon, NSE.lat, PHONE() ? 1300 : 900, 25, -28, 3); openCard(stationCard(ST.bom)); }
      else if (!ST[id].real) openCard(stationCard(ST[id]));
      else { const s = ST[id], p = s.real; flyBuilding(p[1], p[0], 40000, 0, -50, 3); openCard(stationCard(s)); }
    });
    chips.appendChild(b);
  });
}
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function hostLink(h) { return /^[a-z0-9.-]+\.[a-z]{2,}$/i.test(h) ? `<a href="https://${esc(h)}" target="_blank" rel="noopener">${esc(h)}</a>` : esc(h); }
const FLAG_NAME = { IN: "India", US: "United States", DE: "Germany", AE: "United Arab Emirates" };
function botCard(b) {
  const st = ST[b.st], flag = D.FLAG[b.st];
  return `<button class="close" type="button" aria-label="Close">×</button><h3>${esc(b.name)}</h3>
    <p class="role">${esc(b.role)}${st ? " · " + esc(st.name) : ""}${flag ? " · wears " + FLAG_NAME[flag] + "'s flag" : ""}</p>
    <span class="tag ${b.use}">${esc(D.USE_TEXT[b.use])}</span><blockquote>“${esc(b.says)}”</blockquote>
    <dl><div><dt>What I carry</dt><dd><ul>${b.carries.map(c => `<li>${esc(c)}</li>`).join("")}</ul></dd></div>
    <div><dt>Picked up from</dt><dd>${b.from.map(hostLink).join(", ")}</dd></div><div><dt>When</dt><dd>${esc(b.when)}</dd></div>
    <div><dt>Drops it at</dt><dd class="mono">${esc(b.writes)}</dd></div><div><dt>Used for</dt><dd>${esc(b.used)}</dd></div></dl>`;
}
function stationCard(s) {
  const list = byStation[s.id] || [];
  const pin = s.real ? "The pin marks the city of the organisation that publishes the data. Its files are served from cloud servers." : "No pin: we haven't confirmed where this company works from.";
  return `<button class="close" type="button" aria-label="Close">×</button><h3>${esc(s.name)}</h3><p class="role">${esc(s.sub)}</p>
    <p style="margin:0 0 12px">${esc(s.about)}</p>
    ${list.length ? `<dl><div><dt>Bots here</dt><dd class="botlist">${list.map(b => `<button type="button" data-bot="${b.id}">${esc(b.name)}</button>`).join("")}</dd></div></dl>` : ""}
    <p class="small">${esc(pin)}</p>`;
}
function openCard(html) {
  const card = $("card"); card.innerHTML = html; card.hidden = false; document.body.classList.add("card-open");
  card.querySelector(".close").addEventListener("click", closeCard);
  card.querySelectorAll("[data-bot]").forEach(btn => btn.addEventListener("click", () => openCard(botCard(D.BOTS.find(b => b.id === btn.dataset.bot)))));
}
function closeCard() { $("card").hidden = true; document.body.classList.remove("card-open"); }
window.addEventListener("keydown", e => { if (e.key === "Escape") closeCard(); });
const handler = new C.ScreenSpaceEventHandler(scene.canvas);
handler.setInputAction(click => {
  const picked = scene.pick(click.position); if (!C.defined(picked)) return;
  const id = picked.id;
  if (id && id.bot) return openCard(botCard(D.BOTS.find(b => b.id === id.bot)));
  if (id instanceof C.Entity) {
    if (id.id.startsWith("st-")) return openCard(stationCard(ST[id.id.slice(3)]));
    if (id.id === "bld-nse") return openCard(stationCard(ST.bom));
    if (id.properties && id.properties.bot) return openCard(botCard(D.BOTS.find(b => b.id === "bhav")));
  }
}, C.ScreenSpaceEventType.LEFT_CLICK);

/* ---------------------------------------------------------------- per-frame: street layer switch, sun line, bots */
function showNote(t) { const n = $("note"); n.textContent = t; n.hidden = false; }
if (!CFG.googleKey) showNote("Preview: real 3D buildings appear once the Google map key is added. Street level uses OpenStreetMap meanwhile.");
function nearestStation(carto) {
  let best = null, bd = Infinity;
  D.STATIONS.forEach(s => { if (!s.real) return; const d = metres([C.Math.toDegrees(carto.longitude), C.Math.toDegrees(carto.latitude)], [s.real[1], s.real[0]]); if (d < bd) { bd = d; best = s; } });
  return bd < 1500000 ? best : null;
}
let lastLine = 0;
scene.preRender.addEventListener((sc, time) => {
  const carto = camera.positionCartographic, h = carto.height, low = h < LOW;
  if (low) ensureTileset();
  const city = !!(tileset && low);
  if (tileset) tileset.show = city;
  globe.show = !city; streetLayer.show = low && !tileset;
  nightLayer.show = !low;                              // the city-lights layer would cover the map once lighting is off
  globe.enableLighting = !low; globe.showGroundAtmosphere = !low; globe.dynamicAtmosphereLighting = !low;                         // real terminator from space; near the ground our own grade keeps streets visible at night
  const ground = new C.Cartographic(carto.longitude, carto.latitude, 0), elev = sunElevation(ground, time);
  updateTint(elev, low);
  updateBots(C.JulianDate.secondsDifference(time, T0), h < 40000);
  if (tileset && !route.heights && h < 20000 && Math.random() < 0.02) sampleRouteHeights();
  const now = performance.now();
  if (now - lastLine > 1000) {
    lastLine = now; const st = nearestStation(ground);
    const part = elev >= 8 ? "day" : elev >= -4 ? (elev >= 0 ? "evening light" : "dusk") : "night";
    const clock = st ? new Intl.DateTimeFormat("en-IN", { hour: "numeric", minute: "2-digit", timeZone: TZ[st.id] }).format(new Date()) : "";
    $("sunline").textContent = st ? `${st.name} · ${clock} · ${part}` : (h > 3e6 ? "The real sky right now" : `Over the sea · ${part}`);
  }
});
window.SGM_WORLD = { viewer };                          // handy for debugging in the console
if (location.search.includes("debug")) scene.renderError.addEventListener((sc, e) => { document.documentElement.dataset.err = String(e && (e.stack || e.message) || e).slice(0, 600); });
if (location.search.includes("debug")) setInterval(() => {   // ?debug: state for headless checks
  const c = camera.positionCartographic;
  document.documentElement.dataset.debug = JSON.stringify({ h: Math.round(c.height), lon: +C.Math.toDegrees(c.longitude).toFixed(4), lat: +C.Math.toDegrees(c.latitude).toFixed(4),
    pitch: +C.Math.toDegrees(camera.pitch).toFixed(1), globe: globe.show, street: streetLayer.show, tiles: globe.tilesLoaded, tint: +tint.amount.toFixed(2), bots: bots.length });
}, 1000);
})();
