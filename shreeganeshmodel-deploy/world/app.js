/* SGM Data World (realistic). CesiumJS Earth in real time (day, evening, night), Google Photorealistic 3D Tiles near
   the ground (key in world/config.js, not in git), cartoon bots and ships moving data along glowing routes with trails,
   plain-language bot cards. The journey (user's drawings) lives in world/journey.js and world/lineup.js. */
(function () {
"use strict";
const C = window.Cesium, D = window.SGM_DATA, CFG = window.SGM_CONFIG || {};
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const PHONE = () => window.innerWidth <= 640;
const $ = id => document.getElementById(id);
const LOW = 250000;                                   // below this height (m) the street-level city shows
const TZ = { bom: "Asia/Kolkata", del: "Asia/Kolkata", web: "Asia/Kolkata", blr: "Asia/Kolkata", stl: "America/Chicago",
             fra: "Europe/Berlin", mtv: "America/Los_Angeles", sfo: "America/Los_Angeles", dxb: "Asia/Dubai" };
const COUNTRY = { IN: "India", US: "USA", DE: "Germany", AE: "UAE" };

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
streetLayer.show = false;                              // stand-in only when Google's 3D city is unavailable
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
    .then(ts => { tileset = ts; scene.primitives.add(ts); })
    .catch(err => showNote("Google's 3D city didn't load (" + (err && err.message || err) + "). Showing the satellite globe instead."));
}
// ground height (ellipsoid metres) under key places, sampled from the 3D city once it is loaded nearby
const GROUND = { bkc: -43 };                           // BKC measured on Google's tiles 2026-10-17
const groundPending = {};
function groundAt(key, lon, lat) {
  if (!(tileset && tileset.show)) return 0;
  if (GROUND[key] === undefined && !groundPending[key]) {
    groundPending[key] = true;
    scene.sampleHeightMostDetailed([C.Cartographic.fromDegrees(lon, lat)]).then(r => {
      if (r[0] && isFinite(r[0].height)) GROUND[key] = r[0].height; else groundPending[key] = false;
    }).catch(() => { groundPending[key] = false; });
  }
  return GROUND[key] !== undefined ? GROUND[key] : 0;
}

/* ---------------------------------------------------------------- real-time sun: evening and night grade near the ground */
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

/* ---------------------------------------------------------------- places, labels */
const ST = Object.fromEntries(D.STATIONS.map(s => [s.id, s]));
const BOT = Object.fromEntries(D.BOTS.map(b => [b.id, b]));
const byStation = {}; D.BOTS.forEach(b => (byStation[b.st] = byStation[b.st] || []).push(b));
const ORANGE = C.Color.fromCssColorString("#e8875c");
D.STATIONS.filter(s => s.real).forEach(s => {          // "On the web" has no confirmed city, so it gets no pin
  viewer.entities.add({
    id: "st-" + s.id, position: C.Cartesian3.fromDegrees(s.real[1], s.real[0], 0),
    point: { pixelSize: 9, color: ORANGE, outlineColor: C.Color.WHITE, outlineWidth: 2, disableDepthTestDistance: 0,
             distanceDisplayCondition: new C.DistanceDisplayCondition(30000, 4e7) },
    label: { text: s.name, font: "600 14px Fredoka, sans-serif", fillColor: C.Color.WHITE, outlineColor: C.Color.BLACK, outlineWidth: 3,
             style: C.LabelStyle.FILL_AND_OUTLINE, pixelOffset: new C.Cartesian2(0, -16), disableDepthTestDistance: 0, heightReference: C.HeightReference.NONE,
             distanceDisplayCondition: new C.DistanceDisplayCondition(30000, 2.6e7) }
  });
});
const B = D.BUILDINGS, NSE = B.nse;
const labelStyle = (text, maxDist) => ({ text, font: "600 15px Fredoka, sans-serif", fillColor: C.Color.WHITE, showBackground: true,
  backgroundColor: C.Color.fromCssColorString("rgba(24,21,28,0.82)"), backgroundPadding: new C.Cartesian2(10, 6),
  disableDepthTestDistance: Number.POSITIVE_INFINITY, distanceDisplayCondition: new C.DistanceDisplayCondition(0, maxDist) });
[["bld-nse", B.nse, "NSE · Exchange Plaza", "bkc", 95, 40000], ["bld-pib", B.pib, "PIB · next to Parliament", "pib", 70, 40000],
 ["bld-sbi", B.sbi, "SBI head office", "sbi", 130, 40000]].forEach(([id, b, text, key, up, max]) =>
  viewer.entities.add({ id, position: new C.CallbackProperty(() => C.Cartesian3.fromDegrees(b.lon, b.lat, groundAt(key, b.lon, b.lat) + up), false), label: labelStyle(text, max) }));
viewer.entities.add({ id: "bld-home", position: new C.CallbackProperty(() => C.Cartesian3.fromDegrees(D.HOME.lon, D.HOME.lat, groundAt("blr", D.HOME.lon, D.HOME.lat) + 260), false),
  label: labelStyle("Your laptop · Bengaluru", 60000) });

/* ---------------------------------------------------------------- paths and movers (bots, ships, notes) with glowing trails */
const T0 = C.JulianDate.now();
const AMBER = C.Color.fromCssColorString("#ffc21a"), CYAN = C.Color.fromCssColorString("#4fd8ff"), MONEY = C.Color.fromCssColorString("#7dff6a"), WAKE = C.Color.WHITE;
function geo(a, b, n, h0, h1, peak) {                   // great-circle points from a to b ([lon, lat]) with an arc height
  const g = new C.EllipsoidGeodesic(C.Cartographic.fromDegrees(a[0], a[1]), C.Cartographic.fromDegrees(b[0], b[1])), out = [];
  for (let i = 0; i <= n; i++) { const t = i / n, c = g.interpolateUsingFraction(t);
    out.push(C.Cartesian3.fromRadians(c.longitude, c.latitude, h0 + (h1 - h0) * t + (peak || 0) * Math.sin(Math.PI * t))); }
  return out;
}
function seaPath(wps) {                                  // [lat, lon] sea waypoints, simplified cable routes
  let pts = [];
  for (let i = 0; i < wps.length - 1; i++) pts = pts.concat(geo([wps[i][1], wps[i][0]], [wps[i + 1][1], wps[i + 1][0]], 14, 0, 0, 0).slice(i ? 1 : 0));
  return pts;
}
function drawRoute(pts, color, width) {
  viewer.entities.add({ polyline: { positions: pts, width: width || 5, arcType: C.ArcType.NONE,
    material: new C.PolylineGlowMaterialProperty({ glowPower: 0.22, taperPower: 1, color: color.withAlpha(0.95) }),
    distanceDisplayCondition: new C.DistanceDisplayCondition(0, 5e7) } });
}
function pathOf(pts) {
  const cum = [0]; for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + C.Cartesian3.distance(pts[i - 1], pts[i]));
  return { pts, cum, len: cum[cum.length - 1] };
}
const movers = [], allModels = [];
// after sunset the toys are drawn in their own colours (unlit), so they stay visible like lit-up toys at night
const UNLIT = new C.CustomShader({ lightingModel: C.LightingModel.UNLIT });
let nightToys = false;
async function loadModel(urls, opts) {
  for (const url of urls) {
    try { const m = await C.Model.fromGltfAsync(Object.assign({ url }, opts)); scene.primitives.add(m); allModels.push(m); if (nightToys) m.customShader = UNLIT; return m; } catch (e) { /* next */ }
  }
  return null;
}
async function addMover(o) {
  const model = await loadModel(o.urls, { scale: o.scale || 1, minimumPixelSize: o.minPx || 0, id: o.pick });
  if (!model) return null;
  model.readyEvent.addEventListener(() => { try { model.activeAnimations.add({ name: o.anim || "run", loop: C.ModelAnimationLoop.REPEAT, multiplier: o.animSpeed || 1 }); } catch (e) {} });
  const m = Object.assign({ model, path: pathOf(o.pts), trail: [], lastTrail: 0, lastU: 0, pos: o.pts[0] }, o);
  if (o.trailColor) m.trailEntity = viewer.entities.add({ polyline: { positions: new C.CallbackProperty(() => m.trail, false), width: o.trailWidth || 6,
    arcType: C.ArcType.NONE, material: new C.PolylineGlowMaterialProperty({ glowPower: 0.35, taperPower: 0.4, color: o.trailColor }) } });
  if (o.label) m.labelEntity = viewer.entities.add({ position: new C.CallbackProperty(() => m.pos, false),
    label: Object.assign(labelStyle(o.label, o.labelMax || 2e7), { pixelOffset: new C.Cartesian2(0, -46), font: "600 13px Fredoka, sans-serif", disableDepthTestDistance: 0 }) });
  movers.push(m); return m;
}
const HPR = new C.HeadingPitchRoll();
function headingAt(a, b) {
  const ca = C.Cartographic.fromCartesian(a), cb = C.Cartographic.fromCartesian(b);
  const y = Math.sin(cb.longitude - ca.longitude) * Math.cos(cb.latitude);
  const x = Math.cos(ca.latitude) * Math.sin(cb.latitude) - Math.sin(ca.latitude) * Math.cos(cb.latitude) * Math.cos(cb.longitude - ca.longitude);
  return Math.atan2(y, x);                              // radians clockwise from north
}
function updateMovers(secs, camH) {
  movers.forEach(m => {
    const cycle = m.secs + (m.pause || 1.5), local = ((secs - (m.delay || 0)) % cycle + cycle) % cycle, u = Math.min(1, local / m.secs);
    if (u < m.lastU) m.trail.length = 0;                 // a new lap starts a fresh trail
    m.lastU = u;
    const L = u * m.path.len, cum = m.path.cum; let i = 1; while (i < cum.length - 1 && cum[i] < L) i++;
    const a = m.path.pts[i - 1], b = m.path.pts[i], t = (L - cum[i - 1]) / Math.max(1e-6, cum[i] - cum[i - 1]);
    const p = C.Cartesian3.lerp(a, b, t, new C.Cartesian3());
    if (m.ground) {                                      // street movers sit on the 3D city
      const c = C.Cartographic.fromCartesian(p); c.height = groundAt(m.ground, C.Math.toDegrees(c.longitude), C.Math.toDegrees(c.latitude)) + 0.5;
      C.Cartographic.toCartesian(c, C.Ellipsoid.WGS84, p);
    }
    m.pos = p;
    HPR.heading = headingAt(a, b) - Math.PI / 2;       // models face +Z, which Cesium turns to +X (east at heading 0)
    C.Transforms.headingPitchRollToFixedFrame(p, HPR, C.Ellipsoid.WGS84, undefined, m.model.modelMatrix);
    const vis = camH >= (m.minH || 0) && camH <= (m.maxH || Infinity);
    m.model.show = vis; if (m.trailEntity) m.trailEntity.show = vis; if (m.labelEntity) m.labelEntity.show = vis;
    if (m.trailColor && secs - m.lastTrail > 0.12) {
      m.lastTrail = secs; m.trail.push(C.Cartesian3.clone(p)); if (m.trail.length > (m.trailN || 60)) m.trail.shift();
    }
  });
}
const statics = [];
async function addStatic(o) {                          // props standing at one place (laptop, you, ML bots)
  const model = await loadModel(o.urls, { scale: o.scale || 1, minimumPixelSize: o.minPx || 0, id: o.pick });
  if (!model) return null;
  model.readyEvent.addEventListener(() => { try { model.activeAnimations.add({ name: o.anim || "idle", loop: C.ModelAnimationLoop.REPEAT, multiplier: o.animSpeed || 1 }); } catch (e) {} });
  const s = Object.assign({ model }, o); statics.push(s); return s;
}
function updateStatics(camH) {
  statics.forEach(s => {
    const h = groundAt(s.ground, s.lon, s.lat) + (s.up || 0);
    HPR.heading = C.Math.toRadians(s.heading || 0) - Math.PI / 2;
    C.Transforms.headingPitchRollToFixedFrame(C.Cartesian3.fromDegrees(s.lon, s.lat, h), HPR, C.Ellipsoid.WGS84, undefined, s.model.modelMatrix);
    s.model.show = camH <= (s.maxH || Infinity);
  });
}

/* the routes (cable routes simplified; see world/README.md) */
const P = { stl: [-90.2, 38.63], nj: [-73.9, 40.3], fra: [8.68, 50.11], mrs: [5.37, 43.3], sfo: [-122.40, 37.79], mtv: [-122.08, 37.39],
            sfPort: [-122.62, 37.80], mumLand: [72.80, 19.10], chennai: [80.30, 13.08], blr: [D.HOME.lon, D.HOME.lat], dxb: [55.27, 25.2],
            nse: [NSE.lon, NSE.lat], pib: [B.pib.lon, B.pib.lat], sbi: [B.sbi.lon, B.sbi.lat] };
const SEA = {
  atlantic: [[40.2, -73.6], [39.5, -60], [38, -40], [37, -20], [36, -9], [35.95, -5.6], [37.5, 2], [37.3, 11], [35, 20], [33.5, 28], [31.4, 32.4], [29.8, 32.6], [27.5, 34], [22, 37.8], [16, 41.5], [12.6, 43.3], [12.2, 48], [13.5, 53], [16, 60], [18.5, 67], [19.1, 72.75]],
  med: [[43.1, 5.4], [40.5, 6.5], [37.6, 10.5], [35, 20], [33.5, 28], [31.4, 32.4], [29.8, 32.6], [27.5, 34], [22, 37.8], [16, 41.5], [12.6, 43.3], [12.2, 48], [13.5, 53], [16, 60], [18.5, 67], [19.0, 72.7]],
  pacific: [[37.8, -122.6], [36, -135], [30, -150], [23, -162], [18, -175], [15, 170], [14, 155], [15, 140], [18, 128], [20.5, 121.5], [17, 116], [12, 112], [7, 108], [3, 105.5], [1.3, 104], [3, 100.6], [6, 96], [9, 88], [13.08, 80.35]]
};
const ROUTES = {
  fedRun: geo(P.stl, P.nj, 60, 0, 0, 0), euroRun: geo(P.fra, P.mrs, 40, 0, 0, 0),
  claudeRun: geo(P.sfo, P.sfPort, 10, 0, 0, 0), gnewsRun: geo(P.mtv, P.sfPort, 12, 0, 0, 0),
  atlantic: seaPath(SEA.atlantic), med: seaPath(SEA.med), pacific: seaPath(SEA.pacific),
  mumToBlr: geo(P.mumLand, P.blr, 60, 0, 0, 90000), nseToBlr: geo(P.nse, P.blr, 60, 0, 0, 70000), pibToBlr: geo(P.pib, P.blr, 80, 0, 0, 160000),
  chennaiToBlr: geo(P.chennai, P.blr, 50, 0, 0, 60000), tgOut: geo(P.blr, P.dxb, 60, 0, 0, 260000), tgBack: geo(P.dxb, P.blr, 60, 0, 0, 140000),
  moneyIn: geo(P.sbi, P.blr, 60, 0, 0, 120000), moneyOut: geo(P.blr, P.sbi, 60, 0, 0, 55000)
};
[["fedRun", AMBER], ["euroRun", AMBER], ["claudeRun", AMBER], ["gnewsRun", AMBER], ["atlantic", AMBER], ["med", AMBER], ["pacific", AMBER],
 ["mumToBlr", AMBER], ["nseToBlr", AMBER], ["pibToBlr", AMBER], ["chennaiToBlr", AMBER], ["tgOut", CYAN], ["tgBack", CYAN], ["moneyIn", MONEY], ["moneyOut", MONEY]]
  .forEach(([k, c]) => drawRoute(ROUTES[k], c, k === "atlantic" || k === "med" || k === "pacific" ? 6 : 4));

const BOTS_DIR = "assets/bots/", PROPS = "assets/props/";
const carry = f => [BOTS_DIR + `bot_${f}_carry.glb`];
async function buildMovers() {
  const M = [
    { urls: carry("US"), pts: ROUTES.fedRun, secs: 14, minPx: 54, scale: 2000, pick: { bot: "fed" }, trailColor: AMBER, label: "Fed · world prices", labelMax: 1.2e7, maxH: 3e7 },
    { urls: carry("DE"), pts: ROUTES.euroRun, secs: 9, minPx: 54, scale: 2000, pick: { bot: "euro" }, trailColor: AMBER, label: "Euro · exchange rates", labelMax: 1.2e7, maxH: 3e7 },
    { urls: carry("US"), pts: ROUTES.claudeRun, secs: 6, minPx: 54, scale: 800, pick: { bot: "claude" }, trailColor: AMBER, label: "Claude · Friday note", labelMax: 1.2e7, maxH: 3e7 },
    { urls: carry("US"), pts: ROUTES.gnewsRun, secs: 7, minPx: 48, scale: 800, pick: { bot: "gnews" }, trailColor: AMBER, maxH: 3e7 },
    { urls: [PROPS + "tanker_oil.glb", PROPS + "ship_cargo.glb", ...carry("US")], pts: ROUTES.atlantic, secs: 70, minPx: 90, scale: 25000, anim: "idle", pick: { ship: "tanker" },
      trailColor: WAKE, trailWidth: 8, trailN: 90, label: "Ship 1 · oil and world prices", labelMax: 3e7 },
    { urls: [PROPS + "submarine.glb", ...carry("DE")], pts: ROUTES.med, secs: 55, delay: 8, minPx: 70, scale: 18000, anim: "idle", pick: { ship: "sub" },
      trailColor: CYAN, trailN: 80, label: "Submarine · the undersea cable", labelMax: 3e7 },
    { urls: [PROPS + "container_ship.glb", PROPS + "ship_cargo.glb", ...carry("US")], pts: ROUTES.pacific, secs: 90, minPx: 90, scale: 30000, anim: "idle", pick: { ship: "container" },
      trailColor: WAKE, trailWidth: 8, trailN: 100, label: "Ship 2 · Claude's note and world headlines", labelMax: 3e7 },
    { urls: carry("IN"), pts: ROUTES.nseToBlr, secs: 6, minPx: 46, scale: 900, pick: { bot: "bhav" }, trailColor: AMBER, maxH: 1.5e7 },
    { urls: carry("IN"), pts: ROUTES.pibToBlr, secs: 8, delay: 2, minPx: 46, scale: 900, pick: { bot: "sarkari" }, trailColor: AMBER, label: "Sarkari · government news", labelMax: 9e6, maxH: 1.5e7 },
    { urls: carry("IN"), pts: ROUTES.mumToBlr, secs: 6, delay: 3, minPx: 42, scale: 900, pick: { bot: "fed" }, trailColor: AMBER, maxH: 1.5e7 },
    { urls: carry("IN"), pts: ROUTES.chennaiToBlr, secs: 5, delay: 1, minPx: 42, scale: 900, pick: { bot: "claude" }, trailColor: AMBER, maxH: 1.5e7 },
    { urls: carry("AE"), pts: ROUTES.tgOut.concat(ROUTES.tgBack.slice(1)), secs: 12, minPx: 44, scale: 900, pick: { bot: "pigeon" }, trailColor: CYAN, label: "Pigeon · your messages", labelMax: 9e6, maxH: 1.5e7 },
    { urls: [PROPS + "rupee_note.glb", PROPS + "data_crate.glb"], pts: ROUTES.moneyIn, secs: 7, minPx: 40, scale: 2500, anim: "idle", pick: { money: "in" }, trailColor: MONEY, label: "₹ salary from SBI", labelMax: 6e6, maxH: 1.2e7 },
    { urls: [PROPS + "rupee_note.glb", PROPS + "data_crate.glb"], pts: ROUTES.moneyOut, secs: 7, delay: 3.5, minPx: 40, scale: 2500, anim: "idle", pick: { money: "out" }, trailColor: MONEY, maxH: 1.2e7 }
  ];
  for (const o of M) await addMover(o);
  // street level: price bots run the real road out of NSE (OSRM route on OpenStreetMap)
  try {
    const j = await (await fetch("world/routes/nse_bkc_exit.json")).json();
    const pts = j.coordinates.map(([lon, lat]) => C.Cartesian3.fromDegrees(lon, lat, 0));
    for (let k = 0; k < 3; k++) await addMover({ urls: carry("IN"), pts, secs: 48, delay: k * 16, pause: 0.3, minPx: 26, scale: 7, ground: "bkc", pick: { bot: "bhav" }, trailColor: AMBER, trailN: 40, maxH: 60000 });
  } catch (e) { /* route missing */ }
  // Bengaluru: you, the laptop and the ML bots (shown at a public landmark, never a home)
  const h = D.HOME;
  await addStatic({ urls: [PROPS + "laptop_giant.glb"], lon: h.lon, lat: h.lat, ground: "blr", scale: 110, minPx: 110, heading: 200, pick: { prop: "laptop" }, maxH: 1.5e7 });
  await addStatic({ urls: [PROPS + "me_figure.glb"], lon: h.lon - 0.0016, lat: h.lat + 0.0012, ground: "blr", scale: 230, minPx: 120, heading: 200, anim: "wave", pick: { prop: "me" }, maxH: 1.5e7 });
  for (let k = 0; k < 3; k++) await addStatic({ urls: [PROPS + "bot_ML.glb"], lon: h.lon + 0.0009 * Math.cos(k * 2.1), lat: h.lat - 0.0008 + 0.0007 * Math.sin(k * 2.1),
    ground: "blr", scale: 34, minPx: 34, heading: 160 + k * 40, anim: "hammer", pick: { prop: "ml" }, maxH: 6e6 });
}
buildMovers();

/* ---------------------------------------------------------------- cards: plain words first, details folded */
let companies = null;
fetch("world/companies.json").then(r => r.json()).then(j => { companies = j; }).catch(() => {});
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const hostLink = h => /^[a-z0-9.-]+\.[a-z]{2,}$/i.test(h) ? `<a href="https://${esc(h)}" target="_blank" rel="noopener">${esc(h)}</a>` : esc(h);
function botCard(id) {
  const b = BOT[id], L = D.LAYMAN[id] || {}, flag = D.FLAG[b.st];
  const cta = L.cta ? (L.cta[0] === "list" && !(companies && companies.groups[L.cta[1]]) ? null : L.cta) : null;
  return `<button class="close" type="button" aria-label="Close">×</button>
    <div class="who"><h3>${esc(b.name)}</h3>${flag ? `<span class="flagchip">${COUNTRY[flag]}</span>` : ""}</div>
    <p class="hello">“${esc(b.says)}”</p>
    <dl class="plain">
      <div><dt>What I bring</dt><dd>${esc(L.brings || b.carries.join(", "))}</dd></div>
      <div><dt>Where I get it</dt><dd>${esc(L.from || "")}${b.from.length ? ` · ${b.from.map(hostLink).join(", ")}` : ""}</dd></div>
      <div><dt>What it means</dt><dd>${esc(L.means || b.used)}</dd></div>
    </dl>
    <div class="ctas">
      ${L.next ? `<button type="button" class="cta" data-bot="${L.next[0]}">${esc(L.next[1])} →</button>` : ""}
      ${cta ? (cta[0] === "list" ? `<button type="button" class="cta" data-list="${cta[1]}">${esc(cta[2])} →</button>`
                                 : `<button type="button" class="cta" data-bot="${cta[1]}">${esc(cta[2])} →</button>`) : ""}
    </div>
    <details><summary>Details</summary><dl>
      <div><dt>Role</dt><dd>${esc(b.role)} · <span class="tag ${b.use}">${esc(D.USE_TEXT[b.use])}</span></dd></div>
      <div><dt>When</dt><dd>${esc(b.when)}</dd></div><div><dt>File it writes</dt><dd class="mono">${esc(b.writes)}</dd></div>
      <div><dt>Used for</dt><dd>${esc(b.used)}</dd></div></dl></details>`;
}
function listCard(key, backBot) {
  const g = companies.groups[key];
  return `<button class="close" type="button" aria-label="Close">×</button><h3>${esc(g.title)}</h3>
    <p class="role">Biggest first · market caps as of ${esc(companies.market_cap_as_of)}${key === "govt" ? ` · government news as of ${esc(companies.policy_as_of)}` : ""}</p>
    <ol class="cos">${g.companies.map(c => `<li><b>${esc(c.name)}</b><span>${esc(c.symbol)} · ${esc(c.industry)}</span></li>`).join("")}</ol>
    <p class="small">Industries: ${g.industries.map(esc).join(", ")}. A list to explore, not a recommendation.</p>
    ${backBot ? `<div class="ctas"><button type="button" class="cta" data-bot="${backBot}">← Back to ${esc(BOT[backBot].name)}</button></div>` : ""}`;
}
const SHIPS = {
  tanker: ["Ship 1 · the oil tanker", "Carries world prices (oil, Brent crude, metals, the dollar) from the US East Coast to Mumbai.", "In real life this data crosses the Atlantic and the Red Sea through undersea fibre cables in a fraction of a second. The tanker is the cartoon of that, and a reminder that oil itself comes to India by ship."],
  sub: ["The submarine", "Marks the undersea cable from Marseille to Mumbai that carries Europe's data.", "Most of India's internet arrives through cables on the sea floor that land in Mumbai and Chennai."],
  container: ["Ship 2 · the container ship", "Brings Claude's Friday note and world headlines from California across the Pacific to Chennai.", "Real route: undersea cables across the Pacific and through Singapore. Simplified here."]
};
const PROPS_INFO = {
  laptop: ["Your laptop", "Where every crate ends up. Every weekday at 6:45 PM it checks the data, and on Fridays at 7:30 PM it builds the buy list."],
  me: ["You, the boss", "You read the Friday list, sign the Saturday review and place the Monday order."],
  ml: ["ML bots", "They turn the data into model scores and run the V3.2 rule: hot industry, ₹1 cr a day, the trend test, top 9, then the financial, fading-theme and trade-for-trade drops."]
};
function simpleCard(title, body, more) {
  return `<button class="close" type="button" aria-label="Close">×</button><h3>${esc(title)}</h3><p class="hello">${esc(body)}</p>${more ? `<p class="small">${esc(more)}</p>` : ""}`;
}
function stationCard(s) {
  const list = byStation[s.id] || [];
  const pin = s.real ? "The pin marks the city of the organisation that publishes the data. Its files are served from cloud servers." : "No pin: we haven't confirmed where this company works from.";
  return `<button class="close" type="button" aria-label="Close">×</button><h3>${esc(s.name)}</h3><p class="role">${esc(s.sub)}</p>
    <p style="margin:0 0 12px">${esc(s.about)}</p>
    ${list.length ? `<dl><div><dt>Bots here</dt><dd class="botlist">${list.map(b => `<button type="button" data-bot="${b.id}">${esc(b.name)}</button>`).join("")}</dd></div></dl>` : ""}
    <p class="small">${esc(pin)}</p>`;
}
let currentBot = null;
function openCard(html) {
  const card = $("card"); card.innerHTML = html; card.hidden = false; card.scrollTop = 0; document.body.classList.add("card-open");
  card.querySelector(".close").addEventListener("click", closeCard);
  card.querySelectorAll("[data-bot]").forEach(btn => btn.addEventListener("click", () => showBot(btn.dataset.bot, true)));
  card.querySelectorAll("[data-list]").forEach(btn => btn.addEventListener("click", () => openCard(listCard(btn.dataset.list, currentBot))));
}
function closeCard() { $("card").hidden = true; document.body.classList.remove("card-open"); }
function showBot(id, fly) {
  currentBot = id; openCard(botCard(id));
  const s = ST[BOT[id].st];
  if (fly && s && s.real) {
    if (id === "bhav") flyBuilding(NSE.lon, NSE.lat, 1100, 15, -26, 2.5, "bkc");
    else flyBuilding(s.real[1], s.real[0], 60000, 15, -50, 2.5);
  }
}
window.addEventListener("keydown", e => { if (e.key === "Escape") closeCard(); });
const handler = new C.ScreenSpaceEventHandler(scene.canvas);
handler.setInputAction(click => {
  const picked = scene.pick(click.position); if (!C.defined(picked)) return;
  const id = picked.id;
  if (id && id.bot) return showBot(id.bot, false);
  if (id && id.ship) { const s = SHIPS[id.ship]; return openCard(simpleCard(s[0], s[1], s[2])); }
  if (id && id.prop) { const s = PROPS_INFO[id.prop]; return openCard(simpleCard(s[0], s[1])); }
  if (id && id.money) return openCard(simpleCard("Money lane", id.money === "in" ? "Salary moves from your SBI account into the weekly stock batches." : "When a batch is sold after six months, the money comes back and rolls into that week's new batch.", "No amounts are shown on this page."));
  if (id instanceof C.Entity) {
    if (id.id.startsWith("st-")) return openCard(stationCard(ST[id.id.slice(3)]));
    if (id.id === "bld-nse") return showBot("bhav", false);
    if (id.id === "bld-pib") return showBot("sarkari", false);
    if (id.id === "bld-sbi") return openCard(simpleCard("SBI head office", "State Bank of India's head office at Nariman Point, Mumbai. Your salary account sits with SBI."));
    if (id.id === "bld-home") return openCard(simpleCard(PROPS_INFO.laptop[0], PROPS_INFO.laptop[1]));
  }
}, C.ScreenSpaceEventType.LEFT_CLICK);

/* ---------------------------------------------------------------- camera helpers */
function flyBuilding(lon, lat, range, heading, pitch, dur, groundKey) {
  const gh = groundKey && GROUND[groundKey] !== undefined ? GROUND[groundKey] : 0;
  return new Promise(res => camera.flyToBoundingSphere(new C.BoundingSphere(C.Cartesian3.fromDegrees(lon, lat, gh + 40), 120), {
    offset: new C.HeadingPitchRange(C.Math.toRadians(heading), C.Math.toRadians(pitch), range), duration: REDUCED ? 0 : dur, complete: res, cancel: res }));
}
function flyTo(lon, lat, h, heading, pitch, dur) {
  return new Promise(res => camera.flyTo({ destination: C.Cartesian3.fromDegrees(lon, lat, h),
    orientation: { heading: C.Math.toRadians(heading || 0), pitch: C.Math.toRadians(pitch === undefined ? -90 : pitch), roll: 0 },
    duration: REDUCED ? 0 : dur, complete: res, cancel: res }));
}

/* ---------------------------------------------------------------- explore mode */
const CHIPS = [["earth", "Whole Earth"], ["nse", "NSE, Mumbai"], ["blr", "Bengaluru"], ["del", "New Delhi"], ["bom", "Mumbai"],
               ["stl", "St. Louis"], ["fra", "Frankfurt"], ["sfo", "San Francisco"], ["mtv", "California"], ["dxb", "Dubai"], ["web", "On the web"]];
function startExplore() {
  $("caption").hidden = true; $("intro").hidden = true; const chips = $("chips"); chips.hidden = false;
  if (!chips.children.length) CHIPS.forEach(([id, label]) => {
    const b = document.createElement("button"); b.type = "button"; b.className = "pill"; b.textContent = label;
    b.addEventListener("click", () => {
      chips.querySelectorAll(".pill").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
      if (id === "earth") { closeCard(); flyTo(65, 18, PHONE() ? 26e6 : 19e6, 0, -90, 2.5); }
      else if (id === "nse") { flyBuilding(NSE.lon, NSE.lat, PHONE() ? 1400 : 1100, 15, -26, 3, "bkc"); showBot("bhav", false); }
      else if (id === "blr") { flyBuilding(D.HOME.lon, D.HOME.lat, PHONE() ? 3200 : 2400, 200, -32, 3, "blr"); openCard(simpleCard("Bengaluru", PROPS_INFO.laptop[1], "Shown at Vidhana Soudha, a public landmark, not a home.")); }
      else if (!ST[id].real) openCard(stationCard(ST[id]));
      else { const s = ST[id], p = s.real; flyBuilding(p[1], p[0], 40000, 0, -50, 3); openCard(stationCard(s)); }
    });
    chips.appendChild(b);
  });
}

/* ---------------------------------------------------------------- per frame */
function showNote(t) { const n = $("note"); n.textContent = t; n.hidden = false; }
if (!CFG.googleKey) showNote("Preview: real 3D buildings appear once the Google map key is added. Street level uses OpenStreetMap meanwhile.");
function metres(a, b) { const R = 6371000, la = (a[1] + b[1]) / 2 * C.Math.RADIANS_PER_DEGREE;
  return Math.hypot((b[0] - a[0]) * C.Math.RADIANS_PER_DEGREE * Math.cos(la) * R, (b[1] - a[1]) * C.Math.RADIANS_PER_DEGREE * R); }
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
  globe.enableLighting = !low; globe.showGroundAtmosphere = !low; globe.dynamicAtmosphereLighting = !low;
  const ground = new C.Cartographic(carto.longitude, carto.latitude, 0), elev = sunElevation(ground, time);
  updateTint(elev, low);
  const night = elev < -2;
  if (night !== nightToys) { nightToys = night; allModels.forEach(m => { m.customShader = night ? UNLIT : undefined; }); }
  const secs = C.JulianDate.secondsDifference(time, T0) * (REDUCED ? 0.3 : 1);
  updateMovers(secs, h); updateStatics(h);
  const now = performance.now();
  if (now - lastLine > 1000) {
    lastLine = now; const st = nearestStation(ground);
    const part = elev >= 8 ? "day" : elev >= -4 ? (elev >= 0 ? "evening light" : "dusk") : "night";
    const clock = st ? new Intl.DateTimeFormat("en-IN", { hour: "numeric", minute: "2-digit", timeZone: TZ[st.id] }).format(new Date()) : "";
    $("sunline").textContent = st ? `${st.name} · ${clock} · ${part}` : (h > 3e6 ? "The real sky right now" : `Over the sea · ${part}`);
  }
});

/* ---------------------------------------------------------------- hooks for the journey and deep links */
window.SGM_WORLD = { viewer, C, D, flyTo, flyBuilding, startExplore, showBot, openCard, closeCard, simpleCard, GROUND, PHONE, REDUCED,
                     showCaption: t => { $("captionText").textContent = t; $("caption").hidden = false; } };
function routeHash() {                                  // #nse jumps to NSE, #explore skips the intro
  if (location.hash === "#nse") { startExplore(); flyBuilding(NSE.lon, NSE.lat, PHONE() ? 1400 : 1100, 15, -26, 0, "bkc"); }
  else if (location.hash === "#explore") startExplore();
}
window.addEventListener("hashchange", routeHash);
setTimeout(routeHash, 0);
if (location.search.includes("debug")) scene.renderError.addEventListener((sc, e) => { document.documentElement.dataset.err = String(e && (e.stack || e.message) || e).slice(0, 600); });
if (location.search.includes("debug")) setInterval(() => {
  const c = camera.positionCartographic;
  document.documentElement.dataset.debug = JSON.stringify({ h: Math.round(c.height), pitch: +C.Math.toDegrees(camera.pitch).toFixed(1), globe: globe.show,
    tiles: !!tileset, movers: movers.length, statics: statics.length, tint: +tint.amount.toFixed(2) });
}, 1000);
})();
