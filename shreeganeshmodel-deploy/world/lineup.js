// Frame 1 of the journey (user's drawing 1): the crew lines up in lanes, each bot with its name, data type and flag
// shirt, then they all start running to the right. three.js overlay above the Cesium Earth; resolves when they've gone.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const COUNTRY = { IN: "India", US: "USA", DE: "Germany", AE: "UAE" };

function play(crew) {
  return new Promise(async resolve => {
    const wrap = document.getElementById("lineup"), canvas = wrap.querySelector("canvas"), labels = wrap.querySelector(".lanes");
    wrap.hidden = false; wrap.classList.remove("gone");
    const W = () => window.innerWidth, H = () => window.innerHeight, phone = W() <= 640;
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2)); renderer.setSize(W(), H(), false);
    renderer.shadowMap.enabled = true; renderer.outputColorSpace = THREE.SRGBColorSpace;
    const scene = new THREE.Scene();
    const lanes = crew.length, gap = 1.6, depth = (lanes - 1) * gap;
    const cam = new THREE.PerspectiveCamera(phone ? 50 : 34, W() / H(), 0.1, 300);
    const cx = phone ? 1.4 : 1.8;
    cam.position.set(cx, phone ? 24 : 18, depth / 2 + (phone ? 16 : 12.5)); cam.lookAt(cx, 0, depth / 2 + (phone ? 3.2 : 2.4));
    scene.add(new THREE.HemisphereLight(0xfff1e2, 0x4a3b33, 1.25));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(-6, 12, 9); sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024); Object.assign(sun.shadow.camera, { left: -20, right: 40, top: 20, bottom: -20 }); scene.add(sun);
    const track = new THREE.Mesh(new THREE.PlaneGeometry(90, depth + 3), new THREE.MeshStandardMaterial({ color: 0xc98a5e, roughness: 0.95 }));
    track.rotation.x = -Math.PI / 2; track.position.set(35, 0, depth / 2); track.receiveShadow = true; scene.add(track);
    const dashMat = new THREE.MeshBasicMaterial({ color: 0xfff6ea });
    for (let i = 0; i <= lanes; i++) for (let x = -1; x < 80; x += 1.3) {
      const d = new THREE.Mesh(new THREE.PlaneGeometry(0.7, 0.05), dashMat); d.rotation.x = -Math.PI / 2; d.position.set(x, 0.005, i * gap - gap / 2); scene.add(d);
    }
    const loader = new GLTFLoader(), bots = [];
    labels.innerHTML = "";
    await Promise.all(crew.map(async (c, i) => {
      const g = await loader.loadAsync(`assets/bots/bot_${c.flag}.glb`);
      const root = g.scene; root.scale.setScalar(1.15); root.rotation.y = 0.75; root.position.set(0, 0, i * gap);   // three-quarter view: faces the camera, ready to run right
      root.traverse(o => { if (o.isMesh) o.castShadow = true; });
      scene.add(root);
      const mixer = new THREE.AnimationMixer(root), clip = n => g.animations.find(a => a.name === n);
      const idle = mixer.clipAction(clip("idle")); idle.play();
      const el = document.createElement("div"); el.className = "lane";
      el.innerHTML = `<b>${c.name}</b><span>${c.what}</span><small>${c.fromShort} · ${COUNTRY[c.flag]}</small>`;
      labels.appendChild(el);
      bots[i] = { root, mixer, idle, run: mixer.clipAction(clip("run")), speed: 7 + (i * 37 % 5) * 0.6, el };
    }));
    let running = false, t0 = 0, done = false;
    const clock = new THREE.Clock(), v = new THREE.Vector3();
    function frame() {
      if (done) return;
      const dt = Math.min(0.05, clock.getDelta()), since = performance.now() - t0;
      bots.forEach(b => {
        b.mixer.update(dt);
        if (running) { b.root.position.x += b.speed * dt * Math.min(1, since / 400); b.root.rotation.y += (Math.PI / 2 - b.root.rotation.y) * Math.min(1, dt * 10); }
        v.set(-0.75, 0.6, b.root.position.z).project(cam);
        b.el.style.transform = `translate(${(v.x + 1) / 2 * W()}px, ${(1 - v.y) / 2 * H()}px) translate(-100%, -50%)`;
        b.el.style.opacity = running ? Math.max(0, 1 - since / 600) : 1;
      });
      if (running) cam.position.x += dt * 2.2;
      renderer.render(scene, cam);
      if (running && since > 3000) { finish(); return; }
      requestAnimationFrame(frame);
    }
    function finish() {
      done = true; wrap.classList.add("gone");
      setTimeout(() => { wrap.hidden = true; renderer.dispose(); resolve(); }, 700);
    }
    window.addEventListener("resize", () => { renderer.setSize(W(), H(), false); cam.aspect = W() / H(); cam.updateProjectionMatrix(); });
    frame();
    go = () => { if (running) return; running = true; t0 = performance.now();
      bots.forEach(b => { b.run.reset().play(); b.idle.crossFadeTo(b.run, 0.2, false); }); };
    if (window.SGM_LINEUP.onReady) window.SGM_LINEUP.onReady();
  });
}
let go = null;
window.SGM_LINEUP = { play, go: () => go && go() };
