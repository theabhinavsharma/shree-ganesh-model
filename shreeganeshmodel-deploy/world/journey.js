/* The "Start the journey" sequence, from the user's drawings (2026-10-17):
   1 the crew lines up in lanes and starts running right (world/lineup.js)
   2 zoom out across several camera angles: bots run from their home cities to the coast
   3 they board the ships (Ship 1 + the submarine for the cable; Ship 2 across the Pacific)
   4 India: Ship 1 docks in Mumbai, Ship 2 in Chennai; NSE in Mumbai, PIB next to Parliament in Delhi; all paths to Bengaluru
   5 Bengaluru: you, the laptop, the ML bots, and money moving between SBI and your stocks
   Then free exploring. Each step waits for "Next". */
(function () {
"use strict";
const W = window.SGM_WORLD; if (!W) return;
const D = W.D, $ = id => document.getElementById(id);
const SHORT = { bhav: "NSE, Mumbai", sarkari: "PIB, New Delhi", fed: "FRED, St. Louis", euro: "ECB, Frankfurt",
                gnews: "Google News, California", claude: "Anthropic, San Francisco", pigeon: "Telegram, Dubai" };
const WHAT = { bhav: "Stock prices", sarkari: "Government news", fed: "Oil and world prices", euro: "Exchange rates",
               gnews: "World headlines", claude: "The Friday market note", pigeon: "Messages to your phone" };
const CREW = Object.keys(SHORT).map(id => { const b = D.BOTS.find(x => x.id === id);
  return { id, name: b.name, flag: D.FLAG[b.st], what: WHAT[id], fromShort: SHORT[id] }; });

function steps() {
  const P = W.PHONE();
  return [
    { text: "Meet the crew. Every evening at 6:45 PM these bots fetch the data behind your Friday list.", lineup: true, next: "Go!" },
    { text: "They set off from home. Fed runs out of the St. Louis Fed with oil and world prices.", go: () => W.flyTo(-83, 31, P ? 1.3e6 : 9e5, 10, -42, 3.5) },
    { text: "Euro leaves the European Central Bank in Frankfurt with exchange rates.", go: () => W.flyTo(6, 38.5, P ? 1.1e6 : 8e5, 0, -45, 4) },
    { text: "At the coast they board the ships. Ship 1 carries oil and world prices across the Atlantic and the Red Sea; the submarine marks the undersea cable.", go: () => W.flyTo(0, 28, P ? 26e6 : 18e6, 0, -90, 4) },
    { text: "From California, Ship 2 brings Claude's Friday note and world headlines across the Pacific.", go: () => W.flyTo(160, 18, P ? 28e6 : 22e6, 0, -90, 4) },
    { text: "India. Ship 1 docks in Mumbai and Ship 2 in Chennai. NSE adds today's stock prices, and PIB in Delhi, next to Parliament, adds government news. Everything heads to Bengaluru.", go: () => W.flyTo(78, 5.5, P ? 3.9e6 : 2.6e6, 0, -55, 4) },
    { text: "Bengaluru. The ML bots turn it all into the Friday list. You decide, and money moves between your SBI account and your stocks.", go: () => W.flyTo(81.5, 9.2, P ? 6.5e5 : 5.0e5, 315, -36, 4), next: "Explore on my own" }
  ];
}
let step = -1, busy = false, lineupDone = null;
async function waitLineup() {
  for (let i = 0; i < 40 && !window.SGM_LINEUP; i++) await new Promise(r => setTimeout(r, 100));
  return window.SGM_LINEUP;
}
async function run(i) {
  const S = steps(); if (i >= S.length) return W.startExplore();
  step = i; busy = true; const s = S[i];
  W.closeCard(); W.showCaption(s.text);
  const next = $("nextBtn"); next.disabled = true; next.textContent = s.next || "Next";
  if (s.lineup) {
    const L = await waitLineup();
    if (!L) { busy = false; return run(i + 1); }
    lineupDone = L.play(CREW);
    next.disabled = false; busy = false;
    return;
  }
  await s.go(); next.disabled = false; busy = false;
}
async function onNext() {
  if (busy) return;
  const S = steps();
  if (S[step] && S[step].lineup && lineupDone) {        // "Go!": the crew runs off, then the Earth takes over
    busy = true; $("nextBtn").disabled = true; window.SGM_LINEUP.go(); await lineupDone; lineupDone = null; busy = false;
    return run(step + 1);
  }
  if (step >= S.length - 1) return W.startExplore();
  run(step + 1);
}
function skip() { const l = $("lineup"); if (l) l.hidden = true; lineupDone = null; W.startExplore(); }
$("startBtn").addEventListener("click", () => { $("intro").hidden = true; run(0); });
$("exploreBtn").addEventListener("click", () => { $("intro").hidden = true; W.startExplore(); });
$("nextBtn").addEventListener("click", onNext);
$("skipBtn").addEventListener("click", skip);
if (location.hash === "#journey") { $("intro").hidden = true; run(0); }          // deep link: start the journey directly
})();
