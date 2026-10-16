/* "Start the journey" (user's call, 2026-10-17): the crew lines up in lanes (world/lineup.js, from the user's first
   drawing). On "Go!" they run off and the follow-cam film takes over (world/chase.js): Euro's whole run on the real
   Earth, by road from Frankfurt and by submarine from Marseille, then a courier to the laptop in Bengaluru. The film's
   end panel offers Fed's journey or free exploring. The earlier slideshow of camera stops is retired (see git history). */
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
const FIRST_FILM = "euro";

let busy = false, lineupDone = null;
async function waitLineup() {
  for (let i = 0; i < 40 && !window.SGM_LINEUP; i++) await new Promise(r => setTimeout(r, 100));
  return window.SGM_LINEUP;
}
function film() {
  W.closeCard(); $("caption").hidden = true;
  if (window.SGM_CHASE && window.SGM_CHASE.has(FIRST_FILM)) window.SGM_CHASE.play(FIRST_FILM); else W.startExplore();   // until a film is live, the lineup leads into exploring
}
async function start() {
  busy = true; W.closeCard();
  W.showCaption("Meet the crew. Every evening at 6:45 PM these bots fetch the data behind your Friday list. Press Go! to see where they go.");
  const next = $("nextBtn"); next.disabled = true; next.textContent = "Go!";
  const L = await waitLineup();
  if (!L) { busy = false; return film(); }
  lineupDone = L.play(CREW);
  if (window.SGM_CHASE && window.SGM_CHASE.has(FIRST_FILM)) window.SGM_CHASE.preload(FIRST_FILM);   // the map loads behind the lineup while the crew is on screen
  next.disabled = false; busy = false;
}
async function onNext() {                               // "Go!": the crew runs off, then the film takes over
  if (busy) return;
  busy = true; $("nextBtn").disabled = true;
  if (lineupDone) { window.SGM_LINEUP.go(); await lineupDone; lineupDone = null; }
  busy = false; film();
}
function skip() { const l = $("lineup"); if (l) l.hidden = true; lineupDone = null; W.startExplore(); }
$("startBtn").addEventListener("click", () => { $("intro").hidden = true; start(); });
$("exploreBtn").addEventListener("click", () => { $("intro").hidden = true; W.startExplore(); });
$("nextBtn").addEventListener("click", onNext);
$("skipBtn").addEventListener("click", skip);
if (location.hash === "#journey") { $("intro").hidden = true; start(); }          // deep link: start the journey directly
})();
