"""Puts each film's captions and beats into follow.html (the FILMS object), from <film>.vtt written by make_video.py.

/usr/bin/python3 docs/sgm-world/follow/fill_page.py"""
import json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
BEATS = {
    "euro": [("From space", "The whole route on the real globe. The dashed line is the road still ahead."),
             ("Frankfurt", "The camera dives to the European Central Bank. Euro, human-sized, runs down the street towards the camera."),
             ("The road", "Street level for the first stretch, then the drone climbs and the run turns into a time-lapse: 1,016 km to Marseille, with the amber streak behind."),
             ("Marseille", "Across the Prado park onto the beach, where Euro jumps into the submarine waiting in the cove."),
             ("The sea", "The Mediterranean, Suez, the Red Sea and the Arabian Sea, leaving a cyan streak."),
             ("Mumbai", "The submarine surfaces off Versova beach and throws the crate to a courier on the sand."),
             ("The courier", "996 km over the Western Ghats and across the Deccan to Bengaluru."),
             ("Bengaluru", "The crate flies into the giant laptop by Vidhana Soudha, 21 data checks tick off, and you cheer.")],
    "fed": [("From space", "The whole route, from St. Louis across the Atlantic to India."),
            ("St. Louis", "The camera dives to the St. Louis Fed, the home of FRED. Fed picks up the prices and runs down the street at dawn."),
            ("The road", "1,532 km east through Indianapolis, Ohio and Pennsylvania, with the amber streak behind."),
            ("Long Branch", "Over the boardwalk onto the New Jersey beach, where Fed jumps aboard Ship 1, the oil tanker."),
            ("The sea", "The Atlantic, Gibraltar, the Mediterranean, Suez and the Red Sea, leaving a white wake."),
            ("Mumbai", "Ship 1 slows off Versova beach and throws the crate to a courier on the sand."),
            ("The courier", "996 km over the Western Ghats and across the Deccan to Bengaluru, at sunset."),
            ("Bengaluru", "The crate lands in the giant laptop. On Friday the ML bots use these prices to mark rising and fading themes.")],
}
SKY = {"euro": "Shown in daylight (9 AM in Frankfurt). Exploring the globe keeps the real time of day.",
       "fed": "Shown at dawn in St. Louis and sunset in India. Exploring the globe keeps the real time of day."}


def sec(t):
    h, m, s = t.split(":"); return round(int(h) * 3600 + int(m) * 60 + float(s), 3)


films = {}
for fid in BEATS:
    vtt = (HERE / f"{fid}.vtt").read_text()
    cues = [[sec(a), sec(b), txt.strip()] for a, b, txt in re.findall(r"(\S+) --> (\S+)\n(.+?)(?:\n\n|\n?$)", vtt, re.S)]
    assert len(cues) == len(BEATS[fid]), (fid, len(cues))   # one beat per caption
    beats = [[f"{int(c[0] // 60)}:{int(c[0] % 60):02d}", b, d] for c, (b, d) in zip(cues, BEATS[fid])]
    films[fid] = {"src": f"{fid}.mp4", "poster": f"{fid}.jpg", "sky": SKY[fid], "cues": cues, "beats": beats}
page = HERE / "follow.html"; s = page.read_text()
s = re.sub(r"const FILMS = .*?;\n", lambda m: "const FILMS = " + json.dumps(films, ensure_ascii=False) + ";\n", s, count=1, flags=re.S)
page.write_text(s)
print("filled", {k: len(v["cues"]) for k, v in films.items()})
