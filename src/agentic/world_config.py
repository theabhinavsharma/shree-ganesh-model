"""Write shreeganeshmodel-deploy/world/config.js (git-ignored) from ~/.config/sgm/google_maps.env.

The env file holds one line, GOOGLE_MAPS_KEY=..., created by the user. The key is a browser key: it ends up in the public
page, so it must stay restricted in Google Cloud to the site's referrers and to the Map Tiles API, with a daily quota cap.
This script never prints the key. Run it before `vercel --prod` from shreeganeshmodel-deploy/.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV, OUT = Path.home() / ".config/sgm/google_maps.env", ROOT / "shreeganeshmodel-deploy/world/config.js"


def main() -> int:
    key = ""
    if ENV.exists():
        for line in ENV.read_text().splitlines():
            if line.strip().startswith("GOOGLE_MAPS_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        print(f"no GOOGLE_MAPS_KEY in {ENV}; the page will run without Google 3D tiles", file=sys.stderr)
        OUT.write_text("window.SGM_CONFIG = {};\n")
        return 1
    OUT.write_text("window.SGM_CONFIG = " + json.dumps({"googleKey": key}) + ";\n")
    print(f"wrote {OUT.relative_to(ROOT)} (key length {len(key)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
