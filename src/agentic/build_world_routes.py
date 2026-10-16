"""Road routes for the Data World follow-cam journeys (shreeganeshmodel-deploy/world/routes/<id>.json).

Each route is a real driving route from the OSRM demo server (router.project-osrm.org) on OpenStreetMap data.
The first and last 3 km keep full detail, because the camera is at street level there. The long middle is
simplified (Douglas-Peucker, 150 m), because the camera is tens of kilometres up by then.
Map data (c) OpenStreetMap contributors (ODbL).

Run: /usr/bin/python3 src/agentic/build_world_routes.py [route id ...]
A rebuilt route needs its heights measured again (world/routes/heights_<journey>.json).
"""
import datetime, json, math, subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "shreeganeshmodel-deploy/world/routes"
# [lon, lat]. Public places only: the data source's building, a public beach or road at the coast, Vidhana Soudha in Bengaluru.
ROUTES = {
    "euro_fra_mrs": ("European Central Bank, Frankfurt", [8.7037, 50.1095], "Plages du Prado, Marseille (Europe's cables meet the sea)", [5.3730, 43.2615]),
    "fed_stl_nj": ("Federal Reserve Bank of St. Louis (FRED)", [-90.1889, 38.6291], "Long Branch beach, New Jersey (Atlantic cable landings)", [-73.9789, 40.2994]),
    "courier_mum_blr": ("Versova beach, Mumbai (cable landing)", [72.8091, 19.1363], "Vidhana Soudha, Bengaluru (public landmark for the laptop)", [77.5907, 12.9796]),
}
# short walks on foot that no road reaches: from the road to the water's edge, picked by eye on Google's 3D map (2026-10-17)
WALK_AFTER = {"euro_fra_mrs": [[5.3730, 43.2613], [5.3725, 43.26145]],      # across the Prado park to the cove's beach
              "fed_stl_nj": [[-73.97839, 40.29950], [-73.97752, 40.29957]]}   # from Ocean Avenue over the boardwalk to the sand
WALK_BEFORE = {"courier_mum_blr": [[72.80843, 19.13568]]}                     # from the sand at Versova up to the coast road
KEEP_M, TOL_M = 3000.0, 150.0


def metres(a, b):
    la = math.radians((a[1] + b[1]) / 2)
    return math.hypot(math.radians(b[0] - a[0]) * math.cos(la) * 6371000, math.radians(b[1] - a[1]) * 6371000)


def seg_dist(p, a, b):
    """Distance from p to segment ab in metres (local flat approximation)."""
    la = math.radians(a[1]); kx, ky = math.cos(la) * 111320, 110540
    ax, ay, bx, by, px, py = 0, 0, (b[0] - a[0]) * kx, (b[1] - a[1]) * ky, (p[0] - a[0]) * kx, (p[1] - a[1]) * ky
    L2 = bx * bx + by * by
    t = 0 if L2 == 0 else max(0, min(1, (px * bx + py * by) / L2))
    return math.hypot(px - t * bx, py - t * by)


def simplify(pts, tol):
    keep, stack = [False] * len(pts), [(0, len(pts) - 1)]
    keep[0] = keep[-1] = True
    while stack:
        i, j = stack.pop()
        best, k = -1, None
        for m in range(i + 1, j):
            d = seg_dist(pts[m], pts[i], pts[j])
            if d > best: best, k = d, m
        if k is not None and best > tol:
            keep[k] = True; stack += [(i, k), (k, j)]
    return [p for p, f in zip(pts, keep) if f]


def build(rid, frm, a, to, b):
    url = f"https://router.project-osrm.org/route/v1/driving/{a[0]},{a[1]};{b[0]},{b[1]}?overview=full&geometries=geojson"
    r = json.loads(subprocess.run(["curl", "-s", "-m", "60", url], capture_output=True, text=True, check=True).stdout)
    if r.get("code") != "Ok": raise SystemExit(f"{rid}: OSRM said {r.get('code')}")
    route = r["routes"][0]; pts = route["geometry"]["coordinates"]
    cum = [0.0]
    for i in range(1, len(pts)): cum.append(cum[-1] + metres(pts[i - 1], pts[i]))
    total = cum[-1]
    i0 = next(i for i, c in enumerate(cum) if c >= KEEP_M); i1 = next(i for i, c in enumerate(cum) if c >= total - KEEP_M)
    coords = pts[:i0] + simplify(pts[i0:i1 + 1], TOL_M) + pts[i1 + 1:]
    coords = WALK_BEFORE.get(rid, []) + [[round(x, 6), round(y, 6)] for x, y in coords] + WALK_AFTER.get(rid, [])
    out = {"id": rid, "from": frm, "to": to, "distance_km": round(route["distance"] / 1000, 1), "driving_hours": round(route["duration"] / 3600, 1),
           "points_full": len(pts), "points_kept": len(coords),
           "detail": f"full detail for the first and last {KEEP_M / 1000:.0f} km; the middle simplified to {TOL_M:.0f} m",
           "walk_on_foot": {"before": WALK_BEFORE.get(rid, []), "after": WALK_AFTER.get(rid, []), "note": "added by hand: the stretch from the road to the water's edge"},
           "source": f"Road route from the OSRM demo server (router.project-osrm.org, driving profile) on OpenStreetMap data, fetched {datetime.date.today()}. Map data © OpenStreetMap contributors (ODbL).",
           "coordinates": coords}
    (OUT / f"{rid}.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"{rid}: {out['distance_km']} km, {len(pts)} -> {len(coords)} points")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    import sys
    only = sys.argv[1:] or list(ROUTES)                 # e.g. `build_world_routes.py fed_stl_nj` rebuilds just that one
    for rid in only: build(rid, *ROUTES[rid])
