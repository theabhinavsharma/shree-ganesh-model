#!/usr/bin/env python3
"""
SGM Data World - worker bot generator.

Builds the little terracotta block bots (Claude Code mascot style) that wear a
flag T-shirt, as binary glTF 2.0 (.glb) files with node-TRS animations
"run" and "idle". No skinning, no extensions.

Pure Python 3 standard library (json, struct, zlib, math). Deterministic:
running it twice produces byte-identical files.

    python3 docs/sgm-world/bots/build_bots.py                  # build all 10 + validate
    python3 docs/sgm-world/bots/build_bots.py --check          # validate existing files only
    python3 docs/sgm-world/bots/build_bots.py --dump-textures DIR   # also write shirt PNGs

Conventions, flag rules and the parts list are in README.md next to this file.
"""

import argparse
import json
import math
import struct
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # docs/sgm-world/bots -> repo root
OUT_DIR = REPO / "shreeganeshmodel-deploy" / "assets" / "bots"

COUNTRIES = ["IN", "US", "DE", "AE"]
VARIANTS = COUNTRIES + ["astro"]

RUN_T, RUN_N = 0.45, 24      # run loop: 0.45 s, 24 linear keys (+ closing key)
IDLE_T, IDLE_N = 2.0, 48     # idle loop: 2.0 s, 48 linear keys (+ closing key)

MAX_BYTES = 150 * 1024
MAX_TRIS = 3000

# ------------------------------------------------------------------ colours

def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def s2l(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def l2s(c):
    c = min(max(c, 0.0), 1.0)
    return c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def lin(h):
    """sRGB hex -> linear RGB list (glTF factors are linear)."""
    return [round(s2l(v / 255), 6) for v in hex_rgb(h)]


COL = {
    "body": "#D97757",        # terracotta
    "eye": "#1F1A17",
    "shirt": "#F7F5F0",       # off-white tee
    "shoe": "#26262B",
    "sole": "#E6E3DD",
    "crate_frame": "#2E3A4B",
    "crate_core": "#8BE9FF",
    "crate_glow": "#2CC6F2",
    "suit": "#F4F5F7",
    "suit_trim": "#C3CAD3",
    "glove": "#9AA4B0",
    "boot": "#8E98A5",
    "pack": "#C9CFD7",
    "tank": "#AEB7C2",
    "nozzle": "#59616B",
    "flame": "#FFB347",
    "flame_glow": "#FF8A2A",
    "glass": "#DDF3FF",
    "panel": "#D3D9E0",
    "btn_r": "#E5484D",
    "btn_y": "#F5C542",
    "btn_b": "#3E8EDE",
}
PANTS = {"IN": "#1F2A5A",     # navy
         "US": "#3F6A9F",     # denim
         "DE": "#3A3E44",     # charcoal
         "AE": "#CDB487",     # sand
         "astro": "#E9ECF0"}  # suit legs

# ------------------------------------------------------------------ flags
# Each flag function takes (x, y) in flag units: height = 1, width = ratio,
# x from the hoist (viewer's left) and y from the top. Returns sRGB 0..255.

IN_SAFFRON, IN_GREEN, IN_NAVY = hex_rgb("#FF9933"), hex_rgb("#138808"), hex_rgb("#000080")
US_RED, US_BLUE = hex_rgb("#B22234"), hex_rgb("#3C3B6E")
DE_RED, DE_GOLD = hex_rgb("#DD0000"), hex_rgb("#FFCE00")
AE_RED, AE_GREEN = hex_rgb("#FF0000"), hex_rgb("#00732F")
WHITE, BLACK = (255, 255, 255), (0, 0, 0)


def flag_in(x, y):
    """India, 2:3. Saffron / white / green; navy Ashoka Chakra, 24 spokes.
    Chakra geometry follows the common reference drawing: band 50 units,
    rim r 17.5..20, hub r 3.5, diamond spokes r 2..17.5 (max half-width 0.6
    at r 7), 24 rim dots r 0.875 between the spokes."""
    if y < 1 / 3:
        return IN_SAFFRON
    if y >= 2 / 3:
        return IN_GREEN
    k = (1 / 3) / 50.0
    dx, dy = (x - 0.75) / k, (y - 0.5) / k
    d = math.hypot(dx, dy)
    if d > 20.0:
        return WHITE
    if d >= 17.5 or d <= 3.5:
        return IN_NAVY
    ang = math.atan2(dx, -dy)
    step = 2 * math.pi / 24
    a = ang - round(ang / step) * step
    along, across = d * math.cos(a), abs(d * math.sin(a))
    if 2.0 <= along <= 17.5:
        hw = 0.6 * (along - 2.0) / 5.0 if along <= 7.0 else 0.6 * (17.5 - along) / 10.5
        if across <= hw:
            return IN_NAVY
    a2 = ang - (round((ang - step / 2) / step) * step + step / 2)
    px, py = d * math.sin(a2), d * math.cos(a2) - 17.5
    if px * px + py * py <= 0.875 ** 2:
        return IN_NAVY
    return WHITE


_STAR_IN = 0.381966  # inner/outer radius of a regular 5-point star
_S36, _C36 = math.sin(math.radians(36)), math.cos(math.radians(36))


def in_star(dx, dy, R):
    """5-point star, one point straight up; dy grows downward."""
    d = math.hypot(dx, dy)
    if d > R:
        return False
    r = R * _STAR_IN
    if d <= r * 0.809:
        return True
    ang = math.degrees(math.atan2(dx, -dy)) % 72.0
    if ang > 36.0:
        ang = 72.0 - ang
    a = math.radians(ang)
    px, py = d * math.sin(a), d * math.cos(a)
    ix, iy = r * _S36, r * _C36
    # inside if P is on the same side of the edge outer(0,R)->inner as the centre
    cp = ix * (py - R) - (iy - R) * px
    co = ix * (0 - R) - (iy - R) * 0
    return cp * co >= 0


def flag_us(x, y):
    """USA, 10:19. 13 stripes (red top and bottom), canton 7 stripes tall and
    0.76 wide, 50 stars in 9 rows alternating 6 and 5 (star diameter 0.0616)."""
    C = 7 / 13
    if x < 0.76 and y < C:
        F, G, R = C / 10, 0.76 / 12, 0.0616 / 2
        base = int(y / F)
        for row in (base, base + 1):
            if 1 <= row <= 9:
                cy = row * F
                cols = (1, 3, 5, 7, 9, 11) if row % 2 == 1 else (2, 4, 6, 8, 10)
                for col in cols:
                    cx = col * G
                    if abs(x - cx) <= R and in_star(x - cx, y - cy, R):
                        return WHITE
        return US_BLUE
    return US_RED if int(y * 13) % 2 == 0 else WHITE


def flag_de(x, y):
    """Germany, 3:5. Black / red / gold."""
    return BLACK if y < 1 / 3 else (DE_RED if y < 2 / 3 else DE_GOLD)


def flag_ae(x, y):
    """UAE, 1:2. Red vertical band (1/4 of the length) at the hoist, then
    green / white / black horizontal bands."""
    if x < 0.5:
        return AE_RED
    return AE_GREEN if y < 1 / 3 else (WHITE if y < 2 / 3 else BLACK)


# Emblem boxes (flag units x0, y0, x1, y1) that are always supersampled.
FLAG_DETAIL = {
    "IN": [(0.75 - 0.15, 0.5 - 0.15, 0.75 + 0.15, 0.5 + 0.15)],
    "US": [(0.0, 0.0, 0.76, 7 / 13)],
    "DE": [],
    "AE": [],
}
# Plain-colour parts of each tee: sleeves (a matching flag colour), shoulder cap, hem underside.
TEE_SOLIDS = {
    "IN": {"sleeve": "#FF9933", "top": "#FF9933", "hem": "#138808"},
    "US": {"sleeve": "#3C3B6E", "top": "#B22234", "hem": "#B22234"},
    "DE": {"sleeve": "#000000", "top": "#000000", "hem": "#FFCE00"},
    "AE": {"sleeve": "#00732F", "top": "#00732F", "hem": "#000000"},
}

# ------------------------------------------------------------------ PNG

def encode_png(w, h, rgb, alpha=False):
    bpp = 4 if alpha else 3
    stride = w * bpp
    raw = bytearray()
    prev = bytes(stride)
    for y in range(h):
        row = rgb[y * stride:(y + 1) * stride]
        cands = [(0, bytes(row))]
        f1, f2, f3, f4 = bytearray(stride), bytearray(stride), bytearray(stride), bytearray(stride)
        for i in range(stride):
            a = row[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            x = row[i]
            f1[i] = (x - a) & 0xFF
            f2[i] = (x - b) & 0xFF
            f3[i] = (x - ((a + b) >> 1)) & 0xFF
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
            f4[i] = (x - pr) & 0xFF
        cands += [(1, f1), (2, f2), (3, f3), (4, f4)]
        best = min(cands, key=lambda t: (sum(v if v < 128 else 256 - v for v in t[1]), t[0]))
        raw.append(best[0])
        raw += best[1]
        prev = row

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6 if alpha else 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
            + chunk(b"IEND", b""))


SS = 4


def raster(tw, th, w_units, h_units, sample, details=(), ss=SS, alpha=False, full=False):
    """Rasterise sample(x, y) -> sRGB(A) 0..255 over a w_units x h_units area
    (x right, y down) into a tw x th PNG. Texels on colour edges or inside
    `details` rectangles are supersampled ss x ss (colour averaged in linear
    light). full=True supersamples every texel (for thin strokes)."""
    sx, sy = w_units / tw, h_units / th
    centers = [[sample((i + 0.5) * sx, (j + 0.5) * sy) for i in range(tw)] for j in range(th)]
    lut = [s2l(v / 255) for v in range(256)]
    nch = 4 if alpha else 3
    out = bytearray()
    for j in range(th):
        for i in range(tw):
            c = centers[j][i]
            need = full
            if not need:
                for jj in (j - 1, j, j + 1):
                    if 0 <= jj < th:
                        row = centers[jj]
                        for ii in (i - 1, i, i + 1):
                            if 0 <= ii < tw and row[ii] != c:
                                need = True
                                break
                    if need:
                        break
            if not need:
                px, py = (i + 0.5) * sx, (j + 0.5) * sy
                for (ax, ay, bx, by) in details:
                    if ax - sx <= px <= bx + sx and ay - sy <= py <= by + sy:
                        need = True
                        break
            if need:
                acc = [0.0] * nch
                for a in range(ss):
                    for b in range(ss):
                        smp = sample((i + (a + 0.5) / ss) * sx, (j + (b + 0.5) / ss) * sy)
                        acc[0] += lut[smp[0]]
                        acc[1] += lut[smp[1]]
                        acc[2] += lut[smp[2]]
                        if alpha:
                            acc[3] += smp[3] / 255
                n = ss * ss
                px = [int(round(l2s(v / n) * 255)) for v in acc[:3]]
                if alpha:
                    px.append(int(round(acc[3] / n * 255)))
                out += bytes(px)
            else:
                out += bytes(c[:nch])
    return encode_png(tw, th, out, alpha=alpha)


def tee_layout():
    """Perimeter of the tee shell, walking round it from the front-left edge to
    the viewer's right (front, right side, back, left side). Returns the 9
    cumulative positions S (S[8] = full perimeter), the front/back face
    lengths, and the flag height H = visible tee height (hem to where the
    sloped shoulder meets the orange body)."""
    hx, hz, cv = SHIRT["w"] / 2, SHIRT["d"] / 2, SHIRT["corner"]
    lf, ls, lc = 2 * (hx - cv), 2 * (hz - cv), cv * math.sqrt(2)
    S = [0.0]
    for seg in (lf, lc, ls, lc, lf, lc, ls, lc):
        S.append(S[-1] + seg)
    y_top = SHIRT["y0"] + SHIRT["h"] - SHIRT["shoulder"] + (hx - BODY["w"] / 2)
    return S, lf, y_top, y_top - SHIRT["y0"]


TEE_W, TEE_H = 2048, 256


def render_tee_png(code):
    """All-round flag print for the tee (panorama: u = position round the body,
    v = height). Bands run all the way round; the emblem (Ashoka Chakra, US
    canton, UAE red hoist band) sits on the front and is repeated on the back,
    hoist on the viewer's left from either side."""
    S, lf, _, H = tee_layout()
    P = S[8]
    panels = ((S[0], S[1]), (S[4], S[5]))   # front, back

    def local(s):
        for a, b in panels:
            if a <= s <= b:
                return s - a
        return None

    def sample(s, yd):
        fy = min(max(yd / H, 0.0), 0.999999)
        loc = local(s)
        if code == "IN":
            return flag_in(0.0 if loc is None else 0.75 + (loc - lf / 2) / H, fy)
        if code == "US":
            if loc is not None and loc < 0.76 * H:
                return flag_us(loc / H, fy)
            return flag_us(1.0, fy)
        if code == "DE":
            return flag_de(0.0, fy)
        if loc is not None and loc < 0.5 * H:   # AE
            return AE_RED
        return flag_ae(1.0, fy)

    details = []
    for a, _ in panels:
        for (x0, y0, x1, y1) in FLAG_DETAIL[code]:
            if code == "IN":
                details.append((a + lf / 2 + (x0 - 0.75) * H, y0 * H, a + lf / 2 + (x1 - 0.75) * H, y1 * H))
            else:
                details.append((a + x0 * H, y0 * H, a + x1 * H, y1 * H))
    return raster(TEE_W, TEE_H, P, H, sample, details)


# ------------------------------------------------------------------ math

def v_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_unit(a):
    n = math.sqrt(v_dot(a, a))
    return (a[0] / n, a[1] / n, a[2] / n)


def q_axis(axis, deg):
    h = math.radians(deg) / 2
    s = math.sin(h)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(h))


def q_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


X_AX, Y_AX, Z_AX = (1, 0, 0), (0, 1, 0), (0, 0, 1)


def q_ypr(yaw=0.0, pitch=0.0, roll=0.0):
    """yaw about Y, then pitch about X, then roll about Z (degrees)."""
    return q_mul(q_mul(q_axis(Y_AX, yaw), q_axis(X_AX, pitch)), q_axis(Z_AX, roll))


def q_to_m(q):
    x, y, z, w = q
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))


def m_apply(m, v):
    return (m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2])

# ------------------------------------------------------------------ meshes

class Prim:
    def __init__(self, mat, textured):
        self.mat = mat
        self.pos, self.nrm, self.idx = [], [], []
        self.uv = [] if textured else None


class Mesh:
    def __init__(self, name):
        self.name = name
        self.prims = []
        self._by = {}

    def prim(self, mat, textured=False):
        key = (mat, textured)
        if key not in self._by:
            self._by[key] = Prim(mat, textured)
            self.prims.append(self._by[key])
        return self._by[key]

    def poly(self, mat, pts, normal, uvs=None):
        p = self.prim(mat, uvs is not None)
        if v_dot(v_cross(v_sub(pts[1], pts[0]), v_sub(pts[2], pts[0])), normal) < 0:
            pts = pts[::-1]
            uvs = uvs[::-1] if uvs else None
        base = len(p.pos)
        for i, pt in enumerate(pts):
            p.pos.append(pt)
            p.nrm.append(normal)
            if p.uv is not None:
                p.uv.append(uvs[i])
        for i in range(1, len(pts) - 1):
            p.idx += [base, base + i, base + i + 1]

    def tris(self, count):
        return sum(len(p.idx) for p in self.prims) // 3


def add_box(mesh, mat, center, size, bevel=0.0, rot=None, faces=None):
    """Box with optional 45-degree chamfer on every edge, flat shaded.
    faces: {'+z': mat, '-z': mat, '+y': mat} -> those main faces get UVs and the
    given (textured) material. +z/-z: u to the viewer's right, v down;
    +y (top): u to +X, v toward +Z (reads upright from the front)."""
    faces = faces or {}
    H = (size[0] / 2, size[1] / 2, size[2] / 2)
    b = min(bevel, H[0] * 0.49, H[1] * 0.49, H[2] * 0.49)
    I = (H[0] - b, H[1] - b, H[2] - b)
    rm = q_to_m(rot) if rot else None

    def emit(pts, n, m, uvs=None):
        if rm:
            pts = [m_apply(rm, p) for p in pts]
            n = m_apply(rm, n)
        pts = [(p[0] + center[0], p[1] + center[1], p[2] + center[2]) for p in pts]
        mesh.poly(m, pts, n, uvs)

    for ax in range(3):
        o1, o2 = [k for k in range(3) if k != ax]
        for s in (1, -1):
            pts = []
            for c1, c2 in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = [0.0, 0.0, 0.0]
                p[ax], p[o1], p[o2] = s * H[ax], c1 * I[o1], c2 * I[o2]
                pts.append(tuple(p))
            n = [0.0, 0.0, 0.0]
            n[ax] = float(s)
            tag = ("+" if s > 0 else "-") + "xyz"[ax]
            if tag in faces and tag in ("+z", "-z", "+y"):
                ix, iy, iz = I
                if tag == "+z":
                    uvs = [((q[0] + ix) / (2 * ix), (iy - q[1]) / (2 * iy)) for q in pts]
                elif tag == "-z":
                    uvs = [((ix - q[0]) / (2 * ix), (iy - q[1]) / (2 * iy)) for q in pts]
                else:
                    uvs = [((q[0] + ix) / (2 * ix), (q[2] + iz) / (2 * iz)) for q in pts]
                emit(pts, tuple(n), faces[tag], uvs)
            else:
                emit(pts, tuple(n), mat)
    if b <= 0:
        return
    for a in range(3):
        for c in range(a + 1, 3):
            o = 3 - a - c
            for sa in (1, -1):
                for sc in (1, -1):
                    pts = []
                    for first, so in ((True, -1), (True, 1), (False, 1), (False, -1)):
                        p = [0.0, 0.0, 0.0]
                        p[a] = sa * (H[a] if first else I[a])
                        p[c] = sc * (I[c] if first else H[c])
                        p[o] = so * I[o]
                        pts.append(tuple(p))
                    n = [0.0, 0.0, 0.0]
                    n[a], n[c] = float(sa), float(sc)
                    emit(pts, v_unit(n), mat)
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                pts = [(sx * H[0], sy * I[1], sz * I[2]),
                       (sx * I[0], sy * H[1], sz * I[2]),
                       (sx * I[0], sy * I[1], sz * H[2])]
                emit(pts, v_unit((sx, sy, sz)), mat)


def add_cyl(mesh, mat, center, r_bot, r_top, h, seg, caps=(True, True), phase=0.0):
    """Vertical (Y) cylinder / frustum / cone, flat shaded."""
    cx, cy, cz = center
    y0, y1 = cy - h / 2, cy + h / 2
    ring = []
    for j in range(seg):
        a = 2 * math.pi * (j + phase) / seg
        ring.append((math.sin(a), math.cos(a)))
    slope = (r_bot - r_top) / h
    for j in range(seg):
        (s0, c0), (s1, c1) = ring[j], ring[(j + 1) % seg]
        am = 2 * math.pi * (j + 0.5 + phase) / seg
        n = v_unit((math.sin(am), slope, math.cos(am)))
        b0 = (cx + r_bot * s0, y0, cz + r_bot * c0)
        b1 = (cx + r_bot * s1, y0, cz + r_bot * c1)
        if r_top <= 1e-9:
            mesh.poly(mat, [b0, b1, (cx, y1, cz)], n)
        elif r_bot <= 1e-9:
            t0 = (cx + r_top * s0, y1, cz + r_top * c0)
            t1 = (cx + r_top * s1, y1, cz + r_top * c1)
            mesh.poly(mat, [(cx, y0, cz), t1, t0], n)
        else:
            t0 = (cx + r_top * s0, y1, cz + r_top * c0)
            t1 = (cx + r_top * s1, y1, cz + r_top * c1)
            mesh.poly(mat, [b0, b1, t1, t0], n)
    if caps[0] and r_bot > 0:
        mesh.poly(mat, [(cx + r_bot * s, y0, cz + r_bot * c) for s, c in ring], (0.0, -1.0, 0.0))
    if caps[1] and r_top > 0:
        mesh.poly(mat, [(cx + r_top * s, y1, cz + r_top * c) for s, c in ring], (0.0, 1.0, 0.0))


def add_dome(mesh, mat, center, R, seg, rings, theta_max_deg):
    """Smooth-shaded sphere cap from the top pole down to theta_max (open below)."""
    p = mesh.prim(mat)
    base = len(p.pos)
    cx, cy, cz = center
    p.pos.append((cx, cy + R, cz))
    p.nrm.append((0.0, 1.0, 0.0))
    tm = math.radians(theta_max_deg)
    for i in range(1, rings + 1):
        th = tm * i / rings
        for j in range(seg):
            ph = 2 * math.pi * j / seg
            n = (math.sin(th) * math.sin(ph), math.cos(th), math.sin(th) * math.cos(ph))
            p.pos.append((cx + R * n[0], cy + R * n[1], cz + R * n[2]))
            p.nrm.append(n)

    def vid(i, j):
        return base + 1 + (i - 1) * seg + (j % seg)

    def tri(a, b, c):
        pa, pb, pc = p.pos[a], p.pos[b], p.pos[c]
        cen = ((pa[0] + pb[0] + pc[0]) / 3 - cx, (pa[1] + pb[1] + pc[1]) / 3 - cy, (pa[2] + pb[2] + pc[2]) / 3 - cz)
        if v_dot(v_cross(v_sub(pb, pa), v_sub(pc, pa)), cen) < 0:
            b, c = c, b
        p.idx += [a, b, c]

    for j in range(seg):
        tri(base, vid(1, j), vid(1, j + 1))
    for i in range(1, rings):
        for j in range(seg):
            tri(vid(i, j), vid(i + 1, j), vid(i + 1, j + 1))
            tri(vid(i, j), vid(i + 1, j + 1), vid(i, j + 1))

# ------------------------------------------------------------------ bot layout
# All numbers in model units; +Y up, bot faces +Z, origin between the feet.

HIP_Y = 0.30            # hips node (legs pivot here, torso pivot here)
LEG_X = 0.13
ARM_X, ARM_Y = 0.345, 0.58
EYE_Y = 0.82
ARM_TILT = 8.0          # resting outward tilt of the free arms (deg)

BODY = dict(w=0.64, h=0.66, d=0.44, y0=0.32, bevel=0.06)     # the orange block
# Tee shell over the lower half of the block. bevel: the astro suit (a chamfered
# box); corner/shoulder/hem: the flag tee (octagonal walls, sloped shoulders).
SHIRT = dict(w=0.665, h=0.38, d=0.465, y0=0.30, bevel=0.065, corner=0.065, shoulder=0.065, hem=0.02)

CRATE = dict(w=0.42, h=0.34, d=0.36, bar=0.035)
CRATE_BOTTOM = {"bot": 0.995, "astro": 1.245}

HELMET = dict(cy=0.83, R=0.41, cut=0.69)


def material(name, hexcol, rough=0.8, emissive=None, alpha=None, texture=None, double=False):
    m = {"name": name,
         "pbrMetallicRoughness": {"baseColorFactor": lin(hexcol) + [1.0 if alpha is None else alpha],
                                  "metallicFactor": 0.0, "roughnessFactor": rough}}
    if texture is not None:
        m["pbrMetallicRoughness"]["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
        m["pbrMetallicRoughness"]["baseColorTexture"] = {"index": texture}
    if emissive:
        m["emissiveFactor"] = lin(emissive)
    if alpha is not None:
        m["alphaMode"] = "BLEND"
    if double:
        m["doubleSided"] = True
    return m


def _oct_ring(ax, az, c, y):
    return [(-ax + c, y, az), (ax - c, y, az), (ax, y, az - c), (ax, y, -az + c),
            (ax - c, y, -az), (-ax + c, y, -az), (-ax, y, -az + c), (-ax, y, az - c)]


def add_tee(mesh, m_print, m_top, m_hem):
    """Flag tee over the lower half of the block: octagonal walls, sloped
    shoulders and a small hem bevel, all carrying the wrap-around print
    (u = position round the body from the front-left edge, v = height).
    Built in torso-local space (y relative to HIP_Y)."""
    S, _, y_top, H = tee_layout()
    P = S[8]
    hx, hz = SHIRT["w"] / 2, SHIRT["d"] / 2
    cv, bt, bb = SHIRT["corner"], SHIRT["shoulder"], SHIRT["hem"]
    y0, y1 = SHIRT["y0"], SHIRT["y0"] + SHIRT["h"]
    k = 2 - math.sqrt(2)   # keeps the corner chamfers parallel on the inset rings
    rings = [_oct_ring(hx - bb, hz - bb, cv - k * bb, y0),
             _oct_ring(hx, hz, cv, y0 + bb),
             _oct_ring(hx, hz, cv, y1 - bt),
             _oct_ring(hx - bt, hz - bt, max(cv - k * bt, 0.004), y1)]

    def loc(p):
        return (p[0], p[1] - HIP_Y, p[2])

    for r in range(3):
        A, B = rings[r], rings[r + 1]
        for e in range(8):
            e1 = (e + 1) % 8
            pts = [A[e], A[e1], B[e1], B[e]]
            us = (S[e] / P, S[e + 1] / P, S[e + 1] / P, S[e] / P)
            uvs = [(u, (y_top - p[1]) / H) for u, p in zip(us, pts)]
            n = v_unit(v_cross(v_sub(pts[1], pts[0]), v_sub(pts[3], pts[0])))
            cx, cz = sum(p[0] for p in pts) / 4, sum(p[2] for p in pts) / 4
            if n[0] * cx + n[2] * cz < 0:
                n = (-n[0], -n[1], -n[2])
            mesh.poly(m_print, [loc(p) for p in pts], n, uvs)
    mesh.poly(m_top, [loc(p) for p in rings[3]], (0.0, 1.0, 0.0))
    mesh.poly(m_hem, [loc(p) for p in rings[0]], (0.0, -1.0, 0.0))


def crate_mesh(M, k=1.0):
    """The glowing data crate, centred on its node. k scales it (1.0 on the bots).
    M(name, hex, **kw) registers a material and returns its name."""
    crate = Mesh("crate")
    W, Hh, D, t = CRATE["w"] * k, CRATE["h"] * k, CRATE["d"] * k, CRATE["bar"] * k
    add_box(crate, M("crate_core", COL["crate_core"], rough=0.4, emissive=COL["crate_glow"]),
            (0, 0, 0), (W - 0.02 * k, Hh - 0.02 * k, D - 0.02 * k), bevel=0.01 * k)
    m_fr = M("crate_frame", COL["crate_frame"], rough=0.6)
    for a in (1, -1):
        for b in (1, -1):
            add_box(crate, m_fr, (0, a * (Hh - t) / 2, b * (D - t) / 2), (W, t, t))
            add_box(crate, m_fr, (a * (W - t) / 2, 0, b * (D - t) / 2), (t, Hh - 2 * t, t))
            add_box(crate, m_fr, (a * (W - t) / 2, b * (Hh - t) / 2, 0), (t, t, D - 2 * t))
    for yb in (-Hh / 6, Hh / 6):   # two slats around the sides: reads as a crate, not a screen
        add_box(crate, m_fr, (0, yb, 0), (W - 0.004 * k, 0.028 * k, D - 0.004 * k))
    return crate


class Node:
    def __init__(self, name, t=(0.0, 0.0, 0.0), r=(0.0, 0.0, 0.0, 1.0), s=(1.0, 1.0, 1.0), mesh=None):
        self.name, self.t, self.r, self.s, self.mesh = name, t, r, s, mesh
        self.children = []

    def add(self, child):
        self.children.append(child)
        return child

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


def build_bot(code, carry, ident=None, pants=None, quirk=None, lift=0.0):
    """code: flag (IN/US/DE/AE) or astro. ident/pants/quirk make a per-character bot
    (see build_crew.py): quirk(M, root, nodes, carry) adds accessory nodes and may
    return {clip name: channels} overrides (e.g. a lying-down idle). lift raises the
    carried crate (and stretches the raised arms) so it can sit on top of a hat."""
    astro = code == "astro"
    name = "bot_%s%s" % (ident or code, "_carry" if carry else "")
    mats = {}

    def M(key, hexcol, **kw):
        if key not in mats:
            mats[key] = material(key, hexcol, **kw)
        return key

    m_body = M("body", COL["body"])
    m_eye = M("eye", COL["eye"], rough=0.45)
    textures = []
    if astro:
        m_shirt = M("suit", COL["suit"])
    else:
        sol = TEE_SOLIDS[code]
        m_tee = "tee_flag_%s" % code
        mats[m_tee] = material(m_tee, "#FFFFFF", texture=0)
        textures.append(("tee_flag_%s" % code, render_tee_png(code)))
        m_shirt = M("tee_sleeve", sol["sleeve"])          # sleeves: a matching flag colour
        m_tee_top = M("tee_shoulder", sol["top"])
        m_tee_hem = M("tee_hem", sol["hem"])
    m_pants = M("pants", pants or PANTS[code])
    m_shoe = M("boot" if astro else "shoe", COL["boot"] if astro else COL["shoe"])
    m_sole = M("sole", COL["sole"])

    # --- legs (pants + shoe + sole), shared by both legs
    leg = Mesh("leg")
    add_box(leg, m_pants, (0, -0.13, 0), (0.13, 0.22, 0.14), bevel=0.02)
    add_box(leg, m_shoe, (0, -0.25, 0.025), (0.15, 0.07, 0.20), bevel=0.025)
    add_box(leg, m_sole, (0, -0.29, 0.025), (0.155, 0.02, 0.205))

    hips_mesh = Mesh("hips")
    add_box(hips_mesh, m_pants, (0, 0, 0), (0.54, 0.10, 0.38), bevel=0.03)

    # --- torso: orange block + tee (flag all round) or astro suit
    torso_mesh = Mesh("torso")
    by = BODY["y0"] + BODY["h"] / 2 - HIP_Y
    add_box(torso_mesh, m_body, (0, by, 0), (BODY["w"], BODY["h"], BODY["d"]), bevel=BODY["bevel"])
    sy = SHIRT["y0"] + SHIRT["h"] / 2 - HIP_Y
    if astro:
        add_box(torso_mesh, m_shirt, (0, sy, 0), (SHIRT["w"], SHIRT["h"], SHIRT["d"]), bevel=SHIRT["bevel"])
    else:
        add_tee(torso_mesh, m_tee, m_tee_top, m_tee_hem)
    fz = SHIRT["d"] / 2
    if astro:
        m_trim = M("suit_trim", COL["suit_trim"])
        m_panel = M("panel", COL["panel"])
        # neck ring the helmet sits in
        add_cyl(torso_mesh, m_trim, (0, HELMET["cut"] - HIP_Y - 0.005, 0), 0.40, 0.40, 0.05, 24)
        # chest control box with three buttons
        add_box(torso_mesh, m_panel, (0, 0.50 - HIP_Y, fz + 0.012), (0.20, 0.10, 0.03), bevel=0.01)
        for k, (bx, bc) in enumerate(((-0.055, "btn_r"), (0.0, "btn_y"), (0.055, "btn_b"))):
            add_box(torso_mesh, M(bc, COL[bc], rough=0.5), (bx, 0.50 - HIP_Y, fz + 0.03), (0.032, 0.032, 0.02))

    eyes_mesh = Mesh("eyes")
    for s in (1, -1):
        add_box(eyes_mesh, m_eye, (s * 0.13, 0, BODY["d"] / 2 + 0.005), (0.075, 0.145, 0.03), bevel=0.012)

    # --- arms
    def arm_mesh(s):
        m = Mesh("arm_%s" % ("L" if s > 0 else "R"))
        skin = m_shirt if astro else m_body   # astro arms are suit sleeves
        if not carry:
            add_box(m, m_shirt, (s * 0.035, -0.02, 0), (0.105, 0.12, 0.17), bevel=0.025)
            add_box(m, skin, (s * 0.04, -0.13, 0), (0.085, 0.16, 0.10), bevel=0.025)
            if astro:
                add_box(m, M("glove", COL["glove"]), (s * 0.04, -0.19, 0), (0.095, 0.07, 0.11), bevel=0.02)
        elif not astro:
            add_box(m, m_shirt, (s * 0.035, 0.0, 0), (0.105, 0.12, 0.17), bevel=0.025)
            add_box(m, m_body, (s * 0.045, 0.24 + lift / 2, 0), (0.09, 0.40 + lift, 0.105), bevel=0.025)
            add_box(m, m_body, (s * -0.06, 0.47 + lift, 0), (0.15, 0.10, 0.12), bevel=0.03)   # hand grips crate side
        else:
            # goal-post arms that clear the round helmet
            add_box(m, m_shirt, (s * 0.035, 0.0, 0), (0.105, 0.12, 0.17), bevel=0.025)
            m_trim = M("suit_trim", COL["suit_trim"])
            add_box(m, m_shirt, (s * 0.085, 0.05, 0), (0.13, 0.11, 0.12))
            add_box(m, m_shirt, (s * 0.12, 0.40, 0), (0.105, 0.64, 0.12), bevel=0.03)
            add_box(m, m_trim, (s * 0.12, 0.36, 0), (0.12, 0.07, 0.135))       # elbow joint
            add_box(m, m_shirt, (s * 0.035, 0.72, 0), (0.26, 0.11, 0.12), bevel=0.03)
            add_box(m, M("glove", COL["glove"]), (s * -0.095, 0.72, 0), (0.08, 0.12, 0.13))
        return m

    # --- hierarchy
    root = Node(name)
    hips = root.add(Node("hips", t=(0.0, HIP_Y, 0.0), mesh=hips_mesh))
    hips.add(Node("leg_L", t=(LEG_X, 0.0, 0.0), mesh=leg))
    hips.add(Node("leg_R", t=(-LEG_X, 0.0, 0.0), mesh=leg))
    torso = hips.add(Node("torso", mesh=torso_mesh))
    torso.add(Node("eyes", t=(0.0, EYE_Y - HIP_Y, 0.0), mesh=eyes_mesh))
    for s, side in ((1, "L"), (-1, "R")):
        rest = (0.0, 0.0, 0.0, 1.0) if carry else q_axis(Z_AX, s * ARM_TILT)
        torso.add(Node("arm_" + side, t=(s * ARM_X, ARM_Y - HIP_Y, 0.0), r=rest, mesh=arm_mesh(s)))

    if astro:
        helmet = Mesh("helmet")
        m_glass = M("glass", COL["glass"], rough=0.06, alpha=0.24)
        th = math.degrees(math.acos((HELMET["cut"] - HELMET["cy"]) / HELMET["R"]))
        add_dome(helmet, m_glass, (0, HELMET["cy"] - HIP_Y, 0), HELMET["R"], 20, 10, th)
        torso.add(Node("helmet", mesh=helmet))

        pack = Mesh("jetpack")
        pz = -(SHIRT["d"] / 2 + 0.06)
        add_box(pack, M("pack", COL["pack"]), (0, 0.47 - HIP_Y, pz), (0.36, 0.30, 0.12), bevel=0.03)
        m_tank, m_noz = M("tank", COL["tank"]), M("nozzle", COL["nozzle"])
        tz = pz - 0.09
        for s in (1, -1):
            add_cyl(pack, m_tank, (s * 0.095, 0.465 - HIP_Y, tz), 0.065, 0.065, 0.29, 10)
            add_cyl(pack, m_tank, (s * 0.095, 0.62 - HIP_Y, tz), 0.065, 0.035, 0.03, 10, caps=(False, True))
            add_cyl(pack, m_noz, (s * 0.095, 0.295 - HIP_Y, tz), 0.052, 0.040, 0.05, 10)
        jet = torso.add(Node("jetpack", mesh=pack))
        flame = Mesh("flame")
        add_cyl(flame, M("flame", COL["flame"], rough=1.0, emissive=COL["flame_glow"]),
                (0, -0.08, 0), 0.0, 0.042, 0.16, 8, caps=(False, False))   # tip points down
        for s, side in ((1, "L"), (-1, "R")):
            jet.add(Node("flame_" + side, t=(s * 0.095, 0.27 - HIP_Y, tz),
                         r=(0.0, 0.0, 0.0, 1.0), mesh=flame))

    if carry:
        bottom = CRATE_BOTTOM["astro" if astro else "bot"] + lift
        torso.add(Node("crate", t=(0.0, bottom + CRATE["h"] / 2 - HIP_Y, 0.0), mesh=crate_mesh(M)))

    overrides = quirk(M, root, {n.name: n for n in root.walk()}, carry) if quirk else None
    anims = build_anims(root, carry, astro, lift)
    if overrides:
        anims.update(overrides)
        anims = fill_coverage(root, anims)
    return name, root, mats, textures, anims

# ------------------------------------------------------------------ animation

def _times(T, N):
    return [round(T * i / N, 6) for i in range(N + 1)]


def build_anims(root, carry, astro, lift=0.0):
    nodes = {n.name: n for n in root.walk()}
    out = {}

    # ---------------- run: one full stride (two footfalls) per loop
    T, N = RUN_T, RUN_N
    times = _times(T, N)
    ch = {}

    def put(node, path, v):
        ch.setdefault((node, path), []).append(tuple(round(x, 6) for x in v))

    for i in range(N + 1):
        p = 2 * math.pi * (i % N) / N           # i == N reproduces i == 0 exactly
        sp, cp = math.sin(p), math.cos(p)
        swing = 36.0 * sp                       # + = left foot back
        put("leg_L", "rotation", q_axis(X_AX, swing))
        put("leg_R", "rotation", q_axis(X_AX, -swing))
        put("leg_L", "translation", (LEG_X, 0.035 * max(0.0, -cp), 0.0))   # lift the swinging leg
        put("leg_R", "translation", (-LEG_X, 0.035 * max(0.0, cp), 0.0))
        put("hips", "translation", (0.0, HIP_Y + 0.05 * abs(sp), 0.0))     # bounce, lowest at mid-stance
        put("torso", "rotation", q_ypr(yaw=-5.0 * sp, pitch=9.0 + 2.5 * math.cos(2 * p), roll=-4.0 * cp))
        syy = 1.0 - 0.06 * cp ** 8 + 0.03 * sp * sp                         # squash on contact, stretch in flight
        sxz = 1.0 / math.sqrt(syy)
        put("torso", "scale", (sxz, syy, sxz))
        put("eyes", "scale", (1.0, 1.0, 1.0))
        if not carry:
            put("arm_L", "rotation", q_mul(q_axis(Z_AX, ARM_TILT), q_axis(X_AX, -10.0 - 55.0 * sp)))
            put("arm_R", "rotation", q_mul(q_axis(Z_AX, -ARM_TILT), q_axis(X_AX, -10.0 + 55.0 * sp)))
        if astro:
            fy = 1.0 + 0.22 * math.sin(2 * p) + 0.10 * math.sin(6 * p + 1.0)
            fx = 1.0 + 0.08 * math.sin(4 * p)
            put("flame_L", "scale", (fx, fy, fx))
            fy2 = 1.0 + 0.22 * math.sin(2 * p + 1.3) + 0.10 * math.sin(6 * p)
            put("flame_R", "scale", (fx, fy2, fx))
    out["run"] = [(n, path, times, vals) for (n, path), vals in ch.items()]

    # ---------------- idle: breathing, tiny sway, one blink
    T, N = IDLE_T, IDLE_N
    times = _times(T, N)
    ch = {}
    for i in range(N + 1):
        p = 2 * math.pi * (i % N) / N
        sp, cp = math.sin(p), math.cos(p)
        put("leg_L", "rotation", (0.0, 0.0, 0.0, 1.0))
        put("leg_R", "rotation", (0.0, 0.0, 0.0, 1.0))
        put("leg_L", "translation", (LEG_X, 0.0, 0.0))
        put("leg_R", "translation", (-LEG_X, 0.0, 0.0))
        put("hips", "translation", (0.0, HIP_Y - 0.004 * (1 - cp) / 2, 0.0))
        put("torso", "rotation", q_ypr(yaw=2.0 * math.sin(p + 0.6), pitch=1.0 * sp, roll=1.6 * math.sin(p + 1.2)))
        syy = 1.0 + 0.022 * sp
        sxz = 1.0 - 0.011 * sp
        put("torso", "scale", (sxz, syy, sxz))
        if not carry:
            put("arm_L", "rotation", q_mul(q_axis(Z_AX, ARM_TILT + 4.0 * sp), q_axis(X_AX, -3.0 * cp)))
            put("arm_R", "rotation", q_mul(q_axis(Z_AX, -ARM_TILT - 4.0 * sp), q_axis(X_AX, -3.0 * cp)))
        if astro:
            f = 0.45 + 0.08 * math.sin(4 * p) + 0.04 * math.sin(10 * p)
            put("flame_L", "scale", (0.8, f, 0.8))
            f2 = 0.45 + 0.08 * math.sin(4 * p + 1.1) + 0.04 * math.sin(10 * p + 0.5)
            put("flame_R", "scale", (0.8, f2, 0.8))
    chans = [(n, path, times, vals) for (n, path), vals in ch.items()]
    blink_t = [0.0, 1.30, 1.36, 1.40, 1.46, 2.0]
    blink_v = [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.08, 1.0), (1.0, 0.08, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]
    chans.append(("eyes", "scale", blink_t, blink_v))
    out["idle"] = chans

    out.update(action_clips(carry, astro, lift))
    out = fill_coverage(root, out)
    for clip in out.values():
        for n, _, _, _ in clip:
            assert n in nodes, n
    return out


# ------------------------------------------------------------------ action clips
# crouch / throw / catch / jump / cheer. Legs have no knees, so a knee bend is the
# hips dropping while both legs scale down in y from the hip pivot (feet stay on the
# ground). Carry bots keep the crate in their raised hands; in "throw" they fling it
# forward (it flies off and shrinks away at the very end so the app can hand over to
# a free-flying data_crate), and in "catch" it flies in from the front into the hands.

ACTION_TIMES = {"crouch": (0.25, 6), "throw": (0.5, 12), "catch": (0.5, 12), "jump": (0.6, 15), "cheer": (1.0, 20)}
ONE_SHOT = ("crouch", "throw", "catch")     # not loops: may end in a different pose than they start


def _ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def kf(t, keys):
    """Smoothstep between (time, value) keys."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * _ease((t - t0) / (t1 - t0))
    return keys[-1][1]


def sample_clip(T, N, fn, loop=True):
    """fn(t) -> {(node, path): value}. loop=True: the last key repeats the first exactly."""
    times = [round(T * i / N, 6) for i in range(N + 1)]
    ch = {}
    for i in range(N + 1):
        t = T * ((i % N) if loop else i) / N
        for key, v in fn(t).items():
            ch.setdefault(key, []).append(tuple(round(x, 6) for x in v))
    return [(n, path, times, vals) for (n, path), vals in ch.items()]


def bot_pose(carry, astro, dy=0.0, leg_rx=0.0, yaw=0.0, pitch=0.0, roll=0.0, sy=1.0, eye=1.0,
             arm_l=(0.0, 0.0), arm_r=(0.0, 0.0), carry_a=0.0, crate=None, flame=0.6, lift=0.0):
    """One frame of a bot pose. arm_l / arm_r: (raise about Z beyond the rest tilt, pitch about X);
    for carry bots both arms pitch together by carry_a. crate: (pos, rot quat, scale) or None = in hands."""
    leg_s = (HIP_Y + dy) / HIP_Y if dy < 0 else 1.0
    sxz = 1.0 / math.sqrt(sy)
    d = {("hips", "translation"): (0.0, HIP_Y + dy, 0.0),
         ("leg_L", "rotation"): q_axis(X_AX, leg_rx), ("leg_R", "rotation"): q_axis(X_AX, leg_rx),
         ("leg_L", "translation"): (LEG_X, 0.0, 0.0), ("leg_R", "translation"): (-LEG_X, 0.0, 0.0),
         ("leg_L", "scale"): (1.0, leg_s, 1.0), ("leg_R", "scale"): (1.0, leg_s, 1.0),
         ("torso", "rotation"): q_ypr(yaw=yaw, pitch=pitch, roll=roll),
         ("torso", "scale"): (sxz, sy, sxz),
         ("eyes", "scale"): (1.0, eye, 1.0)}
    if carry:
        q = q_axis(X_AX, carry_a)
        d[("arm_L", "rotation")] = q
        d[("arm_R", "rotation")] = q
        if crate is None:
            crate = (crate_hold(astro, carry_a, lift), q, 1.0)
        d[("crate", "translation")] = crate[0]
        d[("crate", "rotation")] = crate[1]
        d[("crate", "scale")] = (crate[2],) * 3
    else:
        d[("arm_L", "rotation")] = q_mul(q_axis(Z_AX, ARM_TILT + arm_l[0]), q_axis(X_AX, arm_l[1]))
        d[("arm_R", "rotation")] = q_mul(q_axis(Z_AX, -ARM_TILT - arm_r[0]), q_axis(X_AX, arm_r[1]))
    if astro:
        d[("flame_L", "scale")] = (0.8 + 0.2 * flame, flame, 0.8 + 0.2 * flame)
        d[("flame_R", "scale")] = (0.8 + 0.2 * flame, flame, 0.8 + 0.2 * flame)
    return d


def crate_hold(astro, a, lift=0.0):
    """Torso-local crate centre when the raised arms are pitched by a degrees about the shoulders."""
    piv = ARM_Y - HIP_Y
    r = CRATE_BOTTOM["astro" if astro else "bot"] + lift + CRATE["h"] / 2 - HIP_Y - piv
    ra = math.radians(a)
    return (0.0, piv + r * math.cos(ra), r * math.sin(ra))


def action_clips(carry, astro, lift=0.0):
    P = lambda **kw: bot_pose(carry, astro, lift=lift, **kw)   # noqa: E731
    out = {}

    def crouch(t):
        u = _ease(t / 0.25)
        return P(dy=-0.05 * u, pitch=16.0 * u, sy=1.0 - 0.08 * u, eye=1.0 - 0.25 * u,
                 arm_l=(4.0 * u, 35.0 * u), arm_r=(4.0 * u, 35.0 * u), carry_a=-8.0 * u, flame=0.6 + 0.4 * u)

    def throw(t):
        pitch = kf(t, [(0.0, 0.0), (0.16, -8.0), (0.28, 12.0), (0.5, 0.0)])
        dy = kf(t, [(0.0, 0.0), (0.16, -0.03), (0.28, 0.02), (0.42, 0.0)])
        if carry:
            a = kf(t, [(0.0, 0.0), (0.15, -25.0), (0.3, 75.0), (0.4, 30.0), (0.5, 0.0)])
            tr = 0.24
            crate = None
            if t > tr:
                a_r = kf(tr, [(0.0, 0.0), (0.15, -25.0), (0.3, 75.0), (0.4, 30.0), (0.5, 0.0)])
                p0 = crate_hold(astro, a_r, lift)
                dt = t - tr
                pos = (0.0, p0[1] + 2.2 * dt - 3.5 * dt * dt, p0[2] + 3.6 * dt)
                rot = q_axis(X_AX, a_r + 540.0 * dt)
                crate = (pos, rot, 1.0 - _ease((t - 0.42) / 0.08))
            return P(dy=dy, pitch=pitch, carry_a=a, crate=crate, eye=0.9)
        x = kf(t, [(0.0, 0.0), (0.16, 45.0), (0.3, -130.0), (0.38, -110.0), (0.5, 0.0)])
        return P(dy=dy, pitch=pitch, eye=kf(t, [(0.0, 1.0), (0.25, 0.8), (0.5, 1.0)]),
                 arm_l=(6.0, x), arm_r=(6.0, x))

    def catch(t):
        dy = kf(t, [(0.0, 0.0), (0.2, 0.0), (0.28, -0.05), (0.5, 0.0)])
        sy = kf(t, [(0.0, 1.0), (0.2, 1.0), (0.28, 0.93), (0.5, 1.0)])
        pitch = kf(t, [(0.0, 0.0), (0.2, -4.0), (0.3, 6.0), (0.5, 0.0)])
        eye = kf(t, [(0.0, 1.0), (0.15, 1.15), (0.3, 0.8), (0.5, 1.0)])
        if carry:
            a = kf(t, [(0.0, 60.0), (0.2, 30.0), (0.45, 0.0), (0.5, 0.0)])
            crate = None
            if t < 0.2:
                hc = crate_hold(astro, 30.0, lift)
                dt = t - 0.2
                pos = (0.0, hc[1] - 1.0 * dt - 3.0 * dt * dt, hc[2] - 3.2 * dt)
                crate = (pos, q_axis(X_AX, 30.0 + 400.0 * dt), 1.0)
            return P(dy=dy, sy=sy, pitch=pitch, eye=eye, carry_a=a, crate=crate)
        x = kf(t, [(0.0, 0.0), (0.15, -150.0), (0.3, -125.0), (0.5, 0.0)])
        return P(dy=dy, sy=sy, pitch=pitch, eye=eye, arm_l=(10.0, x), arm_r=(10.0, x))

    def jump(t):
        if t < 0.12:
            u = _ease(t / 0.12)
            dy, sy, rx = -0.05 * u, 1.0 - 0.08 * u, 0.0
        elif t < 0.42:
            u = (t - 0.12) / 0.30
            dy, sy, rx = 0.35 * 4 * u * (1 - u), 1.0 + 0.06 * math.sin(math.pi * u), 14.0 * math.sin(math.pi * u)
        elif t < 0.5:
            u = math.sin(math.pi * (t - 0.42) / 0.16)
            dy, sy, rx = -0.06 * u, 1.0 - 0.12 * u, 0.0
        else:
            u = _ease((t - 0.5) / 0.1)
            dy, sy, rx = -0.06 * (1 - u) * 0.0, 1.0, 0.0
        x = kf(t, [(0.0, 0.0), (0.12, 30.0), (0.22, -150.0), (0.42, -120.0), (0.5, 0.0)])
        a = kf(t, [(0.0, 0.0), (0.12, -10.0), (0.25, 10.0), (0.42, 0.0)])
        boost = kf(t, [(0.0, 0.6), (0.12, 0.8), (0.18, 1.6), (0.32, 1.0), (0.45, 0.6)])
        return P(dy=dy, sy=sy, leg_rx=rx, pitch=kf(t, [(0.0, 0.0), (0.12, 10.0), (0.25, -4.0), (0.46, 8.0), (0.6, 0.0)]),
                 eye=kf(t, [(0.0, 1.0), (0.12, 0.8), (0.25, 1.15), (0.46, 0.7), (0.6, 1.0)]),
                 arm_l=(8.0, x), arm_r=(8.0, x), carry_a=a, flame=boost)

    def cheer(t):
        w = 2 * math.pi * t
        hop = abs(math.sin(w))                       # two hops a second
        dy = 0.06 * hop - 0.03 * max(0.0, 1.0 - 4 * hop) ** 2
        wave = math.sin(2 * w)
        return P(dy=dy, sy=1.0 - 0.06 * max(0.0, 1.0 - 4 * hop) + 0.03 * hop, roll=5.0 * math.sin(w),
                 yaw=4.0 * math.sin(w + 0.7), eye=0.45,
                 arm_l=(132.0 + 14.0 * wave, -10.0), arm_r=(132.0 - 14.0 * wave, -10.0),
                 carry_a=8.0 * math.sin(2 * w), flame=0.9 + 0.4 * hop)

    for name, fn in (("crouch", crouch), ("throw", throw), ("catch", catch), ("jump", jump), ("cheer", cheer)):
        T, N = ACTION_TIMES[name]
        loop = not (name in ONE_SHOT and (carry or name == "crouch"))
        out[name] = sample_clip(T, N, fn, loop=loop)
    return out


def fill_coverage(root, clips):
    """Give every clip the same (node, path) set: a channel a clip lacks is held at the
    node's rest value. Existing channels are untouched, so run / idle move exactly as before."""
    nodes = {n.name: n for n in root.walk()}
    want = []
    for clip in clips.values():
        for n, path, _, _ in clip:
            if (n, path) not in want:
                want.append((n, path))
    out = {}
    for name, clip in clips.items():
        have = {(n, path) for n, path, _, _ in clip}
        dur = max(ts[-1] for _, _, ts, _ in clip)
        extra = []
        for n, path in want:
            if (n, path) not in have:
                node = nodes[n]
                v = {"translation": node.t, "rotation": node.r, "scale": node.s}[path]
                extra.append((n, path, [0.0, dur], [tuple(v), tuple(v)]))
        out[name] = list(clip) + extra
    return out

# ------------------------------------------------------------------ glTF writer

def f32s(x):
    """Shortest decimal that reads back as the same float32 (keeps the JSON small)."""
    for p in range(1, 10):
        v = float("%.*g" % (p, x))
        if struct.unpack("<f", struct.pack("<f", v))[0] == x:
            return v
    return x


class Writer:
    def __init__(self):
        self.bin = bytearray()
        self.views, self.accessors = [], []
        self._time_cache = {}
        self.anim = bytearray()          # animation keys share one bufferView
        self._anim_acc = []

    def anim_floats(self, rows, typ, minmax=False):
        nc = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[typ]
        flat = []
        for r in rows:
            if nc == 1:
                flat.append(r)
            else:
                flat.extend(r)
        data = struct.pack("<%df" % len(flat), *flat)
        f32 = struct.unpack("<%df" % len(flat), data)
        a = {"bufferView": -1, "byteOffset": len(self.anim), "componentType": 5126, "count": len(rows), "type": typ}
        if minmax:
            a["min"] = [f32s(min(f32[k::nc])) for k in range(nc)]
            a["max"] = [f32s(max(f32[k::nc])) for k in range(nc)]
        self.anim += data
        self.accessors.append(a)
        self._anim_acc.append(a)
        return len(self.accessors) - 1

    def close_anim(self):
        if self.anim:
            v = self.view(bytes(self.anim))
            for a in self._anim_acc:
                a["bufferView"] = v

    def view(self, data, target=None):
        while len(self.bin) % 4:
            self.bin.append(0)
        v = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target is not None:
            v["target"] = target
        self.bin += data
        self.views.append(v)
        return len(self.views) - 1

    def floats(self, rows, typ, target=None, minmax=False):
        nc = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[typ]
        flat = []
        for r in rows:
            if nc == 1:
                flat.append(r)
            else:
                flat.extend(r)
        data = struct.pack("<%df" % len(flat), *flat)
        f32 = struct.unpack("<%df" % len(flat), data)
        a = {"bufferView": self.view(data, target), "componentType": 5126, "count": len(rows), "type": typ}
        if minmax:
            a["min"] = [f32s(min(f32[k::nc])) for k in range(nc)]
            a["max"] = [f32s(max(f32[k::nc])) for k in range(nc)]
        self.accessors.append(a)
        return len(self.accessors) - 1

    def indices(self, idx):
        data = struct.pack("<%dH" % len(idx), *idx)
        self.accessors.append({"bufferView": self.view(data, 34963), "componentType": 5123,
                               "count": len(idx), "type": "SCALAR",
                               "min": [min(idx)], "max": [max(idx)]})
        return len(self.accessors) - 1

    def times(self, ts):
        key = tuple(ts)
        if key not in self._time_cache:
            self._time_cache[key] = self.anim_floats(list(ts), "SCALAR", minmax=True)
        return self._time_cache[key]


def write_glb(path, name, root, mats, textures, anims):
    """textures: list of PNG bytes or (image name, PNG bytes); anims: {clip name: channels} in output order."""
    w = Writer()
    gl = {"asset": {"version": "2.0", "generator": "sgm-world build_bots.py"},
          "scene": 0, "scenes": [{"name": name, "nodes": [0]}],
          "nodes": [], "meshes": [], "materials": []}
    mat_index = {}
    for key, m in mats.items():
        mat_index[key] = len(gl["materials"])
        gl["materials"].append(m)

    mesh_index = {}
    node_index = {}
    order = list(root.walk())
    for i, n in enumerate(order):
        node_index[n.name] = i
    for n in order:
        nd = {"name": n.name}
        if n.children:
            nd["children"] = [node_index[c.name] for c in n.children]
        if any(abs(v) > 1e-9 for v in n.t):
            nd["translation"] = [round(v, 6) for v in n.t]
        if abs(n.r[3] - 1.0) > 1e-9:
            nd["rotation"] = [round(v, 7) for v in n.r]
        if any(abs(v - 1.0) > 1e-9 for v in n.s):
            nd["scale"] = [round(v, 6) for v in n.s]
        if n.mesh is not None:
            if id(n.mesh) not in mesh_index:
                prims = []
                for p in n.mesh.prims:
                    attrs = {"POSITION": w.floats(p.pos, "VEC3", 34962, minmax=True),
                             "NORMAL": w.floats(p.nrm, "VEC3", 34962)}
                    if p.uv is not None:
                        attrs["TEXCOORD_0"] = w.floats(p.uv, "VEC2", 34962)
                    prims.append({"attributes": attrs, "indices": w.indices(p.idx),
                                  "material": mat_index[p.mat], "mode": 4})
                mesh_index[id(n.mesh)] = len(gl["meshes"])
                gl["meshes"].append({"name": n.mesh.name, "primitives": prims})
            nd["mesh"] = mesh_index[id(n.mesh)]
        gl["nodes"].append(nd)

    if textures:
        gl["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}]
        gl["images"], gl["textures"] = [], []
        for k, png in enumerate(textures):
            iname = "shirt_print"
            if isinstance(png, tuple):
                iname, png = png
            gl["images"].append({"name": iname, "bufferView": w.view(png), "mimeType": "image/png"})
            gl["textures"].append({"sampler": 0, "source": k})

    gl["animations"] = []
    for aname in anims:
        samplers, channels = [], []
        for node, tpath, ts, vals in anims[aname]:
            if tpath == "rotation":
                fixed, prev = [], None
                for q in vals:
                    if prev is not None and v_dot(q[:3], prev[:3]) + q[3] * prev[3] < 0:
                        q = tuple(-x for x in q)
                    fixed.append(q)
                    prev = q
                vals, typ = fixed, "VEC4"
            else:
                typ = "VEC3"
            samplers.append({"input": w.times(ts), "output": w.anim_floats(vals, typ), "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers) - 1, "target": {"node": node_index[node], "path": tpath}})
        gl["animations"].append({"name": aname, "channels": channels, "samplers": samplers})

    w.close_anim()
    while len(w.bin) % 4:
        w.bin.append(0)
    gl["bufferViews"] = w.views
    gl["accessors"] = w.accessors
    gl["buffers"] = [{"byteLength": len(w.bin)}]

    js = json.dumps(gl, separators=(",", ":")).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(w.bin)
    blob = (struct.pack("<4sII", b"glTF", 2, total)
            + struct.pack("<I4s", len(js), b"JSON") + js
            + struct.pack("<I4s", len(w.bin), b"BIN\x00") + bytes(w.bin))
    path.write_bytes(blob)
    return blob

# ------------------------------------------------------------------ validation

COMP_SIZE = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
TYPE_N = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _mat_trs(t, r, s):
    m = q_to_m(r)
    return [[m[0][0] * s[0], m[0][1] * s[1], m[0][2] * s[2], t[0]],
            [m[1][0] * s[0], m[1][1] * s[1], m[1][2] * s[2], t[1]],
            [m[2][0] * s[0], m[2][1] * s[1], m[2][2] * s[2], t[2]],
            [0, 0, 0, 1]]


def _mm(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def validate(path, required=("run", "idle"), max_bytes=None, max_tris=None, oneshot=(), ground=True):
    """Parse a .glb back and check structure. Returns a stats dict; raises on error."""
    max_bytes = MAX_BYTES if max_bytes is None else max_bytes
    max_tris = MAX_TRIS if max_tris is None else max_tris
    data = path.read_bytes()
    errs = []
    magic, ver, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or ver != 2:
        errs.append("bad header")
    if length != len(data):
        errs.append("header length %d != file %d" % (length, len(data)))
    jlen, jtype = struct.unpack_from("<I4s", data, 12)
    if jtype != b"JSON" or jlen % 4:
        errs.append("bad JSON chunk")
    gl = json.loads(data[20:20 + jlen].decode("utf-8"))
    off = 20 + jlen
    blen, btype = struct.unpack_from("<I4s", data, off)
    if btype != b"BIN\x00" or blen % 4:
        errs.append("bad BIN chunk")
    binbuf = data[off + 8:off + 8 + blen]
    if off + 8 + blen != len(data):
        errs.append("trailing bytes")
    if gl["buffers"][0]["byteLength"] > blen:
        errs.append("buffer larger than BIN chunk")
    for i, v in enumerate(gl["bufferViews"]):
        if v["byteOffset"] + v["byteLength"] > blen:
            errs.append("bufferView %d out of range" % i)
    acc_vals = []
    for i, a in enumerate(gl["accessors"]):
        v = gl["bufferViews"][a["bufferView"]]
        esz = COMP_SIZE[a["componentType"]] * TYPE_N[a["type"]]
        need = a.get("byteOffset", 0) + a["count"] * esz
        if need > v["byteLength"]:
            errs.append("accessor %d out of range" % i)
            acc_vals.append(None)
            continue
        fmt = {5126: "f", 5123: "H", 5125: "I"}[a["componentType"]]
        n = a["count"] * TYPE_N[a["type"]]
        acc_vals.append(struct.unpack_from("<%d%s" % (n, fmt), binbuf, v["byteOffset"] + a.get("byteOffset", 0)))
    tris = 0
    for m in gl["meshes"]:
        for p in m["primitives"]:
            cnt = gl["accessors"][p["attributes"]["POSITION"]]["count"]
            for k, ai in p["attributes"].items():
                if gl["accessors"][ai]["count"] != cnt:
                    errs.append("attribute count mismatch in %s" % m["name"])
            idx = acc_vals[p["indices"]]
            if idx and max(idx) >= cnt:
                errs.append("index out of range in %s" % m["name"])
            tris += len(idx) // 3
    for img in gl.get("images", []):
        v = gl["bufferViews"][img["bufferView"]]
        if binbuf[v["byteOffset"]:v["byteOffset"] + 8] != b"\x89PNG\r\n\x1a\n":
            errs.append("image is not PNG")
    names = [a["name"] for a in gl["animations"]]
    durations = {}
    for a in gl["animations"]:
        dur = 0.0
        for ch in a["channels"]:
            s = a["samplers"][ch["sampler"]]
            ts, vals = acc_vals[s["input"]], acc_vals[s["output"]]
            nc = 4 if ch["target"]["path"] == "rotation" else 3
            if ts[0] != 0.0:
                errs.append("%s: channel does not start at 0" % a["name"])
            if list(ts) != sorted(ts):
                errs.append("%s: times not increasing" % a["name"])
            first, last = vals[:nc], vals[-nc:]
            gap = max(abs(x - y) for x, y in zip(first, last))
            if nc == 4:
                gap = min(gap, max(abs(x + y) for x, y in zip(first, last)))
            if gap > 1e-5 and a["name"] not in oneshot:
                errs.append("%s: %s.%s not seamless" % (a["name"], gl["nodes"][ch["target"]["node"]]["name"],
                                                        ch["target"]["path"]))
            dur = max(dur, ts[-1])
        durations[a["name"]] = dur
    for want in required:
        if want not in names:
            errs.append("missing animation %s" % want)
    # same node/path coverage in every clip (clean switching in any engine)
    cover = [sorted((c["target"]["node"], c["target"]["path"]) for c in a["channels"]) for a in gl["animations"]]
    if any(c != cover[0] for c in cover):
        errs.append("clips animate different node/paths")
    # rest-pose bounding box
    lo, hi = [1e9] * 3, [-1e9] * 3

    def visit(ni, parent):
        nd = gl["nodes"][ni]
        mtx = _mm(parent, _mat_trs(nd.get("translation", [0, 0, 0]), nd.get("rotation", [0, 0, 0, 1]),
                                   nd.get("scale", [1, 1, 1])))
        if "mesh" in nd:
            for p in gl["meshes"][nd["mesh"]]["primitives"]:
                pv = acc_vals[p["attributes"]["POSITION"]]
                for k in range(0, len(pv), 3):
                    for ax in range(3):
                        val = mtx[ax][0] * pv[k] + mtx[ax][1] * pv[k + 1] + mtx[ax][2] * pv[k + 2] + mtx[ax][3]
                        lo[ax] = min(lo[ax], val)
                        hi[ax] = max(hi[ax], val)
        for c in nd.get("children", []):
            visit(c, mtx)

    ident = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    for r in gl["scenes"][gl["scene"]]["nodes"]:
        visit(r, ident)
    if len(data) > max_bytes:
        errs.append("file too big: %d bytes" % len(data))
    if tris > max_tris:
        errs.append("too many triangles: %d" % tris)
    if ground and abs(lo[1]) > 1e-3:
        errs.append("feet not on y=0 (min y %.4f)" % lo[1])
    if errs:
        raise SystemExit("%s: %s" % (path.name, "; ".join(errs)))
    return {"bytes": len(data), "tris": tris, "anims": durations, "bbox": (lo, hi),
            "nodes": len(gl["nodes"]), "materials": len(gl["materials"])}

# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="validate existing files only")
    ap.add_argument("--dump-textures", metavar="DIR", help="also write each tee PNG here")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    if not args.check:
        for code in VARIANTS:
            for carry in (False, True):
                name, root, mats, textures, anims = build_bot(code, carry)
                path = OUT_DIR / (name + ".glb")
                write_glb(path, name, root, mats, textures, anims)
                if args.dump_textures and textures and not carry:
                    d = Path(args.dump_textures)
                    d.mkdir(parents=True, exist_ok=True)
                    (d / ("tee_%s.png" % code)).write_bytes(textures[0][1])
                files.append(path)
    else:
        files = sorted(OUT_DIR.glob("bot_*.glb"))
    print("%-22s %8s %6s %6s  %-24s %s" % ("file", "bytes", "tris", "nodes", "animations (s)", "rest bbox (w x h x d)"))
    for f in files:
        st = validate(f, required=("run", "idle") + tuple(ACTION_TIMES), oneshot=ONE_SHOT)
        lo, hi = st["bbox"]
        an = ", ".join("%s %.2f" % kv for kv in st["anims"].items())
        print("%-22s %8d %6d %6d  %-24s %.2f x %.2f x %.2f (min y %.3f)" % (
            f.name, st["bytes"], st["tris"], st["nodes"], an,
            hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], lo[1]))
    print("ok: %d files valid" % len(files))


if __name__ == "__main__":
    main()
