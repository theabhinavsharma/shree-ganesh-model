#!/usr/bin/env python3
"""
SGM Data World - props generator.

Builds the toy-style props that share the world with the worker bots, as binary
glTF 2.0 (.glb) with node-TRS animations (no skinning, no extensions):

    tanker_oil.glb      crude oil tanker (Ship 1, oil), ~12 long        idle
    container_ship.glb  container ship (Ship 2), ~12 long               idle
    submarine.glb       cartoon yellow-orange submarine, ~8 long        idle
    bot_ML.glb          ML bot: bat-eared helmet, neural-net tee, hammer idle, hammer
    me_figure.glb       toy figure of the site owner, 1.0 tall          idle, wave
    laptop_giant.glb    open laptop with a stock chart on screen        idle
    rupee_note.glb      bundle of generic rupee notes plus a coin       idle
    data_crate.glb      the glowing data crate from the carry bots      idle

Pure Python 3 standard library; reuses the mesh, texture and glTF code in
build_bots.py next to this file. Deterministic output.

    python3 docs/sgm-world/bots/build_props.py            # build all props + validate
    python3 docs/sgm-world/bots/build_props.py --check    # validate existing files only

Conventions: +Y up, each prop faces +Z, origin at the centre of its base.
"""

import argparse
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True   # importing build_bots must not leave __pycache__ in the repo
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_bots as bb  # noqa: E402
from build_bots import (Mesh, Node, add_box, add_cyl, material, write_glb, validate, raster,  # noqa: E402
                        q_axis, q_mul, q_ypr, q_to_m, m_apply, v_unit, v_cross, v_sub, v_dot,
                        X_AX, Y_AX, Z_AX, hex_rgb)

OUT_DIR = bb.REPO / "shreeganeshmodel-deploy" / "assets" / "props"
MAX_BYTES = 150 * 1024
MAX_TRIS = 5000

# ------------------------------------------------------------------ helpers

class Mats:
    """Ordered material registry; callable like build_bots' M()."""

    def __init__(self):
        self.d = {}

    def __call__(self, key, hexcol, **kw):
        if key not in self.d:
            self.d[key] = material(key, hexcol, **kw)
        return key

    def raw(self, key, m):
        self.d[key] = dict(m, name=key)
        return key


def lcg(seed):
    """Tiny deterministic random stream in [0, 1)."""
    s = seed & 0x7FFFFFFF
    while True:
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        yield s / 0x80000000


def merge(dst, src, rot=None, off=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
    """Copy src's primitives into dst: scale, then rotate, then offset."""
    rm = q_to_m(rot) if rot else None
    for p in src.prims:
        q = dst.prim(p.mat, p.uv is not None)
        base = len(q.pos)
        for pos, n in zip(p.pos, p.nrm):
            ps = (pos[0] * scale[0], pos[1] * scale[1], pos[2] * scale[2])
            ns = v_unit((n[0] / scale[0], n[1] / scale[1], n[2] / scale[2]))
            if rm:
                ps, ns = m_apply(rm, ps), m_apply(rm, ns)
            q.pos.append((ps[0] + off[0], ps[1] + off[1], ps[2] + off[2]))
            q.nrm.append(ns)
        if q.uv is not None:
            q.uv.extend(p.uv)
        q.idx.extend(i + base for i in p.idx)


def q_from_to(a, b):
    a, b = v_unit(a), v_unit(b)
    d = v_dot(a, b)
    if d > 0.999999:
        return (0.0, 0.0, 0.0, 1.0)
    if d < -0.999999:
        ax = v_unit(v_cross(a, (1, 0, 0) if abs(a[0]) < 0.9 else (0, 1, 0)))
        return (ax[0], ax[1], ax[2], 0.0)
    c = v_cross(a, b)
    q = (c[0], c[1], c[2], 1.0 + d)
    n = math.sqrt(sum(x * x for x in q))
    return tuple(x / n for x in q)


def add_rod(mesh, mat, p0, p1, r, seg=8, r1=None, caps=(True, True)):
    """Cylinder (or frustum) from p0 to p1."""
    d = v_sub(p1, p0)
    L = math.sqrt(v_dot(d, d))
    tmp = Mesh("tmp")
    add_cyl(tmp, mat, (0, 0, 0), r, r if r1 is None else r1, L, seg, caps=caps)
    mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2)
    merge(mesh, tmp, rot=q_from_to(Y_AX, d), off=mid)


def add_beam(mesh, mat, p0, p1, w, h, bevel=0.0):
    """Box whose long (local Z) axis runs from p0 to p1."""
    d = v_sub(p1, p0)
    L = math.sqrt(v_dot(d, d))
    mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2)
    add_box(mesh, mat, mid, (w, h, L), bevel=bevel, rot=q_from_to(Z_AX, d))


def add_box_faces(mesh, mat, center, size, which):
    """Plain box emitting only the listed faces ('+x', '-x', '+y', '-y', '+z', '-z')."""
    H = (size[0] / 2, size[1] / 2, size[2] / 2)
    for tag in which:
        ax = "xyz".index(tag[1])
        s = 1.0 if tag[0] == "+" else -1.0
        o1, o2 = [k for k in range(3) if k != ax]
        pts = []
        for c1, c2 in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            p = [0.0, 0.0, 0.0]
            p[ax], p[o1], p[o2] = s * H[ax], c1 * H[o1], c2 * H[o2]
            pts.append((p[0] + center[0], p[1] + center[1], p[2] + center[2]))
        n = [0.0, 0.0, 0.0]
        n[ax] = s
        mesh.poly(mat, pts, tuple(n))


def _ico(subdiv):
    t = (1 + 5 ** 0.5) / 2
    verts = [v_unit(v) for v in [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
                                 (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
             (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
             (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(subdiv):
        cache, nf = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                va, vb = verts[a], verts[b]
                verts.append(v_unit(((va[0] + vb[0]) / 2, (va[1] + vb[1]) / 2, (va[2] + vb[2]) / 2)))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = nf
    out = []
    for a, b, c in faces:
        va, vb, vc = verts[a], verts[b], verts[c]
        if v_dot(v_cross(v_sub(vb, va), v_sub(vc, va)), va) < 0:
            b, c = c, b
        out.append((a, b, c))
    return verts, out


_ICO = {}


def add_ico(mesh, mat, center, r, subdiv=1, scale=(1.0, 1.0, 1.0), rot=None):
    """Smooth icosphere / ellipsoid."""
    if subdiv not in _ICO:
        _ICO[subdiv] = _ico(subdiv)
    verts, faces = _ICO[subdiv]
    p = mesh.prim(mat)
    base = len(p.pos)
    rm = q_to_m(rot) if rot else None
    for v in verts:
        pos = (v[0] * r * scale[0], v[1] * r * scale[1], v[2] * r * scale[2])
        n = v_unit((v[0] / scale[0], v[1] / scale[1], v[2] / scale[2]))
        if rm:
            pos, n = m_apply(rm, pos), m_apply(rm, n)
        p.pos.append((pos[0] + center[0], pos[1] + center[1], pos[2] + center[2]))
        p.nrm.append(n)
    for a, b, c in faces:
        p.idx += [base + a, base + b, base + c]


def add_torus(mesh, mat, center, R, r, seg=16, side=6, a0=0.0, a1=360.0, rot=None):
    """Smooth torus in the local XY plane (axis +Z); a0..a1 for an arc."""
    full = abs(a1 - a0) >= 360.0
    nu = seg if full else seg + 1
    p = mesh.prim(mat)
    base = len(p.pos)
    rm = q_to_m(rot) if rot else None
    for i in range(nu):
        a = math.radians(a0 + (a1 - a0) * i / seg)
        ca, sa = math.cos(a), math.sin(a)
        for j in range(side):
            b = 2 * math.pi * j / side
            n = (ca * math.cos(b), sa * math.cos(b), math.sin(b))
            pos = (R * ca + r * n[0], R * sa + r * n[1], r * n[2])
            if rm:
                pos, n = m_apply(rm, pos), m_apply(rm, n)
            p.pos.append((pos[0] + center[0], pos[1] + center[1], pos[2] + center[2]))
            p.nrm.append(n)
    for i in range(seg):
        i1 = (i + 1) % nu
        for j in range(side):
            j1 = (j + 1) % side
            a, b, c, d = (base + i * side + j, base + i1 * side + j, base + i1 * side + j1, base + i * side + j1)
            for t in ((a, b, c), (a, c, d)):
                pa, pb, pc = p.pos[t[0]], p.pos[t[1]], p.pos[t[2]]
                if v_dot(v_cross(v_sub(pb, pa), v_sub(pc, pa)), p.nrm[t[0]]) < 0:
                    t = (t[0], t[2], t[1])
                p.idx += list(t)


def add_lathe(mesh, mat, center, profile, seg):
    """Smooth surface of revolution around the Z axis through center.
    profile: [(z, r), ...] from tail to nose; r == 0 makes a pole."""
    p = mesh.prim(mat)
    rows = []
    n_prof = len(profile)
    for k, (z, r) in enumerate(profile):
        zp, rp = profile[max(k - 1, 0)]
        zn, rn = profile[min(k + 1, n_prof - 1)]
        nz, nr = -(rn - rp), (zn - zp)
        L = math.hypot(nz, nr)
        nz, nr = nz / L, nr / L
        row = []
        if r < 1e-9:
            p.pos.append((center[0], center[1], center[2] + z))
            p.nrm.append((0.0, 0.0, 1.0 if nz > 0 else -1.0))
            row = [len(p.pos) - 1] * seg
        else:
            for j in range(seg):
                ph = 2 * math.pi * j / seg
                p.pos.append((center[0] + r * math.sin(ph), center[1] + r * math.cos(ph), center[2] + z))
                p.nrm.append((nr * math.sin(ph), nr * math.cos(ph), nz))
                row.append(len(p.pos) - 1)
        rows.append(row)
    for k in range(n_prof - 1):
        A, B = rows[k], rows[k + 1]
        for j in range(seg):
            j1 = (j + 1) % seg
            for t in ((A[j], B[j], B[j1]), (A[j], B[j1], A[j1])):
                if len(set(t)) < 3:
                    continue
                pa, pb, pc = p.pos[t[0]], p.pos[t[1]], p.pos[t[2]]
                nn = v_cross(v_sub(pb, pa), v_sub(pc, pa))
                ref = (pa[0] - center[0], pa[1] - center[1], 0.0)
                if v_dot(ref, ref) < 1e-12:
                    ref = (pb[0] - center[0], pb[1] - center[1], 0.0)
                if v_dot(nn, ref) < 0:
                    t = (t[0], t[2], t[1])
                p.idx += list(t)


def add_disc(mesh, mat, center, r, seg, up=True, flip_u=False):
    """Flat textured disc in the XZ plane (u = +x, v = +z)."""
    pts, uvs = [], []
    for j in range(seg):
        a = 2 * math.pi * j / seg
        x, z = r * math.sin(a), r * math.cos(a)
        pts.append((center[0] + x, center[1], center[2] + z))
        uvs.append((0.5 + (-x if flip_u else x) / (2 * r), 0.5 + z / (2 * r)))
    mesh.poly(mat, pts, (0.0, 1.0 if up else -1.0, 0.0), uvs)


def add_pyramid(mesh, mat, base_c, hw, hd, h, apex_dx=0.0):
    """Four-sided pyramid standing on base_c (y up)."""
    x, y, z = base_c
    b = [(x - hw, y, z - hd), (x + hw, y, z - hd), (x + hw, y, z + hd), (x - hw, y, z + hd)]
    apex = (x + apex_dx, y + h, z)
    for i in range(4):
        p0, p1 = b[i], b[(i + 1) % 4]
        n = v_unit(v_cross(v_sub(p1, p0), v_sub(apex, p0)))
        cen = ((p0[0] + p1[0] + apex[0]) / 3 - x, 0.0, (p0[2] + p1[2] + apex[2]) / 3 - z)
        if v_dot(n, cen) < 0:
            n = (-n[0], -n[1], -n[2])
        mesh.poly(mat, [p0, p1, apex], n)
    mesh.poly(mat, b, (0.0, -1.0, 0.0))


def clip(T, N, fn):
    """Sample fn(p, t) -> {(node, path): value} at N+1 keys; the last key repeats the first."""
    times = [round(T * i / N, 6) for i in range(N + 1)]
    ch = {}
    for i in range(N + 1):
        k = i % N
        for key, v in fn(2 * math.pi * k / N, T * k / N).items():
            ch.setdefault(key, []).append(tuple(round(x, 6) for x in v))
    return [(n, path, times, vals) for (n, path), vals in ch.items()]

# ------------------------------------------------------------------ stroke text (for textures)

def arc(cx, cy, rx, ry, a0, a1, n):
    return [(cx + rx * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
             cy + ry * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


# Glyphs in a unit box (x right, y down). Only the characters the props print.
GLYPHS = {
    "M": [[(0.08, 1), (0.08, 0), (0.5, 0.62), (0.92, 0), (0.92, 1)]],
    "N": [[(0.12, 1), (0.12, 0), (0.88, 1), (0.88, 0)]],
    "E": [[(0.86, 0), (0.12, 0), (0.12, 1), (0.86, 1)], [(0.12, 0.5), (0.72, 0.5)]],
    "S": [[(0.88, 0.15), (0.68, 0.02), (0.34, 0.02), (0.13, 0.15), (0.13, 0.35), (0.3, 0.46), (0.7, 0.54),
           (0.88, 0.65), (0.88, 0.85), (0.68, 0.98), (0.32, 0.98), (0.1, 0.85)]],
    "T": [[(0.04, 0), (0.96, 0)], [(0.5, 0), (0.5, 1)]],
    "O": [arc(0.5, 0.5, 0.42, 0.5, 0, 360, 24)],
    "C": [arc(0.55, 0.5, 0.45, 0.5, 45, 315, 18)],
    "K": [[(0.14, 0), (0.14, 1)], [(0.86, 0), (0.14, 0.6)], [(0.4, 0.42), (0.88, 1)]],
    "5": [[(0.84, 0), (0.22, 0), (0.16, 0.44), (0.42, 0.37), (0.68, 0.41), (0.86, 0.57), (0.88, 0.77),
           (0.74, 0.94), (0.46, 1.0), (0.12, 0.9)]],
    "0": [arc(0.5, 0.5, 0.4, 0.5, 0, 360, 24)],
    "R": [[(0.06, 0.03), (0.94, 0.03)], [(0.06, 0.31), (0.94, 0.31)],   # "R" = the rupee sign
          arc(0.44, 0.3, 0.33, 0.27, -90, 90, 12) + [(0.1, 0.57), (0.74, 1.0)]],
}
ASPECT = {"N": 0.66, "M": 0.78, "E": 0.6, "S": 0.62, "T": 0.66, "O": 0.7, "C": 0.62, "K": 0.62, "5": 0.6, "0": 0.6, "R": 0.72}


def text_width(text, h, gap=0.22):
    return sum(ASPECT[c] * h for c in text) + gap * h * (len(text) - 1)


def text_segs(text, x, y, h, gap=0.22):
    segs = []
    for ch in text:
        w = ASPECT[ch] * h
        for line in GLYPHS[ch]:
            for (ax, ay), (bx, by) in zip(line, line[1:]):
                segs.append((x + ax * w, y + ay * h, x + bx * w, y + by * h))
        x += w + gap * h
    return segs


def _seg_d2(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L))
    qx, qy = ax + t * dx - px, ay + t * dy - py
    return qx * qx + qy * qy


class Ink:
    """Round-capped strokes; hit(x, y) tests a point."""

    def __init__(self, segs, width):
        r = width / 2
        self.segs, self.r2 = segs, r * r
        self.sbb = [(min(a, c) - r, min(b, d) - r, max(a, c) + r, max(b, d) + r) for a, b, c, d in segs]
        self.bb = (min(s[0] for s in self.sbb), min(s[1] for s in self.sbb),
                   max(s[2] for s in self.sbb), max(s[3] for s in self.sbb))

    def hit(self, x, y):
        if not (self.bb[0] <= x <= self.bb[2] and self.bb[1] <= y <= self.bb[3]):
            return False
        for (ax, ay, bx, by), (x0, y0, x1, y1) in zip(self.segs, self.sbb):
            if x0 <= x <= x1 and y0 <= y <= y1 and _seg_d2(x, y, ax, ay, bx, by) <= self.r2:
                return True
        return False


def ink_text(text, cx, cy, h, weight=0.14):
    w = text_width(text, h)
    return Ink(text_segs(text, cx - w / 2, cy - h / 2, h), weight * h)

# ------------------------------------------------------------------ ships

def add_hull(mesh, right_top, right_bot, deck_y, ring_ys, band_mat, m_deck, m_keel):
    """Lofted ship hull. right_top/right_bot: starboard half-outlines (x, z) from the
    transom corner to the bow tip (x = 0), same point count, at deck and keel.
    ring_ys: heights of the loft rings (0 .. deck_y); band_mat(y_mid) -> material.
    Width eases out fast above the keel (round bilge); length interpolates linearly."""
    def full(right):
        return right + [(-x, z) for (x, z) in reversed(right[:-1])]

    top, bot = full(right_top), full(right_bot)
    rings = []
    for y in ring_ys:
        t = y / deck_y
        tw = 1 - (1 - t) ** 3
        rings.append([(xb + (xt - xb) * tw, y, zb + (zt - zb) * t) for (xt, zt), (xb, zb) in zip(top, bot)])
    n = len(top)
    for r in range(len(rings) - 1):
        A, B = rings[r], rings[r + 1]
        mat = band_mat((ring_ys[r] + ring_ys[r + 1]) / 2)
        for e in range(n):
            e1 = (e + 1) % n
            pts = [A[e], A[e1], B[e1], B[e]]
            nn = v_cross(v_sub(pts[1], pts[0]), v_sub(pts[3], pts[0]))
            if v_dot(nn, nn) < 1e-14:
                nn = v_cross(v_sub(pts[2], pts[1]), v_sub(pts[3], pts[1]))
            nn = v_unit(nn)
            cx = sum(p[0] for p in pts) / 4
            cz = sum(p[2] for p in pts) / 4
            if nn[0] * cx + nn[2] * cz < 0:
                nn = (-nn[0], -nn[1], -nn[2])
            mesh.poly(mat, pts, nn)
    mesh.poly(m_deck, rings[-1], (0.0, 1.0, 0.0))
    mesh.poly(m_keel, rings[0], (0.0, -1.0, 0.0))


def half_beam(right_top, z):
    """Deck half-width at z (linear between outline points)."""
    pts = sorted(right_top, key=lambda p: p[1])
    if z <= pts[0][1]:
        return pts[0][0]
    for (x0, z0), (x1, z1) in zip(pts, pts[1:]):
        if z0 <= z <= z1:
            return x0 + (x1 - x0) * (z - z0) / (z1 - z0) if z1 > z0 else min(x0, x1)
    return 0.0


def ship_rock_clip(T, bob, roll, pitch, pivot_y, bob_waves=1):
    def fn(p, t):
        return {("hull", "translation"): (0.0, pivot_y + bob * math.sin(bob_waves * p), 0.0),
                ("hull", "rotation"): q_ypr(pitch=pitch * math.sin(2 * p + 1.0), roll=roll * math.sin(p))}
    return clip(T, 48, fn)


def build_tanker():
    M = Mats()
    W = Mesh("tanker")
    m_red = M("hull_bottom", "#6A2C22")
    m_black = M("hull_top", "#1B2230")
    m_deck = M("deck", "#3F5E4A")
    m_white = M("white", "#F1F1EC")
    m_win = M("window", "#1C2A3A", rough=0.3)
    m_pipe = M("pipe", "#C8CDD2")
    m_grey = M("steel", "#8E969F")
    m_crane = M("crane", "#E2BE48")
    m_valve = M("valve", "#C0392B")
    m_dark = M("dark", "#22252A")
    m_boat = M("lifeboat", "#F07D24")

    deck_y = 1.2
    top = [(0.86, -5.9), (1.0, -5.5), (1.0, 3.6), (0.97, 4.4), (0.86, 5.0), (0.62, 5.5), (0.30, 5.85), (0.0, 6.0)]
    bot = [(0.6, -5.25), (0.8, -4.6), (0.8, 3.4), (0.76, 4.2), (0.6, 4.8), (0.36, 5.1), (0.16, 5.25), (0.0, 5.3)]
    add_hull(W, top, bot, deck_y, [0.0, 0.1, 0.24, 0.45, deck_y],
             lambda y: m_red if y < 0.45 else m_black, m_deck, m_red)
    add_ico(W, m_red, (0.0, 0.27, 5.3), 1.0, subdiv=1, scale=(0.2, 0.19, 0.42))   # bulbous bow
    # white rail along the deck edges
    for s in (1, -1):
        add_box(W, m_white, (s * 0.975, deck_y + 0.05, -0.1), (0.02, 0.06, 7.4))

    # centreline pipe run on low supports
    for x in (-0.12, 0.0, 0.12):
        add_rod(W, m_pipe, (x, deck_y + 0.09, -3.6), (x, deck_y + 0.09, 4.4), 0.035, seg=8)
    for z in [-3.3 + 1.05 * k for k in range(8)]:
        add_box(W, m_grey, (0.0, deck_y + 0.03, z), (0.38, 0.06, 0.06))
    # cross manifold midships with valves at the ship's sides
    for z in (0.25, 0.4, 0.55):
        add_rod(W, m_pipe, (-0.9, deck_y + 0.14, z), (0.9, deck_y + 0.14, z), 0.035, seg=8)
    for s in (1, -1):
        add_box(W, m_valve, (s * 0.88, deck_y + 0.14, 0.4), (0.1, 0.16, 0.42), bevel=0.02)
    add_box(W, m_grey, (0.0, deck_y + 0.12, 0.4), (0.3, 0.12, 0.42), bevel=0.02)

    # hose-handling cranes beside the manifold, plus a provision crane aft
    def crane(base, tip):
        bx, bz = base
        add_rod(W, m_crane, (bx, deck_y, bz), (bx, deck_y + 0.5, bz), 0.075, seg=8)
        add_box(W, m_crane, (bx, deck_y + 0.53, bz), (0.18, 0.08, 0.18), bevel=0.02)
        p0 = (bx, deck_y + 0.55, bz)
        add_beam(W, m_crane, p0, tip, 0.06, 0.06)
        add_beam(W, m_dark, tip, (tip[0], tip[1] - 0.35, tip[2]), 0.012, 0.012)
    crane((0.68, -0.35), (0.25, deck_y + 0.95, 0.35))
    crane((-0.68, 1.15), (-0.25, deck_y + 0.95, 0.45))
    crane((0.72, -3.4), (0.95, deck_y + 0.95, -2.8))

    # elevated catwalk running fore and aft
    cx_, cy_ = 0.36, deck_y + 0.36
    add_box(W, m_grey, (cx_, cy_, 0.4), (0.16, 0.025, 8.0))
    for s in (1, -1):
        add_box(W, m_white, (cx_ + s * 0.075, cy_ + 0.1, 0.4), (0.012, 0.012, 8.0))
    for k in range(9):
        z = -3.6 + 1.0 * k
        add_box(W, m_grey, (cx_, deck_y + 0.18, z), (0.03, 0.36, 0.03))

    # forecastle, windlass and foremast
    add_box(W, m_deck, (0.0, deck_y + 0.1, 5.0), (1.25, 0.2, 0.9), bevel=0.03)
    add_box(W, m_dark, (0.0, deck_y + 0.26, 4.9), (0.5, 0.12, 0.2), bevel=0.02)
    add_rod(W, m_white, (0.0, deck_y + 0.2, 5.25), (0.0, deck_y + 0.95, 5.25), 0.03, seg=6)

    # accommodation block at the stern: three decks, bridge with wings, windows
    tiers = [((1.8, 0.45, 1.2), (0.0, deck_y + 0.225, -4.45)),
             ((1.7, 0.42, 1.1), (0.0, deck_y + 0.66, -4.4)),
             ((1.6, 0.42, 1.0), (0.0, deck_y + 1.08, -4.35))]
    for size, c in tiers:
        add_box(W, m_white, c, size, bevel=0.025)
        for k in range(7):
            add_box(W, m_win, (-0.72 + 0.24 * k, c[1] + 0.02, -3.845), (0.13, 0.1, 0.02))
        for sx in (1, -1):
            for k in range(3):
                zc = c[2] + size[2] / 2 - 0.25 - 0.3 * k
                add_box(W, m_win, (sx * (size[0] / 2 + 0.005), c[1] + 0.02, zc), (0.02, 0.1, 0.13))
    bridge_y = deck_y + 1.48
    add_box(W, m_white, (0.0, bridge_y, -4.3), (2.0, 0.38, 0.8), bevel=0.025)
    add_box(W, m_win, (0.0, bridge_y + 0.04, -3.895), (1.92, 0.14, 0.02))
    for s in (1, -1):
        add_box(W, m_win, (s * 1.0, bridge_y + 0.04, -4.3), (0.02, 0.14, 0.6))
    add_box(W, m_white, (0.0, bridge_y + 0.21, -4.3), (2.04, 0.04, 0.84))
    add_rod(W, m_white, (0.0, bridge_y + 0.23, -4.45), (0.0, bridge_y + 0.8, -4.45), 0.03, seg=6)
    add_box(W, m_white, (0.0, bridge_y + 0.66, -4.45), (0.6, 0.03, 0.03))
    add_box(W, m_dark, (0.0, bridge_y + 0.75, -4.45), (0.36, 0.04, 0.07))
    # funnel behind the bridge
    add_box(W, M("funnel", "#ECE6D6"), (0.0, deck_y + 1.35, -5.1), (0.55, 1.5, 0.6), bevel=0.08)
    add_box(W, m_valve, (0.0, deck_y + 1.75, -5.1), (0.57, 0.16, 0.62))
    add_box(W, m_dark, (0.0, deck_y + 2.03, -5.1), (0.56, 0.14, 0.61))
    # free-fall lifeboat on its ramp over the stern
    tilt = q_axis(X_AX, -25.0)
    add_ico(W, m_boat, (0.0, deck_y + 0.42, -5.62), 1.0, subdiv=1, scale=(0.2, 0.17, 0.42), rot=tilt)
    add_box(W, m_grey, (0.0, deck_y + 0.24, -5.6), (0.32, 0.04, 0.9), rot=tilt)

    pivot = 0.6
    hull = Mesh("tanker_hull")
    merge(hull, W, off=(0.0, -pivot, 0.0))
    root = Node("tanker_oil")
    root.add(Node("hull", t=(0.0, pivot, 0.0), mesh=hull))
    anims = {"idle": ship_rock_clip(4.0, 0.035, 2.0, 0.5, pivot, bob_waves=2)}
    return "tanker_oil", root, M.d, [], anims, ("idle",)


def build_container_ship():
    M = Mats()
    W = Mesh("boxship")
    m_red = M("hull_bottom", "#B23A2E")
    m_blue = M("hull_top", "#1F4E8C")
    m_deck = M("deck", "#5C6670")
    m_white = M("white", "#F1F1EC")
    m_win = M("window", "#1C2A3A", rough=0.3)
    m_dark = M("dark", "#22252A")
    m_grey = M("hatch", "#7D8791")
    palette = [M("box_red", "#C8423A"), M("box_blue", "#2F64B3"), M("box_green", "#3E9A55"),
               M("box_yellow", "#E8B931"), M("box_orange", "#E07A2E"), M("box_teal", "#27919A"),
               M("box_grey", "#A3ABB4"), M("box_white", "#E6E6E0"), M("box_maroon", "#7D2E3A")]

    deck_y = 1.25
    top = [(0.9, -5.95), (1.0, -5.6), (1.0, 2.8), (0.96, 3.8), (0.82, 4.6), (0.56, 5.3), (0.26, 5.8), (0.0, 6.0)]
    bot = [(0.55, -5.2), (0.8, -4.4), (0.8, 2.4), (0.74, 3.5), (0.52, 4.4), (0.26, 5.0), (0.1, 5.25), (0.0, 5.35)]
    add_hull(W, top, bot, deck_y, [0.0, 0.1, 0.24, 0.42, deck_y],
             lambda y: m_red if y < 0.42 else m_blue, m_deck, m_red)
    add_ico(W, m_red, (0.0, 0.27, 5.3), 1.0, subdiv=1, scale=(0.19, 0.18, 0.38))

    # container bays: 40 ft boxes (0.82 long), five rows across where the hull is wide enough
    rnd = lcg(20261017)
    CL, CW, CH = 0.82, 0.36, 0.34
    bays = [-4.0] + [-2.15 + 0.88 * k for k in range(8)]
    for bi, z in enumerate(bays):
        hb = min(half_beam(top, z - CL / 2), half_beam(top, z + CL / 2))
        rows = [-0.76, -0.38, 0.0, 0.38, 0.76] if hb >= 0.95 else [-0.38, 0.0, 0.38]
        add_box(W, m_grey, (0.0, deck_y + 0.03, z), (rows[-1] * 2 + CW + 0.04, 0.06, CL + 0.04))
        heights = []
        for ri, x in enumerate(rows):
            n = 2 + int(next(rnd) * 3)            # 2..4 high
            if bi == len(bays) - 1:
                n = min(n, 2)                     # keep the bow-most bay low
            heights.append(n)
        for ri, x in enumerate(rows):
            for tier in range(heights[ri]):
                mat = palette[int(next(rnd) * len(palette))]
                y = deck_y + 0.06 + CH * tier + CH / 2
                faces = ["+z", "-z"]
                left = heights[ri - 1] if ri > 0 else 0
                right = heights[ri + 1] if ri < len(rows) - 1 else 0
                if tier >= left:
                    faces.append("-x")
                if tier >= right:
                    faces.append("+x")
                if tier == heights[ri] - 1:
                    faces.append("+y")
                add_box_faces(W, mat, (x, y, z), (CW - 0.01, CH - 0.012, CL), faces)

    # bridge block about three-quarters aft
    bz = -3.0
    add_box(W, m_white, (0.0, deck_y + 1.0, bz), (1.5, 2.0, 0.7), bevel=0.03)
    for lvl in range(5):
        y = deck_y + 0.3 + 0.36 * lvl
        for k in range(6):
            add_box(W, m_win, (-0.6 + 0.24 * k, y, bz + 0.355), (0.13, 0.1, 0.02))
        for sx in (1, -1):
            for zc in (bz + 0.15, bz - 0.15):
                add_box(W, m_win, (sx * 0.755, y, zc), (0.02, 0.1, 0.13))
    add_box(W, m_white, (0.0, deck_y + 2.2, bz), (2.04, 0.38, 0.78), bevel=0.025)
    add_box(W, m_win, (0.0, deck_y + 2.24, bz + 0.395), (1.96, 0.14, 0.02))
    for s in (1, -1):
        add_box(W, m_win, (s * 1.02, deck_y + 2.24, bz), (0.02, 0.14, 0.6))
    add_box(W, m_white, (0.0, deck_y + 2.41, bz), (2.06, 0.04, 0.82))
    add_rod(W, m_white, (0.0, deck_y + 2.43, bz - 0.1), (0.0, deck_y + 3.0, bz - 0.1), 0.03, seg=6)
    add_box(W, m_white, (0.0, deck_y + 2.85, bz - 0.1), (0.6, 0.03, 0.03))
    add_box(W, m_dark, (0.0, deck_y + 2.94, bz - 0.1), (0.36, 0.04, 0.07))
    # lifeboats on davits either side of the bridge block
    for s in (1, -1):
        add_ico(W, M("lifeboat", "#F07D24"), (s * 0.86, deck_y + 0.75, bz - 0.1), 1.0, subdiv=1,
                scale=(0.13, 0.12, 0.34))

    # funnel aft
    add_box(W, M("funnel", "#E9ECEF"), (0.0, deck_y + 0.75, -5.0), (0.6, 1.5, 0.7), bevel=0.08)
    add_box(W, m_blue, (0.0, deck_y + 1.15, -5.0), (0.62, 0.2, 0.72))
    add_box(W, m_dark, (0.0, deck_y + 1.43, -5.0), (0.61, 0.14, 0.71))

    # forecastle with breakwater and bow mast
    add_box(W, m_deck, (0.0, deck_y + 0.1, 5.15), (1.0, 0.2, 0.9), bevel=0.03)
    for s in (1, -1):
        add_beam(W, m_white, (0.0, deck_y + 0.17, 4.75), (s * 0.85, deck_y + 0.17, 4.35), 0.04, 0.34)
    add_rod(W, m_white, (0.0, deck_y + 0.2, 5.35), (0.0, deck_y + 0.95, 5.35), 0.03, seg=6)

    pivot = 0.6
    hull = Mesh("boxship_hull")
    merge(hull, W, off=(0.0, -pivot, 0.0))
    root = Node("container_ship")
    root.add(Node("hull", t=(0.0, pivot, 0.0), mesh=hull))
    anims = {"idle": ship_rock_clip(3.0, 0.045, 1.2, 0.5, pivot)}
    return "container_ship", root, M.d, [], anims, ("idle",)

# ------------------------------------------------------------------ submarine

def build_submarine():
    M = Mats()
    S = Mesh("sub")
    m_hull = M("hull", "#F6A531")
    m_fin = M("fins", "#E0761F")
    m_ring = M("porthole_ring", "#7F8C99", rough=0.5)
    m_glass = M("porthole_glass", "#1E4F6E", rough=0.2, emissive="#0B2636")
    m_steel = M("steel", "#8E969F")
    R0 = 0.93
    ay = R0                                   # hull axis height (bottom of hull on y = 0)
    prof = [(-4.0, 0.0), (-3.96, 0.13), (-3.45, 0.45), (-2.75, 0.75), (-1.95, 0.9), (-1.25, R0),
            (2.2, R0), (2.9, 0.88), (3.4, 0.74), (3.75, 0.52), (3.95, 0.25), (4.0, 0.0)]
    add_lathe(S, m_hull, (0.0, ay, 0.0), prof, 20)
    # sail (tower) with sail planes, windows and periscope
    ty = 2 * R0 - 0.1
    add_box(S, m_hull, (0.0, ty + 0.35, 0.7), (0.5, 0.7, 1.3), bevel=0.17)
    add_box(S, m_fin, (0.0, ty + 0.4, 1.0), (0.95, 0.045, 0.28), bevel=0.015)
    for s in (1, -1):
        add_torus(S, m_ring, (s * 0.255, ty + 0.38, 0.55), 0.075, 0.018, seg=12, side=5,
                  rot=q_axis(Y_AX, s * 90.0))
        add_rod(S, m_glass, (s * 0.24, ty + 0.38, 0.55), (s * 0.258, ty + 0.38, 0.55), 0.065, seg=10)
    add_rod(S, m_steel, (0.0, ty + 0.65, 0.45), (0.0, ty + 1.15, 0.45), 0.045, seg=8)
    add_box(S, m_steel, (0.0, ty + 1.15, 0.54), (0.1, 0.1, 0.24), bevel=0.02)
    add_box(S, m_glass, (0.0, ty + 1.15, 0.665), (0.07, 0.07, 0.01))
    # hull portholes
    py = ay + 0.12
    px = math.sqrt(R0 ** 2 - 0.12 ** 2)
    for z in (-1.3, -0.25, 0.8, 1.85):
        for s in (1, -1):
            add_torus(S, m_ring, (s * (px + 0.012), py, z), 0.16, 0.035, seg=12, side=5,
                      rot=q_axis(Y_AX, s * 90.0))
            add_rod(S, m_glass, (s * (px - 0.02), py, z), (s * (px + 0.01), py, z), 0.135, seg=12)
    # hatch on the aft deck
    add_cyl(S, m_steel, (0.0, 2 * R0 - 0.01, -1.6), 0.16, 0.15, 0.08, 10)
    # cruciform tail fins
    add_box(S, m_fin, (0.0, ay, -3.3), (0.05, 1.15, 0.55), bevel=0.02)
    add_box(S, m_fin, (0.0, ay, -3.3), (1.25, 0.05, 0.55), bevel=0.02)
    # bow planes
    add_box(S, m_fin, (0.0, ay + 0.05, 2.6), (2.2, 0.04, 0.3), bevel=0.015)

    body = Mesh("sub_body")
    merge(body, S, off=(0.0, -ay, 0.0))
    prop = Mesh("propeller")
    m_brass = M("propeller", "#C9A227", rough=0.45)
    add_rod(prop, m_brass, (0.0, 0.0, 0.12), (0.0, 0.0, -0.16), 0.08, seg=10, r1=0.05)
    for k in range(4):
        rot = q_mul(q_axis(Z_AX, 90.0 * k), q_axis(Y_AX, 30.0))
        c = m_apply(q_to_m(q_axis(Z_AX, 90.0 * k)), (0.0, 0.27, 0.0))
        add_box(prop, m_brass, c, (0.15, 0.4, 0.035), bevel=0.012, rot=rot)

    root = Node("submarine")
    sub = root.add(Node("sub", t=(0.0, ay, 0.0), mesh=body))
    sub.add(Node("propeller", t=(0.0, 0.0, -4.1), mesh=prop))

    def fn(p, t):
        return {("sub", "translation"): (0.0, ay + 0.05 * math.sin(p), 0.0),
                ("sub", "rotation"): q_ypr(pitch=1.2 * math.sin(p + 0.8), roll=0.8 * math.sin(p)),
                ("propeller", "rotation"): q_axis(Z_AX, 720.0 * t / 1.5)}
    anims = {"idle": clip(1.5, 24, fn)}
    return "submarine", root, M.d, [], anims, ("idle",)

# ------------------------------------------------------------------ ML bot

def render_ml_tee():
    """Dark tee, all round; small white neural-network pattern on the chest."""
    S, lf, _, H = bb.tee_layout()
    P = S[8]
    dark = hex_rgb("#343841")
    cx, cy = lf / 2, H * 0.47
    layers = [[-0.06, 0.0, 0.06], [-0.075, -0.025, 0.025, 0.075], [-0.03, 0.03]]
    xs = [-0.085, 0.0, 0.085]
    nodes = [[(cx + xs[i], cy + y) for y in layer] for i, layer in enumerate(layers)]
    segs = []
    for a, b in zip(nodes, nodes[1:]):
        for p in a:
            for q in b:
                segs.append((p[0], p[1], q[0], q[1]))
    lines = Ink(segs, 0.0045)
    dots = [p for layer in nodes for p in layer]
    line_c, dot_c = hex_rgb("#9AA3B0"), hex_rgb("#F2F4F7")

    def sample(s, yd):
        if s < S[0] or s > S[1]:
            return dark
        for (x, y) in dots:
            if (s - x) ** 2 + (yd - y) ** 2 <= 0.012 ** 2:
                return dot_c
        return line_c if lines.hit(s, yd) else dark

    det = [(cx - 0.11, cy - 0.1, cx + 0.11, cy + 0.1)]
    return raster(bb.TEE_W, bb.TEE_H, P, H, sample, det)


def build_ml_bot():
    M = Mats()
    m_body = M("body", bb.COL["body"])
    m_eye = M("eye", bb.COL["eye"], rough=0.45)
    m_tee = M.raw("tee_ml", material("tee_ml", "#FFFFFF", texture=0))
    m_sleeve = M("tee_sleeve", "#343841")
    m_pants = M("pants", "#2C3036")
    m_shoe = M("shoe", "#1B1B1F")
    m_sole = M("sole", "#9AA0A6")
    m_helmet = M("helmet", "#24272D", rough=0.55)
    textures = [("tee_ml", render_ml_tee())]

    leg = Mesh("leg")
    add_box(leg, m_pants, (0, -0.13, 0), (0.13, 0.22, 0.14), bevel=0.02)
    add_box(leg, m_shoe, (0, -0.25, 0.025), (0.15, 0.07, 0.20), bevel=0.025)
    add_box(leg, m_sole, (0, -0.29, 0.025), (0.155, 0.02, 0.205))
    hips_mesh = Mesh("hips")
    add_box(hips_mesh, m_pants, (0, 0, 0), (0.54, 0.10, 0.38), bevel=0.03)

    HY = bb.HIP_Y
    torso_mesh = Mesh("torso")
    B = bb.BODY
    add_box(torso_mesh, m_body, (0, B["y0"] + B["h"] / 2 - HY, 0), (B["w"], B["h"], B["d"]), bevel=B["bevel"])
    bb.add_tee(torso_mesh, m_tee, m_sleeve, m_sleeve)

    helmet = Mesh("helmet")
    add_box(helmet, m_helmet, (0, 0.965 - HY, 0), (0.665, 0.13, 0.465), bevel=0.05)
    for s in (1, -1):
        ear = Mesh("tmp")
        add_pyramid(ear, m_helmet, (0, 0, 0), 0.085, 0.05, 0.24, apex_dx=s * 0.03)
        merge(helmet, ear, rot=q_axis(Z_AX, -s * 14.0), off=(s * 0.2, 1.015 - HY, 0.0))

    eyes_mesh = Mesh("eyes")
    for s in (1, -1):
        add_box(eyes_mesh, m_eye, (s * 0.13, 0, B["d"] / 2 + 0.005), (0.075, 0.145, 0.03), bevel=0.012)

    def arm_mesh(s):
        m = Mesh("arm_%s" % ("L" if s > 0 else "R"))
        add_box(m, m_sleeve, (s * 0.035, -0.02, 0), (0.105, 0.12, 0.17), bevel=0.025)
        add_box(m, m_body, (s * 0.04, -0.13, 0), (0.085, 0.16, 0.10), bevel=0.025)
        return m

    hammer = Mesh("hammer")
    add_box(hammer, M("wood", "#9C6B3E"), (0, 0, 0.18), (0.035, 0.035, 0.46), bevel=0.008)
    add_box(hammer, M("grip", "#8E2B2B"), (0, 0, 0.0), (0.045, 0.045, 0.1), bevel=0.01)
    add_box(hammer, M("steel", "#8E96A0", rough=0.45), (0, 0.0, 0.40), (0.07, 0.19, 0.08), bevel=0.015)
    add_box(hammer, M("steel_dark", "#5E656E", rough=0.45), (0, -0.095, 0.40), (0.075, 0.03, 0.085), bevel=0.008)

    root = Node("bot_ML")
    hips = root.add(Node("hips", t=(0.0, HY, 0.0), mesh=hips_mesh))
    hips.add(Node("leg_L", t=(bb.LEG_X, 0.0, 0.0), mesh=leg))
    hips.add(Node("leg_R", t=(-bb.LEG_X, 0.0, 0.0), mesh=leg))
    torso = hips.add(Node("torso", mesh=torso_mesh))
    torso.add(Node("helmet", mesh=helmet))
    torso.add(Node("eyes", t=(0.0, bb.EYE_Y - HY, 0.0), mesh=eyes_mesh))
    tilt = bb.ARM_TILT
    torso.add(Node("arm_L", t=(bb.ARM_X, bb.ARM_Y - HY, 0.0), r=q_axis(Z_AX, tilt), mesh=arm_mesh(1)))
    arm_r = torso.add(Node("arm_R", t=(-bb.ARM_X, bb.ARM_Y - HY, 0.0),
                           r=q_mul(q_axis(Z_AX, -tilt), q_axis(X_AX, -25.0)), mesh=arm_mesh(-1)))
    arm_r.add(Node("hammer", t=(-0.04, -0.17, 0.0), mesh=hammer))

    def common(d):
        d[("leg_L", "rotation")] = (0.0, 0.0, 0.0, 1.0)
        d[("leg_R", "rotation")] = (0.0, 0.0, 0.0, 1.0)
        d[("leg_L", "translation")] = (bb.LEG_X, 0.0, 0.0)
        d[("leg_R", "translation")] = (-bb.LEG_X, 0.0, 0.0)
        return d

    def idle(p, t):
        sp, cp = math.sin(p), math.cos(p)
        blink = 0.08 if 1.36 <= t <= 1.42 else 1.0
        return common({
            ("hips", "translation"): (0.0, HY - 0.004 * (1 - cp) / 2, 0.0),
            ("torso", "rotation"): q_ypr(yaw=2.0 * math.sin(p + 0.6), pitch=1.0 * sp, roll=1.6 * math.sin(p + 1.2)),
            ("torso", "scale"): (1.0 - 0.011 * sp, 1.0 + 0.022 * sp, 1.0 - 0.011 * sp),
            ("eyes", "scale"): (1.0, blink, 1.0),
            ("arm_L", "rotation"): q_mul(q_axis(Z_AX, tilt + 4.0 * sp), q_axis(X_AX, -3.0 * cp)),
            ("arm_R", "rotation"): q_mul(q_axis(Z_AX, -tilt - 2.0 * sp), q_axis(X_AX, -25.0 + 4.0 * sp)),
        })

    def ease_io(u):
        return u * u * (3 - 2 * u)

    IMPACT, TOP = -55.0, -160.0     # at -55 deg the hammer face lands at y ~0.75 (anvil top), 0.46 ahead

    def swing(t):
        """Right-arm pitch: impact at t=0, lift overhead, hold, strike down (accelerating)."""
        if t < 0.36:
            return IMPACT + (TOP - IMPACT) * ease_io(t / 0.36)
        if t < 0.42:
            return TOP - 4.0 * math.sin(math.pi * (t - 0.36) / 0.06)
        u = (t - 0.42) / 0.18
        return TOP + (IMPACT - TOP) * u ** 3

    def hammer_clip(p, t):
        a = swing(t)
        k = (a - IMPACT) / (TOP - IMPACT)            # 0 at impact, 1 at the top of the lift
        hit = max(0.0, 1.0 - t / 0.1) if t < 0.1 else 0.0
        lean = 7.0 * (1 - k) - 3.0 * k
        return common({
            ("hips", "translation"): (0.0, HY - 0.02 * hit, 0.0),
            ("torso", "rotation"): q_ypr(yaw=-6.0 * k, pitch=lean, roll=2.0 * k),
            ("torso", "scale"): (1.0 + 0.025 * hit, 1.0 - 0.05 * hit, 1.0 + 0.025 * hit),
            ("eyes", "scale"): (1.0, 1.0 - 0.55 * hit, 1.0),
            ("arm_L", "rotation"): q_mul(q_axis(Z_AX, tilt + 6.0 * k), q_axis(X_AX, -10.0 + 14.0 * k)),
            ("arm_R", "rotation"): q_mul(q_axis(Z_AX, -tilt), q_axis(X_AX, a)),
        })

    anims = {"idle": clip(2.0, 60, idle), "hammer": clip(0.6, 36, hammer_clip)}
    anims.update(bb.action_clips(False, False))       # crouch, throw, catch, jump, cheer
    anims = bb.fill_coverage(root, anims)
    return "bot_ML", root, M.d, textures, anims, ("idle", "hammer") + tuple(bb.ACTION_TIMES)

# ------------------------------------------------------------------ the owner's toy figure

def build_me():
    M = Mats()
    m_skin = M("skin", "#B98563", rough=0.7)
    m_skin_d = M("skin_shade", "#A8744F", rough=0.7)
    m_shirt = M.raw("tshirt", material("tshirt", "#F5F5F2", texture=0))
    m_shirt_plain = M("tshirt_plain", "#F5F5F2")
    m_jeans = M("jeans", "#2B4570")
    m_sneaker = M("sneaker", "#F1F1EF")
    m_sole = M("sneaker_sole", "#8C939C")
    m_hair = M("hair", "#1B1714", rough=0.9)
    m_dark = M("eyes", "#1E1A18", rough=0.4)
    m_frame = M("glasses", "#222222", rough=0.4)
    m_lip = M("smile", "#5A2E22")

    # "ME" print on the shirt front
    TW, TH = 0.32, 0.26
    bev = 0.05
    fw, fh = TW - 2 * bev, TH - 2 * bev
    me = ink_text("ME", fw / 2, fh * 0.46, 0.085, weight=0.16)
    white, black = hex_rgb("#F5F5F2"), hex_rgb("#111111")
    tex = raster(256, 128, fw, fh, lambda x, y: black if me.hit(x, y) else white, [me.bb])

    legs = Mesh("legs")
    add_box(legs, m_jeans, (0.0, 0.32, 0.0), (0.3, 0.1, 0.19), bevel=0.03)
    for s in (1, -1):
        add_box(legs, m_jeans, (s * 0.078, 0.185, 0.0), (0.125, 0.24, 0.13), bevel=0.03)
        add_box(legs, m_sneaker, (s * 0.078, 0.06, 0.025), (0.13, 0.07, 0.2), bevel=0.03)
        add_box(legs, m_sole, (s * 0.078, 0.014, 0.025), (0.135, 0.028, 0.205))

    BY = 0.32                                            # body pivot (hips)
    tshirt = Mesh("tshirt")
    add_box(tshirt, m_shirt_plain, (0.0, 0.45 - BY, 0.0), (TW, TH, 0.2), bevel=bev, faces={"+z": m_shirt})

    HY = 0.58                                            # head pivot (neck base)
    HC = (0.0, 0.765 - HY, 0.0)                          # head centre, head-local
    R = 0.19
    head = Mesh("head")
    add_rod(head, m_skin, (0.0, -0.03, 0.0), (0.0, 0.07, 0.0), 0.05, seg=10)
    add_ico(head, m_skin, HC, R, subdiv=2, scale=(1.0, 0.97, 0.96))
    for s in (1, -1):
        add_ico(head, m_skin, (s * 0.183, HC[1] - 0.01, 0.0), 0.042, subdiv=1, scale=(0.55, 1.0, 0.8))

    def on_head(x, y):                                    # z of the head surface in front
        return math.sqrt(max(R * R - x * x - (y - HC[1]) ** 2, 0.0)) * 0.96

    face = Mesh("face")
    ey = HC[1] + 0.02
    for s in (1, -1):
        add_ico(face, m_dark, (s * 0.066, ey, on_head(0.066, ey) - 0.004), 0.02, subdiv=1, scale=(1.0, 1.15, 0.6))
    add_ico(face, m_skin_d, (0.0, HC[1] - 0.02, on_head(0.0, HC[1] - 0.02) + 0.004), 0.021, subdiv=1)
    add_torus(face, m_lip, (0.0, HC[1] - 0.045, on_head(0.0, HC[1] - 0.075) + 0.004), 0.04, 0.0065,
              seg=10, side=5, a0=205.0, a1=335.0)

    glasses = Mesh("glasses")
    gz = on_head(0.066, ey) + 0.022
    for s in (1, -1):
        add_torus(glasses, m_frame, (s * 0.066, ey, gz), 0.05, 0.0075, seg=16, side=5)
        add_beam(glasses, m_frame, (s * 0.116, ey, gz - 0.004), (s * 0.172, ey, gz - 0.13), 0.012, 0.012)
    add_box(glasses, m_frame, (0.0, ey + 0.004, gz + 0.002), (0.036, 0.012, 0.012))

    hair = Mesh("hair")
    rnd = lcg(7)
    N = 70
    golden = math.pi * (3 - math.sqrt(5))
    for i in range(N):
        y = 1 - 2 * (i + 0.5) / N
        rr = math.sqrt(1 - y * y)
        d = (rr * math.cos(golden * i), y, rr * math.sin(golden * i))
        ok = d[1] > 0.3
        if d[2] > 0.4:
            ok = d[1] > 0.62
        if abs(d[0]) > 0.72 and d[2] < 0.35:
            ok = d[1] > 0.0
        if d[2] < -0.35:
            ok = d[1] > -0.25
        if not ok:
            continue
        rad = 0.058 + 0.02 * next(rnd)
        c = (HC[0] + d[0] * R * 0.93, HC[1] + d[1] * R * 0.93, HC[2] + d[2] * R * 0.93)
        add_ico(hair, m_hair, c, rad, subdiv=1, scale=(1.0, 0.85 + 0.2 * next(rnd), 1.0))

    def upper_arm(s):
        m = Mesh("upper_arm_%s" % ("L" if s > 0 else "R"))
        add_box(m, m_shirt_plain, (s * 0.025, -0.035, 0.0), (0.095, 0.1, 0.115), bevel=0.025)
        add_box(m, m_skin, (s * 0.03, -0.11, 0.0), (0.068, 0.08, 0.072), bevel=0.02)
        return m

    fore = Mesh("forearm")
    add_box(fore, m_skin, (0.0, -0.05, 0.0), (0.064, 0.1, 0.068), bevel=0.02)
    add_ico(fore, m_skin, (0.0, -0.115, 0.0), 0.04, subdiv=1)

    root = Node("me_figure")
    rig = root.add(Node("rig"))                       # whole-figure hop / squash from the feet
    rig.add(Node("legs", mesh=legs))
    body = rig.add(Node("body", t=(0.0, BY, 0.0)))
    body.add(Node("tshirt", mesh=tshirt))
    headn = body.add(Node("head", t=(0.0, HY - BY, 0.0), mesh=head))
    headn.add(Node("face", mesh=face))
    headn.add(Node("glasses", mesh=glasses))
    headn.add(Node("hair", mesh=hair))
    for s, side in ((1, "L"), (-1, "R")):
        arm = body.add(Node("arm_" + side, t=(s * 0.185, 0.555 - BY, 0.0), r=q_axis(Z_AX, s * 7.0),
                            mesh=upper_arm(s)))
        arm.add(Node("forearm_" + side, t=(s * 0.03, -0.145, 0.0), r=q_axis(X_AX, -8.0), mesh=fore))

    def idle(p, t):
        sp, cp = math.sin(p), math.cos(p)
        return {("body", "rotation"): q_ypr(yaw=1.5 * math.sin(p + 0.4), roll=1.2 * math.sin(p + 1.0)),
                ("body", "scale"): (1.0 - 0.008 * sp, 1.0 + 0.018 * sp, 1.0 - 0.008 * sp),
                ("head", "rotation"): q_ypr(yaw=3.0 * math.sin(p + 1.5), pitch=2.0 * sp, roll=2.5 * math.sin(p + 0.3)),
                ("arm_L", "rotation"): q_mul(q_axis(Z_AX, 7.0 + 2.5 * sp), q_axis(X_AX, -2.0 * cp)),
                ("arm_R", "rotation"): q_mul(q_axis(Z_AX, -7.0 - 2.5 * sp), q_axis(X_AX, -2.0 * cp)),
                ("forearm_L", "rotation"): q_axis(X_AX, -8.0 - 2.0 * sp),
                ("forearm_R", "rotation"): q_axis(X_AX, -8.0 - 2.0 * sp)}

    def wave(p, t):
        w = math.sin(2 * p)                               # two waves per loop
        return {("body", "rotation"): q_ypr(yaw=-4.0, roll=2.5 + 1.0 * math.sin(p)),
                ("body", "scale"): (1.0, 1.0 + 0.01 * math.sin(p), 1.0),
                ("head", "rotation"): q_ypr(yaw=-6.0, pitch=-3.0 + 2.0 * math.sin(2 * p + 0.5), roll=7.0),
                ("arm_L", "rotation"): q_mul(q_axis(Z_AX, 8.0), q_axis(X_AX, -2.0)),
                ("arm_R", "rotation"): q_mul(q_axis(Z_AX, -128.0 - 4.0 * w), q_axis(X_AX, -12.0)),
                ("forearm_L", "rotation"): q_axis(X_AX, -10.0),
                ("forearm_R", "rotation"): q_axis(Z_AX, -52.0 + 26.0 * w)}

    kf, ease = bb.kf, bb._ease

    def jump(p, t):
        if t < 0.12:
            y, sy = 0.0, 1.0 - 0.08 * ease(t / 0.12)
        elif t < 0.42:
            u = (t - 0.12) / 0.30
            y, sy = 0.3 * 4 * u * (1 - u), 1.0 + 0.05 * math.sin(math.pi * u)
        elif t < 0.5:
            y, sy = 0.0, 1.0 - 0.1 * math.sin(math.pi * (t - 0.42) / 0.16)
        else:
            y, sy = 0.0, 1.0
        sh = kf(t, [(0.0, 7.0), (0.12, -4.0), (0.22, 150.0), (0.42, 125.0), (0.55, 7.0), (0.6, 7.0)])
        el = kf(t, [(0.0, -8.0), (0.12, 10.0), (0.22, -20.0), (0.45, -15.0), (0.6, -8.0)])
        return {("rig", "translation"): (0.0, y, 0.0),
                ("rig", "scale"): (1.0 / math.sqrt(sy), sy, 1.0 / math.sqrt(sy)),
                ("body", "rotation"): q_ypr(pitch=kf(t, [(0.0, 0.0), (0.12, 10.0), (0.25, -4.0), (0.46, 6.0), (0.6, 0.0)])),
                ("body", "scale"): (1.0, 1.0, 1.0),
                ("head", "rotation"): q_ypr(pitch=kf(t, [(0.0, 0.0), (0.12, 8.0), (0.25, -10.0), (0.46, 6.0), (0.6, 0.0)])),
                ("arm_L", "rotation"): q_axis(Z_AX, sh), ("arm_R", "rotation"): q_axis(Z_AX, -sh),
                ("forearm_L", "rotation"): q_axis(X_AX, el), ("forearm_R", "rotation"): q_axis(X_AX, el)}

    def cheer(p, t):
        hop = abs(math.sin(p))                        # two hops a second
        w = math.sin(2 * p)
        sy = 1.0 - 0.06 * max(0.0, 1.0 - 4 * hop)
        return {("rig", "translation"): (0.0, 0.05 * hop, 0.0),
                ("rig", "scale"): (1.0 / math.sqrt(sy), sy, 1.0 / math.sqrt(sy)),
                ("body", "rotation"): q_ypr(yaw=3.0 * math.sin(p + 0.5), roll=4.0 * math.sin(p)),
                ("body", "scale"): (1.0, 1.0, 1.0),
                ("head", "rotation"): q_ypr(pitch=-6.0, roll=-5.0 * math.sin(p)),
                ("arm_L", "rotation"): q_axis(Z_AX, 148.0 + 8.0 * w),
                ("arm_R", "rotation"): q_axis(Z_AX, -148.0 + 8.0 * w),
                ("forearm_L", "rotation"): q_axis(Z_AX, 18.0 * w),
                ("forearm_R", "rotation"): q_axis(Z_AX, 18.0 * w)}

    anims = {"idle": clip(2.4, 48, idle), "wave": clip(1.2, 36, wave),
             "jump": clip(0.6, 15, jump), "cheer": clip(1.0, 20, cheer)}
    anims = bb.fill_coverage(root, anims)
    return "me_figure", root, M.d, [("tshirt_ME", tex)], anims, ("idle", "wave", "jump", "cheer")

# ------------------------------------------------------------------ giant laptop

def render_chart(sw, sh):
    bg, grid, axis = hex_rgb("#0E1828"), hex_rgb("#1F3550"), hex_rgb("#3A5878")
    fill, line_c, dot_c, label_c = hex_rgb("#123B33"), hex_rgb("#3BF08E"), hex_rgb("#A6FFD0"), hex_rgb("#9DB4CC")
    x0, x1, y0, y1 = 0.10, sw - 0.08, 0.08, sh - 0.2
    vals = [0.12, 0.22, 0.17, 0.30, 0.26, 0.38, 0.33, 0.47, 0.55, 0.49, 0.62, 0.74, 0.68, 0.82, 0.93]
    pts = [(x0 + (x1 - x0) * i / (len(vals) - 1), y1 - v * (y1 - y0) * 0.92) for i, v in enumerate(vals)]
    line = Ink([(a[0], a[1], b[0], b[1]) for a, b in zip(pts, pts[1:])], 0.016)
    label = ink_text("STOCKS", sw / 2, sh - 0.1, 0.065, weight=0.15)
    gx, gy = (x1 - x0) / 12, (y1 - y0) / 6
    end = pts[-1]

    def line_y(x):
        for a, b in zip(pts, pts[1:]):
            if a[0] <= x <= b[0]:
                return a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0])
        return None

    def sample(x, y):
        if label.hit(x, y):
            return label_c
        if (x - end[0]) ** 2 + (y - end[1]) ** 2 <= 0.022 ** 2:
            return dot_c
        if line.hit(x, y):
            return line_c
        if not (x0 - 0.004 <= x <= x1 and y0 <= y <= y1 + 0.004):
            return bg
        if abs(x - x0) <= 0.003 or abs(y - y1) <= 0.003:
            return axis
        ly = line_y(x)
        base = fill if (ly is not None and y > ly) else bg
        if abs(((x - x0) / gx) - round((x - x0) / gx)) * gx <= 0.002 or \
           abs(((y - y0) / gy) - round((y - y0) / gy)) * gy <= 0.002:
            return grid if base == bg else hex_rgb("#1D5245")
        return base

    return raster(512, 256, sw, sh, sample, [(x0 - 0.02, y0 - 0.03, x1 + 0.03, y1 + 0.02), label.bb], ss=3)


def render_glow(gw, gh, sw, sh):
    """Soft halo around the screen (RGBA): faint over the screen, fading out past its edges."""
    def sample(x, y):
        dx = max(abs(x - gw / 2) - sw / 2, 0.0)
        dy = max(abs(y - gh / 2) - sh / 2, 0.0)
        d = math.hypot(dx, dy)
        if d <= 0.0:
            a = 0.07
        else:
            a = 0.45 * max(0.0, 1.0 - d / 0.12) ** 2
        return (150, 255, 200, int(round(a * 255)))
    return raster(128, 128, gw, gh, sample, alpha=True, ss=1)


def build_laptop():
    M = Mats()
    m_silver = M("aluminium", "#C9CED5", rough=0.45)
    m_deck = M("keyboard_deck", "#202328")
    m_key = M("keys", "#2E3238", rough=0.7)
    m_pad = M("trackpad", "#B9BEC5", rough=0.4)
    m_bezel = M("bezel", "#121417", rough=0.4)
    m_screen = M.raw("screen", {"pbrMetallicRoughness": {"baseColorFactor": [0.02, 0.02, 0.02, 1.0],
                                                         "metallicFactor": 0.0, "roughnessFactor": 0.35},
                                "emissiveTexture": {"index": 0}, "emissiveFactor": [1.0, 1.0, 1.0]})
    m_glow = M.raw("screen_glow", {"pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                                                            "baseColorTexture": {"index": 1},
                                                            "metallicFactor": 0.0, "roughnessFactor": 1.0},
                                   "emissiveTexture": {"index": 1}, "emissiveFactor": [0.55, 1.0, 0.75],
                                   "alphaMode": "BLEND"})
    SW, SH = 1.44, 0.86
    GW, GH = 1.68, 1.1
    textures = [("stock_chart", render_chart(SW, SH)), ("screen_glow", render_glow(GW, GH, SW, SH))]

    base = Mesh("base")
    add_box(base, m_silver, (0.0, 0.03, 0.0), (1.6, 0.06, 1.1), bevel=0.02)
    add_box(base, m_deck, (0.0, 0.062, -0.13), (1.38, 0.006, 0.46))
    for r in range(4):
        z = -0.31 + 0.085 * r
        for c in range(13):
            add_box(base, m_key, (-0.588 + 0.098 * c, 0.07, z), (0.082, 0.014, 0.07))
    z = -0.31 + 0.085 * 4
    for c in list(range(4)) + list(range(9, 13)):
        add_box(base, m_key, (-0.588 + 0.098 * c, 0.07, z), (0.082, 0.014, 0.07))
    add_box(base, m_key, (0.0, 0.07, z), (0.47, 0.014, 0.07))
    add_box(base, m_pad, (0.0, 0.061, 0.3), (0.5, 0.004, 0.26))
    add_box(base, m_bezel, (0.0, 0.065, -0.53), (1.3, 0.035, 0.04))

    lid = Mesh("lid")
    add_box(lid, m_silver, (0.0, 0.52, -0.0175), (1.6, 1.04, 0.035), bevel=0.015)
    add_box(lid, m_bezel, (0.0, 0.52, 0.002), (1.56, 1.0, 0.004))
    add_box(lid, m_bezel, (0.0, 0.545, 0.0045), (SW, SH, 0.003), faces={"+z": m_screen})

    glow = Mesh("glow")
    gw, gh = GW / 2, GH / 2
    glow.poly(m_glow, [(-gw, -gh, 0.0), (gw, -gh, 0.0), (gw, gh, 0.0), (-gw, gh, 0.0)], (0.0, 0.0, 1.0),
              [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)])

    root = Node("laptop_giant")
    root.add(Node("base", mesh=base))
    lidn = root.add(Node("lid", t=(0.0, 0.06, -0.515), r=q_axis(X_AX, -15.0), mesh=lid))
    lidn.add(Node("glow", t=(0.0, 0.545, 0.014), mesh=glow))

    def fn(p, t):
        s = 1.0 + 0.04 * (1 - math.cos(p)) / 2
        return {("glow", "scale"): (s, s, 1.0)}
    anims = {"idle": clip(2.0, 32, fn)}
    return "laptop_giant", root, M.d, textures, anims, ("idle",)

# ------------------------------------------------------------------ rupee notes + coin

def render_note(nw, nh):
    """Generic note face (not a real banknote design): lavender and pale green panels,
    frame, a rupee sign in a ring, and 500."""
    lav, green = hex_rgb("#D9CFEC"), hex_rgb("#CFE8D6")
    frame, ink1, ink2, ring = hex_rgb("#9C8DBE"), hex_rgb("#5B4A8B"), hex_rgb("#4C3D7A"), hex_rgb("#B9ACD9")
    rupee = ink_text("R", 0.21, nh / 2, 0.24, weight=0.13)
    big = Ink(text_segs("500", 0.93 - text_width("500", 0.15), nh / 2 - 0.075, 0.15), 0.15 * 0.14)
    small = Ink(text_segs("500", 0.045, 0.035, 0.05), 0.05 * 0.16)

    def sample(x, y):
        if rupee.hit(x, y):
            return ink1
        if big.hit(x, y) or small.hit(x, y):
            return ink2
        e = min(x, y, nw - x, nh - y)
        if 0.016 <= e <= 0.022:
            return frame
        d = math.hypot(x - 0.21, y - nh / 2)
        if 0.155 <= d <= 0.163 or 0.175 <= d <= 0.179:
            return ring
        return green if x > nw * 0.66 else lav
    return raster(512, 256, nw, nh, sample, [rupee.bb, big.bb, small.bb, (0.0, 0.0, 0.4, nh)], ss=3)


def render_coin():
    gold, dark, ring = hex_rgb("#E9B941"), hex_rgb("#A0731F"), hex_rgb("#B98A2A")
    sym = ink_text("R", 0.5, 0.5, 0.5, weight=0.14)

    def sample(x, y):
        if sym.hit(x, y):
            return dark
        d = math.hypot(x - 0.5, y - 0.5)
        if 0.39 <= d <= 0.43:
            return ring
        return gold
    return raster(128, 128, 1.0, 1.0, sample, [sym.bb, (0.05, 0.05, 0.95, 0.95)], ss=3)


def build_rupee():
    M = Mats()
    m_paper = M("note_paper", "#CFC3E6")
    m_print = M.raw("note_print", material("note_print", "#FFFFFF", texture=0))
    m_band = M("band", "#EFE3C2")
    m_coin = M.raw("coin_face", dict(material("coin_face", "#FFFFFF", texture=1, rough=0.35), **{}))
    m_rim = M("coin_rim", "#D4A22E", rough=0.35)
    NW, NH, NT = 1.0, 0.44, 0.012
    textures = [("note_face", render_note(NW, NH)), ("coin_face", render_coin())]

    notes = Mesh("notes")
    jitter = [(-2.5, -0.01, 0.008), (1.8, 0.012, -0.006), (-1.2, -0.006, -0.01), (2.2, 0.008, 0.01), (0.0, 0.0, 0.0)]
    for i, (yaw, dx, dz) in enumerate(jitter):
        y = NT / 2 + i * (NT + 0.0015)
        faces = {"+y": m_print} if i == len(jitter) - 1 else None
        add_box(notes, m_paper, (dx - 0.16, y, dz), (NW, NT, NH), rot=q_axis(Y_AX, yaw), faces=faces)
    stack_h = len(jitter) * (NT + 0.0015)
    # paper band between the rupee medallion and the 500 (note-local x = -0.01)
    add_box(notes, m_band, (-0.17, (stack_h + 0.004) / 2, 0.0), (0.12, stack_h + 0.004, NH + 0.012))

    coin = Mesh("coin")
    CR, CT = 0.15, 0.04
    tmp = Mesh("tmp")
    add_cyl(tmp, m_rim, (0.0, 0.0, 0.0), CR, CR, CT, 28, caps=(False, False))
    add_disc(tmp, m_coin, (0.0, CT / 2, 0.0), CR, 28, up=True)
    add_disc(tmp, m_coin, (0.0, -CT / 2, 0.0), CR, 28, up=False, flip_u=True)
    merge(coin, tmp, rot=q_axis(X_AX, 90.0))

    root = Node("rupee_note")
    spin = root.add(Node("spin", mesh=notes))
    spin.add(Node("coin", t=(0.52, CR, 0.1), mesh=coin))     # stands beside the stack

    def fn(p, t):
        return {("spin", "rotation"): q_axis(Y_AX, 360.0 * t / 2.0)}
    anims = {"idle": clip(2.0, 12, fn)}
    return "rupee_note", root, M.d, textures, anims, ("idle",)

# ------------------------------------------------------------------ standalone data crate

def build_crate():
    M = Mats()
    k = 0.5 / bb.CRATE["w"]
    h = bb.CRATE["h"] * k
    root = Node("data_crate")
    root.add(Node("crate", t=(0.0, h / 2, 0.0), mesh=bb.crate_mesh(M, k)))

    def fn(p, t):
        return {("crate", "translation"): (0.0, h / 2 + 0.06 * (1 - math.cos(p)) / 2, 0.0),
                ("crate", "rotation"): q_axis(Y_AX, 6.0 * math.sin(p))}
    anims = {"idle": clip(1.6, 32, fn)}
    return "data_crate", root, M.d, [], anims, ("idle",)

# ------------------------------------------------------------------ main

BUILDERS = [build_tanker, build_container_ship, build_submarine, build_ml_bot, build_me, build_laptop,
            build_rupee, build_crate]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="validate existing files only")
    ap.add_argument("--only", help="comma-separated prop names to build")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for fn in BUILDERS:
        if args.check:
            continue
        name, root, mats, textures, anims, required = fn()
        if args.only and name not in args.only.split(","):
            continue
        path = OUT_DIR / (name + ".glb")
        write_glb(path, name, root, mats, textures, anims)
        jobs.append((path, required))
    if args.check:
        req = {"bot_ML": ("idle", "hammer") + tuple(bb.ACTION_TIMES), "me_figure": ("idle", "wave", "jump", "cheer")}
        jobs = [(p, req.get(p.stem, ("idle",))) for p in sorted(OUT_DIR.glob("*.glb"))]
    print("%-20s %8s %6s %6s  %-26s %s" % ("file", "bytes", "tris", "nodes", "animations (s)", "rest bbox (w x h x d)"))
    for path, required in jobs:
        st = validate(path, required=required, max_bytes=MAX_BYTES, max_tris=MAX_TRIS, oneshot=bb.ONE_SHOT)
        lo, hi = st["bbox"]
        an = ", ".join("%s %.2f" % kv for kv in st["anims"].items())
        print("%-20s %8d %6d %6d  %-26s %.2f x %.2f x %.2f (min y %.3f)" % (
            path.name, st["bytes"], st["tris"], st["nodes"], an,
            hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], lo[1]))
    print("ok: %d files valid" % len(jobs))


if __name__ == "__main__":
    main()
