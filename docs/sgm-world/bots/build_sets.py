#!/usr/bin/env python3
"""
SGM Data World - set pieces generator (toy diorama scale: a bot is 1.0 tall,
buildings 3-6). Binary glTF 2.0 with node-TRS animations, no extensions.

    nse_plaza  fed_stl  ecb_tower  pib_office  sbi_bank          buildings      idle
    port_crane                                                   quay crane     idle, work
    cable_landing                                                beach + cable  idle (buoy bob)
    desk  anvil  list_scroll  phone_stand  calendar  broker_desk  props          idle (+ hit / buzz / flip)
    telegram_tower                                               lattice tower  idle (beacon blink)
    palm_tree  oak_tree  rock  lamp_post  bench  bush            dressing       idle

Pure Python 3 standard library; reuses build_bots.py / build_props.py next to
this file. Deterministic. +Y up, front faces +Z, origin at the centre of the base.

    python3 docs/sgm-world/bots/build_sets.py            # build all + validate
    python3 docs/sgm-world/bots/build_sets.py --check    # validate existing files only
"""

import argparse
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_bots as bb  # noqa: E402
import build_props as bp  # noqa: E402
from build_bots import Mesh, Node, add_box, add_cyl, write_glb, validate, raster, q_axis, q_mul, q_ypr, \
    v_unit, v_cross, v_sub, v_dot, X_AX, Y_AX, Z_AX, hex_rgb  # noqa: E402
from build_props import Mats, merge, add_rod, add_beam, add_ico, add_torus, add_lathe, lcg, clip, Ink  # noqa: E402

OUT_DIR = bb.REPO / "shreeganeshmodel-deploy" / "assets" / "sets"
MAX_BYTES = 150 * 1024
MAX_TRIS = 5000
HIDDEN = (1e-4, 1e-4, 1e-4)
kf, ease = bb.kf, bb._ease

# ------------------------------------------------------------------ helpers

def static_idle(node_name, node):
    """'idle' for a still prop: the main node held at its rest pose."""
    return {"idle": [(node_name, "scale", [0.0, 2.0], [tuple(node.s), tuple(node.s)])]}


def add_text3d(mesh, mat, text, cx, y0, z, h, depth, stroke):
    """Chunky 3D letters from the stroke font (front face at z + depth/2)."""
    w = bp.text_width(text, h)
    for ax, ay, bx, by in bp.text_segs(text, cx - w / 2, 0.0, h):
        x0, y_0, x1, y_1 = ax, y0 + h - ay, bx, y0 + h - by
        L = math.hypot(x1 - x0, y_1 - y_0)
        ang = math.degrees(math.atan2(y_1 - y_0, x1 - x0))
        add_box(mesh, mat, ((x0 + x1) / 2, (y_0 + y_1) / 2, z), (L + stroke, stroke, depth), rot=q_axis(Z_AX, ang))


def add_prism_x(mesh, mat, center, length, width, height):
    """Gable roof: ridge along X, triangle in the ZY plane, base on center's y."""
    cx, cy, cz = center
    hx, hz = length / 2, width / 2
    for s in (1, -1):
        mesh.poly(mat, [(s * hx + cx, cy, cz - hz), (s * hx + cx, cy, cz + hz), (s * hx + cx, cy + height, cz)],
                  (float(s), 0.0, 0.0))
    for s in (1, -1):
        p = [(cx - hx, cy, cz + s * hz), (cx + hx, cy, cz + s * hz), (cx + hx, cy + height, cz), (cx - hx, cy + height, cz)]
        n = v_unit(v_cross(v_sub(p[1], p[0]), v_sub(p[3], p[0])))
        if n[2] * s < 0:
            n = (-n[0], -n[1], -n[2])
        mesh.poly(mat, p, n)
    mesh.poly(mat, [(cx - hx, cy, cz - hz), (cx + hx, cy, cz - hz), (cx + hx, cy, cz + hz), (cx - hx, cy, cz + hz)],
              (0.0, -1.0, 0.0))


def add_prism_z(mesh, mat, center, width, height, depth):
    """Pediment: triangle in the XY plane (base width, apex height), extruded along Z."""
    cx, cy, cz = center
    hw, hd = width / 2, depth / 2
    for s in (1, -1):
        mesh.poly(mat, [(cx - hw, cy, cz + s * hd), (cx + hw, cy, cz + s * hd), (cx, cy + height, cz + s * hd)],
                  (0.0, 0.0, float(s)))
    for s in (1, -1):
        p = [(cx + s * hw, cy, cz - hd), (cx + s * hw, cy, cz + hd), (cx, cy + height, cz + hd), (cx, cy + height, cz - hd)]
        n = v_unit(v_cross(v_sub(p[1], p[0]), v_sub(p[3], p[0])))
        if n[0] * s < 0:
            n = (-n[0], -n[1], -n[2])
        mesh.poly(mat, p, n)
    mesh.poly(mat, [(cx - hw, cy, cz - hd), (cx + hw, cy, cz - hd), (cx + hw, cy, cz + hd), (cx - hw, cy, cz + hd)],
              (0.0, -1.0, 0.0))


def add_rock(mesh, mat, center, r, seed, scale=(1.0, 0.65, 0.85)):
    """Flat-shaded jittered icosphere boulder sitting on center's y."""
    verts, faces = bp._ico(1)
    rnd = lcg(seed)
    jit = [1.0 + 0.36 * (next(rnd) - 0.5) for _ in verts]
    cx, cy, cz = center
    lift = r * scale[1] * 0.78
    pts = [(cx + v[0] * r * j * scale[0], cy + lift + v[1] * r * j * scale[1], cz + v[2] * r * j * scale[2])
           for v, j in zip(verts, jit)]
    for a, b, c in faces:
        pa, pb, pc = pts[a], pts[b], pts[c]
        n = v_unit(v_cross(v_sub(pb, pa), v_sub(pc, pa)))
        cen = ((pa[0] + pb[0] + pc[0]) / 3 - cx, (pa[1] + pb[1] + pc[1]) / 3 - cy - lift, (pa[2] + pb[2] + pc[2]) / 3 - cz)
        if v_dot(n, cen) < 0:
            n = (-n[0], -n[1], -n[2])
        mesh.poly(mat, [pa, pb, pc], n)


# ------------------------------------------------------------------ buildings

def build_nse():
    M = Mats()
    W = Mesh("nse_plaza")
    conc, conc2 = M("concrete", "#E4E1DA"), M("concrete_dark", "#C9C5BC")
    glass, glass_d = M("glass", "#5FA8C8", rough=0.25), M("glass_dark", "#2F5F80", rough=0.25)
    pave, navy, leaf = M("paving", "#D8D2C6"), M("sign_letters", "#1F3A6E", rough=0.5), M("leaf", "#5E9E4A")
    add_box(W, pave, (0.0, 0.04, 0.95), (5.6, 0.08, 1.9))
    bz, bd, bw = -0.85, 2.0, 4.8
    add_box(W, glass, (0.0, 0.08 + 1.55, bz), (bw - 0.12, 3.1, bd - 0.12))
    for k in range(7):
        add_box(W, conc, (0.0, 0.12 + 0.5 * k, bz), (bw + 0.04, 0.09, bd + 0.04))
    add_box(W, conc, (0.0, 3.22, bz), (bw + 0.06, 0.14, bd + 0.06))
    add_box(W, conc2, (0.9, 3.5, bz - 0.2), (1.6, 0.42, 0.9), bevel=0.03)
    fz = bz + bd / 2 + 0.07
    for k in range(12):
        x = -2.2 + 0.4 * k
        if abs(x) < 0.5:
            continue
        add_box(W, conc, (x, 0.08 + 1.57, fz), (0.07, 3.14, 0.16))
    for s in (1, -1):                                   # side fins
        for k in range(4):
            add_box(W, conc, (s * (bw / 2 + 0.05), 0.08 + 1.57, bz - 0.75 + 0.5 * k), (0.12, 3.14, 0.07))
    # central entrance: dark glass doors, canopy on posts, tall atrium strip above
    add_box(W, glass_d, (0.0, 0.08 + 0.3, fz - 0.05), (1.0, 0.6, 0.04))
    add_box(W, glass_d, (0.0, 0.08 + 1.8, fz - 0.06), (0.8, 2.4, 0.04))
    add_box(W, conc, (0.0, 0.76, fz + 0.32), (1.6, 0.06, 0.7), bevel=0.015)
    for s in (1, -1):
        add_rod(W, conc2, (s * 0.7, 0.08, fz + 0.6), (s * 0.7, 0.74, fz + 0.6), 0.03, seg=6)
    # sign block with plain letters
    add_box(W, conc2, (-1.45, 0.08 + 0.22, 1.25), (1.5, 0.44, 0.34), bevel=0.02)
    add_text3d(W, navy, "NSE", -1.45, 0.08 + 0.1, 1.25 + 0.17 + 0.03, 0.24, 0.06, 0.05)
    for x in (1.2, 2.2):
        add_box(W, conc2, (x, 0.08 + 0.15, 1.25), (0.5, 0.3, 0.5), bevel=0.02)
        add_ico(W, leaf, (x, 0.08 + 0.42, 1.25), 0.26, subdiv=1, scale=(1.0, 0.8, 1.0))
    root = Node("nse_plaza")
    n = root.add(Node("building", mesh=W))
    return "nse_plaza", root, M.d, [], static_idle("building", n), ("idle",), True


def build_fed():
    M = Mats()
    W = Mesh("fed_stl")
    st, st2 = M("stone", "#E9E1CF"), M("stone_shade", "#D3C8B1")
    roof, win = M("roof", "#9AA3AD"), M("window", "#33404D", rough=0.4)
    metal, glow = M("lamp_metal", "#2A2D33"), M("lamp_glass", "#FFE7A3", emissive="#FFD27A")
    add_box(W, st2, (0.0, 0.25, -0.5), (4.2, 0.5, 2.6), bevel=0.02)              # podium z -1.8 .. 0.8
    for k in range(4):                                                         # steps forward of it
        dz = 0.25 * (4 - k)
        add_box(W, st, (0.0, 0.0625 + 0.125 * k, 0.8 + dz / 2), (3.0, 0.125, dz))
    add_box(W, st, (0.0, 0.5 + 1.0, -0.75), (3.6, 2.0, 1.7))                     # hall behind the colonnade
    add_box(W, win, (0.0, 0.5 + 0.6, 0.11), (0.6, 1.2, 0.04))
    for x in (-1.25, -0.75, 0.75, 1.25):
        add_box(W, win, (x, 0.5 + 0.95, 0.11), (0.28, 0.9, 0.04))
    for k in range(6):                                                         # columns
        x = -1.5 + 0.6 * k
        add_box(W, st, (x, 0.54, 0.45), (0.34, 0.08, 0.34))
        add_cyl(W, st, (x, 0.58 + 0.79, 0.45), 0.13, 0.115, 1.58, 12)
        add_box(W, st, (x, 2.2, 0.45), (0.36, 0.1, 0.36))
    add_box(W, st, (0.0, 2.4, -0.45), (3.9, 0.3, 2.6))                           # entablature
    add_box(W, st2, (0.0, 2.29, 0.86), (3.92, 0.06, 0.02))
    add_prism_z(W, st, (0.0, 2.55, -0.45), 3.9, 0.62, 2.6)                       # pediment + roof
    add_prism_z(W, st2, (0.0, 2.62, 0.86), 3.2, 0.42, 0.02)                      # recessed tympanum
    add_box(W, roof, (0.0, 2.555, -0.45), (3.92, 0.01, 2.62))
    for s in (1, -1):                                                          # lamp posts
        x, z = s * 1.75, 1.9
        add_cyl(W, metal, (x, 0.06, z), 0.09, 0.08, 0.12, 8)
        add_rod(W, metal, (x, 0.1, z), (x, 1.35, z), 0.035, seg=8)
        add_box(W, glow, (x, 1.48, z), (0.17, 0.24, 0.17), bevel=0.02)
        bp.add_pyramid(W, metal, (x, 1.6, z), 0.11, 0.11, 0.12)
    root = Node("fed_stl")
    n = root.add(Node("building", mesh=W))
    return "fed_stl", root, M.d, [], static_idle("building", n), ("idle",), True


def build_ecb():
    M = Mats()
    W = Mesh("ecb_tower")
    glass, glass_l = M("glass", "#6FA3C8", rough=0.25), M("glass_light", "#A9D2E8", rough=0.2)
    mull, hall = M("mullion", "#EEF2F5"), M("hall", "#D8CCB8")
    roof, plaza, dark = M("hall_roof", "#8E969F"), M("plaza", "#CFCAC0"), M("dark_glass", "#33404D", rough=0.4)
    add_box(W, plaza, (0.0, 0.04, 0.0), (5.0, 0.08, 3.4))
    add_box(W, hall, (0.0, 0.08 + 0.42, 1.0), (4.4, 0.84, 1.0), bevel=0.02)       # Grossmarkthalle in front
    add_box(W, dark, (0.0, 0.5, 1.51), (4.0, 0.26, 0.02))
    add_box(W, roof, (0.0, 0.98, 1.0), (4.5, 0.12, 1.1), bevel=0.04)
    for cx, cz, H, sgn in ((-0.72, -0.55, 6.0, 1), (0.74, -0.75, 5.4, -1)):        # twin twisted towers
        sec = H / 4
        for k in range(4):
            rot = q_axis(Y_AX, sgn * (4.0 * k + 6.0))
            y = 0.08 + sec * k + sec / 2
            add_box(W, glass, (cx, y, cz), (0.95, sec + 0.01, 0.75), rot=rot)
            for j in range(3):
                add_box(W, mull, (cx, 0.08 + sec * k + sec * (j + 1) / 3, cz), (0.97, 0.035, 0.77), rot=rot)
        add_box(W, mull, (cx, 0.08 + H + 0.05, cz), (0.8, 0.1, 0.6), rot=q_axis(Y_AX, sgn * 18.0))
    add_box(W, glass_l, (0.02, 0.08 + 2.3, -0.65), (0.62, 4.6, 0.85))             # atrium between them
    for k in range(4):
        add_box(W, mull, (0.02, 0.08 + 0.9 + 1.1 * k, -0.65), (0.64, 0.05, 0.87))
    root = Node("ecb_tower")
    n = root.add(Node("building", mesh=W))
    return "ecb_tower", root, M.d, [], static_idle("building", n), ("idle",), True


def render_india_flag(fw, fh):
    return raster(128, 64, fw, fh, lambda x, y: bb.flag_in(x / fh, min(y / fh, 0.99999)),
                  [(0.75 * fh - 0.16 * fh, 0.34 * fh, 0.75 * fh + 0.16 * fh, 0.66 * fh)])


def build_pib():
    M = Mats()
    W = Mesh("pib_office")
    sand, red = M("sandstone", "#D9A86C"), M("red_sandstone", "#B5653F")
    win, dome, gold = M("window", "#3A2E2A", rough=0.4), M("dome", "#D2A066"), M("finial", "#D4A22E", rough=0.4)
    pole = M("flagpole", "#F1F1EC")
    m_flag = M.raw("flag_india", bb.material("flag_india", "#FFFFFF", texture=0))
    FW, FH = 0.6, 0.4
    textures = [("flag_india", render_india_flag(FW, FH))]
    add_box(W, red, (0.0, 0.2, 0.0), (4.1, 0.4, 2.1))
    add_box(W, sand, (0.0, 0.4 + 0.75, 0.0), (4.0, 1.5, 2.0))
    add_box(W, red, (0.0, 1.95, 0.0), (4.16, 0.1, 2.16))
    add_box(W, sand, (0.0, 2.06, 0.0), (4.0, 0.12, 2.0))
    for row, y in enumerate((0.75, 1.45)):                                     # arched windows
        for k in range(6):
            x = -1.65 + 0.66 * k
            if abs(x) < 0.5:
                continue
            add_box(W, win, (x, y, 1.005), (0.3, 0.42, 0.03))
            add_rod(W, win, (x, y + 0.21, 0.99), (x, y + 0.21, 1.02), 0.15, seg=12)
    add_box(W, sand, (0.0, 0.4 + 0.85, 1.15), (1.2, 1.7, 0.36), bevel=0.02)       # portico
    add_box(W, red, (0.0, 2.15, 1.15), (1.3, 0.1, 0.42))
    add_box(W, win, (0.0, 0.4 + 0.45, 1.335), (0.5, 0.9, 0.03))
    add_rod(W, win, (0.0, 1.3, 1.32), (0.0, 1.3, 1.35), 0.25, seg=14)
    add_cyl(W, sand, (0.0, 2.12 + 0.17, 0.0), 0.46, 0.46, 0.34, 16)              # drum
    add_cyl(W, red, (0.0, 2.48, 0.0), 0.5, 0.5, 0.05, 16)
    tmp = Mesh("tmp")
    add_lathe(tmp, dome, (0.0, 0.0, 0.0), [(0.0, 0.46), (0.12, 0.47), (0.26, 0.42), (0.38, 0.3), (0.46, 0.15), (0.5, 0.04), (0.52, 0.0)], 16)
    merge(W, tmp, rot=q_axis(X_AX, -90.0), off=(0.0, 2.5, 0.0))
    add_rod(W, gold, (0.0, 3.0, 0.0), (0.0, 3.2, 0.0), 0.02, seg=6)
    add_ico(W, gold, (0.0, 3.12, 0.0), 0.045, subdiv=1)
    px = 1.55
    add_cyl(W, sand, (px, 2.17, 0.0), 0.12, 0.1, 0.1, 8)
    add_rod(W, pole, (px, 2.12, 0.0), (px, 3.5, 0.0), 0.025, seg=8)
    add_ico(W, gold, (px, 3.52, 0.0), 0.04, subdiv=1)
    flag = Mesh("flag")
    pts = [(0.0, -FH, 0.0), (FW, -FH, 0.0), (FW, 0.0, 0.0), (0.0, 0.0, 0.0)]
    uvs = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    flag.poly(m_flag, pts, (0.0, 0.0, 1.0), uvs)
    flag.poly(m_flag, pts, (0.0, 0.0, -1.0), uvs)        # back face shows the mirror image, like real cloth
    root = Node("pib_office")
    root.add(Node("building", mesh=W))
    root.add(Node("flag", t=(px + 0.025, 3.45, 0.0), mesh=flag))

    def fn(p, t):
        return {("flag", "rotation"): q_mul(q_axis(Y_AX, 7.0 * math.sin(p)), q_axis(X_AX, 2.0 * math.sin(2 * p + 0.5)))}
    return "pib_office", root, M.d, textures, {"idle": clip(2.4, 24, fn)}, ("idle",), True


def build_sbi():
    M = Mats()
    W = Mesh("sbi_bank")
    glass, glass_d = M("glass", "#3E7CC9", rough=0.25), M("glass_dark", "#24497D", rough=0.25)
    white, plaza = M("white", "#F1F1EC"), M("plaza", "#D8D2C6")
    planter, leaf = M("planter", "#8E969F"), M("leaf", "#5E9E4A")
    add_box(W, plaza, (0.0, 0.04, 0.0), (4.2, 0.08, 3.6))
    add_box(W, glass_d, (0.0, 0.08 + 0.32, -0.3), (2.5, 0.64, 2.0))
    add_box(W, white, (0.0, 0.78, 0.85), (2.7, 0.06, 0.7), bevel=0.015)
    tz, tw, td, th = -0.4, 2.0, 1.7, 4.2
    y0 = 0.72
    add_box(W, glass, (0.0, y0 + th / 2, tz), (tw, th, td))
    for k in range(11):
        add_box(W, white, (0.0, y0 + 0.4 * k + 0.02, tz), (tw + 0.04, 0.06, td + 0.04))
    for sx in (1, -1):
        for sz in (1, -1):
            add_box(W, white, (sx * tw / 2, y0 + th / 2, tz + sz * td / 2), (0.09, th, 0.09))
    for k in range(4):
        add_box(W, white, (-0.6 + 0.4 * k, y0 + th / 2, tz + td / 2 + 0.02), (0.035, th, 0.035))
    add_box(W, white, (0.0, y0 + th + 0.15, tz), (tw + 0.1, 0.3, td + 0.1), bevel=0.02)
    add_beam(W, glass_d, (-0.9, y0 + th + 0.3, tz + 0.6), (0.9, y0 + th + 0.75, tz + 0.6), 0.1, 0.06)
    add_box(W, white, (1.35, 0.08 + 0.2, 1.25), (0.9, 0.4, 0.18), bevel=0.02)       # blank sign plinth
    for x in (-1.5, -0.6):
        add_box(W, planter, (x, 0.08 + 0.13, 1.3), (0.5, 0.26, 0.5), bevel=0.02)
        add_ico(W, leaf, (x, 0.08 + 0.4, 1.3), 0.27, subdiv=1, scale=(1.0, 0.85, 1.0))
    root = Node("sbi_bank")
    n = root.add(Node("building", mesh=W))
    return "sbi_bank", root, M.d, [], static_idle("building", n), ("idle",), True

# ------------------------------------------------------------------ port crane

def build_crane():
    M = Mats()
    W = Mesh("quay_crane")
    conc, yel, dark = M("concrete", "#CFCAC0"), M("edge_yellow", "#F2C230"), M("rail", "#3A3E44")
    crane, white, cab = M("crane", "#E4572E"), M("crane_white", "#F4F4F0"), M("cab_glass", "#2F3B4C", rough=0.3)
    black, blue = M("rubber", "#1E1F22"), M("container", "#2F64B3")
    add_box(W, conc, (0.0, 0.2, 0.0), (4.4, 0.4, 3.0))
    add_box(W, yel, (0.0, 0.405, 1.44), (4.4, 0.012, 0.1))
    for x in (-1.6, 0.0, 1.6):
        add_cyl(W, dark, (x, 0.46, 1.3), 0.07, 0.08, 0.12, 8)
        add_box(W, black, (x, 0.2, 1.54), (0.36, 0.26, 0.08), bevel=0.02)
    for z in (-0.7, 0.8):
        add_box(W, dark, (0.0, 0.41, z), (4.4, 0.02, 0.07))
    legs_z, legs_x, top = (-0.7, 0.8), (-1.0, 1.0), 3.8
    for z in legs_z:
        for x in legs_x:
            add_box(W, crane, (x, 0.4 + (top - 0.4) / 2, z), (0.16, top - 0.4, 0.16))
            add_box(W, dark, (x, 0.5, z), (0.42, 0.18, 0.3), bevel=0.02)
        add_box(W, crane, (0.0, top - 0.05, z), (2.16, 0.18, 0.18))
        add_beam(W, crane, (-1.0, 1.4, z), (1.0, top - 0.2, z), 0.08, 0.08)
    for x in legs_x:
        add_box(W, crane, (x, top - 0.05, 0.05), (0.18, 0.2, 1.7))
        add_box(W, crane, (x, 1.6, 0.05), (0.12, 0.12, 1.5))
    bz0, bz1, by = -2.0, 3.8, 3.98
    for x in (-0.45, 0.45):
        add_box(W, white, (x, by, (bz0 + bz1) / 2), (0.14, 0.22, bz1 - bz0))
    for k in range(8):
        add_box(W, white, (0.0, by + 0.08, bz0 + 0.3 + 0.75 * k), (0.9, 0.06, 0.06))
    for x in (-0.55, 0.55):                                                      # apex posts and stays
        add_box(W, crane, (x, (top + 5.1) / 2, -0.25), (0.14, 5.1 - top, 0.14))
        add_beam(W, white, (x, 5.05, -0.25), (x * 0.82, by + 0.1, bz1 - 0.1), 0.05, 0.05)
        add_beam(W, white, (x, 5.05, -0.25), (x * 0.82, by + 0.1, bz0 + 0.1), 0.05, 0.05)
    add_box(W, crane, (0.0, 5.1, -0.25), (1.26, 0.14, 0.18))
    add_box(W, white, (0.0, by + 0.38, -1.45), (1.0, 0.55, 0.9), bevel=0.03)        # machinery house
    add_box(W, cab, (0.0, by + 0.42, -0.99), (0.8, 0.12, 0.02))

    trolley = Mesh("trolley")
    add_box(trolley, dark, (0.0, 0.0, 0.0), (1.05, 0.2, 0.5), bevel=0.02)
    add_box(trolley, white, (0.42, -0.3, 0.0), (0.34, 0.3, 0.32), bevel=0.02)
    add_box(trolley, cab, (0.42, -0.3, 0.165), (0.28, 0.16, 0.01))
    cable = Mesh("cable")
    for sx in (1, -1):
        for sz in (1, -1):
            add_box(cable, black, (sx * 0.22, -0.5, sz * 0.12), (0.02, 1.0, 0.02))
    spreader = Mesh("spreader")
    add_box(spreader, yel, (0.0, -0.04, 0.0), (1.2, 0.08, 0.36), bevel=0.01)
    add_box(spreader, blue, (0.0, -0.08 - 0.21, 0.0), (1.22, 0.42, 0.42), bevel=0.015)
    for k in range(6):
        add_box(spreader, M("container_rib", "#28579C"), (-0.5 + 0.2 * k, -0.29, 0.215), (0.03, 0.38, 0.012))

    root = Node("port_crane")
    root.add(Node("crane", mesh=W))
    ty = by - 0.22
    trol = root.add(Node("trolley", t=(0.0, ty, 1.0), mesh=trolley))
    trol.add(Node("cable", t=(0.0, -0.1, 0.0), s=(1.0, 0.8, 1.0), mesh=cable))
    trol.add(Node("spreader", t=(0.0, -0.1 - 0.8, 0.0), mesh=spreader))

    def pose(z, L, sway=0.0):
        return {("trolley", "translation"): (0.0, ty, z),
                ("cable", "scale"): (1.0, L, 1.0),
                ("spreader", "translation"): (0.0, -0.1 - L, 0.0),
                ("spreader", "rotation"): q_axis(X_AX, sway)}

    land_L = ty - 0.1 - 0.5 - 0.41          # container resting on the quay top (y 0.41)
    sea_L = ty - 0.1 - 0.5 - 0.05           # down to ship-deck level over the water

    def work(p, t):
        z = kf(t, [(0.0, 3.0), (1.2, 3.0), (2.0, -0.2), (3.2, -0.2), (4.0, 3.0)])
        L = kf(t, [(0.0, 0.8), (0.6, sea_L), (1.2, 0.8), (2.0, 0.8), (2.6, land_L), (3.2, 0.8), (4.0, 0.8)])
        sway = 2.5 * math.sin(2 * p) * (1.0 if 1.2 < t < 2.0 or t > 3.2 else 0.3)
        return pose(z, L, sway)

    def idle(p, t):
        return pose(1.0, 0.8, 1.2 * math.sin(p))
    return ("port_crane", root, M.d, [], {"idle": clip(2.0, 16, idle), "work": clip(4.0, 64, work)},
            ("idle", "work"), True)

# ------------------------------------------------------------------ cable landing

def build_cable_landing():
    M = Mats()
    W = Mesh("cable_landing")
    sand, sand_d = M("sand", "#EED9A4"), M("sand_dark", "#D9BF85")
    wall, trim, roof = M("hut_wall", "#8CC4E0"), M("hut_trim", "#F4F4F0"), M("hut_roof", "#C8423A")
    door, wood, cab = M("hut_door", "#5A3A28"), M("spool_wood", "#B07A45"), M("cable", "#1A1A1A", rough=0.5)
    add_box(W, sand, (0.0, 0.125, -0.8), (3.6, 0.25, 2.0))
    # beach ramp down to the waterline (y 0 at z 1.8)
    for s in (1, -1):
        W.poly(sand_d, [(s * 1.8, 0.0, 0.2), (s * 1.8, 0.25, 0.2), (s * 1.8, 0.0, 1.8)], (float(s), 0.0, 0.0))
    top = [(-1.8, 0.25, 0.2), (1.8, 0.25, 0.2), (1.8, 0.0, 1.8), (-1.8, 0.0, 1.8)]
    W.poly(sand, top, v_unit((0.0, 1.6, 0.25)))
    W.poly(sand_d, [(-1.8, 0.0, 0.2), (1.8, 0.0, 0.2), (1.8, 0.0, 1.8), (-1.8, 0.0, 1.8)], (0.0, -1.0, 0.0))
    hx, hz = -0.85, -1.05
    add_box(W, wall, (hx, 0.25 + 0.4, hz), (1.0, 0.8, 0.9), bevel=0.02)
    for sx in (1, -1):
        for sz in (1, -1):
            add_box(W, trim, (hx + sx * 0.5, 0.65, hz + sz * 0.45), (0.06, 0.82, 0.06))
    add_prism_x(W, roof, (hx, 1.05, hz), 1.2, 1.15, 0.42)
    add_box(W, door, (hx + 0.15, 0.25 + 0.3, hz + 0.455), (0.3, 0.55, 0.02))
    add_box(W, M("hut_window", "#2F3B4C", rough=0.3), (hx - 0.25, 0.25 + 0.48, hz + 0.455), (0.24, 0.2, 0.02))
    sx_, sy_, sz_ = 0.65, 0.25 + 0.44, -1.05                                    # cable spool on its rim
    for x0 in (sx_ - 0.24, sx_ + 0.18):
        add_rod(W, wood, (x0, sy_, sz_), (x0 + 0.06, sy_, sz_), 0.44, seg=16)
    add_rod(W, cab, (sx_ - 0.18, sy_, sz_), (sx_ + 0.18, sy_, sz_), 0.33, seg=16)
    path = [(-0.35, 0.33, -0.6), (-0.05, 0.31, -0.25), (0.22, 0.29, 0.15), (0.4, 0.22, 0.6), (0.55, 0.12, 1.15),
            (0.66, 0.03, 1.6), (0.76, -0.12, 2.0), (0.86, -0.38, 2.45)]
    for a, b in zip(path, path[1:]):
        add_rod(W, cab, a, b, 0.065, seg=8)
    for p in path[1:-1]:
        add_ico(W, cab, p, 0.065, subdiv=0)
    buoy = Mesh("buoy")
    org, wht = M("buoy_orange", "#F07D24"), M("buoy_white", "#F4F4F0")
    add_cyl(buoy, org, (0.0, -0.1, 0.0), 0.06, 0.22, 0.2, 12)
    add_cyl(buoy, wht, (0.0, 0.06, 0.0), 0.22, 0.22, 0.12, 12)
    add_cyl(buoy, org, (0.0, 0.235, 0.0), 0.22, 0.07, 0.23, 12)
    add_ico(buoy, M("buoy_light", "#FFE27A", emissive="#FFC23D"), (0.0, 0.39, 0.0), 0.05, subdiv=1)
    root = Node("cable_landing")
    root.add(Node("beach", mesh=W))
    root.add(Node("buoy", t=(1.35, 0.0, 2.35), mesh=buoy))

    def fn(p, t):
        return {("buoy", "translation"): (1.35, 0.04 * math.sin(p), 2.35),
                ("buoy", "rotation"): q_ypr(pitch=6.0 * math.sin(p + 0.7), roll=5.0 * math.sin(p + 2.0))}
    return "cable_landing", root, M.d, [], {"idle": clip(2.0, 24, fn)}, ("idle",), False

# ------------------------------------------------------------------ desk-scale props

def build_desk():
    M = Mats()
    W = Mesh("desk")
    wood, wood_d, drawer = M("wood", "#A0703C"), M("wood_dark", "#7A5230"), M("drawer", "#B88350")
    metal = M("metal", "#C9CED5", rough=0.45)
    add_box(W, wood, (0.0, 0.8, 0.0), (2.4, 0.08, 1.5), bevel=0.02)
    add_box(W, wood_d, (-0.85, 0.38, 0.0), (0.55, 0.76, 1.3), bevel=0.02)
    for y in (0.16, 0.4, 0.62):
        add_box(W, drawer, (-0.85, y, 0.655), (0.48, 0.19, 0.02))
        add_box(W, metal, (-0.85, y + 0.02, 0.675), (0.16, 0.025, 0.025))
    for z in (0.62, -0.62):
        add_box(W, wood_d, (1.1, 0.38, z), (0.08, 0.76, 0.08))
    add_box(W, wood_d, (0.3, 0.56, -0.68), (1.5, 0.36, 0.03))
    # mug
    mx, my, mz = 0.95, 0.84, 0.45
    white, band = M("mug", "#F4F4F0"), M("mug_band", "#D97757")
    add_cyl(W, white, (mx, my + 0.075, mz), 0.075, 0.075, 0.15, 14)
    add_cyl(W, band, (mx, my + 0.08, mz), 0.077, 0.077, 0.04, 14, caps=(False, False))
    add_cyl(W, M("coffee", "#4A2E1E"), (mx, my + 0.146, mz), 0.066, 0.066, 0.004, 14)
    add_torus(W, white, (mx + 0.085, my + 0.08, mz), 0.04, 0.012, seg=10, side=5)
    # desk lamp
    lm, shade = M("lamp_metal", "#2A2D33"), M("lamp_shade", "#D97757")
    add_cyl(W, lm, (-0.9, 0.86, -0.45), 0.13, 0.12, 0.04, 12)
    add_rod(W, lm, (-0.9, 0.88, -0.45), (-0.95, 1.38, -0.55), 0.022, seg=6)
    add_rod(W, lm, (-0.95, 1.38, -0.55), (-0.75, 1.48, -0.25), 0.022, seg=6)
    add_ico(W, lm, (-0.95, 1.38, -0.55), 0.035, subdiv=0)
    add_rod(W, shade, (-0.75, 1.48, -0.25), (-0.7, 1.3, -0.1), 0.05, seg=12, r1=0.14, caps=(True, False))
    add_ico(W, M("bulb", "#FFF3C4", emissive="#FFE29A"), (-0.71, 1.33, -0.13), 0.045, subdiv=1)
    root = Node("desk")
    root.add(Node("desk_body", mesh=W))
    steam = Mesh("steam")
    add_ico(steam, M("steam", "#FFFFFF", alpha=0.55), (0.0, 0.0, 0.0), 0.035, subdiv=1)
    for k in range(3):
        root.add(Node("steam_%d" % k, t=(mx, my + 0.16, mz), s=HIDDEN, mesh=steam))

    def fn(p, t):
        d = {}
        for k in range(3):
            u = ((t / 2.0) + k / 3.0) % 1.0
            sc = 0.4 + 1.1 * u if u < 0.6 else max(1.06 * (1.0 - u) / 0.4 * 1.0, 1e-4)
            d[("steam_%d" % k, "translation")] = (mx + 0.03 * math.sin(2 * math.pi * u + k), my + 0.16 + 0.4 * u, mz)
            d[("steam_%d" % k, "scale")] = (sc, sc, sc)
        return d
    return "desk", root, M.d, [], {"idle": clip(2.0, 30, fn)}, ("idle",), True


def build_anvil():
    M = Mats()
    stump = Mesh("stump")
    bark, ring = M("bark", "#6B4A2E"), M("stump_top", "#C49A6C")
    add_cyl(stump, bark, (0.0, 0.21, 0.0), 0.3, 0.27, 0.42, 12)
    add_cyl(stump, ring, (0.0, 0.423, 0.0), 0.255, 0.255, 0.006, 12)
    add_torus(stump, M("stump_ring", "#A97F55"), (0.0, 0.427, 0.0), 0.16, 0.006, seg=14, side=4, rot=q_axis(X_AX, 90.0))
    anvil = Mesh("anvil")
    steel, top = M("steel", "#4A4F57", rough=0.5), M("steel_face", "#7A818B", rough=0.4)
    add_box(anvil, steel, (0.0, 0.04, 0.0), (0.36, 0.08, 0.26), bevel=0.015)
    add_box(anvil, steel, (0.0, 0.14, 0.0), (0.2, 0.13, 0.16), bevel=0.01)
    add_box(anvil, steel, (-0.02, 0.265, 0.0), (0.46, 0.13, 0.2), bevel=0.015)
    add_rod(anvil, steel, (0.21, 0.28, 0.0), (0.47, 0.3, 0.0), 0.075, seg=8, r1=0.012)
    add_box(anvil, steel, (-0.27, 0.29, 0.0), (0.1, 0.08, 0.16), bevel=0.01)
    add_box(anvil, top, (-0.02, 0.333, 0.0), (0.44, 0.008, 0.19))
    spark = Mesh("spark")
    add_box(spark, M("spark", "#FFD24A", emissive="#FFB020", rough=1.0), (0.0, 0.0, 0.0), (0.035, 0.035, 0.035))
    root = Node("anvil")
    root.add(Node("stump", mesh=stump))
    an = root.add(Node("anvil_body", t=(0.0, 0.42, 0.0), mesh=anvil))
    sp = an.add(Node("sparks", t=(0.03, 0.34, 0.0)))
    dirs = []
    rnd = lcg(11)
    for k in range(8):
        a = 2 * math.pi * k / 8 + 0.3 * next(rnd)
        dirs.append((math.cos(a) * (0.6 + 0.4 * next(rnd)), 0.9 + 0.6 * next(rnd), math.sin(a) * (0.6 + 0.4 * next(rnd))))
        sp.add(Node("spark_%d" % k, s=HIDDEN, mesh=spark))

    def pose(t, hit):
        d = {("anvil_body", "translation"): (0.0, 0.42 - (0.012 * (1 - t / 0.08) if hit and t < 0.08 else 0.0), 0.0)}
        for k, (dx, dy, dz) in enumerate(dirs):
            if hit and t < 0.25:
                u = t / 0.25
                tt = t * 1.6
                d[("spark_%d" % k, "translation")] = (dx * tt, dy * tt - 2.2 * tt * tt, dz * tt)
                sc = 1.0 - u
                d[("spark_%d" % k, "scale")] = (max(sc, 1e-4),) * 3
                d[("spark_%d" % k, "rotation")] = q_axis(v_unit((dz, 0.3, -dx)), 600.0 * t)
            else:
                d[("spark_%d" % k, "translation")] = (0.0, 0.0, 0.0)
                d[("spark_%d" % k, "scale")] = HIDDEN
                d[("spark_%d" % k, "rotation")] = (0.0, 0.0, 0.0, 1.0)
        return d
    anims = {"idle": clip(2.0, 8, lambda p, t: pose(t, False)), "hit": clip(0.6, 24, lambda p, t: pose(t, True))}
    return "anvil", root, M.d, [], anims, ("idle", "hit"), True


def render_list(pw, ph):
    cream, grey, dot, ink, tick = (hex_rgb("#FBF6E6"), hex_rgb("#9AA0A6"), hex_rgb("#8A8F96"),
                                   hex_rgb("#3A3A3A"), hex_rgb("#2EAD5B"))
    tk = Ink([(0.07, 0.11, 0.11, 0.15), (0.11, 0.15, 0.19, 0.055)], 0.03)
    title = Ink([(0.26, 0.105, 0.6, 0.105)], 0.032)
    lens = [0.42, 0.31, 0.47, 0.36, 0.28, 0.44, 0.33, 0.39, 0.25]
    rows = [0.25 + 0.075 * k for k in range(9)]
    lines = Ink([(0.135, y, 0.135 + L, y) for y, L in zip(rows, lens)], 0.022)

    def sample(x, y):
        if tk.hit(x, y):
            return tick
        if title.hit(x, y):
            return ink
        for yy in rows:
            if (x - 0.085) ** 2 + (y - yy) ** 2 <= 0.014 ** 2:
                return dot
        return grey if lines.hit(x, y) else cream
    return raster(256, 512, pw, ph, sample, [tk.bb, title.bb, (0.06, 0.22, 0.7, 0.88)], ss=3)


def build_list():
    M = Mats()
    W = Mesh("easel")
    wood, board_c, metal = M("easel_wood", "#A0703C"), M("board", "#8A5A36"), M("clip_metal", "#C9CED5", rough=0.4)
    m_paper = M.raw("paper", bb.material("paper", "#FFFFFF", texture=0))
    m_paper_back = M("paper_back", "#F3EBD3")
    PW, PH = 0.72, 0.95
    textures = [("buy_list", render_list(PW, PH))]
    for s in (1, -1):
        add_beam(W, wood, (s * 0.45, 0.012, 0.16), (s * 0.3, 1.45, -0.1), 0.05, 0.05)
        add_box(W, wood, (s * 0.45, 0.012, 0.16), (0.07, 0.024, 0.07))          # foot pad
    add_beam(W, wood, (0.0, 0.016, -0.5), (0.0, 1.36, -0.12), 0.05, 0.05)
    add_box(W, wood, (0.0, 0.012, -0.5), (0.07, 0.024, 0.07))
    add_box(W, wood, (0.0, 0.43, 0.1), (0.98, 0.05, 0.14), bevel=0.01)
    board = Mesh("board")
    add_box(board, board_c, (0.0, 0.55, 0.0), (0.82, 1.1, 0.035), bevel=0.012)
    add_box(board, metal, (0.0, 1.04, 0.03), (0.3, 0.08, 0.04), bevel=0.01)
    add_torus(board, metal, (0.0, 1.1, 0.03), 0.04, 0.008, seg=10, side=4)
    paper = Mesh("paper")
    add_box(paper, m_paper_back, (0.0, -PH / 2, 0.0), (PW, PH, 0.004), faces={"+z": m_paper})
    root = Node("list_scroll")
    root.add(Node("easel", mesh=W))
    bn = root.add(Node("board", t=(0.0, 0.455, 0.1), r=q_axis(X_AX, -12.0), mesh=board))
    bn.add(Node("paper", t=(0.0, 1.02, 0.022), mesh=paper))

    def fn(p, t):
        return {("paper", "rotation"): q_ypr(yaw=1.2 * math.sin(p + 0.4), pitch=-(1.5 + 1.5 * math.sin(p)),
                                             roll=0.8 * math.sin(2 * p))}
    return "list_scroll", root, M.d, textures, {"idle": clip(2.4, 24, fn)}, ("idle",), True


def build_phone():
    M = Mats()
    W = Mesh("stand")
    dark, body = M("stand", "#2A2D33"), M("phone_body", "#2B2F36", rough=0.5)
    add_box(W, dark, (0.0, 0.04, -0.05), (0.92, 0.08, 0.62), bevel=0.02)
    add_box(W, dark, (0.0, 0.13, 0.22), (0.88, 0.1, 0.06), bevel=0.015)
    add_beam(W, dark, (0.0, 0.08, -0.3), (0.0, 0.92, -0.18), 0.3, 0.05)
    phone = Mesh("phone")
    add_box(phone, body, (0.0, 0.75, -0.045), (0.8, 1.5, 0.09), bevel=0.07)
    add_box(phone, M("camera", "#111316"), (0.0, 1.43, 0.002), (0.14, 0.025, 0.006))
    for y in (1.15, 0.95):
        add_box(phone, body, (0.405, y, -0.045), (0.02, 0.12, 0.04))
    screen = Mesh("screen")
    m_scr = M("screen", "#10254A", rough=0.3, emissive="#0B2A5C")
    add_box(screen, m_scr, (0.0, 0.75, 0.003), (0.72, 1.38, 0.006))
    bub_a, bub_b = M("bubble_in", "#24467D", emissive="#0E2A57"), M("bubble_out", "#3E78C9", emissive="#163E7A")
    for y, x, w_, m in ((0.32, -0.12, 0.42, bub_a), (0.5, 0.12, 0.38, bub_b), (0.68, -0.1, 0.46, bub_a)):
        add_box(screen, m, (x, y, 0.007), (w_, 0.12, 0.004), bevel=0.002)
    notif = Mesh("notif")
    add_box(notif, M("card", "#FFFFFF", rough=0.6), (0.0, 0.0, 0.0), (0.64, 0.24, 0.025), bevel=0.02)
    add_rod(notif, M("icon", "#2E7BD6"), (-0.22, 0.0, 0.01), (-0.22, 0.0, 0.02), 0.07, seg=14)
    add_box(notif, M("text_dark", "#4A4F57"), (0.06, 0.045, 0.014), (0.32, 0.035, 0.006))
    add_box(notif, M("text_light", "#A3ABB4"), (0.02, -0.04, 0.014), (0.24, 0.03, 0.006))
    rest = q_axis(X_AX, -12.0)
    root = Node("phone_stand")
    root.add(Node("stand", mesh=W))
    ph = root.add(Node("phone", t=(0.0, 0.1, 0.17), r=rest, mesh=phone))
    ph.add(Node("screen", mesh=screen))
    ph.add(Node("notif", t=(0.0, 1.14, 0.03), s=HIDDEN, mesh=notif))

    def idle(p, t):
        return {("phone", "rotation"): rest, ("notif", "scale"): HIDDEN}

    def buzz(p, t):
        wig = 4.0 * math.sin(2 * math.pi * 14 * t) * max(0.0, 1.0 - t / 0.6)
        s = kf(t, [(0.0, 1e-4), (0.15, 1e-4), (0.28, 1.15), (0.36, 0.95), (0.44, 1.0), (1.0, 1.0)])
        return {("phone", "rotation"): q_mul(rest, q_axis(Z_AX, wig)), ("notif", "scale"): (s, s, s)}
    anims = {"idle": clip(2.0, 4, idle), "buzz": bb.sample_clip(1.0, 40, lambda t: buzz(0.0, t), loop=False)}
    return "phone_stand", root, M.d, [], anims, ("idle", "buzz"), True


def render_month(pw, ph):
    white, red, grey, hi = hex_rgb("#FFFFFF"), hex_rgb("#D9483B"), hex_rgb("#C9CED5"), hex_rgb("#D97757")
    bar = Ink([(0.25, 0.08, 0.59, 0.08)], 0.035)

    def sample(x, y):
        if y < 0.16:
            return white if bar.hit(x, y) else red
        cw, chh = pw / 7, (ph - 0.2) / 5
        c, r = int(x / cw), int((y - 0.2) / chh)
        if y < 0.2 or r > 4:
            return white
        lx, ly = x - c * cw, (y - 0.2) - r * chh
        if 0.015 < lx < cw - 0.015 and 0.015 < ly < chh - 0.015:
            return hi if (c, r) == (4, 2) else grey
        return white
    return raster(128, 128, pw, ph, sample, ss=3, full=True)


def build_calendar():
    M = Mats()
    W = Mesh("calendar")
    stand, metal = M("stand", "#2E3A4B"), M("rings", "#C9CED5", rough=0.4)
    m_page = M.raw("month_page", bb.material("month_page", "#FFFFFF", texture=0))
    m_back = M("page_back", "#F1F1EC")
    PW, PH = 0.84, 0.66
    textures = [("month_page", render_month(PW, PH))]
    top = 0.8 * math.cos(math.radians(15.0)) + 0.015 * math.sin(math.radians(15.0))   # boards meet here, feet on y=0
    for a in (-15.0, 15.0):
        q = q_axis(X_AX, a)
        d = bb.m_apply(bb.q_to_m(q), (0.0, -0.4, 0.0))
        add_box(W, stand, (0.0, top + d[1], d[2]), (0.9, 0.8, 0.03), rot=q)
    qf = q_axis(X_AX, -15.0)
    nrm = bb.m_apply(bb.q_to_m(qf), (0.0, 0.0, 1.0))
    c = bb.m_apply(bb.q_to_m(qf), (0.0, -0.06 - PH / 2, 0.0))
    add_box(W, m_back, (c[0] + 0.03 * nrm[0], top + c[1] + 0.03 * nrm[1], c[2] + 0.03 * nrm[2]), (PW, PH, 0.025),
            rot=qf, faces={"+z": m_page})
    for x in (-0.3, -0.1, 0.1, 0.3):
        add_torus(W, metal, (x, top, 0.0), 0.045, 0.009, seg=12, side=4, rot=q_axis(Y_AX, 90.0))
    page = Mesh("page")
    add_box(page, m_back, (0.0, -0.06 - PH / 2, 0.047), (PW, PH, 0.004), faces={"+z": m_page})
    root = Node("calendar")
    root.add(Node("stand", mesh=W))
    for k in range(3):
        root.add(Node("page_%d" % k, t=(0.0, top, 0.0), r=qf, s=HIDDEN, mesh=page))

    def flip(p, t):
        d = {}
        for k in range(3):
            tt = (t - 0.4 * k) % 1.2
            if tt < 0.45:
                a, s = -15.0 - 330.0 * ease(tt / 0.45), 1.0
            elif tt < 0.5:
                a, s = -345.0, max(1.0 - (tt - 0.45) / 0.05, 1e-4)
            else:
                a, s = -15.0, 1e-4
            d[("page_%d" % k, "rotation")] = q_axis(X_AX, a)
            d[("page_%d" % k, "scale")] = (s, s, s)
        return d

    def idle(p, t):
        d = {}
        for k in range(3):
            d[("page_%d" % k, "rotation")] = qf
            d[("page_%d" % k, "scale")] = HIDDEN
        return d
    anims = {"idle": clip(2.0, 4, idle), "flip": clip(1.2, 48, flip)}
    return "calendar", root, M.d, textures, anims, ("idle", "flip"), True


def render_candles(seed):
    bg, grid, up, dn = hex_rgb("#0E1626"), hex_rgb("#1C2B40"), hex_rgb("#2EE87A"), hex_rgb("#F0524F")
    rnd = lcg(seed)
    v, bars = 0.5, []
    for i in range(11):
        o = v
        v = min(0.85, max(0.15, v + (next(rnd) - 0.42) * 0.22))
        hi_, lo_ = max(o, v) + 0.06 * next(rnd), min(o, v) - 0.06 * next(rnd)
        bars.append((o, v, hi_, lo_))
    W_, H_ = 2.0, 1.0

    def sample(x, y):
        yy = 1.0 - y / H_
        i = int(x / (W_ / 12)) - 0
        if 0 <= i < 11:
            o, c, h_, l_ = bars[i]
            cx = (i + 0.5) * W_ / 12
            col = up if c >= o else dn
            if abs(x - cx) < 0.05 and min(o, c) <= yy <= max(o, c) + 0.01:
                return col
            if abs(x - cx) < 0.008 and l_ <= yy <= h_:
                return col
        if abs((yy * 5) - round(yy * 5)) < 0.02:
            return grid
        return bg
    return raster(128, 64, W_, H_, sample, ss=3, full=True)


def build_broker():
    M = Mats()
    W = Mesh("kiosk")
    navy, white, acc = M("kiosk", "#2E3A4B"), M("counter", "#F1F1EC"), M("accent", "#D97757")
    mon, dark = M("monitor", "#1C1F24", rough=0.5), M("dark", "#2A2D33")
    add_box(W, navy, (0.0, 0.425, -0.2), (1.4, 0.85, 0.6), bevel=0.03)
    add_box(W, white, (0.0, 0.875, -0.18), (1.5, 0.05, 0.74), bevel=0.015)
    add_box(W, acc, (0.0, 0.62, 0.105), (1.42, 0.08, 0.01))
    add_box(W, dark, (0.0, 0.91, 0.06), (0.5, 0.02, 0.16), bevel=0.005)
    scr = []
    for k, seed in enumerate((3, 8)):
        m_scr = M.raw("screen_%d" % k, {"pbrMetallicRoughness": {"baseColorFactor": [0.02, 0.02, 0.02, 1.0],
                                                                 "metallicFactor": 0.0, "roughnessFactor": 0.35},
                                         "emissiveTexture": {"index": k}, "emissiveFactor": [1.0, 1.0, 1.0]})
        scr.append(m_scr)
    textures = [("candles_a", render_candles(3)), ("candles_b", render_candles(8))]
    tilt = q_axis(X_AX, -8.0)
    for k, x in enumerate((-0.35, 0.35)):
        add_cyl(W, dark, (x, 0.915, -0.3), 0.1, 0.09, 0.03, 10)
        add_rod(W, dark, (x, 0.92, -0.32), (x, 1.08, -0.32), 0.025, seg=6)
        add_box(W, mon, (x, 1.22, -0.3), (0.62, 0.38, 0.04), bevel=0.012, rot=tilt, faces={"+z": scr[k]})
    stool = Mesh("stool")
    add_cyl(stool, M("seat", "#8E2B2B"), (0.0, 0.64, 0.0), 0.2, 0.19, 0.07, 14)
    add_rod(stool, dark, (0.0, 0.05, 0.0), (0.0, 0.61, 0.0), 0.03, seg=8)
    add_cyl(stool, dark, (0.0, 0.02, 0.0), 0.2, 0.2, 0.04, 14)
    add_torus(stool, M("chrome", "#C9CED5", rough=0.35), (0.0, 0.3, 0.0), 0.15, 0.012, seg=14, side=4, rot=q_axis(X_AX, 90.0))
    live = Mesh("live_bar")
    add_box(live, M("live", "#2EE87A", emissive="#2EE87A"), (0.0, 0.05, 0.0), (0.03, 0.1, 0.004))
    root = Node("broker_desk")
    root.add(Node("kiosk", mesh=W))
    root.add(Node("stool", t=(0.0, 0.0, 0.62), mesh=stool))
    lives = []
    for k, x in enumerate((-0.35, 0.35)):
        c = bb.m_apply(bb.q_to_m(tilt), (0.24, -0.07, 0.024))
        name = "live_%s" % ("L" if k == 0 else "R")
        root.add(Node(name, t=(x + c[0], 1.22 + c[1], -0.3 + c[2]), r=tilt, mesh=live))
        lives.append(name)

    def fn(p, t):
        return {(lives[0], "scale"): (1.0, 0.7 + 0.5 * (0.5 + 0.5 * math.sin(p)), 1.0),
                (lives[1], "scale"): (1.0, 0.7 + 0.5 * (0.5 + 0.5 * math.sin(p + 2.1)), 1.0)}
    return "broker_desk", root, M.d, textures, {"idle": clip(1.2, 16, fn)}, ("idle",), True


def build_tower():
    M = Mats()
    W = Mesh("lattice")
    conc, red, white = M("base", "#CFCAC0"), M("red", "#D9483B"), M("white", "#F1F1EC")
    metal = M("metal", "#8E969F")
    add_box(W, conc, (0.0, 0.15, 0.0), (1.3, 0.3, 1.3), bevel=0.03)
    y0, y1, b0, b1, n = 0.3, 5.0, 0.5, 0.13, 7

    def hw(y):
        return b0 + (b1 - b0) * (y - y0) / (y1 - y0)
    for sx in (1, -1):
        for sz in (1, -1):
            add_beam(W, metal, (sx * b0, y0, sz * b0), (sx * b1, y1, sz * b1), 0.07, 0.07)
    corners = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
    for k in range(n):
        ya, yb = y0 + (y1 - y0) * k / n, y0 + (y1 - y0) * (k + 1) / n
        m = red if k % 2 == 0 else white
        ha, hb = hw(ya), hw(yb)
        for i in range(4):
            (ax_, az_), (bx_, bz_) = corners[i], corners[(i + 1) % 4]
            add_beam(W, m, (ax_ * hb, yb, az_ * hb), (bx_ * hb, yb, bz_ * hb), 0.045, 0.045)
            add_beam(W, m, (ax_ * ha, ya, az_ * ha), (bx_ * hb, yb, bz_ * hb), 0.035, 0.035)
            add_beam(W, m, (bx_ * ha, ya, bz_ * ha), (ax_ * hb, yb, az_ * hb), 0.035, 0.035)
    add_box(W, metal, (0.0, y1 + 0.03, 0.0), (0.5, 0.05, 0.5))
    add_rod(W, white, (0.0, y1, 0.0), (0.0, y1 + 0.7, 0.0), 0.03, seg=8)
    for k in range(3):
        add_box(W, white, (0.0, y1 + 0.25 + 0.15 * k, 0.07), (0.1, 0.12, 0.03))
    dish = Mesh("tmp")
    m_dish = M.raw("dish", dict(bb.material("dish", "#F1F1EC"), doubleSided=True))
    add_lathe(dish, m_dish, (0.0, 0.0, 0.0), [(-0.07, 0.0), (-0.06, 0.12), (-0.03, 0.24), (0.02, 0.34)], 16)
    merge(W, dish, off=(0.0, 4.25, 0.3))
    add_rod(W, metal, (0.0, 4.25, 0.15), (0.0, 4.25, 0.23), 0.03, seg=6)
    for a in (0, 120, 240):
        ra = math.radians(a)
        add_rod(W, metal, (0.3 * math.cos(ra), 4.25 + 0.3 * math.sin(ra), 0.34), (0.0, 4.25, 0.55), 0.008, seg=4)
    add_ico(W, metal, (0.0, 4.25, 0.55), 0.03, subdiv=0)
    beacon = Mesh("beacon")
    add_ico(beacon, M("beacon", "#FF4A3D", emissive="#FF2A1A", rough=0.4), (0.0, 0.0, 0.0), 0.08, subdiv=1)
    root = Node("telegram_tower")
    root.add(Node("tower", mesh=W))
    root.add(Node("beacon", t=(0.0, y1 + 0.76, 0.0), mesh=beacon))

    def fn(p, t):
        s = kf(t, [(0.0, 1.0), (0.25, 1.0), (0.35, 0.35), (1.1, 0.35), (1.2, 1.0)])
        return {("beacon", "scale"): (s, s, s)}
    return "telegram_tower", root, M.d, [], {"idle": clip(1.2, 24, fn)}, ("idle",), True

# ------------------------------------------------------------------ dressing

def build_palm():
    M = Mats()
    trunk = Mesh("trunk")
    tr, tr_d = M("trunk", "#9A6B42"), M("trunk_ring", "#7E5534")
    pts = [(0.28 * (i / 6) ** 2, 0.45 * i, 0.0) for i in range(7)]
    for i, (a, b) in enumerate(zip(pts, pts[1:])):
        r0, r1 = 0.14 - 0.01 * i, 0.13 - 0.01 * i
        add_rod(trunk, tr, a, b, r0, seg=8, r1=r1)
        add_rod(trunk, tr_d, (b[0], b[1] - 0.04, b[2]), (b[0], b[1] + 0.02, b[2]), r1 + 0.02, seg=8)
    crown = Mesh("crown")
    leaf, leaf_d, nut = M("frond", "#4FA34A"), M("frond_dark", "#3E8A3C"), M("coconut", "#6B4A2E")
    for k in range(7):
        yaw = 360.0 * k / 7 + 10.0
        m = leaf if k % 2 == 0 else leaf_d
        segs = [(0.0, 0.05), (0.35, 0.12), (0.7, 0.05), (1.0, -0.15), (1.2, -0.42)]
        ry = q_axis(Y_AX, yaw)
        for j, ((r0, y0_), (r1, y1_)) in enumerate(zip(segs, segs[1:])):
            p0 = bb.m_apply(bb.q_to_m(ry), (0.0, y0_, r0))
            p1 = bb.m_apply(bb.q_to_m(ry), (0.0, y1_, r1))
            w = 0.26 - 0.05 * j
            d = v_sub(p1, p0)
            L = math.sqrt(v_dot(d, d))
            mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2)
            pitch = -math.degrees(math.atan2(d[1], math.hypot(d[0], d[2])))
            add_box(crown, m, mid, (w, 0.025, L + 0.03), rot=q_mul(ry, q_axis(X_AX, pitch)))
    for k in range(3):
        a = math.radians(120 * k + 30)
        add_ico(crown, nut, (0.1 * math.cos(a), -0.08, 0.1 * math.sin(a)), 0.075, subdiv=1)
    root = Node("palm_tree")
    t = root.add(Node("trunk", mesh=trunk))
    t.add(Node("crown", t=pts[-1], mesh=crown))

    def fn(p, tt):
        return {("trunk", "rotation"): q_ypr(roll=-1.5 * math.sin(p), pitch=0.8 * math.sin(p + 1.0)),
                ("crown", "rotation"): q_ypr(yaw=3.0 * math.sin(p + 0.5), roll=-4.0 * math.sin(p - 0.6),
                                             pitch=2.0 * math.sin(2 * p))}
    return "palm_tree", root, M.d, [], {"idle": clip(3.0, 30, fn)}, ("idle",), False   # trunk foot sinks a hair


def build_oak():
    M = Mats()
    trunk = Mesh("trunk")
    bark = M("bark", "#7A5534")
    add_rod(trunk, bark, (0.0, 0.0, 0.0), (0.0, 1.3, 0.0), 0.17, seg=8, r1=0.12)
    add_rod(trunk, bark, (0.0, 0.9, 0.0), (0.38, 1.45, 0.1), 0.07, seg=6, r1=0.04)
    add_rod(trunk, bark, (0.0, 1.0, 0.0), (-0.32, 1.5, -0.12), 0.07, seg=6, r1=0.04)
    foliage = Mesh("foliage")
    g1, g2, g3 = M("leaf", "#5E9E4A"), M("leaf_dark", "#4A8A3E"), M("leaf_light", "#79B95C")
    blobs = [((0.0, 0.55, 0.0), 0.62, g1), ((0.45, 0.35, 0.15), 0.45, g2), ((-0.45, 0.35, -0.1), 0.48, g2),
             ((0.1, 0.25, 0.45), 0.42, g3), ((-0.15, 0.3, -0.45), 0.42, g1), ((0.2, 0.9, -0.1), 0.4, g3)]
    for c, r, m in blobs:
        add_ico(foliage, m, c, r, subdiv=1, scale=(1.0, 0.85, 1.0))
    root = Node("oak_tree")
    t = root.add(Node("trunk", mesh=trunk))
    t.add(Node("foliage", t=(0.0, 1.4, 0.0), mesh=foliage))

    def fn(p, tt):
        s = 1.0 + 0.015 * math.sin(2 * p)
        return {("trunk", "rotation"): q_ypr(roll=-0.8 * math.sin(p)),
                ("foliage", "rotation"): q_ypr(roll=-2.0 * math.sin(p - 0.5), pitch=1.2 * math.sin(p + 1.2)),
                ("foliage", "scale"): (s, 1.0 / s, s)}
    return "oak_tree", root, M.d, [], {"idle": clip(3.0, 30, fn)}, ("idle",), False


def build_rock():
    M = Mats()
    W = Mesh("rock")
    add_rock(W, M("rock", "#8C8F94", rough=0.9), (0.0, 0.0, 0.0), 0.5, 5)
    add_rock(W, M("rock_dark", "#73777D", rough=0.9), (0.55, 0.0, 0.25), 0.25, 9, scale=(1.0, 0.7, 0.9))
    root = Node("rock")
    n = root.add(Node("rock_body", mesh=W))
    return "rock", root, M.d, [], static_idle("rock_body", n), ("idle",), False


def build_lamp():
    M = Mats()
    W = Mesh("lamp_post")
    metal, glow = M("lamp_metal", "#2A2D33"), M("lamp_glass", "#FFE7A3", emissive="#FFD27A")
    add_cyl(W, metal, (0.0, 0.08, 0.0), 0.13, 0.11, 0.16, 10)
    add_rod(W, metal, (0.0, 0.15, 0.0), (0.0, 2.1, 0.0), 0.045, seg=8, r1=0.035)
    add_beam(W, metal, (0.0, 2.05, 0.0), (0.0, 2.15, 0.38), 0.04, 0.04)
    add_rod(W, metal, (0.0, 2.15, 0.38), (0.0, 2.06, 0.38), 0.012, seg=4)
    add_box(W, glow, (0.0, 1.93, 0.38), (0.2, 0.25, 0.2), bevel=0.02)
    bp.add_pyramid(W, metal, (0.0, 2.05, 0.38), 0.13, 0.13, 0.12)
    root = Node("lamp_post")
    n = root.add(Node("lamp", mesh=W))
    return "lamp_post", root, M.d, [], static_idle("lamp", n), ("idle",), True


def build_bench():
    M = Mats()
    W = Mesh("bench")
    wood, iron = M("slats", "#A0703C"), M("iron", "#2A2D33")
    for z in (-0.1, 0.03, 0.16):
        add_box(W, wood, (0.0, 0.45, z), (1.5, 0.04, 0.11), bevel=0.01)
    back = q_axis(X_AX, -12.0)
    for y in (0.66, 0.82):
        add_box(W, wood, (0.0, y, -0.2 - (y - 0.6) * 0.21), (1.5, 0.1, 0.035), bevel=0.01, rot=back)
    for s in (1, -1):
        x = s * 0.62
        add_beam(W, iron, (x, 0.02, 0.18), (x, 0.43, 0.12), 0.05, 0.05)
        add_beam(W, iron, (x, 0.02, -0.18), (x, 0.9, -0.3), 0.05, 0.05)
        for z in (0.18, -0.18):
            add_box(W, iron, (x, 0.015, z), (0.08, 0.03, 0.08))                # feet
        add_beam(W, iron, (x, 0.43, 0.18), (x, 0.43, -0.22), 0.05, 0.05)
    root = Node("bench")
    n = root.add(Node("bench_body", mesh=W))
    return "bench", root, M.d, [], static_idle("bench_body", n), ("idle",), True


def build_bush():
    M = Mats()
    W = Mesh("bush")
    g1, g2 = M("leaf", "#5E9E4A"), M("leaf_dark", "#4A8A3E")
    for c, r, m in (((0.0, 0.32, 0.0), 0.34, g1), ((0.3, 0.25, 0.1), 0.26, g2), ((-0.28, 0.24, -0.05), 0.27, g2),
                    ((0.05, 0.22, 0.28), 0.24, g1)):
        add_ico(W, m, c, r, subdiv=1, scale=(1.0, 0.9, 1.0))
    root = Node("bush")
    root.add(Node("bush_body", mesh=W))

    def fn(p, t):
        s = 1.0 + 0.02 * math.sin(p)
        return {("bush_body", "scale"): (s, 1.0 / s, s), ("bush_body", "rotation"): q_ypr(roll=1.5 * math.sin(p + 0.5))}
    return "bush", root, M.d, [], {"idle": clip(2.5, 20, fn)}, ("idle",), False

# ------------------------------------------------------------------ main

BUILDERS = [build_nse, build_fed, build_ecb, build_pib, build_crane, build_cable_landing, build_desk, build_anvil,
            build_list, build_phone, build_calendar, build_sbi, build_broker, build_tower,
            build_palm, build_oak, build_rock, build_lamp, build_bench, build_bush]
ONESHOT = ("buzz",)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", help="comma-separated names to build")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("%-20s %8s %6s %6s  %-24s %s" % ("file", "bytes", "tris", "nodes", "animations (s)", "rest bbox (w x h x d)"))
    n = 0
    for fn in BUILDERS:
        name, root, mats, textures, anims, required, grounded = fn()
        if args.only and name not in args.only.split(","):
            continue
        anims = bb.fill_coverage(root, anims)
        path = OUT_DIR / (name + ".glb")
        write_glb(path, name, root, mats, textures, anims)
        st = validate(path, required=required, max_bytes=MAX_BYTES, max_tris=MAX_TRIS, oneshot=ONESHOT,
                      ground=grounded)
        lo, hi = st["bbox"]
        an = ", ".join("%s %.2f" % kv for kv in st["anims"].items())
        print("%-20s %8d %6d %6d  %-24s %.2f x %.2f x %.2f (min y %.3f)" % (
            path.name, st["bytes"], st["tris"], st["nodes"], an, hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], lo[1]))
        n += 1
    print("ok: %d files valid" % n)


if __name__ == "__main__":
    main()
