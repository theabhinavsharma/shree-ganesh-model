#!/usr/bin/env python3
"""
SGM Data World - one model per crew character.

Every character is the standard worker bot (build_bots.py) wearing its country's
full-flag tee, plus its own quirk: something on the head or face that hints at
the job, and usually a prop in the right hand. Each id gets

    bot_<id>.glb         hands free; the prop is in the right hand
    bot_<id>_carry.glb   holding the data crate overhead; the prop sits on the
                         crate or is put away (QUIRKS[id]["carry"]), and the crate
                         rides on top of any hat (QUIRKS[id]["lift"])

with the same clips as the base bots: run, idle, crouch, throw, catch, jump, cheer.
Accessories are child nodes of the torso / arm / crate, so they move with the
body in every clip. Dalal sleeps: his idle lies him down on a pillow with
floating Zzz (he wakes up for every other clip).

    python3 docs/sgm-world/bots/build_crew.py              # build all 46 files + validate
    python3 docs/sgm-world/bots/build_crew.py --only bhav,dalal

Stations decide the tee: bom / del / web / blr / courier = IN, stl / mtv / sfo = US,
fra = DE, dxb = AE. Pants colours match world/data.js. No ethnic or religious
headwear, no caricatures; no logos or brand marks.
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
from build_bots import Mesh, Node, add_box, add_cyl, q_axis, q_mul, X_AX, Y_AX, Z_AX  # noqa: E402
from build_props import add_ico, add_torus, add_rod, add_beam, add_pyramid, add_lathe, merge  # noqa: E402

OUT_DIR = bb.OUT_DIR
HIDDEN = (1e-4, 1e-4, 1e-4)

# torso-local landmarks (y measured from the hips pivot at HIP_Y)
TOP = bb.BODY["y0"] + bb.BODY["h"] - bb.HIP_Y     # head top        0.68
EY = bb.EYE_Y - bb.HIP_Y                           # eye centre      0.52
EX = 0.13                                          # eye x
FZ = bb.BODY["d"] / 2                              # face front z    0.22
NECK = 0.345                                       # where the tee meets the orange block
HAND = (-0.04, -0.17, 0.07)                        # right fist, arm-local (left fist: mirror x)
CRATE_TOP = bb.CRATE["h"] / 2


def qx(d):
    return q_axis(X_AX, d)


def qy(d):
    return q_axis(Y_AX, d)


def qz(d):
    return q_axis(Z_AX, d)


def band(mesh, mat, y, h, grow=0.012, bevel=0.03):
    """A ring hugging the orange block at height y (headbands, straps, scarves)."""
    add_box(mesh, mat, (0.0, y, 0.0), (bb.BODY["w"] + 2 * grow, h, bb.BODY["d"] + 2 * grow), bevel=bevel)


def temples(mesh, mat, y, z_front, x_front):
    for s in (1, -1):
        add_beam(mesh, mat, (s * x_front, y, z_front), (s * 0.335, y, FZ - 0.02), 0.022, 0.022)
        add_box(mesh, mat, (s * 0.335, y, 0.08), (0.02, 0.022, 0.26))


def add_star(mesh, mat, c, R, depth, r_in=0.45):
    """Five-point star plate facing +Z, one point up."""
    pts = []
    for k in range(10):
        a = math.pi / 2 + k * math.pi / 5
        r = R if k % 2 == 0 else R * r_in
        pts.append((c[0] + r * math.cos(a), c[1] + r * math.sin(a)))
    zf, zb = c[2] + depth / 2, c[2] - depth / 2
    for k in range(10):
        p0, p1 = pts[k], pts[(k + 1) % 10]
        mesh.poly(mat, [(c[0], c[1], zf), (p0[0], p0[1], zf), (p1[0], p1[1], zf)], (0.0, 0.0, 1.0))
        mesh.poly(mat, [(c[0], c[1], zb), (p1[0], p1[1], zb), (p0[0], p0[1], zb)], (0.0, 0.0, -1.0))
        e = (p1[0] - p0[0], p1[1] - p0[1])
        n = bb.v_unit((e[1], -e[0], 0.0))
        mid = ((p0[0] + p1[0]) / 2 - c[0], (p0[1] + p1[1]) / 2 - c[1])
        if n[0] * mid[0] + n[1] * mid[1] < 0:
            n = (-n[0], -n[1], 0.0)
        mesh.poly(mat, [(p0[0], p0[1], zf), (p1[0], p1[1], zf), (p1[0], p1[1], zb), (p0[0], p0[1], zb)], n)


def letter_z(mesh, mat, h):
    """A chunky 3D 'Z' centred on the origin."""
    w = 0.62 * h
    st = 0.16 * h
    for (ax, ay), (bx_, by_) in (((0.1, 0), (0.9, 0)), ((0.9, 0), (0.1, 1)), ((0.1, 1), (0.9, 1))):
        x0, y0, x1, y1 = (ax - 0.5) * w, (0.5 - ay) * h, (bx_ - 0.5) * w, (0.5 - by_) * h
        L = math.hypot(x1 - x0, y1 - y0)
        add_box(mesh, mat, ((x0 + x1) / 2, (y0 + y1) / 2, 0.0), (L + st, st, 0.05),
                rot=qz(math.degrees(math.atan2(y1 - y0, x1 - x0))))

# ------------------------------------------------------------------ head quirks (torso-local)

def head_bhav(M, m):
    band(m, M("visor_band", "#1F7A4A"), 0.6, 0.05)
    add_box(m, M("visor", "#35C27A", rough=0.3, alpha=0.8), (0.0, 0.6, FZ + 0.13), (0.54, 0.02, 0.28), rot=qx(14.0))


def head_split(M, m):
    for s, col, key in ((1, "#FF4FA0", "hair_pink"), (-1, "#29C5F2", "hair_cyan")):
        mat = M(key, col)
        add_box(m, mat, (s * 0.165, TOP + 0.035, 0.0), (0.3, 0.09, 0.45), bevel=0.03)
        add_box(m, mat, (s * 0.16, TOP - 0.02, FZ + 0.005), (0.29, 0.07, 0.04), bevel=0.015)
        for x, z, h in ((0.08, 0.1, 0.17), (0.21, -0.03, 0.15), (0.11, -0.15, 0.13)):
            add_pyramid(m, mat, (s * x, TOP + 0.07, z), 0.065, 0.065, h, apex_dx=s * 0.05)


def head_filer(M, m):
    f = M("thick_frame", "#1A1A1A", rough=0.4)
    for s in (1, -1):
        add_torus(m, f, (s * EX, EY, 0.262), 0.1, 0.024, seg=16, side=5)
    add_box(m, f, (0.0, EY + 0.025, 0.262), (0.08, 0.028, 0.028))
    temples(m, f, EY + 0.04, 0.255, 0.225)


def head_promo(M, m):
    k, p, g = M("hat_black", "#1C1C1F", rough=0.5), M("hat_band", "#6A3FA0"), M("gold", "#D4A22E", rough=0.35)
    add_cyl(m, k, (0.0, TOP + 0.012, 0.0), 0.27, 0.27, 0.025, 20)
    add_cyl(m, k, (0.0, TOP + 0.2, 0.0), 0.17, 0.185, 0.36, 20)
    add_cyl(m, p, (0.0, TOP + 0.065, 0.0), 0.176, 0.178, 0.06, 20, caps=(False, False))
    add_torus(m, g, (EX, EY, 0.255), 0.09, 0.013, seg=16, side=4)
    add_rod(m, M("monocle_glass", "#CFE8FF", rough=0.1, alpha=0.35), (EX, EY, 0.25), (EX, EY, 0.256), 0.085, seg=16)
    add_beam(m, g, (EX + 0.07, EY - 0.06, 0.255), (0.25, 0.3, 0.238), 0.012, 0.012)


def head_tally(M, m):
    r = M("headband", "#D9302E")
    band(m, r, 0.632, 0.07)
    add_ico(m, r, (0.27, 0.632, -0.23), 0.045, subdiv=1)
    add_beam(m, r, (0.29, 0.62, -0.24), (0.45, 0.5, -0.33), 0.055, 0.015)
    add_beam(m, r, (0.29, 0.62, -0.24), (0.41, 0.44, -0.27), 0.05, 0.015)


def head_nsdl(M, m):
    lens, g = M("aviator_mirror", "#2E8FE0", rough=0.12, emissive="#0B2E57"), M("gold", "#D4A22E", rough=0.35)
    for s in (1, -1):
        add_box(m, lens, (s * 0.135, EY - 0.01, 0.255), (0.18, 0.15, 0.02), bevel=0.055, rot=qz(s * 10.0))
    add_box(m, g, (0.0, EY + 0.065, 0.267), (0.46, 0.018, 0.018))
    add_box(m, g, (0.0, EY + 0.025, 0.265), (0.06, 0.016, 0.016))
    temples(m, g, EY + 0.06, 0.262, 0.23)


def head_amfi(M, m):
    add_box(m, M("beanie", "#2A9D8F"), (0.0, 0.705, 0.0), (0.67, 0.19, 0.47), bevel=0.09)
    add_box(m, M("beanie_cuff", "#1F7A70"), (0.0, 0.635, 0.0), (0.685, 0.065, 0.485), bevel=0.03)
    add_ico(m, M("pompom", "#F4EFE2"), (0.0, 0.83, 0.0), 0.085, subdiv=1)


def head_khabri(M, m):
    h, d, k = M("fedora", "#4A4A52"), M("fedora_dent", "#34343A"), M("fedora_band", "#1C1C1F")
    add_cyl(m, h, (0.0, TOP + 0.012, 0.0), 0.31, 0.31, 0.022, 20)
    add_cyl(m, h, (0.0, TOP + 0.11, 0.0), 0.18, 0.15, 0.2, 10)
    add_box(m, d, (0.0, TOP + 0.21, 0.0), (0.18, 0.02, 0.06))
    add_cyl(m, k, (0.0, TOP + 0.05, 0.0), 0.183, 0.181, 0.05, 10, caps=(False, False))
    card = q_mul(qy(-30.0), qz(12.0))
    add_box(m, M("press_card", "#FFFFFF"), (0.09, TOP + 0.11, 0.15), (0.11, 0.08, 0.008), rot=card)
    add_box(m, M("press_red", "#D9483B"), (0.09, TOP + 0.135, 0.154), (0.11, 0.02, 0.006), rot=card)


def head_khabri2(M, m):
    cup, dark = M("headphone_cup", "#E4572E"), M("headphone_band", "#2A2D33")
    for s in (1, -1):
        add_rod(m, cup, (s * 0.322, EY, 0.0), (s * 0.4, EY, 0.0), 0.13, seg=16)
        add_rod(m, dark, (s * 0.4, EY, 0.0), (s * 0.415, EY, 0.0), 0.09, seg=12)
        add_box(m, dark, (s * 0.37, 0.6, 0.0), (0.04, 0.12, 0.05))
    pts = [(0.37 * math.cos(math.pi * k / 8), 0.64 + 0.2 * math.sin(math.pi * k / 8)) for k in range(9)]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        add_beam(m, dark, (x0, y0, 0.0), (x1, y1, 0.0), 0.06, 0.045)


def head_sarkari(M, m):
    k, n, g = M("cap_band", "#1C1C1F", rough=0.4), M("cap_navy", "#1F2F55"), M("gold", "#D4A22E", rough=0.35)
    add_cyl(m, k, (0.0, TOP + 0.03, 0.0), 0.22, 0.22, 0.06, 18)
    add_cyl(m, n, (0.0, TOP + 0.105, -0.01), 0.21, 0.28, 0.09, 18)
    add_box(m, M("cap_peak", "#111214", rough=0.2), (0.0, TOP + 0.0, 0.27), (0.32, 0.018, 0.15), bevel=0.006, rot=qx(18.0))
    add_ico(m, g, (0.0, TOP + 0.075, 0.27), 0.035, subdiv=1)
    add_box(m, g, (0.0, TOP + 0.02, 0.222), (0.26, 0.012, 0.01))


def head_screeny(M, m):
    band(m, M("strap", "#3A3E44"), EY, 0.07)
    add_box(m, M("goggles", "#EDEFF2", rough=0.5), (0.0, EY, FZ + 0.07), (0.56, 0.22, 0.16), bevel=0.05)
    add_box(m, M("goggle_glass", "#1C2230", rough=0.2), (0.0, EY, FZ + 0.152), (0.5, 0.16, 0.012), bevel=0.02)
    add_box(m, M("goggle_glow", "#39E1FF", emissive="#39E1FF"), (0.0, EY, FZ + 0.16), (0.42, 0.035, 0.006))


def head_tikki(M, m):
    kh, kd = M("bucket_hat", "#B8A06A"), M("bucket_brim", "#A08A55")
    add_cyl(m, kh, (0.0, TOP + 0.08, 0.0), 0.24, 0.21, 0.16, 16)
    add_cyl(m, kd, (0.0, TOP - 0.005, 0.0), 0.38, 0.25, 0.08, 16, caps=(True, False))
    add_cyl(m, M("bucket_band", "#5E4B2A"), (0.0, TOP + 0.03, 0.0), 0.244, 0.242, 0.035, 16, caps=(False, False))


def head_dalal(M, m):
    c, w = M("nightcap", "#3E6DB5"), M("nightcap_white", "#F4F4F0")
    add_cyl(m, w, (0.0, TOP + 0.015, 0.0), 0.262, 0.258, 0.045, 14)
    add_cyl(m, c, (0.0, TOP + 0.11, 0.0), 0.25, 0.15, 0.17, 14)
    add_cyl(m, w, (0.0, TOP + 0.13, 0.0), 0.208, 0.2, 0.03, 14, caps=(False, False))
    add_rod(m, c, (0.0, TOP + 0.19, 0.0), (0.17, TOP + 0.3, -0.02), 0.15, seg=10, r1=0.075)
    add_rod(m, c, (0.17, TOP + 0.3, -0.02), (0.36, TOP + 0.23, -0.04), 0.075, seg=8, r1=0.025)
    add_ico(m, w, (0.38, TOP + 0.21, -0.04), 0.065, subdiv=1)


def head_fed(M, m):
    br, dk = M("cowboy", "#8A5A36"), M("cowboy_dark", "#5E3B22")
    add_cyl(m, br, (0.0, TOP + 0.02, 0.0), 0.33, 0.33, 0.025, 20)
    for s in (1, -1):
        add_box(m, br, (s * 0.31, TOP + 0.07, 0.0), (0.13, 0.024, 0.44), rot=qz(s * 38.0))
    add_cyl(m, br, (0.0, TOP + 0.14, 0.0), 0.17, 0.15, 0.22, 10)
    add_box(m, dk, (0.0, TOP + 0.25, 0.0), (0.06, 0.03, 0.22))
    add_cyl(m, dk, (0.0, TOP + 0.055, 0.0), 0.172, 0.17, 0.045, 10, caps=(False, False))


def head_euro(M, m):
    g = M("gold", "#D4A22E", rough=0.35)
    for s in (1, -1):
        add_torus(m, g, (s * EX, EY, 0.255), 0.078, 0.011, seg=16, side=4)
        add_rod(m, M("tint", "#F0A030", rough=0.1, alpha=0.6), (s * EX, EY, 0.25), (s * EX, EY, 0.257), 0.075, seg=16)
    add_box(m, g, (0.0, EY + 0.02, 0.255), (0.1, 0.014, 0.014))
    temples(m, g, EY + 0.03, 0.255, 0.21)
    sc, st = M("scarf", "#1F3F99"), M("scarf_stripe", "#F2C230")
    band(m, sc, NECK, 0.09, grow=0.025, bevel=0.04)
    band(m, st, NECK, 0.022, grow=0.03, bevel=0.01)
    add_box(m, sc, (0.15, 0.2, FZ + 0.04), (0.11, 0.26, 0.03), rot=qz(-8.0))
    add_box(m, st, (0.165, 0.07, FZ + 0.04), (0.11, 0.025, 0.032), rot=qz(-8.0))


def head_gnews(M, m):
    metal, red = M("mast", "#8E969F", rough=0.4), M("feed", "#D9483B")
    add_rod(m, metal, (0.0, TOP - 0.01, -0.05), (0.0, TOP + 0.15, -0.05), 0.025, seg=8)
    add_ico(m, metal, (0.0, TOP + 0.15, -0.05), 0.04, subdiv=0)
    dish = Mesh("tmp")
    mat = M("dish_2s", "#F1F1EC", double=True)
    add_lathe(dish, mat, (0.0, 0.0, 0.0), [(0.0, 0.0), (0.012, 0.08), (0.04, 0.15), (0.075, 0.2)], 16)
    add_rod(dish, metal, (0.0, 0.0, 0.0), (0.0, 0.0, 0.17), 0.012, seg=6)
    add_ico(dish, red, (0.0, 0.0, 0.18), 0.03, subdiv=0)
    merge(m, dish, rot=qx(-55.0), off=(0.0, TOP + 0.18, -0.05))


def head_drive(M, m):
    w = M("cloud", "#F7FAFF", rough=0.9)
    for c, r in (((0.0, TOP + 0.11, 0.0), 0.17), ((0.16, TOP + 0.06, 0.03), 0.13), ((-0.16, TOP + 0.06, 0.02), 0.13),
                 ((0.06, TOP + 0.07, 0.13), 0.11), ((-0.07, TOP + 0.08, -0.13), 0.12), ((0.22, TOP + 0.03, -0.09), 0.09)):
        add_ico(m, w, c, r, subdiv=1, scale=(1.0, 0.85, 1.0))


def head_pigeon(M, m):
    grey, dark = M("pigeon_grey", "#8D97A6"), M("pigeon_dark", "#5E6670")
    add_box(m, grey, (0.0, TOP + 0.0, 0.0), (0.66, 0.15, 0.46), bevel=0.07)
    add_ico(m, grey, (0.0, TOP + 0.06, -0.02), 1.0, subdiv=1, scale=(0.24, 0.1, 0.18))
    band(m, M("neck_green", "#3FAF8F", rough=0.4), NECK, 0.07, grow=0.022)
    band(m, M("neck_purple", "#7A4FA0", rough=0.4), NECK + 0.05, 0.03, grow=0.018)
    beak = Mesh("tmp")
    add_pyramid(beak, M("beak", "#E0A35A"), (0.0, 0.0, 0.0), 0.065, 0.05, 0.17)
    merge(m, beak, rot=qx(90.0), off=(0.0, 0.415, FZ - 0.01))
    add_ico(m, M("cere", "#F4F4F0"), (0.0, 0.44, FZ + 0.01), 0.035, subdiv=1, scale=(1.2, 0.7, 0.8))
    ring = M("pigeon_eye", "#F08A24")
    for s in (1, -1):
        add_torus(m, ring, (s * EX, EY, 0.245), 0.088, 0.018, seg=14, side=4)
        wing = Mesh("tmp")
        for k in range(3):
            L = 0.32 - 0.07 * k
            add_box(wing, grey if k < 2 else dark, (s * L / 2, -0.07 * k, -0.012 * k), (L, 0.085, 0.025), bevel=0.01)
        merge(m, wing, rot=qy(s * 38.0), off=(s * 0.22, 0.5, -0.24))


def head_clock(M, m):
    red, g = M("bell", "#D9483B", rough=0.35), M("gold", "#D4A22E", rough=0.35)
    for s in (1, -1):
        add_rod(m, g, (s * 0.1, TOP - 0.01, 0.0), (s * 0.19, TOP + 0.12, 0.0), 0.02, seg=6)
        bell = Mesh("tmp")
        bb.add_dome(bell, red, (0.0, 0.0, 0.0), 0.11, 12, 5, 90.0)
        add_torus(bell, g, (0.0, 0.0, 0.0), 0.11, 0.014, seg=12, side=4, rot=qx(90.0))
        merge(m, bell, rot=qz(-s * 28.0), off=(s * 0.21, TOP + 0.12, 0.0))
    add_rod(m, g, (0.0, TOP - 0.01, 0.0), (0.0, TOP + 0.2, 0.0), 0.015, seg=6)
    add_ico(m, g, (0.0, TOP + 0.21, 0.0), 0.04, subdiv=1)
    # wind-up key in the back
    add_rod(m, g, (0.0, 0.45, -0.22), (0.0, 0.45, -0.34), 0.025, seg=8)
    for s in (1, -1):
        add_box(m, g, (s * 0.075, 0.45, -0.35), (0.11, 0.09, 0.025), bevel=0.02)


def head_watch(M, m):
    ear, tan = M("dog_ear", "#7A4E2D"), M("dog_snout", "#D9B48A")
    for s in (1, -1):
        add_box(m, ear, (s * 0.345, TOP - 0.13, 0.02), (0.06, 0.3, 0.17), bevel=0.03, rot=qz(s * 14.0))
    add_box(m, tan, (0.0, 0.37, FZ + 0.06), (0.22, 0.13, 0.14), bevel=0.04)
    add_ico(m, M("dog_nose", "#1A1A1A", rough=0.3), (0.0, 0.42, FZ + 0.135), 0.04, subdiv=1, scale=(1.2, 0.9, 0.9))
    add_box(m, M("tongue", "#F07A8A"), (0.035, 0.305, FZ + 0.11), (0.06, 0.02, 0.06), bevel=0.008)
    band(m, M("collar", "#D9302E"), NECK, 0.05, grow=0.02)
    add_ico(m, M("gold", "#D4A22E", rough=0.35), (0.0, NECK - 0.05, FZ + 0.04), 0.035, subdiv=1)


def head_paisa(M, m):
    pink, g = M("star_glasses", "#FF4FA0", rough=0.4), M("gold", "#D4A22E", rough=0.35)
    for s in (1, -1):
        add_star(m, pink, (s * 0.14, EY, 0.255), 0.125, 0.025)
    add_box(m, g, (0.0, EY + 0.02, 0.258), (0.07, 0.016, 0.016))
    temples(m, g, EY + 0.03, 0.255, 0.24)


def head_claude(M, m):
    f = M("tortoise", "#8A5A36", rough=0.4)
    y = EY - 0.035
    for s in (1, -1):
        add_torus(m, f, (s * EX, y, 0.258), 0.075, 0.012, seg=12, side=4, a0=180.0, a1=360.0)
        add_box(m, f, (s * EX, y, 0.258), (0.16, 0.016, 0.016))
        add_rod(m, M("reading_lens", "#E8F2FF", rough=0.1, alpha=0.35), (s * EX, y - 0.035, 0.254), (s * EX, y - 0.035, 0.258), 0.05, seg=10)
    add_box(m, f, (0.0, y + 0.01, 0.258), (0.1, 0.014, 0.014))
    temples(m, f, y + 0.01, 0.255, 0.21)


def head_courier(M, m):
    cap = M("courier_cap", "#F2C230")
    add_box(m, cap, (0.0, TOP + 0.035, -0.02), (0.6, 0.13, 0.44), bevel=0.06)
    add_ico(m, cap, (0.0, TOP + 0.1, -0.02), 0.03, subdiv=0)
    add_box(m, M("cap_peak_dark", "#2E3A4B"), (0.0, TOP + 0.0, 0.28), (0.42, 0.022, 0.2), bevel=0.01, rot=qx(8.0))
    strap, bag = M("bag_strap", "#5E3B22"), M("bag", "#8A5A36")
    add_beam(m, strap, (-0.27, 0.37, FZ + 0.02), (0.29, 0.03, FZ + 0.02), 0.06, 0.015)
    add_beam(m, strap, (-0.27, 0.37, -FZ - 0.02), (0.27, 0.12, -FZ - 0.02), 0.06, 0.015)
    add_box(m, bag, (0.3, 0.05, -0.2), (0.12, 0.24, 0.3), bevel=0.03)
    add_box(m, M("bag_flap", "#6E4528"), (0.365, 0.11, -0.2), (0.02, 0.12, 0.3), bevel=0.008)

# ------------------------------------------------------------------ hand props (grip at the origin)

def prop_tape(M, m):
    tape, gr, rd = M("ticker_tape", "#F4EFD8"), M("tick_green", "#2EAD5B"), M("tick_red", "#D9483B")
    pts = [(0.0, 0.0, 0.0)] + [((0.05 if k % 2 else -0.05), -0.075 * k, 0.05 * k) for k in range(1, 5)]
    for k, (a, b) in enumerate(zip(pts, pts[1:])):
        add_beam(m, tape, a, b, 0.11, 0.01)
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2 + 0.008)
        add_box(m, gr if k % 2 == 0 else rd, mid, (0.035, 0.022, 0.004))


def prop_scissors(M, m):
    steel, red = M("steel", "#B8C0C8", rough=0.3), M("scissor_handle", "#D9302E")
    for s in (1, -1):
        add_torus(m, red, (s * 0.045, 0.0, 0.0), 0.045, 0.015, seg=12, side=4)
        add_beam(m, steel, (s * 0.03, 0.05, 0.0), (-s * 0.045, 0.42, 0.0), 0.04, 0.012)
    add_ico(m, M("screw", "#5E656E"), (0.0, 0.16, 0.012), 0.022, subdiv=0)


def prop_folder(M, m):
    add_box(m, M("folder", "#E8C27A"), (0.0, 0.16, 0.0), (0.27, 0.33, 0.03), bevel=0.008)
    add_box(m, M("folder", "#E8C27A"), (-0.07, 0.34, 0.0), (0.09, 0.04, 0.03))
    add_box(m, M("paper_white", "#FFFFFF"), (0.02, 0.19, 0.01), (0.23, 0.31, 0.02))


def prop_abacus(M, m):
    wood = M("abacus_wood", "#8A5A36")
    add_box(m, wood, (0.0, 0.0, 0.0), (0.34, 0.03, 0.04))
    add_box(m, wood, (0.0, 0.26, 0.0), (0.34, 0.03, 0.04))
    for s in (1, -1):
        add_box(m, wood, (s * 0.155, 0.13, 0.0), (0.03, 0.29, 0.04))
    cols = [M("bead_red", "#D9302E"), M("bead_yellow", "#F2C230"), M("bead_green", "#2EAD5B"), M("bead_blue", "#2E7BD6")]
    for r, y in enumerate((0.065, 0.13, 0.195)):
        add_box(m, M("rod", "#C9CED5"), (0.0, y, 0.0), (0.3, 0.008, 0.008))
        for k, x in enumerate((-0.11, -0.07, -0.03, 0.05, 0.09)):
            add_box(m, cols[(r + k) % 4], (x + 0.02 * (r % 2), y, 0.0), (0.035, 0.04, 0.04))


def prop_globe(M, m):
    g = M("gold", "#D4A22E", rough=0.35)
    add_cyl(m, g, (0.0, 0.015, 0.0), 0.06, 0.07, 0.03, 12)
    add_rod(m, g, (0.0, 0.03, 0.0), (0.0, 0.08, 0.0), 0.012, seg=6)
    add_ico(m, M("ocean", "#2F7FD1", rough=0.5), (0.0, 0.19, 0.0), 0.11, subdiv=1)
    land = M("land", "#58B05A")
    for d in ((0.5, 0.6, 0.62), (-0.7, 0.1, 0.7), (0.2, -0.5, 0.85), (-0.3, 0.7, -0.6)):
        n = bb.v_unit(d)
        add_ico(m, land, (0.1 * n[0], 0.19 + 0.1 * n[1], 0.1 * n[2]), 0.045, subdiv=1, scale=(1.0, 0.8, 1.0))
    add_torus(m, g, (0.0, 0.19, 0.0), 0.125, 0.009, seg=18, side=4, rot=q_mul(qz(23.0), qy(90.0)))


def prop_piggy(M, m):
    pink, dark = M("piggy", "#F59AB5"), M("piggy_dark", "#E07A99")
    add_ico(m, pink, (0.0, 0.12, 0.0), 1.0, subdiv=1, scale=(0.1, 0.095, 0.14))
    add_rod(m, dark, (0.0, 0.12, 0.125), (0.0, 0.12, 0.165), 0.045, seg=10)
    for s in (1, -1):
        add_pyramid(m, dark, (s * 0.05, 0.195, 0.06), 0.025, 0.02, 0.05)
        for z in (0.07, -0.07):
            add_box(m, dark, (s * 0.05, 0.03, z), (0.035, 0.05, 0.035))
        add_ico(m, M("piggy_eye", "#1A1A1A"), (s * 0.04, 0.16, 0.115), 0.012, subdiv=0)
    add_box(m, M("slot", "#3A2E2A"), (0.0, 0.215, 0.0), (0.06, 0.01, 0.012))


def prop_mic(M, m):
    add_rod(m, M("mic_handle", "#1C1C1F"), (0.0, -0.02, 0.0), (0.0, 0.17, 0.0), 0.022, seg=8)
    add_box(m, M("mic_flag", "#D9483B"), (0.0, 0.14, 0.0), (0.075, 0.07, 0.075), bevel=0.01)
    add_ico(m, M("mic_head", "#3A3E44", rough=0.4), (0.0, 0.22, 0.0), 0.055, subdiv=1)


def prop_newspaper(M, m):
    add_rod(m, M("newsprint", "#EDE9DE"), (0.0, -0.02, 0.0), (0.0, 0.3, 0.0), 0.045, seg=10)
    for y in (0.06, 0.2):
        add_rod(m, M("newsprint_band", "#9AA0A6"), (0.0, y, 0.0), (0.0, y + 0.03, 0.0), 0.047, seg=10, caps=(False, False))
    add_rod(m, M("newsprint_edge", "#C9C3B4"), (0.0, 0.3, 0.0), (0.0, 0.305, 0.0), 0.035, seg=10)


def prop_stamp(M, m):
    add_rod(m, M("stamp_wood", "#A0703C"), (0.0, -0.03, 0.0), (0.0, 0.08, 0.0), 0.02, seg=8)
    add_ico(m, M("stamp_knob", "#8A3A2E"), (0.0, 0.1, 0.0), 0.045, subdiv=1)
    add_box(m, M("stamp_base", "#3A3E44"), (0.0, -0.05, 0.0), (0.14, 0.045, 0.1), bevel=0.01)
    add_box(m, M("stamp_ink", "#C0392B"), (0.0, -0.078, 0.0), (0.13, 0.012, 0.09))


def prop_vr_controller(M, m):
    add_rod(m, M("controller", "#2A2D33"), (0.0, -0.02, 0.0), (0.0, 0.12, 0.0), 0.026, seg=8)
    add_torus(m, M("controller", "#2A2D33"), (0.0, 0.17, 0.0), 0.06, 0.013, seg=12, side=4, rot=qx(70.0))
    add_ico(m, M("goggle_glow", "#39E1FF", emissive="#39E1FF"), (0.0, 0.12, 0.022), 0.015, subdiv=0)


def prop_binoculars(M, m):
    b, glass = M("binoculars", "#2F3A2E"), M("bino_glass", "#1D2733", rough=0.1)
    for s in (1, -1):
        add_rod(m, b, (s * 0.048, 0.06, -0.07), (s * 0.048, 0.06, 0.08), 0.042, seg=10)
        add_rod(m, b, (s * 0.048, 0.06, -0.11), (s * 0.048, 0.06, -0.07), 0.028, seg=8)
        add_rod(m, glass, (s * 0.048, 0.06, 0.08), (s * 0.048, 0.06, 0.085), 0.036, seg=10)
    add_box(m, b, (0.0, 0.06, 0.0), (0.06, 0.03, 0.08))


def prop_pillow(M, m):
    add_box(m, M("pillow", "#F4F4F0"), (0.0, 0.0, 0.0), (0.36, 0.26, 0.11), bevel=0.05)
    add_box(m, M("pillow_stripe", "#9CC3E6"), (0.0, 0.0, 0.0), (0.37, 0.04, 0.115))


def prop_drum(M, m):
    d, r = M("oil_drum", "#2F5FA8", rough=0.5), M("drum_rib", "#244C88", rough=0.5)
    add_cyl(m, d, (0.0, -0.1, 0.0), 0.08, 0.08, 0.18, 14)
    for y in (-0.06, -0.14):
        add_cyl(m, r, (0.0, y, 0.0), 0.084, 0.084, 0.014, 14, caps=(False, False))
    add_cyl(m, M("drum_cap", "#C9CED5"), (0.03, -0.005, 0.0), 0.015, 0.015, 0.012, 8)


def prop_coin(M, m):
    g, dg = M("coin_gold", "#E9B941", rough=0.35), M("coin_dark", "#A0731F", rough=0.35)
    add_rod(m, g, (0.0, 0.13, -0.015), (0.0, 0.13, 0.015), 0.13, seg=20)
    add_torus(m, dg, (0.0, 0.13, 0.016), 0.118, 0.01, seg=20, side=4)
    add_torus(m, dg, (0.005, 0.13, 0.018), 0.06, 0.013, seg=12, side=4, a0=50.0, a1=310.0)   # euro sign
    for y in (0.115, 0.145):
        add_box(m, dg, (-0.025, y, 0.02), (0.085, 0.016, 0.01))


def prop_box(M, m):
    add_box(m, M("cardboard", "#C49A6C"), (0.0, -0.095, 0.0), (0.22, 0.17, 0.2), bevel=0.01)
    add_box(m, M("tape", "#E8D9A8"), (0.0, -0.009, 0.0), (0.05, 0.004, 0.205))
    add_box(m, M("tape", "#E8D9A8"), (0.0, -0.06, 0.101), (0.05, 0.07, 0.004))


def prop_envelope(M, m):
    w = M("envelope", "#FFFFFF")
    add_box(m, w, (0.0, 0.08, 0.0), (0.22, 0.15, 0.012))
    for s in (1, -1):
        add_beam(m, M("envelope_line", "#C9CED5"), (s * 0.105, 0.15, 0.008), (0.0, 0.08, 0.008), 0.008, 0.004)
    add_ico(m, M("seal", "#D9302E"), (0.0, 0.08, 0.012), 0.022, subdiv=0)


def prop_flashlight(M, m):
    add_rod(m, M("torch", "#2A2D33"), (0.0, -0.02, 0.0), (0.0, 0.16, 0.0), 0.03, seg=10)
    add_rod(m, M("torch", "#2A2D33"), (0.0, 0.16, 0.0), (0.0, 0.21, 0.0), 0.03, seg=10, r1=0.048)
    add_rod(m, M("torch_lens", "#FFF3A8", emissive="#FFE066"), (0.0, 0.21, 0.0), (0.0, 0.215, 0.0), 0.042, seg=10)


def prop_purse(M, m):
    add_ico(m, M("purse", "#A83246"), (0.0, -0.075, 0.0), 1.0, subdiv=1, scale=(0.11, 0.085, 0.06))
    g = M("gold", "#D4A22E", rough=0.35)
    add_box(m, g, (0.0, -0.005, 0.0), (0.13, 0.016, 0.035))
    for s in (1, -1):
        add_ico(m, g, (s * 0.022, 0.012, 0.0), 0.018, subdiv=0)


def prop_quill(M, m):
    cream, tip = M("feather", "#F4EFE2"), M("feather_tip", "#D97757")
    add_rod(m, M("quill_shaft", "#C9B48A"), (0.0, -0.02, 0.0), (0.0, 0.42, 0.0), 0.008, seg=6)
    add_ico(m, cream, (0.0, 0.25, 0.0), 1.0, subdiv=1, scale=(0.016, 0.17, 0.07))
    add_ico(m, tip, (0.0, 0.39, 0.0), 1.0, subdiv=1, scale=(0.016, 0.07, 0.045))
    add_rod(m, M("nib", "#2A2D33"), (0.0, -0.02, 0.0), (0.0, -0.07, 0.0), 0.012, seg=6, r1=0.002)


def prop_note(M, m):
    add_box(m, M("note_paper", "#FBF6E6"), (0.0, 0.14, 0.0), (0.24, 0.3, 0.01))
    add_box(m, M("feather_tip", "#D97757"), (0.0, 0.26, 0.007), (0.18, 0.03, 0.004))
    for k in range(4):
        add_box(m, M("note_line", "#9AA0A6"), (-0.01 + 0.01 * (k % 2), 0.2 - 0.05 * k, 0.007), (0.16 - 0.03 * (k % 2), 0.014, 0.004))

# ------------------------------------------------------------------ the crew

# id: station, pants (world/data.js), head quirk, right-hand prop (+ grip rotation, offset),
#     carry: "crate" = the prop rides on the crate (rotation, offset, scale) or "drop"; lift = hat height
QUIRKS = {
    "bhav":    dict(st="bom", pants="#35588a", head=head_bhav, prop=prop_tape, grip=(0.0, 0.0), carry="drop", lift=0.0),
    "split":   dict(st="bom", pants="#5a4a7a", head=head_split, prop=prop_scissors, grip=(35.0, 0.0), carry="drop", lift=0.15),
    "filer":   dict(st="bom", pants="#2f6b6b", head=head_filer, prop=prop_folder, grip=(60.0, -0.05), carry=("crate", (-90.0, 0.0), 0.9), lift=0.0),
    "promo":   dict(st="bom", pants="#6b4e2e", head=head_promo, prop=None, lift=0.38),
    "tally":   dict(st="bom", pants="#3c3c3c", head=head_tally, prop=prop_abacus, grip=(0.0, -0.08), carry=("crate", (0.0, 0.0), 0.75), lift=0.0),
    "nsdl":    dict(st="bom", pants="#7a6a3a", head=head_nsdl, prop=prop_globe, grip=(0.0, -0.02), carry=("crate", (0.0, 0.0), 0.9), lift=0.0),
    "amfi":    dict(st="bom", pants="#35588a", head=head_amfi, prop=prop_piggy, grip=(0.0, -0.04), carry=("crate", (0.0, 0.0), 1.0), lift=0.22),
    "khabri":  dict(st="bom", pants="#444444", head=head_khabri, prop=prop_mic, grip=(25.0, 0.0), carry="drop", lift=0.22),
    "khabri2": dict(st="del", pants="#6b4e2e", head=head_khabri2, prop=prop_newspaper, grip=(55.0, 0.0), carry=("crate", (90.0, 90.0), 1.0, 0.045), lift=0.17),
    "sarkari": dict(st="del", pants="#2f4f7f", head=head_sarkari, prop=prop_stamp, grip=(0.0, 0.0), carry=("crate", (0.0, 0.0), 1.0), lift=0.16),
    "screeny": dict(st="web", pants="#4a4a6a", head=head_screeny, prop=prop_vr_controller, grip=(20.0, 0.0), carry="drop", lift=0.0),
    "tikki":   dict(st="web", pants="#2f6b6b", head=head_tikki, prop=prop_binoculars, grip=(0.0, 0.0), carry="drop", lift=0.17),
    "dalal":   dict(st="bom", pants="#5d5d5d", head=head_dalal, prop=None, lift=0.32, sleeper=True),
    "fed":     dict(st="stl", pants="#3b3b55", head=head_fed, prop=prop_drum, grip=(0.0, 0.0), carry=("crate", (0.0, 0.0), 1.0, 0.18), lift=0.27),
    "euro":    dict(st="fra", pants="#6b5a3a", head=head_euro, prop=prop_coin, grip=(0.0, -0.02), carry=("crate", (0.0, 0.0), 0.8), lift=0.0),
    "gnews":   dict(st="mtv", pants="#35588a", head=head_gnews, prop=None, lift=0.34),
    "drive":   dict(st="mtv", pants="#4a4a4a", head=head_drive, prop=prop_box, grip=(0.0, 0.0), carry=("crate", (0.0, 0.0), 1.0, 0.17), lift=0.27),
    "pigeon":  dict(st="dxb", pants="#2f6fae", head=head_pigeon, prop=prop_envelope, grip=(15.0, -0.02), carry="drop", lift=0.04),
    "clock":   dict(st="blr", pants="#3b3b55", head=head_clock, prop=None, lift=0.25),
    "watch":   dict(st="blr", pants="#5a4a3a", head=head_watch, prop=prop_flashlight, grip=(75.0, 0.0), carry="drop", lift=0.0),
    "paisa":   dict(st="blr", pants="#2f4f7f", head=head_paisa, prop=prop_purse, grip=(0.0, 0.0), carry=("crate", (0.0, 0.0), 1.0, 0.1), lift=0.0),
    "claude":  dict(st="sfo", pants="#3b3b55", head=head_claude, prop=prop_quill, grip=(25.0, 0.0), carry="drop", lift=0.0,
                    prop_l=prop_note, carry_l=("crate", (-90.0, 0.0), 1.0)),
    "courier": dict(st="blr", pants="#5B6B3A", head=head_courier, prop=None, lift=0.12, flag="IN"),
}
FLAG_OF = {"bom": "IN", "del": "IN", "web": "IN", "blr": "IN", "stl": "US", "mtv": "US", "sfo": "US", "fra": "DE", "dxb": "AE"}


def _rest_points(root, start):
    """Rest-pose vertex positions under `start` (in root space), sampled from POSITION data."""
    out = []

    def visit(n, mtx):
        m2 = bb._mm(mtx, bb._mat_trs(n.t, n.r, n.s))
        if n.mesh is not None:
            for p in n.mesh.prims:
                for v in p.pos[::3]:
                    out.append(tuple(m2[a][0] * v[0] + m2[a][1] * v[1] + m2[a][2] * v[2] + m2[a][3] for a in range(3)))
        for c in n.children:
            visit(c, m2)
    visit(start, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return out


def make_quirk(ident, spec):
    def quirk(M, root, nodes, carry):
        torso = nodes["torso"]
        hm = Mesh("quirk_" + ident)
        spec["head"](M, hm)
        torso.add(Node("quirk", mesh=hm))
        for key, arm, carry_key, sx in (("prop", "arm_R", "carry", 1.0), ("prop_l", "arm_L", "carry_l", -1.0)):
            fn = spec.get(key)
            if not fn:
                continue
            pm = Mesh(key)
            fn(M, pm)
            if not carry:
                tilt, dx = spec.get("grip", (0.0, 0.0)) if key == "prop" else (20.0, 0.0)
                t = (HAND[0] * sx + dx * sx, HAND[1], HAND[2])
                nodes[arm].add(Node(key, t=t, r=qx(tilt), mesh=pm))
            else:
                how = spec.get(carry_key, "drop")
                if how != "drop":
                    rx, ry = how[1]
                    k = how[2]
                    up = how[3] if len(how) > 3 else 0.0
                    off = (0.1 * -sx if key == "prop_l" else 0.0, CRATE_TOP + up, 0.0)
                    nodes["crate"].add(Node(key, t=off, r=q_mul(qy(ry), qx(rx)), s=(k, k, k), mesh=pm))
        if spec.get("sleeper"):
            return dalal_sleep(M, root, nodes, carry)
        return None
    return quirk


def dalal_sleep(M, root, nodes, carry):
    """Dalal sleeps in idle: lies on his back across the view on a pillow, eyes shut,
    slow breathing, Zzz rising. A pillow is in his left hand whenever he is awake."""
    hips, torso = nodes["hips"], nodes["torso"]
    sleeper = Node("sleeper")
    root.children = [sleeper]
    sleeper.add(hips)
    pm = Mesh("pillow")
    prop_pillow(M, pm)
    if not carry:
        nodes["arm_L"].add(Node("pillow_hand", t=(0.06, -0.13, 0.12), r=q_mul(qx(-15.0), qy(10.0)), mesh=pm))
    torso.add(Node("pillow_head", t=(0.0, 0.6, -0.27), s=HIDDEN, mesh=pm))
    zm = Mesh("zzz")
    zmat = M("zzz", "#3A5FC8", rough=0.5, emissive="#1A2F70")
    letter_z(zm, zmat, 1.0)
    for k in range(3):
        root.add(Node("zzz_%d" % k, s=HIDDEN, mesh=zm))
    lie = q_mul(qy(90.0), qx(-82.0))
    # rest the lowest point of the lying body (with the head pillow) on the ground
    pts = _rest_points(root, sleeper)
    rm = bb.q_to_m(lie)
    ys = [bb.m_apply(rm, p)[1] for p in pts]
    xs = [bb.m_apply(rm, p)[0] for p in pts]
    pil = bb._mm(bb._mat_trs((0, bb.HIP_Y, 0), (0, 0, 0, 1), (1, 1, 1)), bb._mat_trs(torso.t, torso.r, torso.s))
    pc = (0.0, 0.6 + pil[1][3], -0.27)
    pc_r = bb.m_apply(rm, pc)
    lift = -min(min(ys), pc_r[1] - 0.06)
    shift = -(min(xs) + max(xs)) / 2
    tr = (shift, lift, 0.0)
    head = bb.m_apply(rm, (0.0, 0.85, 0.0))
    hx, hy = head[0] + shift, head[1] + lift

    def idle(t):
        p = 2 * math.pi * t / bb.IDLE_T
        breath = math.sin(p)
        d = {("sleeper", "translation"): tr, ("sleeper", "rotation"): lie,
             ("torso", "scale"): (1.0 + 0.012 * breath, 1.0 + 0.03 * breath, 1.0 + 0.012 * breath),
             ("eyes", "scale"): (1.0, 0.1, 1.0),
             ("pillow_head", "scale"): (1.0, 1.0, 1.0)}
        if not carry:
            d[("pillow_hand", "scale")] = HIDDEN
        for k in range(3):
            u = (t / bb.IDLE_T + k / 3.0) % 1.0
            sc = (0.13 + 0.13 * u) * (1.0 if u < 0.8 else max((1.0 - u) / 0.2, 1e-3))
            d[("zzz_%d" % k, "translation")] = (hx + 0.12 + 0.25 * u, hy + 0.35 + 0.55 * u, 0.0)
            d[("zzz_%d" % k, "scale")] = (sc, sc, sc)
        return d
    return {"idle": bb.sample_clip(bb.IDLE_T, bb.IDLE_N, idle)}


def build(ident, carry):
    spec = QUIRKS[ident]
    flag = spec.get("flag") or FLAG_OF[spec["st"]]
    return bb.build_bot(flag, carry, ident=ident, pants=spec["pants"], quirk=make_quirk(ident, spec),
                        lift=spec.get("lift", 0.0) if carry else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", help="comma-separated ids")
    args = ap.parse_args()
    ids = args.only.split(",") if args.only else list(QUIRKS)
    required = ("run", "idle") + tuple(bb.ACTION_TIMES)
    print("%-22s %8s %6s %6s  %s" % ("file", "bytes", "tris", "nodes", "rest bbox (w x h x d)"))
    n = 0
    for ident in ids:
        for carry in (False, True):
            name, root, mats, textures, anims = build(ident, carry)
            path = OUT_DIR / (name + ".glb")
            bb.write_glb(path, name, root, mats, textures, anims)
            st = bb.validate(path, required=required, oneshot=bb.ONE_SHOT)
            lo, hi = st["bbox"]
            print("%-22s %8d %6d %6d  %.2f x %.2f x %.2f (min y %.3f)" % (
                path.name, st["bytes"], st["tris"], st["nodes"], hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], lo[1]))
            n += 1
    print("ok: %d files valid" % n)


if __name__ == "__main__":
    main()
