#!/usr/bin/env python3
"""AAA scene -- PROPS agent.  "The Water Court" set dressing.

This tier did not exist.  The first build shipped with scripts/aaa/props.py absent while
assemble.py already listed "props" in AGENTS, so it silently read two missing fragments
and nobody noticed until three critics counted the geoms.  What was missing:

  * 341 of the 1250 scenery-geom budget and 635 of the 1600 BE, unspent
  * ZERO geoms carrying Lapis Cobalt -- palette entry 10 and bible section 2 rule 3's
    "ONE saturated object ... and scarcity is the point".  The court's only chromatic
    focal point simply did not exist; every accent in the built scene was warm.
  * bible section 7's "one prop cluster per 1.8 m of aisle run" -- that is ~16 clusters,
    and there were none, so both side aisles and the west yard were bare floor
  * the near band.  In the hero corridor exactly THREE geoms had a top below 60 mm, and
    all three were atmosphere's curb strips.  The bottom quarter of every frame is the
    15 cm of floor between 0.148 m and 0.296 m, and there was nothing in it -- which is
    the direct cause of pass-criterion 1 (near-band contrast std >= 28) failing.

Writes (idempotently -- byte-identical on re-run) into src/mjlab_microduck/robot/microduck/:
    aaa_props_assets.xml    <material> fragment -- NO new textures
    aaa_props_body.xml      <geom> fragment, all contype=0 conaffinity=0

NO NEW TEXTURE FAMILY.  materials.py authored a 20-map library (wood, canvas, paint,
iron, terra, sand, glass, lapis, olive, stone, ...) sized for large surfaces and it was
referenced by zero geoms in the shipped scene -- 12 MB of PNG built for a consumer that
never ran.  This agent is that consumer.  It declares its own props_* materials, but every
one of them points at an existing materials_tex_* texture; the only thing it changes is
texrepeat, because the library's periods were authored for walls (a 1.40 m wood period on
a 0.12 m crate shows 9% of one tile, i.e. a flat patch).

MATERIAL PERIODS.  texuniform="true" gives a repeat period of 2/texrepeat METRES
(probe-verified by the terrain agent, and it is not 1/texrepeat).  Prop-scale periods:
    crate boarding      0.09 m  -> texrepeat 22
    amphora wheel-throw 0.05 m  -> texrepeat 40
    sacking weave       0.06 m  -> texrepeat 33
    ironwork            0.07 m  -> texrepeat 29
    lapis glaze crazing 0.05 m  -> texrepeat 40

PLACEMENT LAW (bible sections 6, 7, 9):
    * nothing inside r <= 0.9 m of the origin -- the spawn apron stays empty
    * the hero corridor |y| < 1.195 carries only ground scatter under 45 mm, and none of
      it on the walking centre-line: everything sits at |y| in [0.28, 1.10]
    * side aisles get one cluster per 1.8 m of run, never two of the same silhouette
      adjacent
    * west yard runs at half the aisle density -- it is the breathing space
    * every repeated element varies on three axes: yaw over the full circle, scale +/-18%,
      and a per-instance rgba tint of +/-6% on value.  No two adjacent instances share a
      yaw within 0.3 rad.
    * EVERY geom is contype="0" conaffinity="0".  The duck may walk through a crate; it
      may not walk through the fountain, and architecture owns that distinction.
"""
import math
import os
from pathlib import Path

import numpy as np

MD = Path(__file__).resolve().parents[2] / "src/mjlab_microduck/robot/microduck"
P = "props_"

BE_COST = {"box": 1.0, "cylinder": 5.3, "sphere": 2.8, "ellipsoid": 2.7, "capsule": 6.1}

GE: list[str] = []
_names: set[str] = set()
_stats = {"box": 0, "cylinder": 0, "ellipsoid": 0}

rng = np.random.default_rng(31337)


# ----------------------------------------------------------------------------------
def _q(yaw, pitch=0.0, roll=0.0):
    """Quaternion for ZYX euler (yaw about z, then pitch about y, then roll about x)."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return f'quat="{w:.6f} {x:.6f} {y:.6f} {z:.6f}"'


def g(name, mat, pos, size, typ="box", yaw=0.0, pitch=0.0, roll=0.0, tint=None):
    """One visible, non-colliding prop geom."""
    nm = P + name
    assert nm not in _names, f"duplicate {nm}"
    assert min(size) > 1e-4, f"{nm}: non-positive half-size {size}"
    r = math.hypot(pos[0], pos[1])
    assert r > 0.9, f"{nm}: inside the spawn apron at r={r:.3f}"
    _names.add(nm)
    _stats[typ] = _stats.get(typ, 0) + 1
    q = "" if (abs(yaw) < 1e-9 and abs(pitch) < 1e-9 and abs(roll) < 1e-9) else " " + _q(yaw, pitch, roll)
    c = "" if tint is None else f' rgba="{tint[0]:.3f} {tint[1]:.3f} {tint[2]:.3f} 1"'
    GE.append(
        f'<geom name="{nm}" type="{typ}"'
        f' pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"'
        f' size="{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}"{q}'
        f' material="{P}{mat}"{c} contype="0" conaffinity="0"/>'
    )


def note(txt=""):
    GE.append(f"<!-- {txt} -->" if txt else "")


def tint(v=0.06, warm=0.0):
    """Per-instance albedo tint: +/-6% on value (bible section 7's variation law) with an
    optional warm/cool skew, applied as a per-geom rgba MULTIPLIER on the shared texture."""
    k = 1.0 + rng.uniform(-v, v)
    return (k * (1 + warm), k, k * (1 - warm * 0.8))


def band(x, y):
    """Aerial-perspective suffix by true distance from the spawn point."""
    r = math.hypot(x, y)
    return "" if r < 3.4 else ("_mid" if r < 6.2 else "_far")


# ==================================================================================
#                                 P R O P   K I T
# ==================================================================================
def crate(nm, x, y, z0, s, yaw, mat="wood"):
    """A boarded crate: 4 side boards + a lid + 2 corner battens.  7 boxes.
    Boarding is what makes a crate read as built rather than as a cube -- a bare box at
    0.12 m is exactly the flat-colour box the brief bans."""
    h = 0.055 * s
    w = 0.062 * s
    b = band(x, y)
    for k, (dx, dy, sx, sy) in enumerate((
            (0, -w, w, 0.006), (0, w, w, 0.006),
            (-w, 0, 0.006, w), (w, 0, 0.006, w))):
        c, si = math.cos(yaw), math.sin(yaw)
        g(f"{nm}_s{k}", mat + b, (x + dx * c - dy * si, y + dx * si + dy * c, z0 + h),
          (sx, sy, h), yaw=yaw, tint=tint())
    g(f"{nm}_lid", mat + b, (x, y, z0 + 2 * h + 0.005), (w * 1.06, w * 1.06, 0.005),
      yaw=yaw, tint=tint())
    for k, sgn in ((0, -1), (1, 1)):
        c, si = math.cos(yaw), math.sin(yaw)
        dx, dy = sgn * w * 0.98, sgn * w * 0.98
        g(f"{nm}_b{k}", "iron" + b, (x + dx * c - dy * si, y + dx * si + dy * c, z0 + h * 0.55),
          (0.008, 0.008, h * 0.55), yaw=yaw, tint=tint(0.04))


def amphora(nm, x, y, z0, s, yaw, lean=0.0, mat="terra"):
    """Wheel-thrown storage jar: foot, belly, shoulder, neck, rim, two handles.  7 geoms,
    3 of them cylinders -- the one silhouette in this kit that a box cannot fake."""
    b = band(x, y)
    R = 0.038 * s
    # ONE ellipsoid for the belly, boxes for everything else.  A cylinder costs 5.3 boxes
    # and at a 0.04 m radius, 2 m away, nothing in frame can tell one from the other; the
    # belly is the only part whose silhouette actually has to curve.
    g(f"{nm}_foot", mat + b, (x, y, z0 + 0.008 * s), (R * 0.40, R * 0.40, 0.008 * s),
      yaw=yaw + 0.39, pitch=lean, tint=tint(0.05, 0.02))
    g(f"{nm}_belly", mat + b, (x, y, z0 + 0.048 * s), (R, R, 0.040 * s),
      typ="ellipsoid", yaw=yaw, pitch=lean, tint=tint(0.05, 0.02))
    g(f"{nm}_shldr", mat + b, (x, y, z0 + 0.090 * s), (R * 0.58, R * 0.58, 0.016 * s),
      yaw=yaw + 0.39, pitch=lean, tint=tint(0.05))
    g(f"{nm}_neck", mat + b, (x, y, z0 + 0.116 * s), (R * 0.28, R * 0.28, 0.016 * s),
      yaw=yaw + 0.39, pitch=lean, tint=tint(0.05))
    g(f"{nm}_rim", mat + b, (x, y, z0 + 0.134 * s), (R * 0.40, R * 0.40, 0.005 * s),
      typ="box", yaw=yaw + 0.78, tint=tint(0.05))
    for k, sgn in ((0, -1), (1, 1)):
        c, si = math.cos(yaw), math.sin(yaw)
        dx = sgn * R * 0.72
        g(f"{nm}_h{k}", mat + b, (x + dx * c, y + dx * si, z0 + 0.108 * s),
          (0.006 * s, 0.006 * s, 0.020 * s), yaw=yaw, pitch=sgn * 0.42, tint=tint(0.05))


def sherds(nm, x, y, n, spread, s=1.0):
    """Broken pot lying where it fell.  Flat angular shards, all under 12 mm."""
    b = band(x, y)
    for k in range(n):
        a = rng.uniform(0, 2 * math.pi)
        d = spread * math.sqrt(rng.random())
        sz = rng.uniform(0.010, 0.024) * s
        g(f"{nm}_{k}", "terra" + b,
          (x + d * math.cos(a), y + d * math.sin(a), 0.0035 + 0.002 * k % 0.004),
          (sz, sz * rng.uniform(0.45, 0.85), rng.uniform(0.0025, 0.0055)),
          yaw=rng.uniform(0, 2 * math.pi), roll=rng.uniform(-0.35, 0.35), tint=tint(0.07, 0.02))


def sack(nm, x, y, s, yaw, spilled=False):
    """A slumped canvas sack: three stacked ellipsoids of falling size, plus a tied neck.
    An ellipsoid costs 2.7 boxes and this is one of the two places it earns that."""
    b = band(x, y)
    g(f"{nm}_a", "canvas" + b, (x, y, 0.026 * s), (0.052 * s, 0.040 * s, 0.026 * s),
      typ="ellipsoid", yaw=yaw, tint=tint())
    g(f"{nm}_b", "canvas" + b, (x + 0.008 * s, y - 0.006 * s, 0.062 * s),
      (0.040 * s, 0.032 * s, 0.020 * s), typ="ellipsoid", yaw=yaw + 0.5, tint=tint())
    g(f"{nm}_neck", "canvas" + b, (x + 0.012 * s, y - 0.010 * s, 0.088 * s),
      (0.016 * s, 0.014 * s, 0.012 * s), yaw=yaw + 0.9, tint=tint())
    g(f"{nm}_tie", "iron" + b, (x + 0.012 * s, y - 0.010 * s, 0.081 * s),
      (0.018 * s, 0.016 * s, 0.003 * s), yaw=yaw + 0.9, tint=tint(0.04))
    if spilled:
        for k in range(5):
            a = rng.uniform(-1.2, 1.2) + yaw
            d = 0.05 + 0.055 * k
            g(f"{nm}_sp{k}", "sand" + b,
              (x + d * math.cos(a), y + d * math.sin(a), 0.0032),
              (0.030 - 0.004 * k, 0.022 - 0.003 * k, 0.0030),
              yaw=a + rng.uniform(-0.4, 0.4), tint=tint(0.05))


def rope_coil(nm, x, y, s, yaw):
    """Coiled rope: four flattened rings of falling radius, plus a loose tail.
    Cylinders, and worth it -- a coil of rope is the one prop with no straight edge."""
    b = band(x, y)
    for k in range(3):
        g(f"{nm}_r{k}", "canvas" + b, (x + 0.004 * k, y - 0.003 * k, 0.008 + 0.013 * k),
          ((0.062 - 0.011 * k) * s, (0.062 - 0.011 * k) * s, 0.0075 * s),
          typ="cylinder", yaw=yaw + 0.4 * k, tint=tint(0.05))
    for k in range(3):
        a = yaw + 2.1 + 0.55 * k
        g(f"{nm}_t{k}", "canvas" + b,
          (x + (0.075 + 0.055 * k) * math.cos(a), y + (0.075 + 0.055 * k) * math.sin(a), 0.008),
          (0.032 * s, 0.008 * s, 0.007 * s), yaw=a + rng.uniform(-0.3, 0.3), tint=tint(0.05))


def basket(nm, x, y, s, yaw):
    """Woven basket: a splayed drum of six staves plus a rim hoop."""
    b = band(x, y)
    R = 0.048 * s
    for k in range(6):
        th = yaw + k * math.pi / 3
        g(f"{nm}_s{k}", "canvas" + b,
          (x + R * math.cos(th), y + R * math.sin(th), 0.030 * s),
          (R * 0.56, 0.006, 0.030 * s), yaw=th + math.pi / 2, tint=tint(0.05))
    g(f"{nm}_rim", "wood" + b, (x, y, 0.064 * s), (R * 1.08, R * 1.08, 0.006 * s),
      yaw=yaw + 0.39, tint=tint(0.05))


def ladder(nm, x, y, yaw, lean, L=0.62):
    """Leaning against the wall: 2 rails + 7 rungs.  The rungs read as a rhythm of hard
    little shadows on the wall behind, which is what makes it worth 9 geoms."""
    b = band(x, y)
    c, s = math.cos(yaw), math.sin(yaw)
    for k, sgn in ((0, -1), (1, 1)):
        dx, dy = 0.0, sgn * 0.052
        g(f"{nm}_rail{k}", "wood" + b,
          (x + dx * c - dy * s, y + dx * s + dy * c, L * 0.5 * math.cos(lean)),
          (0.010, 0.008, L * 0.5), yaw=yaw, pitch=lean, tint=tint())
    for k in range(7):
        t = (k + 0.6) / 7.6
        zz = L * t * math.cos(lean)
        off = -L * t * math.sin(lean)
        g(f"{nm}_rung{k}", "wood" + b,
          (x + off * c, y + off * s, zz), (0.006, 0.050, 0.005), yaw=yaw, tint=tint(0.05))


def cart_wheel(nm, x, y, yaw, s=1.0):
    """A wheel off its axle, leaning on a wall: felloe ring approximated by 8 boxes on a
    circle, a hub, and 4 spokes.  Reads as a circle from any angle without a cylinder."""
    b = band(x, y)
    R = 0.115 * s
    for k in range(6):
        th = k * math.pi / 3
        px = R * math.cos(th)
        pz = R * math.sin(th) + R + 0.006
        g(f"{nm}_f{k}", "wood" + b,
          (x + px * math.cos(yaw), y + px * math.sin(yaw), pz),
          (R * 0.56, 0.010, 0.013), yaw=yaw, roll=0.0, pitch=-th + math.pi / 2,
          tint=tint(0.05))
    for k in range(3):
        th = k * math.pi / 3 + 0.2
        g(f"{nm}_sp{k}", "wood" + b, (x, y, R + 0.006),
          (R * 0.86, 0.007, 0.007), yaw=yaw, pitch=-th, tint=tint(0.05))
    g(f"{nm}_hub", "iron" + b, (x, y, R + 0.006), (0.022, 0.020, 0.020), yaw=yaw,
      tint=tint(0.04))


def brazier(nm, x, y, s, yaw):
    """Iron fire-basket on three legs, with a bed of embers.  The embers are the only
    emissive this agent spends, and they go in the shadow aisle where they read."""
    b = band(x, y)
    R = 0.052 * s
    for k in range(3):
        th = yaw + k * 2 * math.pi / 3
        g(f"{nm}_leg{k}", "iron" + b,
          (x + R * 0.62 * math.cos(th), y + R * 0.62 * math.sin(th), 0.030 * s),
          (0.007, 0.007, 0.030 * s), yaw=th, pitch=0.16, tint=tint(0.04))
    g(f"{nm}_bowl", "iron" + b, (x, y, 0.072 * s), (R, R, 0.020 * s), typ="cylinder",
      yaw=yaw, tint=tint(0.04))
    g(f"{nm}_rim", "iron" + b, (x, y, 0.092 * s), (R * 1.08, R * 1.08, 0.005 * s),
      yaw=yaw + 0.39, tint=tint(0.04))
    g(f"{nm}_ember", "ember", (x, y, 0.086 * s), (R * 0.78, R * 0.78, 0.004 * s),
      yaw=yaw + 0.39)


def broom(nm, x, y, yaw, lean):
    b = band(x, y)
    c, s = math.cos(yaw), math.sin(yaw)
    g(f"{nm}_shaft", "wood" + b, (x, y, 0.145 * math.cos(lean)),
      (0.007, 0.007, 0.145), yaw=yaw, pitch=lean, tint=tint())
    off = -0.29 * math.sin(lean)
    for k in range(4):
        g(f"{nm}_br{k}", "canvas" + b,
          (x + off * c + 0.006 * k * c, y + off * s + 0.006 * k * s, 0.026),
          (0.030, 0.007, 0.026), yaw=yaw, pitch=lean + 0.10 * (k - 1.5), tint=tint(0.05))


def trestle(nm, x, y, yaw, s=1.0):
    """A work trestle: two A-frames and a plank.  Gives the west yard a horizontal at
    0.16 m -- the height band the bible says breaks the horizon without cropping."""
    b = band(x, y)
    c, si = math.cos(yaw), math.sin(yaw)
    for k, sgn in ((0, -1), (1, 1)):
        dx = sgn * 0.115 * s
        for m, ss in ((0, -1), (1, 1)):
            g(f"{nm}_l{k}{m}", "wood" + b,
              (x + dx * c, y + dx * si, 0.075 * s), (0.008, 0.008, 0.078 * s),
              yaw=yaw + math.pi / 2, pitch=ss * 0.22, tint=tint())
    g(f"{nm}_top", "wood" + b, (x, y, 0.158 * s), (0.150 * s, 0.038 * s, 0.007 * s),
      yaw=yaw, tint=tint())


def scatter(nm, x, y, kind="sherd"):
    """Near-band ground furniture: nothing over 45 mm, nothing on the walking line."""
    b = band(x, y)
    if kind == "tile":
        g(nm, "terra" + b, (x, y, 0.006), (0.060, 0.042, 0.006),
          yaw=rng.uniform(0, 2 * math.pi), roll=rng.uniform(-0.12, 0.12), tint=tint(0.07))
    elif kind == "pebble":
        for k in range(4):
            a, d = rng.uniform(0, 6.28), rng.uniform(0.02, 0.075)
            sz = rng.uniform(0.008, 0.018)
            g(f"{nm}_{k}", "stone" + b, (x + d * math.cos(a), y + d * math.sin(a), sz * 0.55),
              (sz, sz * rng.uniform(0.6, 0.95), sz * 0.55), yaw=rng.uniform(0, 6.28),
              tint=tint(0.07))
    elif kind == "tool":
        yw = rng.uniform(0, 6.28)
        g(f"{nm}_h", "wood" + b, (x, y, 0.008), (0.085, 0.008, 0.008), yaw=yw, tint=tint())
        g(f"{nm}_b", "iron" + b, (x + 0.082 * math.cos(yw), y + 0.082 * math.sin(yw), 0.010),
          (0.022, 0.026, 0.010), yaw=yw + 0.25, tint=tint(0.04))
    elif kind == "cloth":
        yw = rng.uniform(0, 6.28)
        g(f"{nm}_a", "canvas" + b, (x, y, 0.005), (0.058, 0.044, 0.005), yaw=yw, tint=tint())
        g(f"{nm}_b", "canvas" + b, (x + 0.030 * math.cos(yw + 1.1),
                                    y + 0.030 * math.sin(yw + 1.1), 0.013),
          (0.032, 0.026, 0.009), yaw=yw + 0.8, tint=tint())


# ==================================================================================
def build():
    note("=============================== aaa_props_body.xml ===============================")
    note("PROPS agent.  Set dressing: aisle clusters, west-yard working gear, near-band")
    note("ground scatter, and THE lapis vessel.  Every geom is contype=0 conaffinity=0 --")
    note("the duck may walk through a crate; architecture owns everything it may not.")
    note("Nothing inside r = 0.9 m of the origin.")
    note()

    # ---------------------------------------------------------------- THE LAPIS VESSEL
    note("=================================================================================")
    note("THE ONE SATURATED OBJECT.  Palette entry 10, Lapis Cobalt, and bible section 2")
    note("rule 3: it appears on EXACTLY ONE prop in the whole 16 x 16 m, and scarcity is")
    note("the point.  The shipped scene had zero geoms carrying it, so the direction had no")
    note("chromatic focal point at all -- every accent in it was warm ochre, rose or amber.")
    note("Placed IN SHADE on the north (shadow-wall) aisle, 2.05 m off the corridor centre")
    note("line, because the blue sky fill is what keeps a blue object saturated there; in")
    note("the sun a 2800 K key would drag it grey.  It stands on a bleached stone plinth so")
    note("it is lifted clear of the floor and reads against the dark arcade behind it, and")
    note("it sits on a guaranteed walking route so the duck passes within 0.8 m of it.")
    LX, LY = 2.58, 1.86
    g("lap_plinth", "stone", (LX, LY, 0.028), (0.105, 0.105, 0.028), yaw=0.18)
    g("lap_socle", "stone", (LX, LY, 0.066), (0.082, 0.082, 0.010), yaw=0.18)
    g("lap_foot", "lapis", (LX, LY, 0.089), (0.040, 0.040, 0.013), typ="cylinder", yaw=0.18)
    g("lap_belly", "lapis", (LX, LY, 0.148), (0.072, 0.072, 0.052), typ="ellipsoid", yaw=0.18)
    g("lap_shldr", "lapis", (LX, LY, 0.206), (0.047, 0.047, 0.014), typ="cylinder", yaw=0.18)
    g("lap_neck", "lapis", (LX, LY, 0.238), (0.028, 0.028, 0.018), typ="cylinder", yaw=0.18)
    g("lap_rim", "lapis", (LX, LY, 0.262), (0.040, 0.040, 0.007), yaw=0.18 + 0.39)
    for k, sgn in ((0, -1), (1, 1)):
        g(f"lap_h{k}", "lapis", (LX + sgn * 0.063, LY, 0.198), (0.009, 0.009, 0.034),
          yaw=0.18, pitch=sgn * 0.5)
    note()

    # ---------------------------------------------------------------- AISLE CLUSTERS
    note("=================================================================================")
    note("SIDE-AISLE CLUSTERS.  Bible section 7: one cluster per 1.8 m of aisle run, never")
    note("two of the same silhouette adjacent.  Both aisles run the full 14 m, so that is")
    note("eight clusters a side.  The south (sun) aisle gets the storage and the traffic --")
    note("it is the lit side and the props are read by their cast shadows.  The north")
    note("(shadow) aisle gets the working gear and the brazier, because on that side only")
    note("silhouette and emission survive.")
    # BIG / SMALL alternate.  Bible section 7 wants a cluster every 1.8 m and never two of
    # the same silhouette adjacent; it does not require every cluster to be a pile.  Four
    # full clusters and four single objects per side keeps the rhythm at a third of the
    # geom cost, and alternating mass with a single vertical is better composition anyway.
    KINDS_S = ["crates", "jar", "amphorae", "pots", "sack", "jar", "basket", "pots"]
    KINDS_N = ["rope", "pots", "crates", "jar", "brazier", "pots", "amphorae", "jar"]
    last_yaw = 0.0
    for side, sgn, kinds in ((("s", -1, KINDS_S)), (("n", +1, KINDS_N))):
        for i, kind in enumerate(kinds):
            x = -6.30 + i * 1.80 + rng.uniform(-0.16, 0.16)
            y = sgn * (2.34 + rng.uniform(-0.34, 0.34))
            if math.hypot(x, y) < 1.05:
                x += 0.9
            yaw = rng.uniform(0, 2 * math.pi)
            while abs(((yaw - last_yaw + math.pi) % (2 * math.pi)) - math.pi) < 0.30:
                yaw = rng.uniform(0, 2 * math.pi)
            last_yaw = yaw
            nm = f"cl{side}{i}"
            s = rng.uniform(0.86, 1.18)          # +/-18% scale, per the variation law
            if kind == "crates":
                crate(f"{nm}a", x, y, 0.0, s, yaw)
                crate(f"{nm}b", x + 0.13 * math.cos(yaw + 1.2), y + 0.13 * math.sin(yaw + 1.2),
                      0.0, s * 0.78, yaw + 0.9)
                sherds(f"{nm}d", x - 0.19 * math.cos(yaw), y - 0.19 * math.sin(yaw), 3, 0.07)
            elif kind == "amphorae":
                amphora(f"{nm}a", x, y, 0.0, s, yaw)
                amphora(f"{nm}b", x + 0.11 * math.cos(yaw + 2.1), y + 0.11 * math.sin(yaw + 2.1),
                        0.0, s * 0.84, yaw + 1.7, lean=0.28)
                sherds(f"{nm}d", x + 0.20 * math.cos(yaw - 1.1), y + 0.20 * math.sin(yaw - 1.1),
                       4, 0.09)
            elif kind == "jar":      # SMALL: one jar and the sherds of its neighbour
                amphora(f"{nm}a", x, y, 0.0, s * 1.06, yaw)
                sherds(f"{nm}b", x + 0.19 * math.cos(yaw), y + 0.19 * math.sin(yaw), 3, 0.07)
            elif kind == "pots":     # SMALL: a broken drift and one upright
                sherds(f"{nm}a", x, y, 6, 0.16, s)
                g(f"{nm}b", "terra" + band(x, y),
                  (x + 0.17 * math.cos(yaw), y + 0.17 * math.sin(yaw), 0.030 * s),
                  (0.034 * s, 0.034 * s, 0.030 * s), typ="ellipsoid", yaw=yaw, tint=tint(0.06))
                g(f"{nm}c", "terra" + band(x, y),
                  (x + 0.17 * math.cos(yaw), y + 0.17 * math.sin(yaw), 0.062 * s),
                  (0.024 * s, 0.024 * s, 0.006 * s), yaw=yaw + 0.4, tint=tint(0.06))
            elif kind == "sack":
                sack(f"{nm}a", x, y, s, yaw, spilled=True)
                sack(f"{nm}b", x + 0.13 * math.cos(yaw + 1.9), y + 0.13 * math.sin(yaw + 1.9),
                     s * 0.86, yaw + 2.4)
            elif kind == "basket":
                basket(f"{nm}a", x, y, s, yaw)
                basket(f"{nm}b", x + 0.13 * math.cos(yaw + 2.6), y + 0.13 * math.sin(yaw + 2.6),
                       s * 0.80, yaw + 0.8)
                sherds(f"{nm}c", x - 0.16 * math.cos(yaw), y - 0.16 * math.sin(yaw), 3, 0.06)
            elif kind == "rope":
                rope_coil(f"{nm}a", x, y, s, yaw)
                crate(f"{nm}b", x + 0.18 * math.cos(yaw + 1.5), y + 0.18 * math.sin(yaw + 1.5),
                      0.0, s * 0.9, yaw + 1.1)
            elif kind == "brazier":
                brazier(f"{nm}a", x, y, s, yaw)
                sack(f"{nm}b", x + 0.19 * math.cos(yaw + 2.2), y + 0.19 * math.sin(yaw + 2.2),
                     s * 0.8, yaw + 1.4)
    note()

    # ---------------------------------------------------------------- WEST YARD
    note("=================================================================================")
    note("WEST YARD.  Half the aisle density -- it is the breathing space, and it is where")
    note("the long shadows have room to read.  So: a small number of BIG silhouettes rather")
    note("than many small ones, all beyond 2.5 m, all in the 0.12-0.35 m band that breaks")
    note("the horizon without cropping at a 0.12 m eye.")
    ladder("lad", -6.86, 1.72, yaw=0.06, lean=0.26, L=0.66)
    cart_wheel("whl", -5.35, -1.28, yaw=1.15)
    trestle("trs", -4.05, 1.62, yaw=0.28)
    rope_coil("rc1", -4.62, -1.94, 1.15, 0.7)
    sack("sk1", -3.30, -1.55, 1.10, 2.1, spilled=True)
    broom("brm", -6.72, -0.55, yaw=2.9, lean=0.34)
    crate("wc1", -5.90, 0.92, 0.0, 1.16, 0.42)
    crate("wc2", -5.74, 1.08, 0.0, 0.94, 1.85)
    amphora("wa1", -3.62, 1.98, 0.0, 1.14, 0.9)
    amphora("wa2", -3.48, 2.14, 0.0, 0.96, 2.4, lean=0.22)
    sherds("wsh", -2.55, -1.86, 7, 0.24, 1.1)
    basket("wbk", -6.15, -2.32, 1.05, 1.4)
    note()

    # ---------------------------------------------------------------- GATE APPROACH
    note("=================================================================================")
    note("EAST GATE APPROACH.  Two clusters flanking the light slot, so the vista terminus")
    note("has something in front of it to read the slot's brightness against.")
    amphora("ga1", 6.42, 1.02, 0.0, 1.12, 0.35)
    amphora("ga2", 6.55, 1.18, 0.0, 0.90, 2.05, lean=0.18)
    crate("gc1", 6.48, -1.06, 0.0, 1.10, 1.65)
    crate("gc2", 6.62, -1.22, 0.0, 0.86, 0.35)
    sherds("gsh", 6.30, -0.86, 5, 0.11)
    note()

    # ---------------------------------------------------------------- NEAR BAND
    note("=================================================================================")
    note("THE NEAR BAND.  This is the fix for pass-criterion 1.  Measured on the shipped")
    note("scene: in the hero corridor (x 0.9-7.1, |y| < 0.7) exactly THREE geoms had a top")
    note("below 60 mm and all three were atmosphere's curb strips -- so the bottom quarter")
    note("of every frame, which the bible's own framing maths says is the 15 cm of floor")
    note("between 0.148 and 0.296 m, was bare flagstone in every render.")
    note("Everything below is under 45 mm and sits at |y| in [0.28, 1.10]: beside the")
    note("walking line, never on it, so a duck crossing the corridor end to end passes")
    note("within 0.2 m of something at 1.5-2.5 m intervals without any of it ever filling")
    note("the frame.")
    NEAR = [(1.42, 0.52, "pebble"), (1.88, -0.61, "sherd"), (2.35, 0.86, "tile"),
            (2.90, -0.44, "pebble"), (3.34, 0.63, "tool"), (3.86, -0.92, "sherd"),
            (4.30, 0.41, "pebble"), (4.78, -0.58, "cloth"), (5.24, 0.95, "sherd"),
            (5.70, -0.37, "pebble"), (6.16, 0.66, "tile"), (6.64, -0.83, "sherd"),
            (-1.35, 0.58, "pebble"), (-1.92, -0.72, "sherd"), (-2.48, 0.44, "tile"),
            (-3.70, 0.79, "sherd"), (-4.36, -0.51, "cloth"), (-5.75, -0.88, "sherd")]
    for k, (x, y, kind) in enumerate(NEAR):
        if kind == "sherd":
            sherds(f"nb{k}", x, y, 4, 0.075, 0.9)
        else:
            scatter(f"nb{k}", x, y, kind)
    note()

    # ---------------------------------------------------------------- WALL FURNITURE
    note("=================================================================================")
    note("WALL FURNITURE.  Small iron at the wall bases -- a tethering ring and its staple,")
    note("a wedge under a settled coping, a dropped bucket.  These exist so the walls have")
    note("something at duck height to be read against.")
    for k, (x, sgn) in enumerate(((-5.10, -1), (3.40, -1), (-4.25, +1), (4.95, +1))):
        y = sgn * 3.16
        yw = rng.uniform(-0.25, 0.25)
        g(f"ring{k}_st", "iron" + band(x, y), (x, y, 0.088), (0.014, 0.010, 0.010),
          yaw=yw, tint=tint(0.04))
        g(f"ring{k}_r", "iron" + band(x, y), (x, y - sgn * 0.020, 0.070),
          (0.016, 0.005, 0.016), yaw=yw, pitch=0.3, tint=tint(0.04))
    bkt = [(-2.20, -3.02, 0.6), (5.62, 3.04, 2.3)]
    for k, (x, y, yw) in enumerate(bkt):
        g(f"bkt{k}_b", "iron" + band(x, y), (x, y, 0.032), (0.042, 0.042, 0.032),
          typ="cylinder", yaw=yw, tint=tint(0.04))
        g(f"bkt{k}_r", "iron" + band(x, y), (x, y, 0.066), (0.046, 0.046, 0.004),
          typ="cylinder", yaw=yw, tint=tint(0.04))
        g(f"bkt{k}_h", "iron" + band(x, y), (x, y, 0.084), (0.046, 0.005, 0.020),
          yaw=yw + 0.7, tint=tint(0.04))
    return "\n".join(GE) + "\n"


# ==================================================================================
def assets():
    """props_* materials only.  NO new textures: every one of these points at a
    materials_tex_* map that materials.py already authored and that nothing referenced.

    texuniform="true" -> period = 2/texrepeat metres, so the numbers below are chosen for
    prop-sized geometry, not for walls.  The three aerial-perspective variants per family
    follow bible section 8: full chroma inside 3.4 m, value up / chroma down at 3.4-6.2 m,
    value up further and shifted cool beyond 6.2 m."""
    AP = [("", None), ("_mid", "0.93 0.94 0.97 1"), ("_far", "0.86 0.89 0.97 1")]
    FAM = [
        # name      texture         texrepeat   spec  shin  what it is on
        ("wood", "wood", "22 22", 0.08, 0.06),      # 0.09 m boarding
        ("terra", "terra", "40 40", 0.10, 0.06),    # 0.05 m wheel-throw ridges
        ("canvas", "canvas", "33 33", 0.05, 0.04),  # 0.06 m sacking weave
        ("iron", "iron", "29 29", 0.30, 0.35),      # 0.07 m forged surface
        ("stone", "stone", "26 26", 0.10, 0.06),    # 0.08 m pebbles, plinth
        ("sand", "sand", "24 24", 0.06, 0.05),      # 0.08 m spilled grain
    ]
    out = [
        "<!-- ============================= aaa_props_assets.xml =============================",
        "     PROPS agent.  Materials only == this agent authors NO textures.",
        "     materials.py built a 20-map library (12 MB of PNG) sized for large surfaces and",
        "     the shipped scene referenced NONE of it: every downstream agent authored its",
        "     own maps instead, so the whole library was dead weight decoded to the GPU for",
        "     zero pixels.  This agent is the consumer it was built for.  All that changes",
        "     here is texrepeat: the library's periods were authored for walls, and a 1.40 m",
        "     wood period on a 0.12 m crate shows 9% of one tile, i.e. a flat patch.",
        "     Under texuniform the period is 2/texrepeat METRES (probe-verified), so e.g.",
        "     texrepeat 22 == a 0.09 m board.",
        "     ============================================================================== -->",
    ]
    for base, tex, rep, spec, shin in FAM:
        for sfx, rgba in AP:
            c = f' rgba="{rgba}"' if rgba else ""
            out.append(
                f'<material name="{P}{base}{sfx}" texture="materials_tex_{tex}"'
                f' texuniform="true" texrepeat="{rep}"{c}'
                f' specular="{spec}" shininess="{shin}"/>'
            )
    out.append("")
    out.append("<!-- THE one saturated object.  Glaze crazing at a 0.05 m period.")
    out.append("     NO `reflectance`, and this one is measured, not inherited: the bible's")
    out.append("     material table gives aaa_lapis reflectance 0.05, and shipping it cost")
    out.append("     TEN MILLISECONDS a frame at 320x240 -- 37.5 ms with the nine lapis geoms")
    out.append("     visible, 27.3 ms with them hidden, on a scene where hiding all 394 props")
    out.append("     only saved 13.2.  The classic renderer runs an extra full scene pass per")
    out.append("     reflective geom, so the price scales with the WHOLE scene's geom count,")
    out.append("     not with the mirror's size: 0.05 reflectance on a 0.11 m jar was costing")
    out.append("     more than the entire props tier.  It is also invisible at that level.")
    out.append("     Specular and shininess carry the glaze instead, for free. -->")
    out.append(f'<material name="{P}lapis" texture="materials_tex_lapis" texuniform="true"'
               f' texrepeat="40 40" specular="0.52" shininess="0.62"/>')
    out.append("")
    out.append("<!-- brazier embers.  The only emissive this agent spends: a dull red coal")
    out.append("     bed, well under the amber of atmosphere's lamps so it cannot compete")
    out.append("     with the curb strips for the shadow aisle's one legible incident. -->")
    out.append(f'<material name="{P}ember" rgba="1.00 0.42 0.15 1" emission="0.62"/>')
    return "\n".join(out) + "\n"


def main():
    print("PROPS agent - generating")
    body = build()
    with open(os.path.join(MD, "aaa_props_assets.xml"), "w") as f:
        f.write(assets())
    with open(os.path.join(MD, "aaa_props_body.xml"), "w") as f:
        f.write(body)
    be = sum(BE_COST.get(k, 1.0) * v for k, v in _stats.items())
    print(f"  geoms {len(_names)} / 330   BE {be:.1f} / 460   "
          f"cylinders {_stats.get('cylinder', 0)} / 20")
    print(f"  types {_stats}")
    print(f"  wrote aaa_props_assets.xml, aaa_props_body.xml")


if __name__ == "__main__":
    main()
