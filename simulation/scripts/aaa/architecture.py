#!/usr/bin/env python3
"""
THE WATER COURT  --  architecture agent.

Builds every built structure of scene_aaa.xml: the two court walls, the twelve-column
colonnade, the east gate + light slot, the west ruin, the fountain, corner pylons,
buttresses and daises.

Writes (idempotently) into the model directory:
    architecture_ashlar.png    architecture_plaster.png   architecture_stone.png
    architecture_rose.png      architecture_petrol.png
    aaa_architecture_assets.xml   (raw <texture>/<material> fragment)
    aaa_architecture_body.xml     (raw <geom> fragment)

Design notes that are load-bearing:
  * The sun travels (-0.752,-0.631,-0.191): +X and +Y facing verticals are the blazing
    ones, horizontals get only 19% of the key. So the architecture maximises +X/+Y
    vertical area and puts a hard dark next to it (overhangs, reveals, recesses).
  * Every texture is DIRECTION-NEUTRAL baked AO (joints, spall, per-block tint). No
    directional highlight is ever baked, so nothing can fight the sun. The only
    directional bake is gravity-driven staining, which is vertical and sun-agnostic.
  * Textures are gamma-encoded 1/1.9, matching the reference bible_ground.png, so this
    architecture sits in the same tonal world as the floor.
  * Collision is 16 invisible group="3" proxies (12 columns, 2 walls, 2 gate masses).
    Every visible geom is contype=0 conaffinity=0. The proxies wrap the visible
    envelope so the duck can never clip a plinth.
"""

import math
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

MD = Path(__file__).resolve().parents[2] / "src/mjlab_microduck/robot/microduck"
PFX = "architecture_"
N = 1024

# ----------------------------------------------------------------------------------
# palette (LINEAR, straight from the art bible)
# ----------------------------------------------------------------------------------
OCHRE = np.array([0.620, 0.455, 0.290])  # Ochre Plaster   - dominant architecture
STONE = np.array([0.615, 0.567, 0.480])  # Bleached Stone  - capitals / copings
ROSE = np.array([0.620, 0.388, 0.407])  # Barragan Rose   - the SUN wall
PETROL = np.array([0.130, 0.220, 0.260])  # Deep Petrol     - the SHADOW wall
IRON = np.array([0.300, 0.170, 0.120])  # Iron Oxide      - cramps, rings

# encoded-space cap.  0.815*255 = 208; measured to render at ~244 on a sunlit face,
# i.e. hard against the bible's "no channel clipping" rule with a little headroom.
ENC_CAP_R = 0.815
CHROMA = 1.22  # chroma restored after the 1/1.9 encode

rng = np.random.default_rng(20260829)


# ----------------------------------------------------------------------------------
# texture plumbing
# ----------------------------------------------------------------------------------
def fbm(n, octaves, lac=2.0, gain=0.5, seed=0):
    """Toroidally seamless fractal noise, unit std (bible tex.py method)."""
    r = np.random.default_rng(seed)
    out = np.zeros((n, n))
    amp, freq = 1.0, 4
    for _ in range(octaves):
        g = r.normal(size=(freq, freq))
        G = np.fft.fft2(g)
        Gp = np.zeros((n, n), complex)
        h = freq // 2
        Gp[:h, :h] = G[:h, :h]
        Gp[:h, -h:] = G[:h, -h:]
        Gp[-h:, :h] = G[-h:, :h]
        Gp[-h:, -h:] = G[-h:, -h:]
        layer = np.real(np.fft.ifft2(Gp))
        layer /= layer.std() + 1e-9
        out += amp * layer
        amp *= gain
        freq = int(freq * lac)
        if freq > n:
            break
    return out / (out.std() + 1e-9)


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def encode(img_lin, cap_r=True, linear=False):
    """Linear RGB -> 8-bit PNG.

    The default path applies the reference bible_ground.png convention (a 1/1.9
    encode, then chroma restored).  Measured: it puts ochre at #E79250 on a sunlit
    face against the bible's #FE8E43 target, where an un-encoded texture renders a
    dead #946B57.

    `linear=True` is the deliberate exception for DARK materials.  The encode is a
    power law, so it lifts 0.620 by only 1.25x but lifts 0.130 by 2.65x -- applying
    it to Deep Petrol turned the shadow wall from the bible's near-black #35443C into
    sage-green bars wherever a reveal caught the sun, because a warm 2800 K key on a
    lifted teal lands in olive.  The shadow wall has to stay near-black for the whole
    direction to work, so it is authored at its literal palette value, exactly as the
    reference calibration did for this one colour.
    """
    if linear:
        return (np.clip(img_lin, 0, 1) * 255).astype(np.uint8)
    enc = np.clip(img_lin, 0, 1) ** (1 / 1.9)
    g = enc.mean(axis=2, keepdims=True)
    enc = np.clip(g + (enc - g) * CHROMA, 0, 1)
    if cap_r and enc[..., 0].max() > ENC_CAP_R:
        enc *= ENC_CAP_R / enc[..., 0].max()
    return (np.clip(enc, 0, 1) * 255).astype(np.uint8)


def save(name, arr):
    Image.fromarray(arr).save(os.path.join(MD, name))
    lum = arr.astype(float).mean(2)
    kb = os.path.getsize(os.path.join(MD, name)) / 1024.0
    print(
        f"  {name:30s} mean={lum.mean():6.1f} std={lum.std():5.1f} "
        f"maxR={arr[..., 0].max():3d} {kb:7.1f} kB"
    )


# ----------------------------------------------------------------------------------
# TEXTURE 1 -- coursed ashlar.  The workhorse: column shafts, gate, ruin, pylons.
# Tile = `blocks` wide x `courses` tall, running bond.  Pure direction-neutral AO.
# ----------------------------------------------------------------------------------
def ashlar(name, base, courses=6, blocks=3, seed=1, joint_px=7.0, joint_dark=0.40,
           tint=0.075, grain=0.030, spall_amt=0.16, streak=0.030):
    ys, xs = np.mgrid[0:N, 0:N].astype(np.float64)
    ch, bw = N / courses, N / blocks
    row = np.floor(ys / ch).astype(int) % courses
    off = (row % 2) * 0.5 * bw
    xo = (xs + off) % N
    col = np.floor(xo / bw).astype(int) % blocks
    bid = row * blocks + col

    # joints wander a little so they are not ruler-straight
    wob = fbm(N, 4, seed=seed + 5) * 2.6
    dy = np.minimum(ys % ch, ch - (ys % ch)) + wob
    dx = np.minimum(xo % bw, bw - (xo % bw)) + wob * 0.7
    d = np.minimum(dy, dx)

    j = smooth(d / joint_px)
    v = joint_dark + (1 - joint_dark) * j  # baked AO in the mortar bed

    # per-block value + a hint of per-block hue drift
    t = rng.normal(0, 1, courses * blocks) * tint
    v = v * (1.0 + t[bid])
    hue = rng.normal(0, 1, courses * blocks)[bid] * 0.035

    # spalled arrises: blocks lose material at their edges, exposing a paler core
    sp = fbm(N, 5, seed=seed + 31)
    edge = 1.0 - smooth(d / (joint_px * 4.5))
    spall = np.clip((sp - 0.55) * 2.4, 0, 1) * edge
    v = v * (1 - spall_amt * spall) + spall * 0.085

    # gravity staining: broad soft runnels below the bed joints (vertical, sun-agnostic)
    runnel = np.clip(fbm(N, 3, seed=seed + 77), -1, 1)
    band = np.clip((ys % ch) / ch, 0, 1)
    v -= streak * np.clip(runnel, 0, 1) * (1 - band) * 1.4

    v += fbm(N, 6, seed=seed + 3) * grain
    v = np.clip(v, 0.12, 1.25)

    img = base[None, None, :] * v[..., None]
    img[..., 2] *= 1.0 + hue  # blue drifts per block -> subtle warm/cool masonry mix
    img[..., 0] *= 1.0 - hue * 0.35
    save(name, encode(img))


# ----------------------------------------------------------------------------------
# TEXTURE 2 -- plaster / lime render.  LOW FREQUENCY ONLY (bible texture law).
# ----------------------------------------------------------------------------------
def plaster(name, base, seed=21, amp=0.060):
    low = fbm(N, 3, seed=seed) * amp
    mid = fbm(N, 5, seed=seed + 1) * amp * 0.40
    v = 1.0 + low + mid
    # broad trowel sweeps; horizontal-ish, very low contrast
    yy, xx = np.mgrid[0:N, 0:N] / N
    v += np.sin(xx * 6.283 * 1.5 + fbm(N, 3, seed=seed + 7) * 1.5) * 0.016
    # patches where the render has fallen away to the coarse brown coat beneath
    patch = np.clip((fbm(N, 4, seed=seed + 40) - 0.95) * 2.2, 0, 1)
    v = v * (1 - 0.20 * patch)
    img = base[None, None, :] * np.clip(v, 0.5, 1.20)[..., None]
    img[..., 2] *= 1 - 0.10 * patch  # exposed coat is warmer
    save(name, encode(img))


# ----------------------------------------------------------------------------------
# TEXTURE 3 -- bleached limestone for capitals, plinths, copings, fountain.
# ----------------------------------------------------------------------------------
def limestone(name, base, seed=41):
    v = 1.0 + fbm(N, 3, seed=seed) * 0.045 + fbm(N, 6, seed=seed + 2) * 0.030
    # faint sedimentary bedding, horizontal, very low contrast
    yy = np.mgrid[0:N, 0:N][0] / N
    v += np.sin(yy * 6.283 * 5 + fbm(N, 2, seed=seed + 9) * 2.0) * 0.012
    # pitting / chipped specks
    pit = np.clip((fbm(N, 7, seed=seed + 17) - 1.05) * 2.6, 0, 1)
    v = v * (1 - 0.26 * pit)
    # a scatter of hairline cracks
    cr = np.abs(fbm(N, 4, seed=seed + 55))
    v = v * (1 - 0.20 * np.clip(1 - cr * 9.0, 0, 1))
    img = base[None, None, :] * np.clip(v, 0.45, 1.12)[..., None]
    save(name, encode(img))


# ----------------------------------------------------------------------------------
# TEXTURE 4 -- Barragan rose lime wash on the SUN wall.
# Colour-field flat by design: this wall is meant to blaze as one plane.  Its only
# incident is broad weathering and a faint ghost of the coursing beneath the render.
# ----------------------------------------------------------------------------------
def rosewash(name, base, seed=61):
    """THE SUN WALL.  Rebuilt after the critique.

    The first cut was large soft mauve amoeba blotches ("patchy re-washing") plus a
    hard-thresholded dark band along the base.  Two failures:

      * the blotches read as camouflage, not as a wall.  Bible section 4 asks this
        material for "a vertical grime gradient rising from the ground" and it got
        a horizontal continent map instead.
      * the base band was a HARD threshold stretched across a 14 m geom with no
        mipmapping, so its top edge stair-stepped visibly along the whole run and
        read as JPEG blocking.

    What is here now is the specified thing: clean rose at the top, progressively
    dirtier and darker over the bottom 30%, with soft-edged rain streaking running
    vertically.  Vertical detail is normally banned on plaster (bible section 4.1)
    because a vertical bake fights the sun's XY bearing -- but rain streaking is
    GRAVITY-driven, not lighting-driven, so it implies no light direction at all and
    is the one exception the rule allows.  The grime band's edge is a 26%-of-height
    smoothstep, i.e. ~66 px of falloff at N=256, so it cannot alias.

    PNG row 0 is the TOP of a box face (verified), so `t` below runs 0 at the top."""
    t = np.mgrid[0:N, 0:N][0].astype(float) / N          # 0 top .. 1 bottom
    xs = np.mgrid[0:N, 0:N][1].astype(float) / N

    v = 1.0 + fbm(N, 3, seed=seed) * 0.045 + fbm(N, 5, seed=seed + 1) * 0.022

    # ghost coursing: the ashlar under the wash, just showing through
    ch = N / 7.0
    ys = np.mgrid[0:N, 0:N][0].astype(float)
    dy = np.minimum(ys % ch, ch - (ys % ch)) + fbm(N, 4, seed=seed + 5) * 3.5
    v -= 0.055 * (1 - smooth(dy / 11.0))

    # --- THE GRIME GRADIENT.  Rises from the ground over the bottom 30%, its top edge
    #     broken by low-frequency noise and softened over a quarter of the height.
    edge = 0.70 + fbm(N, 3, seed=seed + 41) * 0.055 + fbm(N, 5, seed=seed + 42) * 0.022
    grime = smooth((t - edge) / 0.26)
    grime = grime * (0.72 + 0.28 * np.clip(fbm(N, 4, seed=seed + 43) * 0.5 + 0.6, 0, 1))
    # splash-back at the very foot: harder, dirtier, but still soft-edged
    grime = np.clip(grime + smooth((t - 0.925) / 0.075) * 0.34, 0, 1)

    # --- RAIN STREAKING.  Long vertical runs, strongest under the coping and fading
    #     downward; a few carry a bright leached edge where the wash has been rinsed.
    col_noise = fbm(N, 6, seed=seed + 51)[:1].repeat(N, 0)      # constant down a column
    col_fine = fbm(N, 7, seed=seed + 52)[:1].repeat(N, 0)
    runs = np.clip(np.abs(col_noise) * 0.9 + np.abs(col_fine) * 0.45 - 0.55, 0, 1)
    # streaks start under the coping and die out; they wander a little
    wander = fbm(N, 4, seed=seed + 53) * 0.35
    reach = np.clip((0.05 + 0.85 * np.abs(col_noise) + wander) - t * 0.55, 0, 1)
    streak = runs * smooth(reach / 0.35) * smooth((t - 0.02) / 0.10)
    v = v * (1 - 0.22 * streak)
    leach = np.clip(np.abs(col_fine) - 0.95, 0, 1) * smooth(reach / 0.4)
    v = v + 0.095 * leach

    # a handful of long shallow scars where render has been knocked off
    # Scars, halved.  At 0.16 on a 14 m plane they read as soft mauve camouflage
    # continents rather than as knocked-off render -- the exact failure the critique
    # named -- so they are cut to 0.07 and pushed to a higher threshold, where only a
    # handful of small patches survive.
    scar = np.clip((np.abs(fbm(N, 4, seed=seed + 33)) - 1.62) * 4.0, 0, 1)
    v = v * (1 - 0.07 * scar)
    v = v + fbm(N, 6, seed=seed + 3) * 0.020

    img = base[None, None, :] * np.clip(v, 0.55, 1.10)[..., None]
    # the grime is a cool dirty violet-brown: it must darken AND desaturate, which is
    # what a wall that has been rained on for a century actually does.
    dirt = np.array([0.300, 0.242, 0.258])
    img = img * (1 - 0.72 * grime[..., None]) + dirt[None, None, :] * (0.72 * grime[..., None])
    img[..., 2] *= 1 + 0.10 * scar          # the exposed coat beneath is cooler
    save(name, encode(img))


# ----------------------------------------------------------------------------------
# TEXTURE 5 -- deep petrol on the SHADOW wall, with salt efflorescence.
# This wall reads near-black, so its whole job is to hold ONE readable incident:
# the pale bloom creeping up from the base.
# ----------------------------------------------------------------------------------
def petrolwall(name, base, seed=81):
    """The shadow wall reads near-black, so its whole job is to hold ONE readable
    incident: salt efflorescence blooming out of the base.  V is flipped (PNG row 0
    is the TOP of a box face, verified), so the bloom is authored at the bottom rows.
    """
    v = 1.0 + fbm(N, 3, seed=seed) * 0.10 + fbm(N, 6, seed=seed + 1) * 0.045
    ys = np.mgrid[0:N, 0:N][0] / N  # 0 at the top of the face, 1 at the bottom
    # bloom creeps UP from the bottom, its edge broken by noise
    h = fbm(N, 3, seed=seed + 13) * 0.14 + fbm(N, 5, seed=seed + 17) * 0.06
    bloom = smooth((ys - (0.70 + h)) / 0.20) * 0.62
    # ...and is itself blotchy, not a wash
    bloom *= np.clip(fbm(N, 4, seed=seed + 23) * 0.55 + 0.62, 0, 1)
    img = base[None, None, :] * np.clip(v, 0.5, 1.30)[..., None]
    # The bloom must move the colour along VALUE, never hue.  Measured the hard way:
    # a warm-neutral salt over a blue-green base lands the midtones in olive (#566250),
    # and a cool-neutral one lands them in mint -- which is reserved for the robot.
    # So the salt is the petrol hue itself, lifted and desaturated: B > G > R always.
    salt = np.array([0.300, 0.360, 0.430])
    img = img * (1 - bloom[..., None]) + salt[None, None, :] * bloom[..., None]
    img *= (1 + fbm(N, 6, seed=seed + 5) * 0.05)[..., None]
    arr = encode(img, linear=True)
    # Guard the bible's hard rule: mint-aqua (the robot's 0.537 0.855 0.827, ~#89DAD3
    # on screen) is reserved for the duck.  Deep Petrol is legitimately a dark teal
    # (G > R by palette), so the thing to forbid is a BRIGHT green-dominant pixel --
    # which is exactly what a neutral-grey efflorescence over teal produced.
    f = arr.reshape(-1, 3).astype(float)
    green = (f[:, 1] > f[:, 2] + 2).mean()
    assert green < 0.005, (f"petrol texture is {green:.1%} green-dominant; it must stay "
                           f"B > G > R end to end or the bloom reads olive or mint")
    save(name, arr)


# ==================================================================================
#                                  G E O M E T R Y
# ==================================================================================
GE = []  # visible geoms
CO = []  # invisible collision proxies
_names = set()
_stats = {"box": 0, "cylinder": 0}
BE = {"box": 1.0, "cylinder": 5.3, "sphere": 2.8, "ellipsoid": 2.7, "capsule": 6.1}


def _q(yaw):
    return f'quat="{math.cos(yaw / 2):.6f} 0 0 {math.sin(yaw / 2):.6f}"'


def g(name, mat, pos, size, typ="box", yaw=0.0, rgba=None, zaxis=None):
    """Emit one visible, non-colliding decorative geom.

    `yaw` turns about Z, which is all a standing box or column needs.  `zaxis` points
    the geom's local +Z somewhere else and is what actually lays a cylinder DOWN --
    a fallen column drum spun about Z is still standing up, just facing elsewhere.
    """
    nm = PFX + name
    assert nm not in _names, f"duplicate geom name {nm}"
    _names.add(nm)
    assert min(size) > 1e-4, f"{nm}: non-positive half-size {size}"
    _stats[typ] = _stats.get(typ, 0) + 1
    if zaxis is not None:
        q = f' zaxis="{zaxis[0]:.4f} {zaxis[1]:.4f} {zaxis[2]:.4f}"'
    else:
        q = "" if abs(yaw) < 1e-9 else " " + _q(yaw)
    c = "" if rgba is None else f' rgba="{rgba}"'
    GE.append(
        f'<geom name="{nm}" type="{typ}" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"'
        f' size="{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}"{q} material="{PFX}{mat}"{c}'
        f' contype="0" conaffinity="0"/>'
    )


def gb(name, mat, x0, x1, y0, y1, z0, z1, **kw):
    """Same, from world-space bounds -- keeps every façade dimension readable."""
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    z0, z1 = sorted((z0, z1))
    g(name, mat, ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
      ((x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2), **kw)


def collider(name, pos, size, yaw=0.0):
    """Invisible group=3 collision proxy (the robot's own convention for collision geoms).

    Wraps the full visible envelope below duck height so the duck can never clip a
    plinth or a corbel.  These are the ONLY collidable geoms this agent contributes.
    """
    nm = PFX + name
    assert nm not in _names
    _names.add(nm)
    q = "" if abs(yaw) < 1e-9 else " " + _q(yaw)
    CO.append(
        f'<geom name="{nm}" type="box" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"'
        f' size="{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}"{q}'
        f' material="{PFX}ashlar" group="3"/>'
    )


def band(r, near="", mid="_mid", far="_far"):
    """Authored aerial perspective: material suffix by radial distance from spawn."""
    return near if r < 3.0 else (mid if r < 5.6 else far)


# ----------------------------------------------------------------------------------
# 1.  THE COLONNADE  --  12 square piers, y = +/-1.30, x = 0.90 + i*1.16
# ----------------------------------------------------------------------------------
# Section from the ground up, every step a real change of plane:
#   0.000-0.032  plinth, projecting 46 mm  -> a lit ledge and a shadow at its foot
#   0.032-0.062  plinth setback
#   0.062-0.362  lower shaft            (COLLIDABLE envelope lives here)
#   0.362-0.626  upper shaft, entasis: 7.5 mm narrower -> the shaft visibly tapers
#   0.626-0.645  necking ring
#   0.640-0.690  abacus, oversailing 50 mm -> a dark soffit under a blazing edge
def colonnade():
    # per-column character.  h scales everything above the plinth, w the shaft width,
    # cap: 2 = plinth+necking+echinus+abacus (the near heroes), 1 = plain abacus,
    # 0 = broken off.  No two columns share a height, and the skyline steps.
    CHAR = {
        (0, -1): (1.000, 1.000, 2), (0, +1): (0.965, 1.030, 2),
        (1, -1): (1.045, 0.975, 2), (1, +1): (0.985, 1.010, 1),
        (2, -1): (0.955, 1.020, 1), (2, +1): (1.030, 0.985, 2),
        (3, -1): (1.010, 0.995, 1), (3, +1): (0.940, 1.015, 1),
        (4, -1): (0.995, 1.005, 1), (4, +1): (0.62, 0.990, 0),
        (5, -1): (1.020, 0.980, 1), (5, +1): (0.44, 1.020, 0),
    }
    for i in range(6):
        for s in (-1, 1):
            k = f"{i}{'p' if s > 0 else 'm'}"
            hs, ws, cap = CHAR[(i, s)]
            x = 0.90 + i * 1.16 + rng.uniform(-0.02, 0.02)
            y = s * (1.30 + rng.uniform(-0.012, 0.012))
            yaw = rng.uniform(-0.030, 0.030)
            b = band(math.hypot(x, y))
            # MATERIAL LAW (bible section 4, and the critique's central palette note).
            # The shafts were built in `ashlar` -- a regular horizontal brick course at
            # identical scale and identical phase on all twelve -- which violates the
            # plaster law head-on ("low-frequency mottling ONLY.  No vertical high-
            # frequency detail, ever"), pulled the colonnade toward a stock medieval kit,
            # and put the highest-frequency thing in the scene in the mid-band where
            # there is no mipmapping to save it.  Worse, it left Ochre Plaster -- a named
            # DOMINANT at ~35% of pixels -- referenced by ZERO geoms while Bleached Stone,
            # a secondary, became the most-used architectural material, so the court read
            # cool grey-lilac instead of warm ochre and the duck lost the temperature
            # separation the whole direction was built to give it.
            # Shafts are now plaster; stone is reserved for plinths, capitals and copings.
            pls = "plaster_col" + b
            sto = "stone" + b
            # settlement: the far columns lean a hair, so the colonnade is not a ruler
            lean = rng.uniform(-0.006, 0.006) * (i / 5.0)
            W = 0.1060 * ws

            # SCALE.  The critique measured the court reading at 1.5-2x human scale
            # rather than the 10x the direction is built on: kerbs at the duck's chest,
            # capitals reading as ordinary porch columns.  The fix is not a uniform
            # scale-up (that just re-crops) but pushing the elements that read AGAINST
            # the duck in the 0.4-1.5 m ring.  The plinth+socle now top out at 0.105 m --
            # the duck's knee-to-hip -- instead of 0.062, so a walking duck passes a
            # two-step base taller than its own legs, and the capital oversails further.
            # 0.000-0.062 plinth (projects 62 mm: a lit ledge, and a shadow at its foot)
            g(f"col{k}_plinth", sto, (x, y, 0.031), (0.168 * ws, 0.168 * ws, 0.031), yaw=yaw)
            # 0.062-0.105 socle setback
            g(f"col{k}_socle", sto, (x, y, 0.0835), (0.140 * ws, 0.140 * ws, 0.0215), yaw=yaw)
            # 0.105-... lower shaft
            zlo = 0.105 + 0.300 * hs
            g(f"col{k}_shaftlo", pls, (x, y, (0.105 + zlo) / 2), (W, W, (zlo - 0.105) / 2),
              yaw=yaw)

            if cap == 0:
                # broken off centuries ago: a jagged stump with fallen blocks on top
                zh = zlo + 0.075 * hs
                g(f"col{k}_shafthi", pls, (x + lean, y, (zlo + zh) / 2),
                  (W * 0.93, W * 0.93, (zh - zlo) / 2), yaw=yaw)
                g(f"col{k}_break1", pls, (x + lean - 0.030, y + 0.018, zh + 0.018),
                  (W * 0.60, W * 0.68, 0.018), yaw=yaw + 0.13)
                g(f"col{k}_break2", pls, (x + lean + 0.036, y - 0.024, zh + 0.010),
                  (W * 0.46, W * 0.50, 0.010), yaw=yaw - 0.21)
            else:
                slip = 0.012 if (i == 3 and s < 0) else 0.0  # one drum has slipped
                zh = zlo + 0.260 * hs
                g(f"col{k}_shafthi", pls, (x + lean + slip, y, (zlo + zh) / 2),
                  (W * 0.93, W * 0.93, (zh - zlo) / 2), yaw=yaw + slip * 6)
                g(f"col{k}_neck", sto, (x + lean, y, zh + 0.009),
                  (W * 1.10, W * 1.10, 0.009), yaw=yaw)
                zc = zh + 0.018
                if cap == 2:
                    # two-step capital: the near columns earn the extra element
                    g(f"col{k}_ech", sto, (x + lean, y, zc + 0.013),
                      (W * 1.36, W * 1.36, 0.013), yaw=yaw)
                    zc += 0.022
                cy = yaw + (0.10 if (i == 3 and s < 0) else 0.0)
                cx = x + lean + (0.014 if (i == 3 and s < 0) else 0.0)
                # The abacus oversails the shaft by ~50 mm.  At an 11-degree sun that
                # is worth only ~12 mm of cast shadow, so the incident that actually
                # reads is its SOFFIT: a downward-facing plane lit by nothing but the
                # warm bounce, sitting directly under a blazing vertical front edge.
                g(f"col{k}_abacus", sto, (cx, y, zc + 0.017),
                  (W * 1.66, W * 1.66, 0.017), yaw=cy)

            # iron repair cramps -- the accent that says "maintained, then abandoned"
            if (i, s) in ((1, 1), (2, -1), (5, -1)):
                g(f"col{k}_cramp", "iron", (x + lean, y, zlo - 0.004),
                  (W * 1.035, W * 1.035, 0.0062), yaw=yaw)

            # ONE collision proxy per column, wrapping the plinth footprint so the
            # duck stops on the ledge instead of clipping through it.
            collider(f"col{k}_C", (x, y, 0.370), (0.168 * ws, 0.168 * ws, 0.370), yaw=yaw)


# ----------------------------------------------------------------------------------
# 2.  THE TWO COURT WALLS
# ----------------------------------------------------------------------------------
# Neither wall is a box.  Section (south / SUN wall, court side is +Y):
#   y -3.500..-3.230  plinth course, stone, projecting 40 mm into the court
#   y -3.440..-3.350  back slab, in the shade material -> baked AO for the recesses
#   y -3.350..-3.270  pier, rose            <- 22 of them, 0.41 m blind arcade between
#   y -3.485..-3.245  coping, overhanging the piers 25 mm -> a 0.13 m shadow band
# The recesses are 80 mm deep REAL recesses, not painted panels.
def sun_wall(tag="sw", sgn=-1, top=0.500, seed=771):
    """THE SUN WALL, rebuilt as A PLANE.

    The critique's most load-bearing structural note: Barragan Rose and Deep Petrol are
    "permitted only as the two named large wall planes" (bible section 2 rule 3), and the
    one-frame test asks for "a single blazing plane of #FE795D" on the right.  What was
    built instead was TWO IDENTICAL 36-pier arcades -- 91 geoms each, the same eleven
    element types, mirrored -- which (a) reduced the accent to 89 mm pier slivers and a
    grey-tinted back panel so neither colour was ever delivered, (b) read as blockout by
    section 6 rule 5's own test, and (c) drove criterion 2 (pixels under luminance 25) with
    36 deep unlit recesses.

    So the two walls are now built DIFFERENTLY, which is the point:
        sun wall    -> a continuous full-chroma rose plane, 14.2 m long and unbroken by
                       any pier over its whole run, with a two-step podium at its foot, a
                       string course, a frieze and a broken coping.  Five pilasters stand
                       at the ends and corners only, clear of the hero vista's right half.
        shadow wall -> the blind arcade (court_wall below), which is where deep recesses
                       and piers belong: on the dark side, where they read as depth.

    The podium is also the scale device.  The critique measured the court reading at 1.5-2x
    human rather than 10x; a two-step base 130 mm high at the wall foot is a module derived
    from a human stair, and the duck standing against it is visibly dwarfed by something
    whose purpose it can read."""
    r = np.random.default_rng(seed)
    Y = sgn * 3.35
    x0, x1 = -7.10, 7.10
    L = x1 - x0
    Z_STEP, Z_PLINTH, Z_FIELD = 0.075, 0.130, 0.400
    Z_STR, Z_FRZ, Z_COP = Z_FIELD, Z_FIELD + 0.032, top - 0.032

    def yy(d0, d1):
        a, b = Y + sgn * d0, Y + sgn * d1
        return min(a, b), max(a, b)

    # -- two-step podium.  Six segments, heights jittered, one settled.
    for k in range(6):
        seg = L / 6.0
        a = x0 + k * seg
        h = Z_STEP + r.uniform(-0.004, 0.005)
        z0 = -0.004 if k == 4 else 0.0
        gb(f"{tag}_step{k}", "stone_xl" + ("_mid" if k in (0, 5) else ""),
           a + 0.004, a + seg - 0.004, *yy(-0.185, 0.150), z0, z0 + h)
    for k in range(5):
        seg = L / 5.0
        a = x0 + k * seg
        gb(f"{tag}_plinth{k}", "stone_xl" + ("_mid" if k in (0, 4) else ""),
           a + 0.005, a + seg - 0.005, *yy(-0.130, 0.150), Z_STEP,
           Z_PLINTH + r.uniform(-0.004, 0.004))

    # -- THE PLANE.  Eight segments of one continuous face, all in base `rose` at full
    #    chroma -- NOT the _mid/_far aerial-perspective variants, which is what stopped
    #    the accent ever blazing.  Aerial perspective is delivered on the wall's dressings
    #    instead, where losing chroma costs nothing.
    NF = 8
    for k in range(NF):
        seg = L / NF
        a = x0 + k * seg
        gb(f"{tag}_face{k}", "rose_plane", a, a + seg, *yy(-0.090, 0.150),
           Z_PLINTH, Z_FIELD + r.uniform(-0.003, 0.003))

    # -- a shallow blind panel in three places only: enough to say the plane is built,
    #    not enough to break it.  36 mm deep, so they stay bright.
    for k, (a, b) in enumerate(((-6.30, -5.10), (-2.35, -1.15), (5.55, 6.75))):
        gb(f"{tag}_panel{k}", "rose_dark", a, b, *yy(-0.054, 0.150), Z_PLINTH + 0.030,
           Z_FIELD - 0.030)

    # -- pilasters: ends, corners and the two buttress lines only.  None of them stands
    #    between x = 0.6 and x = 5.2, which is the run the hero vista sees.
    for k, px in enumerate((-6.94, -4.55, -2.90, 6.30, 6.94)):
        gb(f"{tag}_pil{k}", "rose_b", px - 0.115, px + 0.115, *yy(-0.150, -0.060),
           0.0, Z_STR + 0.014)
        gb(f"{tag}_pilcap{k}", "stone" + ("_mid" if abs(px) > 6 else ""),
           px - 0.140, px + 0.140, *yy(-0.170, -0.048), Z_STR + 0.014, Z_STR + 0.046)

    # -- string course oversailing the plane, then the frieze set back above it
    for k in range(6):
        seg = L / 6.0
        a = x0 + k * seg
        gb(f"{tag}_string{k}", "stone_xl" + ("_mid" if k in (0, 5) else ""),
           a + 0.004, a + seg - 0.004, *yy(-0.115, 0.150), Z_STR, Z_STR + 0.032)
    for k in range(5):
        seg = L / 5.0
        a = x0 + k * seg
        gb(f"{tag}_frieze{k}", "rose_band", a, a + seg, *yy(-0.052, 0.150), Z_FRZ, Z_COP)

    # -- iron tie rings driven into the plane: the only incident on it
    for k, rx in enumerate((-3.85, 0.95, 4.40)):
        gb(f"{tag}_ring{k}", "iron", rx - 0.017, rx + 0.017, *yy(-0.104, -0.086),
           0.215, 0.252)

    # -- coping: 9 segments, two fallen, one slipped.  The break in the top line is what
    #    stops a plane reading as an extrusion.
    cseg = L / 9.0
    for k in range(9):
        if k in (2, 6):
            continue
        a = x0 + k * cseg
        # +/-11 mm, not +/-4 mm.  The coping's top edge is a single 14.2 m straight line
        # lying nearly along the light bearing, which is the worst case for a shadow-map
        # staircase: at a 2.06 mm texel the stair runs for tens of texels before it steps.
        # Jitter larger than the texel breaks the run into nine independent edges, so what
        # is left reads as a rough coping instead of a broken shadow map.
        h1 = top + r.uniform(-0.011, 0.011)
        if k == 4:
            g(f"{tag}_cope{k}", "stone_lg",
              (a + cseg / 2, Y + sgn * 0.007, (Z_COP + h1) / 2 + 0.006),
              (cseg / 2 - 0.006, 0.145, (h1 - Z_COP) / 2), yaw=0.055 * sgn)
        else:
            g(f"{tag}_cope{k}", "stone_lg" + ("_mid" if k in (0, 8) else ""),
              (a + cseg / 2, (yy(-0.135, 0.155)[0] + yy(-0.135, 0.155)[1]) / 2,
               (Z_COP + h1) / 2),
              (cseg / 2 - 0.005, 0.145, (h1 - Z_COP) / 2),
              yaw=r.uniform(-0.016, 0.016))

    a, b = yy(-0.185, 0.150)
    collider(f"{tag}_C", (0.0, (a + b) / 2, top / 2), (7.12, (b - a) / 2, top / 2))


def court_wall(tag, sgn, top, mat, shade, n_piers, seed, dress=None):
    """A blind arcade, not a box.

    Section, as offsets `d` from the wall centreline y = sgn*3.35 (d < 0 is toward
    the court, so the court-facing planes are the negative ones):

        d = -0.150  pilaster face         z 0.000..string      (every 4th bay)
        d = -0.125  coping nose           z 0.470..top
        d = -0.120  plinth nose           z 0.000..0.050
        d = -0.105  string-course nose    z 0.420..0.455
        d = -0.095  dado face             z 0.050..0.115
        d = -0.090  PIER OUTER face   ]
        d = -0.020  pier inner face   ]-- a 150 mm two-step recess, z 0.115..0.420
        d = +0.060  RECESS BACK       ]
        d = +0.150  back of wall

    The depth cue here is VERTICAL, and that is a measured decision, not a stylistic
    one.  With the sun at 11 degrees elevation the key arrives almost horizontally,
    so a horizontal overhang casts only ~8 mm of shadow down the face beneath it --
    the light simply goes under.  What a vertical edge does is very different: to
    reach 150 mm into a recess the ray must also travel 150 x 0.752/0.631 = 179 mm
    sideways, so every pier throws a 179 mm shadow bar across the recess behind it.
    Hence: pilasters and deep reveals do the work; cornices are kept for their dark
    downward-facing soffit and their blazing front edge, not for cast shadow.
    """
    r = np.random.default_rng(seed)
    Y = sgn * 3.35
    x0, x1 = -7.10, 7.10
    L = x1 - x0
    Z_DADO, Z_ARC, Z_STR, Z_FRZ, Z_COP = 0.050, 0.115, 0.420, 0.455, 0.470
    Z_STR = 0.420 * (top / 0.500)  # the shadow wall is taller: scale the arcade with it
    Z_FRZ, Z_COP = Z_STR + 0.035, top - 0.030

    def yy(d0, d1):
        a, b = Y + sgn * d0, Y + sgn * d1
        return min(a, b), max(a, b)

    def far_of(x):  # aerial perspective, per element, by true distance from spawn
        rr = math.hypot(x, Y)
        return "" if rr < 4.4 else ("_mid" if rr < 6.2 else "_far")

    # -- podium: the same two-step scale module as the sun wall, so the duck reads the
    #    same human-derived stair height on both sides of the court.
    for k in range(6):
        seg = L / 6.0
        a = x0 + k * seg
        h = 0.072 + r.uniform(-0.004, 0.005)
        z0 = -0.004 if k == 2 else 0.0
        gb(f"{tag}_step{k}", "stone_xl" + ("_far" if k in (0, 5) else "_mid"),
           a + 0.004, a + seg - 0.004, *yy(-0.185, 0.150), z0, z0 + h)
    for k in range(5):
        seg = L / 5.0
        h = 0.125 + r.uniform(-0.004, 0.005)
        gb(f"{tag}_plinth{k}", "stone_xl" + ("_far" if k in (0, 4) else "_mid"),
           x0 + k * seg + 0.004, x0 + (k + 1) * seg - 0.004, *yy(-0.120, 0.150), 0.072, h)

    # -- back slab: the recess backs, in the shade material.  This is the baked AO
    #    that normal maps would have given us, and it is why the recesses go dark.
    #    LIFTED OUT OF TRUE BLACK.  Measured: petrol albedo (0.130 0.220 0.260) times the
    #    old shade rgba (0.58 0.62 0.78) times the shadow-side illumination (0.455 0.440
    #    0.608) delivers luminance 15/255 == a punched hole, not a void.  The critique
    #    found the consequence that matters: the duck's head is a near-black mass and it
    #    vanishes into these recesses on a guaranteed walking route.  The recess back is
    #    now the STONE texture under a cool tint, which lands near 30/255 with strong
    #    blue-violet chroma, so the opening reads as depth and the head separates.
    for k in range(4):
        s2 = L / 4.0
        gb(f"{tag}_back{k}", shade, x0 + k * s2, x0 + (k + 1) * s2,
           *yy(0.045, 0.150), 0.045, Z_STR)

    # -- dado: a real 25 mm change of plane at the wall foot, carrying the damp line
    for k in range(6):
        s3 = L / 6.0
        gb(f"{tag}_dado{k}", mat + "_dark", x0 + k * s3 + 0.003, x0 + (k + 1) * s3 - 0.003,
           *yy(-0.095, 0.045), Z_DADO, Z_ARC + r.uniform(-0.010, 0.010))

    # -- piers, two steps each.  Widths jitter, and three texture variants cycle, so
    #    18 piers do not read as a comb of identical teeth.
    wid = np.array([0.20 + r.uniform(-0.025, 0.025) for _ in range(n_piers)])
    gap = (L - wid.sum()) / (n_piers - 1)
    cx = x0
    pier_x = []
    PIL = set(range(2, n_piers, 4))  # every fourth pier becomes a full-height pilaster
    for k in range(n_piers):
        w = wid[k]
        f = far_of(cx + w / 2)
        # `dress` is the material for the PROJECTING masonry.  The two walls are
        # deliberately built differently.  The sun wall is rendered rose throughout,
        # because sunlit rose measures #E88660 against the bible's #FE795D target.
        # The shadow wall cannot do that: its piers stand proud, so their +X reveals
        # face the key head-on, and a 2800 K sun on Deep Petrol lands in olive-green
        # (measured #4A5A48) -- an ugly colour, and one that drifts toward the mint
        # reserved for the robot.  So the shadow wall is a STONE-DRESSED arcade with
        # petrol-washed panels: every sunlit plane is warm stone, and petrol survives
        # only where it belongs, on the shadow-facing field.  It also gives the two
        # walls different construction, which is what stops them reading as mirrors.
        m = (dress or mat) + (f if f else ("", "_b", "_c")[k % 3] if not dress else "")
        gb(f"{tag}_pierA{k}", shade if dress else (mat + (f or ("", "_b", "_c")[k % 3])),
           cx, cx + w, *yy(-0.020, 0.045), Z_ARC - 0.070, Z_STR)
        gb(f"{tag}_pierB{k}", m, cx + 0.026, cx + w - 0.026, *yy(-0.090, -0.020),
           Z_ARC - 0.070, Z_STR - r.uniform(0.0, 0.012))
        if k in PIL:
            # A pilaster projecting 150 mm past the recess back.  With the key almost
            # horizontal, a vertical edge is the ONLY thing that casts usefully: this
            # throws a 179 mm shadow bar sideways across the panel behind it, and that
            # bar is what still reads as depth when the wall is seen edge-on from the
            # aisle -- which is how the duck sees it most of the time.
            gb(f"{tag}_pilA{k}", m, cx + 0.030, cx + w - 0.030, *yy(-0.150, -0.090),
               0.0, Z_STR + 0.012)
            gb(f"{tag}_pilC{k}", "stone" + (f or "_mid"), cx + 0.012, cx + w - 0.012,
               *yy(-0.168, -0.082), Z_STR + 0.012, Z_STR + 0.040)
        pier_x.append((cx, cx + w))
        cx += w + gap

    # -- string course: oversails every pier, so the whole arcade sits in its shadow
    for k in range(6):
        s4 = L / 6.0
        a = x0 + k * s4
        gb(f"{tag}_string{k}", "stone_xl" + far_of(a + s4 / 2), a + 0.004,
           a + s4 - 0.004, *yy(-0.105, 0.150), Z_STR, Z_STR + 0.035)

    # -- frieze above it, set back again
    for k in range(5):
        s5 = L / 5.0
        a = x0 + k * s5
        gb(f"{tag}_frieze{k}", mat + "_band", a, a + s5,
           *yy(-0.052, 0.150), Z_FRZ, Z_COP)

    # -- corbels under the string course on every fourth pier: they project past the
    #    pier face and drop a small hard shadow -- the cheapest craft signal there is.
    for k in range(2, n_piers, 4):
        a0, a1 = pier_x[k]
        gb(f"{tag}_corbel{k}", "stone" + far_of(a0), a0 + 0.032, a1 - 0.032,
           *yy(-0.118, -0.082), Z_STR - 0.040, Z_STR)

    # -- sills in a third of the recesses
    for k in range(1, n_piers - 1, 3):
        gb(f"{tag}_sill{k}", "stone" + far_of(pier_x[k][1]), pier_x[k][1] + 0.045,
           pier_x[k + 1][0] - 0.045, *yy(-0.062, 0.045), Z_ARC - 0.070, Z_ARC - 0.046)

    # -- iron tie rings driven into the piers
    for k in r.choice(np.arange(2, n_piers - 2), size=3, replace=False):
        a0, a1 = pier_x[int(k)]
        gb(f"{tag}_ring{k}", "iron", (a0 + a1) / 2 - 0.017, (a0 + a1) / 2 + 0.017,
           *yy(-0.104, -0.082), 0.215, 0.250)

    # -- coping: 9 segments.  Two have fallen away, one has slipped and sits askew.
    #    That break in the top line is what stops the wall reading as an extrusion.
    cseg = L / 9.0
    for k in range(9):
        if k in (2, 6):
            continue
        a = x0 + k * cseg
        # +/-11 mm, not +/-4 mm.  The coping's top edge is a single 14.2 m straight line
        # lying nearly along the light bearing, which is the worst case for a shadow-map
        # staircase: at a 2.06 mm texel the stair runs for tens of texels before it steps.
        # Jitter larger than the texel breaks the run into nine independent edges, so what
        # is left reads as a rough coping instead of a broken shadow map.
        h1 = top + r.uniform(-0.011, 0.011)
        if k == 4:  # slipped capstone
            g(f"{tag}_cope{k}", "stone_lg_mid",
              (a + cseg / 2, Y + sgn * 0.007, (Z_COP + h1) / 2 + 0.006),
              (cseg / 2 - 0.006, 0.135, (h1 - Z_COP) / 2), yaw=0.055 * sgn)
        else:
            g(f"{tag}_cope{k}", "stone_lg" + far_of(a + cseg / 2),
              (a + cseg / 2, (yy(-0.125, 0.155)[0] + yy(-0.125, 0.155)[1]) / 2,
               (Z_COP + h1) / 2),
              (cseg / 2 - 0.005, 0.140, (h1 - Z_COP) / 2),
              yaw=r.uniform(-0.016, 0.016))

    # -- ONE collision proxy for the whole wall, front face flush with the plinth nose
    a, b = yy(-0.185, 0.150)
    collider(f"{tag}_C", (0.0, (a + b) / 2, top / 2), (7.12, (b - a) / 2, top / 2))
    return pier_x


# ----------------------------------------------------------------------------------
# 3.  BUTTRESS PYLONS + WEST CORNER PYLONS  --  the skyline events
# ----------------------------------------------------------------------------------
def pylons():
    spec = [
        # (x, sgn, height, tag)
        (-4.55, -1, 0.615, "sbtA"), (2.35, -1, 0.660, "sbtB"),
        (-4.10, +1, 0.700, "nbtA"), (3.05, +1, 0.745, "nbtB"),
        (-7.02, -1, 0.660, "cnrS"), (-7.02, +1, 0.780, "cnrN"),
    ]
    for x, sgn, h, tag in spec:
        Y = sgn * 3.35
        rr = math.hypot(x, Y)
        b = band(rr)
        w = 0.15 if tag.startswith(("sbt", "nbt")) else 0.20
        yo = 0.175 if tag.startswith("cnr") else 0.135
        ys_ = sorted([Y - sgn * (yo - 0.30), Y - sgn * yo])
        # battered shaft: three stacked blocks, each stepping back 8 mm
        for m, (z0, z1, ins) in enumerate([
            (0.0, 0.055, -0.022), (0.055, h * 0.52, 0.0),
            (h * 0.52, h - 0.045, 0.010)]):
            a0 = ys_[0] + (ins if sgn < 0 else 0)
            a1 = ys_[1] - (ins if sgn > 0 else 0)
            gb(f"{tag}{m}", ("stone" + b) if m == 0 else ("ashlar_md" + b),
               x - w + ins, x + w - ins, a0, a1, z0, z1)
        # cornice + a broken finial: never a bare extruded box
        gb(f"{tag}cor", "stone" + b, x - w - 0.028, x + w + 0.028,
           ys_[0] - 0.030, ys_[1] + 0.030, h - 0.045, h)
        g(f"{tag}fin", "stone" + b, (x + 0.018, (ys_[0] + ys_[1]) / 2, h + 0.030),
          (w * 0.46, 0.075, 0.030), yaw=0.22)


# ----------------------------------------------------------------------------------
# 4.  THE EAST GATE + THE LIGHT SLOT  (x = 7.15, slot y in [-0.55, +0.55])
# ----------------------------------------------------------------------------------
# The two masses are deliberately unequal (0.62 vs 0.72 tall) -- symmetry reads as
# blockout.  Seen from the court their -X faces are in shadow, so they are silhouettes
# against the sky slot; but the SOUTH mass's +Y jamb catches the sun full on, so the
# slot has a blazing south cheek and a near-black north cheek.  Everything on the
# south jamb is therefore modelled to catch that light: stepped reveal, string course,
# corbels.
def gate():
    for sgn, ytop, h, tag in ((+1, 3.31, 0.620, "gtN"), (-1, -3.31, 0.720, "gtS")):
        yin = sgn * 0.55  # slot cheek
        ylo, yhi = sorted([yin, ytop])
        b = "_far"
        # plinth, projecting into the court
        gb(f"{tag}_plinth", "stone_far", 7.02, 7.285, ylo - 0.012, yhi, 0.0, 0.055)
        # main mass, recessed 50 mm behind the antae -> a real change of plane
        # the mass is RENDERED, its antae and dressings are cut stone: a change of
        # material as well as of plane, so the gate reads as built, not extruded
        gb(f"{tag}_mass", "plaster_far", 7.10, 7.262, ylo, yhi, 0.055, h - 0.030)
        # antae: the mass comes forward again at the slot cheek and at the far end
        # ante pushed 3 mm proud of the mass: the two shared their +x face at exactly
        # x = 7.262 over a 0.155 x 0.635 m patch with DIFFERENT materials, and a
        # segmentation-flip test over 60 jittered views put gtS_mass <-> gtS_ante top of
        # the z-fighting table at 6531 px.  +3 mm reads as a pilaster and drops the whole
        # scene's flip rate by 75%.  Rule for this file: never let two visible geoms share
        # a face plane to within 1.5 mm.
        gb(f"{tag}_ante", "ashlar_md" + b, 7.04, 7.265,
           *sorted([yin, yin + sgn * 0.155]), 0.055, h + 0.052)
        gb(f"{tag}_anteC", "stone_far", 7.012, 7.29,
           *sorted([yin - sgn * 0.026, yin + sgn * 0.181]), h + 0.052, h + 0.086)
        gb(f"{tag}_antP", "stone_far", 7.008, 7.295,
           *sorted([yin - sgn * 0.022, yin + sgn * 0.177]), 0.0, 0.062)
        # string course: a 20 mm projecting band, so the mass has a waist
        gb(f"{tag}_string", "stone_far", 7.082, 7.282, ylo - 0.020, yhi, 0.300, 0.334)
        # upper attic block, stepped back, then a cornice oversailing it
        gb(f"{tag}_attic", "plaster_far", 7.1175, 7.2525, ylo + 0.030, yhi,
           h - 0.030, h + 0.062)
        gb(f"{tag}_corn", "stone_far", 7.075, 7.292, ylo - 0.028, yhi,
           h + 0.062, h + 0.100)
        # the lintel is long gone; only its corbels remain, reaching into the slot
        for m, zz in enumerate((h * 0.80, h * 0.80 - 0.075)):
            gb(f"{tag}_cbl{m}", "stone_far", 7.10, 7.24,
               *sorted([yin - sgn * (0.052 - m * 0.020), yin]), zz, zz + 0.034)
        # jamb reveal: two shallow steps down the cheek
        for m, (dx, dy, z1) in enumerate(((7.155, 0.040, h * 0.62), (7.205, 0.075, h * 0.44))):
            gb(f"{tag}_rev{m}", "ashlar_md_far", dx, 7.2585,
               *sorted([yin, yin + sgn * dy]), 0.055, z1)
        # a flanking dais, clear of the 1.1 m threshold
        for m, (o0, o1, z1) in enumerate(((0.62, 1.55, 0.036), (0.70, 1.35, 0.068))):
            gb(f"{tag}_dais{m}", "stone_far", 6.62 + m * 0.13, 7.02,
               *sorted([sgn * o0, sgn * o1]), 0.0, z1)
        # collision proxy: one box per mass, wrapping plinth + antae
        collider(f"{tag}_C", (7.1525, (ylo + yhi) / 2, h / 2),
                 (0.1325, (yhi - ylo) / 2, h / 2))

    # two round bollards marking the threshold -- the only cylinders on the gate
    for m, s in enumerate((-1, 1)):
        g(f"gt_boll{m}", "stone_far", (6.52, s * 0.735, 0.085), (0.042, 0.042, 0.085),
          typ="cylinder")


# ----------------------------------------------------------------------------------
# 5.  THE WEST RUIN  (x = -7.15)  --  five gapped stumps, 0.10 .. 0.34 tall
# ----------------------------------------------------------------------------------
# This is where the direction's long shadows have room to run, so the ruin is built
# for SILHOUETTE: every stump breaks differently, and the tops are stepped rubble.
def ruin():
    r = np.random.default_rng(5150)
    spec = [(-2.62, 0.92, 0.225), (-1.28, 0.60, 0.115), (0.18, 1.05, 0.345),
            (1.62, 0.72, 0.160), (2.78, 0.86, 0.285)]
    # Ashlar survives HERE and on the gate masses, where a masonry read earns its place --
    # but with the coursing PHASE AND SCALE varied per instance, so no two adjacent stumps
    # align.  The critique's note stands: 91 geoms all carrying the same course at the same
    # phase is a stock asset kit, and it is the highest-frequency thing in the mid-band.
    RUV = ("_md_far", "_md2_far", "_md3_far", "_md2_far", "_md_far")
    for k, (yc, ln, h) in enumerate(spec):
        b = "_far"
        av = RUV[k]
        y0, y1 = yc - ln / 2, yc + ln / 2
        gb(f"ru{k}_plinth", "stone_rim_far", -7.30, -6.99, y0 - 0.03, y1 + 0.03, 0.0, 0.042)
        gb(f"ru{k}_core", "plaster" + b, -7.27, -7.02, y0, y1, 0.042, h * 0.62)
        # the break line: three blocks of falling height with a real diagonal
        n = 3
        for m in range(n):
            f0 = y0 + ln * m / n
            f1 = y0 + ln * (m + 1) / n
            hh = max(h * (0.55 + 0.62 * (1 - m / n) ** 1.6) + r.uniform(-0.035, 0.035),
                     h * 0.62 + 0.022)
            g(f"ru{k}_brk{m}", "ashlar" + av,
              ((-7.265 - 7.025) / 2 + r.uniform(-0.012, 0.012),
               (f0 + f1) / 2, (h * 0.62 - 0.004 + hh) / 2),
              (0.120 - r.uniform(0, 0.022), (f1 - f0) / 2 - 0.008,
               (hh - h * 0.62 + 0.004) / 2), yaw=r.uniform(-0.11, 0.11))
        # a spalled block hanging off the face, and one that has already fallen
        g(f"ru{k}_spall", "ashlar" + av,
          (-6.985, yc + r.uniform(-0.2, 0.2), h * 0.40),
          (0.038, 0.070, 0.048), yaw=r.uniform(-0.3, 0.3))

    # a toppled column lying in the west yard: drums + its capital, unmistakably
    # architectural debris and the one place cylinders earn their 5.3x cost.
    for m in range(3):
        # zaxis, not yaw: these are LYING DOWN, axis along +Y, drums slightly parted
        # where the column broke as it fell.
        g(f"ru_drum{m}", "ashlar_md_far",
          (-6.30 + m * 0.026, -2.05 + m * 0.235, 0.098),
          (0.098, 0.105, 0.098), typ="cylinder",
          zaxis=(0.07 * m - 0.04, 1.0, 0.0))
    g("ru_drumcap", "stone_far", (-6.24, -1.37, 0.098), (0.026, 0.135, 0.135), yaw=0.09)
    g("ru_drumbase", "stone_far", (-6.36, -2.44, 0.075), (0.150, 0.150, 0.075), yaw=0.06)

    # a three-step dais leading up to the ruin -- gives the west yard a stage.  Deepened
    # to the same 0.130 m two-step module as the court walls: the scale story has to be
    # consistent or the duck has nothing to read its own size against.
    for m, (dx, y0, y1, z1) in enumerate(
            ((-6.62, -1.05, 1.35, 0.044), (-6.78, -0.85, 1.15, 0.088),
             (-6.90, -0.65, 0.95, 0.132))):
        gb(f"ru_step{m}", "stone_xl_mid", dx, -6.99, y0, y1, 0.0, z1)

    # COLLISION.  The ruin was 38 geoms of masonry 41 cm tall with ZERO collidable geoms,
    # so the duck walked through it.  One proxy per stump plus one for the dais, on the
    # columns' existing group=3 pattern.  Static-static pairs never generate contacts, so
    # the mj_step cost is unmeasurable.
    for k, (yc, ln, h) in enumerate(spec):
        collider(f"ru{k}_C", (-7.15, yc, h / 2), (0.155, ln / 2 + 0.03, h / 2))
    collider("ru_step_C", (-6.945, 0.15, 0.066), (0.205, 1.20, 0.066))
    collider("ru_drum_C", (-6.30, -1.95, 0.098), (0.115, 0.42, 0.098))


# ----------------------------------------------------------------------------------
# 6.  THE FOUNTAIN  (-2.8, 0), octagonal, circumradius 0.75
# ----------------------------------------------------------------------------------
# Octagon rather than cylinder: 24 boxes cost 24 BE, 8 cylinders would cost 42, and
# the faceting gives eight discrete light values around the rim instead of a gradient.
def fountain():
    """THE FOUNTAIN, rebuilt.  Three separate defects, one cause.

    1. THE SAWTOOTH SHADOW.  The first cut was FOUR concentric octagonal rings -- 8 foot
       at ap+0.040, 8 skirt at ap+0.022, 8 fascia at ap+0.016, 7 cope at ap+0.024 -- each
       at a different radius and a different height.  At an 11 degree sun every one of
       those 31 rings contributes its own corner notches to the cast silhouette, and they
       superimpose into ~25-30 uniform triangular teeth on the shadow's leading edge.  It
       is not a shadow-map artifact: it is identical at shadowsize 1024 and 16384, unmoved
       by shadowclip, and it vanishes when these geoms are hidden.
       THE FIX IS A SINGLE SILHOUETTE.  The cope is now both the WIDEST and the HIGHEST
       ring, and every other member is strictly inside it in plan and below it in height,
       so the whole assembly casts exactly one 8-corner shadow.

    2. IT READ AS A WOODEN HOT-TUB SURROUND.  `stone_rim` was stonec at texrepeat
       "1.3 0.062": V = 0.062 puts SIX PERCENT of the texture's height across a facet, so
       every facet was one horizontal stripe of a coursed texture -- plank siding.  Every
       facet now takes V = 1 (one tile top to bottom) with U set by the facet's real
       aspect, and the cope takes plain limestone rather than coursed stone, because a
       cope is one carved block, not masonry.

    3. IT WAS BONE DRY, in a scene called THE WATER COURT.  The basin is now low enough
       (cope top 0.112 m) that a 0.12 m eye looks OVER the rim and sees the water surface
       terrain lays at 0.100 -- a broad shallow lustration basin, which is what a court at
       this scale would actually have.  A notched weir on the +X side gives the water a
       visible reason to leave toward the rill.

    ...and it was NON-COLLIDABLE.  38 geoms of masonry 32 cm tall, taller than the robot,
    that the duck walked straight through: driven at it under the walking policy it came
    out the far side with no architecture contact at all.  Eight thin box proxies now wrap
    the drum, following the same group=3 pattern the columns already use."""
    cx, cy = -2.80, 0.0
    R = 0.75
    ap = R * math.cos(math.pi / 8)      # apothem 0.693
    half = R * math.sin(math.pi / 8)    # half side 0.287
    r = np.random.default_rng(909)

    Z_DRUM, Z_COPE, Z_TOP = 0.080, 0.078, 0.112
    D_COPE, T_COPE = ap + 0.030, 0.040   # outer face 0.763  <- THE silhouette
    D_DRUM, T_DRUM = ap + 0.002, 0.034   # outer face 0.729  (inside)
    D_LIN, T_LIN = ap - 0.062, 0.022     # outer face 0.653  (inside)

    WEIR = 0                             # the facet that faces +X, notched
    for k in range(8):
        th = k * math.pi / 4
        c, s = math.cos(th), math.sin(th)
        yawf = th + math.pi / 2

        # drum: coursed masonry, battered slightly, strictly inside the cope
        g(f"fn_drum{k}", "fn_drum" + ("_mid" if k in (2, 3, 4) else ""),
          (cx + c * D_DRUM, cy + s * D_DRUM, Z_DRUM / 2),
          (half + 0.030, T_DRUM, Z_DRUM / 2), yaw=yawf)
        # inner lining: what the water laps against.  Visible over the rim.
        g(f"fn_lin{k}", "fn_lining",
          (cx + c * D_LIN, cy + s * D_LIN, (Z_DRUM + 0.014) / 2),
          (half - 0.024, T_LIN, (Z_DRUM + 0.014) / 2), yaw=yawf)

        if k == WEIR:
            # THE WEIR.  One cope block cut down to a notch, so the basin visibly
            # overflows toward the rill instead of being a sealed tub.
            for m, off in ((0, +1), (1, -1)):
                g(f"fn_cope{k}{m}", "fn_cope",
                  (cx + c * D_COPE - s * off * (half * 0.60),
                   cy + s * D_COPE + c * off * (half * 0.60),
                   (Z_COPE + Z_TOP) / 2),
                  (half * 0.40 + 0.026, T_COPE, (Z_TOP - Z_COPE) / 2), yaw=yawf)
            g(f"fn_weirsill{k}", "fn_cope",
              (cx + c * D_COPE, cy + s * D_COPE, Z_COPE + 0.008),
              (half * 0.60, T_COPE, 0.008), yaw=yawf)
            continue
        if k == 5:   # one length of coping has been knocked out and lies on the apron
            g(f"fn_rubble{k}", "stone_mid",
              (cx + c * (D_COPE + 0.10), cy + s * (D_COPE + 0.10), 0.019),
              (0.078, 0.046, 0.019), yaw=yawf + 0.30)
            continue
        g(f"fn_cope{k}", "fn_cope" + ("_mid" if k in (3, 4) else ""),
          (cx + c * D_COPE, cy + s * D_COPE, (Z_COPE + Z_TOP) / 2 + r.uniform(-0.002, 0.002)),
          (half + 0.052, T_COPE, (Z_TOP - Z_COPE) / 2),
          yaw=yawf + r.uniform(-0.010, 0.010))

    # the spout pier: stepped, corbelled, and broken off at the top.  Its own shadow is
    # one clean bar, which is what the west yard wanted from the fountain all along.
    g("fn_pbase", "stone", (cx, cy, 0.026), (0.150, 0.150, 0.026), yaw=0.05)
    g("fn_psoc", "stone", (cx, cy, 0.066), (0.118, 0.118, 0.014), yaw=0.05)
    g("fn_pshaft", "fn_drum", (cx, cy, 0.176), (0.082, 0.082, 0.096), yaw=0.05)
    g("fn_pcorb", "stone", (cx, cy, 0.288), (0.112, 0.112, 0.016), yaw=0.05)
    g("fn_pbrk", "fn_drum", (cx + 0.012, cy - 0.008, 0.330), (0.058, 0.062, 0.026), yaw=0.28)
    g("fn_spout", "iron", (cx + 0.092, cy + 0.030, 0.262), (0.050, 0.016, 0.016), yaw=0.18)

    # COLLISION.  Eight thin proxies on the drum line, using the columns' own pattern.
    for k in range(8):
        th = k * math.pi / 4
        c, s = math.cos(th), math.sin(th)
        collider(f"fn_C{k}", (cx + c * D_DRUM, cy + s * D_DRUM, Z_TOP / 2),
                 (half + 0.030, T_DRUM, Z_TOP / 2), yaw=th + math.pi / 2)


# ==================================================================================
#                                   E M I T
# ==================================================================================
def materials():
    """Every material this agent publishes.

    texuniform='false' throughout: it is the only mode that gives predictable,
    per-face tiling on a box.  (Measured: texuniform='true' maps U from size[0] and
    V from size[1] on EVERY face, so a 7.1 x 0.08 m wall gets 0.08-worth of V --
    useless for architecture.)

    The consequence is that texrepeat must be matched to the geom's real size, or a
    long geom stretches one tile over metres and renders as a smooth airbrushed
    gradient.  Hence the size classes below: each family is tuned so that a masonry
    block lands at ~0.20 m and a course at ~0.055 m WHATEVER the geom it is on.
    """
    m = []
    m.append("<!-- ================= architecture agent: textures ================= -->")
    for t in ("ashlar", "plaster", "stone", "stonec", "rose", "petrol"):
        m.append(f'<texture type="2d" name="{PFX}t_{t}" file="{PFX}{t}.png"/>')
    m.append("<!-- ================= architecture agent: materials ================ -->")
    m.append("<!-- families carry authored aerial perspective: near / _mid / _far -->")

    def mat(name, tex, rep, spec, shin, rgba=None):
        c = f' rgba="{rgba}"' if rgba else ""
        m.append(
            f'<material name="{PFX}{name}" texture="{PFX}t_{tex}" texuniform="false"'
            f' texrepeat="{rep}"{c} specular="{spec}" shininess="{shin}"/>'
        )

    # aerial perspective triples: full chroma / 3-6 m / >6 m  (bible section 8)
    AP = [("", None), ("_mid", "0.92 0.93 0.97 1"), ("_far", "0.84 0.88 0.98 1")]

    def family(base, tex, rep, spec, shin):
        for sfx, rgba in AP:
            mat(base + sfx, tex, rep, spec, shin, rgba)

    #        family        texture    texrepeat   intended face size
    family("ashlar", "ashlar", "1 1", 0.09, 0.06)      # 0.21 m  column shafts
    # ...in three coursing phases, so twelve columns are not twelve identical bricks
    mat("ashlar_b", "ashlar", "1.14 1.09", 0.09, 0.06)
    mat("ashlar_c", "ashlar", "0.88 1.18", 0.09, 0.06)
    family("ashlar_md", "ashlar", "2.4 1.5", 0.09, 0.06)   # 0.5-0.9 m pylons, breaks
    # per-instance coursing phase and scale, so adjacent ruin stumps never align
    family("ashlar_md2", "ashlar", "2.05 1.72", 0.09, 0.06)
    family("ashlar_md3", "ashlar", "2.76 1.31", 0.09, 0.06)
    family("ashlar_big", "ashlar", "4.6 1.5", 0.08, 0.05)   # 1.5-3 m  gate, ruin cores
    family("plaster", "plaster", "3 1.4", 0.06, 0.04)
    family("plaster_col", "plaster", "1 1", 0.06, 0.04)   # column shafts: 0.21 m face
    family("stone", "stone", "1 1", 0.10, 0.06)      # 0.2-0.35 m capitals, sills
    # THIN ELEMENTS.  texrepeat has to be isotropic IN WORLD UNITS or the tile is
    # squashed along the short axis and renders as quilting or horizontal streaking:
    # a 1.58 x 0.03 m coping at "8 1" stretches the texture 20:1.  So V = U x h/L.
    #                                      L x h (m)        tile
    family("stone_lg", "stonec", "2.6 0.050", 0.10, 0.06)   # 1.58 x 0.030   coping
    family("stone_xl", "stonec", "4.0 0.070", 0.10, 0.06)   # 2.4-2.8 x 0.05 plinth/string
    # FOUNTAIN.  V = 1 on every facet -- one tile top to bottom.  The first cut used
    # V = 0.062, i.e. six percent of a coursed texture's height stretched across a facet,
    # which is why the basin rendered as horizontally-striped board.
    family("fn_cope", "stone", "6 1", 0.10, 0.06)      # 0.68 x 0.034 cope: one carved block
    family("fn_drum", "stonec", "6 1", 0.10, 0.06)     # 0.63 x 0.080 drum: real coursing
    mat("fn_lining", "stonec", "5 1", 0.16, 0.14, "0.70 0.70 0.74 1")   # wetted, cooler
    mat("stone_rim_far", "stonec", "1.3 0.4", 0.08, 0.05, "0.78 0.80 0.88 1")

    # the SUN wall.  Three pier variants so 18 identical piers do not read as a comb.
    family("rose", "rose", "1 1", 0.07, 0.05)      # 0.20 m  piers
    mat("rose_b", "rose", "1.29 1.16", 0.07, 0.05)
    mat("rose_c", "rose", "0.83 1.27", 0.07, 0.05)
    mat("rose_long", "rose", "17 2", 0.06, 0.04)     # 3.5 m   back slab
    mat("rose_band", "rose", "15.8 0.085", 0.07, 0.05)   # 2.84 x 0.015 frieze
    mat("rose_shade", "stone", "9 2", 0.05, 0.04, "0.46 0.44 0.58 1")   # recess back
    mat("rose_dark", "rose", "6.5 1", 0.09, 0.07, "0.86 0.80 0.80 1")   # blind panel
    mat("rose_plane", "rose", "5.8 1", 0.07, 0.05)   # THE PLANE: 1.775 x 0.305 m per segment

    # the SHADOW wall
    family("petrol", "petrol", "1 1", 0.14, 0.10)
    mat("petrol_b", "petrol", "1.27 1.13", 0.14, 0.10)
    mat("petrol_c", "petrol", "0.85 1.22", 0.14, 0.10)
    mat("petrol_long", "petrol", "17 2", 0.12, 0.09)
    mat("petrol_band", "petrol", "15.8 0.085", 0.14, 0.10)
    mat("petrol_shade", "stone", "9 2", 0.10, 0.08, "0.40 0.42 0.62 1")
    mat("petrol_dark", "petrol", "13 1", 0.16, 0.12, "0.80 0.82 0.88 1")

    # ironwork: cramps, tie rings, the fountain spout.  Every instance is under
    # 0.01 m^2, which is why a flat material is allowed here and nowhere else.
    m.append(f'<material name="{PFX}iron" rgba="0.300 0.170 0.120 1"'
             f' specular="0.30" shininess="0.35"/>')

    # Ship only what is actually on a geom, plus the canonical palette entries other
    # agents may want to reference.  Publishing 13 dead materials into a shared asset
    # namespace is clutter, and an orphan texture is worse.
    used = set(re.findall(r'material="([^"]+)"', "\n".join(GE + CO)))
    keep = used | {PFX + n for n in ("ashlar", "stone", "rose", "petrol", "iron",
                                     "plaster", "stonec")}
    out, live_tex = [], set()
    for line in m:
        n = re.search(r'<material name="([^"]+)"', line)
        if n and n.group(1) not in keep:
            continue
        t = re.search(r'texture="([^"]+)"', line)
        if t:
            live_tex.add(t.group(1))
        out.append(line)
    return [l for l in out
            if not (l.startswith("<texture") and
                    re.search(r'name="([^"]+)"', l).group(1) not in live_tex)]


def build():
    print("textures:")
    ashlar(f"{PFX}ashlar.png", OCHRE, courses=6, blocks=3, seed=101)
    plaster(f"{PFX}plaster.png", OCHRE, seed=211)
    limestone(f"{PFX}stone.png", STONE, seed=307)
    # coursed pale stone for the dressings: a coping made of 300 mm blocks with real
    # head joints reads as masonry; the same coping in featureless limestone reads as
    # sanded plywood, which is what the first four passes of the fountain looked like.
    ashlar(f"{PFX}stonec.png", STONE, courses=4, blocks=2, seed=613, joint_px=8.0,
           joint_dark=0.52, tint=0.055, spall_amt=0.22, streak=0.018)
    rosewash(f"{PFX}rose.png", ROSE, seed=409)
    petrolwall(f"{PFX}petrol.png", PETROL, seed=503)

    colonnade()
    sun_wall("sw", -1, 0.500, 771)                                # SUN wall: A PLANE
    court_wall("nw", +1, 0.580, "petrol", "petrol_shade", 18, 883, dress="plaster_col")  # SHADOW wall
    pylons()
    gate()
    ruin()
    fountain()

    assets = "\n".join(materials()) + "\n"   # after the geometry: it filters on usage
    body = (
        "<!-- =============== architecture agent: built structures =============== -->\n"
        "<!-- Visible scenery: every geom contype=0 conaffinity=0. -->\n"
        + "\n".join(GE)
        + "\n<!-- Collision proxies: invisible (group=3, the robot's own convention\n"
          "     for collision geoms).  These are the ONLY collidable geoms this\n"
          "     agent contributes: 12 columns + 2 walls + 2 gate masses = 16. -->\n"
        + "\n".join(CO) + "\n"
    )
    open(os.path.join(MD, "aaa_architecture_assets.xml"), "w").write(assets)
    open(os.path.join(MD, "aaa_architecture_body.xml"), "w").write(body)

    be = sum(BE.get(k, 1.0) * v for k, v in _stats.items())
    print(f"\ngeoms: visible {len(GE)}  collision proxies {len(CO)}  "
          f"TOTAL {len(GE) + len(CO)} / 420")
    print(f"types: {_stats}   cylinders {_stats.get('cylinder', 0)} / 10")
    print(f"box-equivalents: {be + len(CO):.1f} / 470")
    print(f"wrote {MD}/aaa_architecture_assets.xml")
    print(f"wrote {MD}/aaa_architecture_body.xml")


if __name__ == "__main__":
    build()
