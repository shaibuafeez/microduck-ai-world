#!/usr/bin/env python3
"""AAA scene — the ATMOSPHERE layer of "The Water Court".

Owns: the skybox cubemap, the three-light rig, the <visual>/<statistic> block,
the background hfield ranges, the distant silhouette masses, and every emissive
geom in the scene.

Writes (idempotently) into the model directory:
    atmosphere_sky_{right,left,up,down,front,back}.png   1024^2 cubemap faces
    atmosphere_distant.png                               albedo for distant masses
    atmosphere_ridge.png                                 albedo for the hfield ranges
    aaa_atmosphere_assets.xml   <texture>/<material>/<hfield> fragment
    aaa_atmosphere_body.xml     <geom>/<light> fragment
    aaa_atmosphere_visual.xml   <statistic>/<visual> fragment (composer must include)

The four background ranges carry their elevation INLINE in the hfield element
rather than as PNGs -- see build_hfield for why a file path cannot work here.

    python scripts/aaa/atmosphere.py

CUBEMAP FACE CONVENTION — measured on this machine (mujoco 3.12.0), not assumed.
The brief and scripts/make_sky_texture.py both claim "fileright is -X and fileleft
is +X".  That is FALSE here.  Probed with solid-colour faces plus marker geoms at
(3,0,0)/(0,3,0)/(0,0,3) to pin the view direction beyond doubt, then with u/v
ramp faces to pin the in-face axes.  The engine uses the plain OpenGL convention:

    with s = image column mapped to [-1,+1] (left -> right)
         t = image row    mapped to [-1,+1] (TOP  -> BOTTOM)

    fileright  +X   d = ( 1,  s, -t)
    fileleft   -X   d = (-1, -s, -t)
    fileback   +Y   d = (-s,  1, -t)
    filefront  -Y   d = ( s, -1, -t)
    fileup     +Z   d = ( s, -t,  1)
    filedown   -Z   d = ( s,  t, -1)

Every pixel is a pure function of the unit direction d, so the faces are
continuous across all twelve cube edges by construction.
"""
from __future__ import annotations

import math
import os
import numpy as np
from PIL import Image

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODEL = os.path.join(ROOT, "src", "mjlab_microduck", "robot", "microduck")
P = "atmosphere_"

FACE = 1024          # published face resolution: 1 texel/pixel at 1280x720/fovy78
SS = 2               # supersample factor, box-filtered down -> real prefilter

# --------------------------------------------------------------------------
# the sun.  Must agree EXACTLY with light aaa_sun's dir, negated.
#   dir = -0.752 -0.631 -0.191   ->  sun sits at azimuth 40.0 deg, elevation 11.0 deg
# --------------------------------------------------------------------------
SUN_AZ = np.deg2rad(40.0)
SUN_EL = np.deg2rad(11.0)
SUN = np.array([np.cos(SUN_EL) * np.cos(SUN_AZ),
                np.cos(SUN_EL) * np.sin(SUN_AZ),
                np.sin(SUN_EL)], np.float64)
SUN /= np.linalg.norm(SUN)
SUN_H = np.array([np.cos(SUN_AZ), np.sin(SUN_AZ)])       # horizontal sun bearing

# --------------------------------------------------------------------------
# palette (art bible section 2).  Authored in FINAL DISPLAY SPACE: the classic
# renderer draws the skybox unlit and ungraded, so these bytes are the pixels.
# --------------------------------------------------------------------------
# Three vertical stops, each with a sunward and an anti-solar variant.  The gold
# is deliberately confined to a ~4 deg skirt: v1 spread it over the whole lower
# sky and the result was milk -- no band, and no room for the bloom to register.
ZEN_SUN = np.array([0.415, 0.525, 0.835])       # Sky Cobalt, a touch warm sunward
ZEN_ANT = np.array([0.270, 0.360, 0.720])       # deeper anti-solar zenith
MID_SUN = np.array([0.930, 0.615, 0.430])       # the salmon shoulder over the sun
MID_ANT = np.array([0.470, 0.420, 0.640])       # anti-solar violet shoulder
HOR_SUN = np.array([1.000, 0.845, 0.585])       # Sun Gold  #FFD799
HOR_ANT = np.array([0.590, 0.495, 0.575])       # anti-solar dusty rose
GROUNDHAZE = np.array([0.760, 0.660, 0.530])    # under the horizon; the floor hides it

CLOUD_HOT = np.array([1.000, 0.610, 0.370])     # deck underside facing the sun
CLOUD_MID = np.array([0.795, 0.585, 0.545])     # the turn: warmed off neutral grey
CLOUD_COOL = np.array([0.395, 0.372, 0.548])    # deck underside away from the sun --
# violet, not grey.  The critique found the one large cloud reading as a lens smudge
# because it was desaturated neutral in a peach-and-periwinkle sky.
CIRRUS_HOT = np.array([1.000, 0.780, 0.610])
CIRRUS_COOL = np.array([0.560, 0.590, 0.760])

RIDGE_FAR = np.array([0.600, 0.610, 0.720])     # baked far range, pre-haze
RIDGE_MID = np.array([0.455, 0.450, 0.570])


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def sstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def mix(a, b, t):
    """Blend colours. a,b end in a size-3 axis (or are a bare rgb triple);
    t is a scalar or an array shaped like the pixel grid."""
    t = np.asarray(t, np.float32)
    if t.ndim > 0:
        t = t[..., None]
    return a * (1.0 - t) + b * t


def lattice(seed, n):
    return np.random.default_rng(seed).random((n, n)).astype(np.float32) * 2.0 - 1.0


def vnoise(px, py, lat):
    """Tileable value noise, C1 via smootherstep. Pure function of (px,py)."""
    n = lat.shape[0]
    x0 = np.floor(px)
    y0 = np.floor(py)
    fx = (px - x0).astype(np.float32)
    fy = (py - y0).astype(np.float32)
    fx = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0)
    fy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0)
    i0 = (x0.astype(np.int64) % n).astype(np.int32)
    j0 = (y0.astype(np.int64) % n).astype(np.int32)
    i1 = (i0 + 1) % n
    j1 = (j0 + 1) % n
    a = lat[i0, j0]
    b = lat[i1, j0]
    c = lat[i0, j1]
    d = lat[i1, j1]
    return (a + (b - a) * fx) * (1.0 - fy) + (c + (d - c) * fx) * fy


def fbm2(px, py, lats, octaves, lac=2.03, gain=0.5):
    tot = np.zeros(px.shape, np.float32)
    amp, frq, nrm = 1.0, 1.0, 0.0
    for k in range(octaves):
        tot += amp * vnoise(px * frq, py * frq, lats[k % len(lats)])
        nrm += amp
        amp *= gain
        frq *= lac
    return tot / nrm


def fbm_tile(x, y, base, octaves, seed, gain=0.5):
    """fbm that tiles EXACTLY on [0,1)^2.

    vnoise wraps at its lattice size, so an octave tiles only when its
    coordinates span a whole number of lattice cells.  Sizing each octave's
    lattice to its own integer period gives that for free.  Getting this wrong
    is what put a visible plaid of tile seams across every distant mass.
    """
    tot = np.zeros(np.shape(x), np.float32)
    amp, nrm = 1.0, 0.0
    for k in range(octaves):
        per = base * (2 ** k)
        tot += amp * vnoise(x * per, y * per, lattice(seed + 13 * k, per))
        nrm += amp
        amp *= gain
    return tot / nrm


def ridge_lut(seed, nmax=19, power=1.35, size=4096):
    """Ridged skyline height in [0,1] as a periodic function of azimuth.

    Summing 1-|sin| per harmonic puts a cusp at every zero crossing, which is
    what makes a skyline read as peaks rather than dunes.  Normalised from its
    own measured min/max so the caller's amplitude means exactly what it says.
    """
    r = np.random.default_rng(seed)
    az = np.linspace(-np.pi, np.pi, size, endpoint=False)
    s = np.zeros(size)
    for n in range(1, nmax):
        s += (r.uniform(0.45, 1.0) / n ** 0.62) * (1.0 - np.abs(np.sin(n * az + r.uniform(0, 2 * np.pi))))
    s = (s - s.min()) / (s.max() - s.min())
    return az, (s ** power)


def sample_lut(az, lut):
    grid, val = lut
    return np.interp(az, grid, val, period=2 * np.pi)


# --------------------------------------------------------------------------
# THE SKY
# --------------------------------------------------------------------------
FACES = {
    "right": lambda s, t: np.stack([np.ones_like(s), s, -t], -1),    # +X
    "left":  lambda s, t: np.stack([-np.ones_like(s), -s, -t], -1),  # -X
    "back":  lambda s, t: np.stack([-s, np.ones_like(s), -t], -1),   # +Y
    "front": lambda s, t: np.stack([s, -np.ones_like(s), -t], -1),   # -Y
    "up":    lambda s, t: np.stack([s, -t, np.ones_like(s)], -1),    # +Z
    "down":  lambda s, t: np.stack([s, t, -np.ones_like(s)], -1),    # -Z
}

_DECK = [lattice(101 + i, 256) for i in range(6)]
_CIRR = [lattice(211 + i, 256) for i in range(5)]
_RIDGE_A = ridge_lut(3301, nmax=17, power=1.45)
_RIDGE_B = ridge_lut(5507, nmax=23, power=1.20)

def sky_colour(d):
    """d: (...,3) unit directions -> (...,3) display-space RGB in [0,1].

    Every term is a function of d alone, so the six faces meet exactly.  The
    azimuthal terms use cah = dot(d_horizontal, sun_horizontal) WITHOUT
    normalising by |d_horizontal|: dividing by it (v1 did) makes azimuth
    undefined at the zenith and paints a radial pinwheel of streaks into the
    top face.  cah goes smoothly to 0 overhead, which is what we want anyway.
    """
    e = np.clip(d[..., 2], -1.0, 1.0).astype(np.float32)
    hx, hy = d[..., 0].astype(np.float32), d[..., 1].astype(np.float32)
    az = np.arctan2(hy, hx).astype(np.float32)
    cah = (hx * SUN_H[0] + hy * SUN_H[1]).astype(np.float32)     # smooth everywhere
    warm = np.clip(cah * 0.5 + 0.5, 0.0, 1.0) ** 2.40

    # ---- three-stop vertical gradient -------------------------------------
    zen = mix(ZEN_ANT, ZEN_SUN, warm)
    mid = mix(MID_ANT, MID_SUN, warm)
    hor = mix(HOR_ANT, HOR_SUN, warm)
    up = np.clip(e, 0.0, 1.0)
    col = mix(zen, mid, np.exp(-up / 0.30).astype(np.float32))
    col = mix(col, hor, np.exp(-up / 0.055).astype(np.float32))

    # Belt of Venus: rose band ~3-13 deg up on the ANTI-solar side only.
    belt = sstep(0.02, 0.075, e) * (1.0 - sstep(0.14, 0.32, e)) * np.clip(-cah, 0.0, 1.0) ** 0.8
    col = mix(col, np.array([0.905, 0.660, 0.690]), belt * 0.46)

    # ---- sun bloom.  No disc: the tightest term is wide (cos^38 ~ 13 deg)
    # and the shoulder below keeps it a saturated gold instead of a white hole.
    # An 8-bit skybox cannot hold a sun 100x brighter than the sky, and the bible
    # bans a disc anyway.  So the sun is delivered as a BROAD swell -- wide, low
    # powers -- over a sky that is dark enough elsewhere for the swell to show.
    # v3 used a cos^38 core; the shoulder squashed it to the same value as the
    # horizon band and the sun was literally invisible in the texture.
    cs = np.clip((d * SUN).sum(-1), 0.0, 1.0).astype(np.float32)
    glow = 0.46 * cs ** 2.0 + 0.50 * cs ** 7.0 + 0.40 * cs ** 22.0
    glow *= sstep(-0.05, 0.03, e)
    col = col + glow[..., None] * np.array([0.60, 0.36, 0.12])
    # a touch of achromatic lift in the very core so it reads as a light source
    col = col + (0.20 * cs ** 26.0)[..., None] * np.array([0.30, 0.30, 0.30])

    # ---- cloud deck: a real horizontal slab, projected --------------------
    # p = H*(dx,dy)/e is where the view ray pierces the deck, so cells compress
    # toward the horizon for free.  The zenith maps to the single point of deck
    # directly overhead, which is correct -- but it means the cell size there is
    # whatever the noise frequency says, so the frequency has to be high enough
    # that overhead reads as cloud and not as one enormous smudge.
    ec = np.maximum(e, 0.055)
    px = (hx / ec).astype(np.float32)
    py = (hy / ec).astype(np.float32)
    rad = np.sqrt(px * px + py * py)
    lim = 19.0
    # SMOOTH radial compression, not a hard clamp.  `sc = min(1, lim/rad)` mapped every
    # direction outside radius 19 onto the SAME circle, so along each azimuth the deck
    # value became constant below a threshold elevation == a fan of radial streaks
    # converging on the zenith.  That is the "stray god-ray decal" the critique found in
    # 5_raking_light.png, and it is baked into the sky_up/back faces.  tanh keeps the
    # mapping strictly monotonic, so the gradient never dies and no streak can form.
    radc = (lim * np.tanh(rad / lim)).astype(np.float32)
    sc = (radc / np.maximum(rad, 1e-6)).astype(np.float32)
    px, py = px * sc, py * sc
    rad = radc
    # ...and an asymmetric domain warp before any thresholding, so nothing in the deck can
    # come out mirror-symmetric about a view axis.  The critique found a bilaterally
    # symmetric two-lobed grey blot sitting exactly on the hero vista's vanishing point.
    wpx = px + 2.10 * vnoise(px * 0.21 + 5.3, py * 0.17 - 2.9, _DECK[1])
    wpy = py + 1.55 * vnoise(px * 0.19 - 8.1, py * 0.23 + 6.7, _DECK[3])
    px, py = wpx.astype(np.float32), wpy.astype(np.float32)

    F = 0.80
    n = fbm2(px * F + 11.0, py * F - 7.0, _DECK, 5)
    n2 = fbm2(px * F * 2.6 - 3.0, py * F * 2.6 + 5.0, _DECK[2:], 3)
    dens = n * 0.78 + n2 * 0.22

    # Directional relief: sample the same field a step toward and away from the
    # sun.  Where density RISES toward the sun that slope turns away from it and
    # goes dark; where it falls, the slope faces the sun and catches fire.  This
    # is the only thing that gives the deck form -- a low sun rakes cloud base
    # almost horizontally, so the gradient along the sun bearing is the whole
    # lighting model.
    eps = 0.55
    dp = fbm2((px + eps * SUN_H[0]) * F + 11.0, (py + eps * SUN_H[1]) * F - 7.0, _DECK, 3)
    dm = fbm2((px - eps * SUN_H[0]) * F + 11.0, (py - eps * SUN_H[1]) * F - 7.0, _DECK, 3)
    form = np.clip(0.5 - (dp - dm) * 2.6, 0.0, 1.0)

    fade = np.exp(-rad / 6.0).astype(np.float32)
    cover = sstep(-0.06, 0.26, dens) * (0.30 + 0.70 * fade)
    cover *= sstep(0.050, 0.26, e)          # deck base ~13 deg up; clear band below
    cover *= 0.92

    thick = np.clip(dens * 1.4 + 0.30, 0.0, 1.0)
    lit = (0.34 + 0.66 * form) * (0.55 + 0.45 * np.clip(cah * 0.5 + 0.5, 0, 1))
    lit = np.clip(lit * (1.0 - 0.42 * thick) + 0.26 * fade * np.clip(cah, 0, 1), 0.0, 1.0)
    cc = np.where(lit[..., None] < 0.5,
                  mix(CLOUD_COOL, CLOUD_MID, np.clip(lit * 2.0, 0, 1)),
                  mix(CLOUD_MID, CLOUD_HOT, np.clip((lit - 0.5) * 2.0, 0, 1)))
    # incandescent rim where the deck thins to nothing
    rim = (1.0 - sstep(0.00, 0.22, dens)) * sstep(-0.06, 0.12, dens)
    cc = cc + (rim * 0.55 * np.clip(cah * 0.5 + 0.55, 0, 1))[..., None] * np.array([0.70, 0.40, 0.16])
    cc = mix(cc, hor, (1.0 - fade) * 0.82)              # far deck dissolves into the band
    col = mix(col, cc, cover)

    # ---- cirrus, five times higher, drawn out along the wind ---------------
    q = 5.2
    cx, cy = px / q, py / q
    wx = cx * 0.94 - cy * 0.34
    wy = cx * 0.34 + cy * 0.94
    # Anisotropy 6.2 -> 2.9, plus a warp.  Parallel streaks on a horizontal deck really
    # do converge at the zenith -- that is correct perspective, not a bug -- but at 6.2:1
    # they were straight, high-contrast and perfectly regular, and the result read as a
    # stray god-ray decal pinned to a point with no sun in it.  Softer anisotropy and a
    # domain warp keep the perspective and lose the starburst.
    wwx = wx + 0.55 * fbm2(wx * 0.42 - 3.1, wy * 0.38 + 7.4, _DECK[2:], 2)
    wwy = wy + 0.40 * fbm2(wx * 0.36 + 9.7, wy * 0.44 - 5.2, _DECK[1:], 2)
    cn = fbm2(wwx * 1.15 + 40.0, wwy * 2.9 - 12.0, _CIRR, 4)
    cfade = np.exp(-rad / (q * 7.0)).astype(np.float32)
    ccov = sstep(0.10, 0.46, cn) * cfade * sstep(0.09, 0.36, e) * 0.50
    ccol = mix(CIRRUS_COOL, CIRRUS_HOT, np.clip(cah * 0.5 + 0.5, 0, 1) ** 1.5)
    col = mix(col, ccol, ccov * (1.0 - cover * 0.92))   # deck sits in front of cirrus

    # ---- baked far ranges: continuous horizon depth for zero geoms ---------
    # The hfield ranges are islands with gaps between them; this is the
    # unbroken range behind them, so a gap reads as recession, not as a hole.
    for lut, base, amp, tint, hazemix, dark in (
        (_RIDGE_A, 0.0130, 0.0470, RIDGE_FAR, 0.72, 1.00),
        (_RIDGE_B, 0.0035, 0.0230, RIDGE_MID, 0.55, 0.94),
    ):
        top = base + amp * sample_lut(az, lut)
        drop = np.clip((top - e) / (amp + 1e-9), 0.0, 1.0)
        body = mix(tint * dark * (0.82 + 0.26 * drop[..., None]), hor, hazemix)
        body = body + (np.clip(cah, 0, 1) ** 2 * 0.11)[..., None] * np.array([0.34, 0.20, 0.06])
        band = sstep(0.0, 0.0030, top - e) * sstep(-0.028, -0.004, e)
        col = mix(col, body, band)

    # ---- below the horizon: the infinite floor covers this, keep it warm ---
    col = mix(col, GROUNDHAZE * 0.86, sstep(0.0, 0.045, -e))

    # ---- hue-preserving shoulder ------------------------------------------
    # Clipping per channel turns the bloom into a white hole.  Compressing the
    # MAX channel and rescaling all three keeps the sun gold at full saturation.
    col = np.clip(col, 0.0, None)
    knee = 0.86
    mx = col.max(-1)
    over = np.clip(mx - knee, 0.0, None)
    mx2 = np.minimum(mx, knee) + over / (1.0 + over / (1.0 - knee))
    col = col * (mx2 / np.maximum(mx, 1e-6))[..., None]
    return np.clip(col, 0.0, 1.0)


def build_sky():
    w = FACE * SS
    ax = (np.arange(w, dtype=np.float32) + 0.5) / w * 2.0 - 1.0
    s = np.broadcast_to(ax[None, :], (w, w))
    t = np.broadcast_to(ax[:, None], (w, w))
    out = {}
    for name, fn in FACES.items():
        d = fn(s, t).astype(np.float32)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        col = sky_colour(d)
        img = (col * 255.0 + 0.5).astype(np.uint8)
        img = img.reshape(FACE, SS, FACE, SS, 3).mean(axis=(1, 3))   # box prefilter
        img = img.astype(np.uint8)
        path = os.path.join(MODEL, f"{P}sky_{name}.png")
        Image.fromarray(img).save(path, optimize=True)
        out[name] = img
        print(f"  {P}sky_{name}.png  {os.path.getsize(path)/1e6:.2f} MB  "
              f"mean={img.mean():6.1f} std={img.std():5.1f}")
    return out


# --------------------------------------------------------------------------
# ALBEDO for the distant masses and the ranges.
# Both are seen at 9-26 m through a 3.4 px/deg policy camera, so they must be
# LOW frequency and LOW contrast or they alias into noise (no mipmapping here).
# --------------------------------------------------------------------------
def build_distant():
    n = 512
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    base = fbm_tile(x, y, 3, 4, 700) * 0.5 + 0.5
    patch = fbm_tile(x, y, 2, 2, 740) * 0.5 + 0.5
    grain = fbm_tile(x, y, 16, 3, 780) * 0.5 + 0.5

    # DIRECTION-NEUTRAL only.  The previous version baked horizontal courses and a
    # vertical pilaster rhythm into this map.  On a box, MuJoCo's per-face UVs put the
    # image's y axis along the world Z on two side faces and along a horizontal world
    # axis on the other two -- so "horizontal courses" rendered as VERTICAL stripes on
    # half of every mass, which is the smearing that made the town read as cardboard.
    # A distant facade may not carry any axis-aligned signal at all; the courses are
    # re-introduced as GEOMETRY in skyline() instead, where they orient correctly.
    #
    # What is left is isotropic: patchy render, blotched repair work, and a soft
    # weathering gradient that is radial rather than vertical so it too has no axis.
    blotch = fbm_tile(x, y, 6, 3, 812) * 0.5 + 0.5
    stain = np.clip((fbm_tile(x, y, 3, 2, 851) * 0.5 + 0.5 - 0.42) * 2.2, 0, 1)
    rx, ry = x - 0.5, y - 0.5
    radial = np.sqrt(rx * rx + ry * ry) * 1.9
    wash = np.clip(radial, 0, 1) ** 2.0 * 0.055

    v = (0.780 + 0.050 * (base - 0.5) * 2.0 + 0.045 * (patch - 0.5) * 2.0
         + 0.030 * (blotch - 0.5) * 2.0 + 0.018 * (grain - 0.5) * 2.0)
    v = v * (1.0 - wash) * (1.0 - 0.045 * stain)
    col = np.stack([v * 1.025, v * 0.995, v * 0.978], -1)
    img = (np.clip(col, 0, 1) * 255).astype(np.uint8)
    path = os.path.join(MODEL, f"{P}distant.png")
    Image.fromarray(img).save(path, optimize=True)
    print(f"  {P}distant.png   {os.path.getsize(path)/1e6:.2f} MB  std={img.std():.1f}")


def build_ridge_tex():
    """WARM HAZED HILL.  The first cut was neutral grey spanning 21 grey levels
    (min 160, max 181, std 3.7) under a material at emission 2.45 over a blue-violet
    rgba == so the ranges self-illuminated at 2.45x, could never take warmth from the
    scene, had nothing left to modulate them, and read as SNOW-CAPPED GLACIERS over an
    arid terracotta court.  They were the most eye-catching element in the establishing
    shot and they were in the wrong genre entirely.

    This is now a warm arid hillside: base near #C9A48C, going warmer and paler toward
    the base where it meets the horizon haze band, low-frequency rock massing at a real
    value range (>80 levels, not 21), and no cool hue anywhere in it."""
    n = 512
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    big = fbm_tile(x, y, 2, 4, 810) * 0.5 + 0.5
    med = fbm_tile(x, y, 5, 3, 830) * 0.5 + 0.5
    fine = fbm_tile(x, y, 12, 3, 850) * 0.5 + 0.5
    haze = np.clip(y, 0, 1) ** 1.5                 # base of the range == palest
    v = (0.690 + 0.185 * (big - 0.5) * 2.0 + 0.085 * (med - 0.5) * 2.0
         + 0.030 * (fine - 0.5) * 2.0)
    v = v * (1.0 - 0.16 * (1.0 - haze))            # crests darker, base paler
    scrub = sstep(0.44, 0.64, big) * 0.075
    WARM = np.array([1.000, 0.812, 0.688], np.float32)   # #C9A48C family
    col = v[..., None] * WARM
    col = col - scrub[..., None] * np.array([0.90, 0.72, 0.52], np.float32)
    img = (np.clip(col, 0, 1) * 255).astype(np.uint8)
    path = os.path.join(MODEL, f"{P}ridge.png")
    Image.fromarray(img).save(path, optimize=True)
    print(f"  {P}ridge.png     {os.path.getsize(path)/1e6:.2f} MB  std={img.std():.1f}")


# --------------------------------------------------------------------------
# HFIELD RANGES.  Islands, not tiles: every field is windowed to zero on all
# four edges so it sinks below the floor plane at its own rim.  That sidesteps
# the per-field [0,1] renormalisation trap entirely (nothing has to line up
# with anything) and leaves the gaps for the baked sky range to fill.
# --------------------------------------------------------------------------
def build_hfield(name, seed, ncol, nrow, crest_bias=0.62, rough=1.0, gain=1.0):
    """rows = y, cols = x.  Ridge runs along +x, crest partway across y.

    Returns INLINE elevation text rather than writing a PNG.  robot_allcollisions_cam.xml
    sets compiler meshdir="assets", and hfield `file` resolves against meshdir while
    texture `file` does not -- so a bare hfield filename that works in a standalone
    test fails the moment the robot is included ("Error opening file
    'assets/atmosphere_hf_n.png'").  Inline data has no path to get wrong, in any
    composition.  MuJoCo renormalises every hfield to [0,1] anyway, and this data
    already spans exactly 0..1, so the renormalisation is the identity.
    """
    u = (np.arange(ncol) + 0.5) / ncol          # along the range
    v = (np.arange(nrow) + 0.5) / nrow          # across it, 0 = court side
    U, V = np.meshgrid(u, v)
    lats = [lattice(seed + i, 64) for i in range(5)]

    az = U * 2.0 * np.pi
    prof = sample_lut(az.ravel(), ridge_lut(seed + 91, nmax=14, power=1.30)).reshape(U.shape)
    prof = 0.42 + 0.58 * prof

    across = np.exp(-((V - crest_bias) ** 2) / (2 * 0.22 ** 2))
    edge_u = sstep(0.0, 0.13, U) * sstep(0.0, 0.13, 1.0 - U)
    edge_v = sstep(0.0, 0.10, V) * sstep(0.0, 0.16, 1.0 - V)

    crag = fbm2(U.astype(np.float32) * 11.0, V.astype(np.float32) * 5.0, lats, 5)
    h = prof * across * (1.0 + 0.42 * rough * crag) * edge_u * edge_v
    h = np.clip(h, 0.0, None)
    h[0, :] = 0.0
    h[-1, :] = 0.0
    h[:, 0] = 0.0
    h[:, -1] = 0.0
    h = h / (h.max() + 1e-9)                     # explicit 0..1 -> normalisation is a no-op
    # `gain` < 1 flattens the profile without changing the hfield's z size, which is what
    # takes the plateau off hf_w: MuJoCo renormalises every field to [0,1], so a raw
    # scale would be undone -- the shape has to change, not the amplitude.
    if gain != 1.0:
        h = h ** (1.0 / gain)
        h = h / (h.max() + 1e-9)
    txt = "\n".join(" ".join(f"{v:.3f}" for v in row) for row in h)
    print(f"  hf_{name}: {nrow}x{ncol} inline, {len(txt)/1e3:.0f} kB")
    return txt


# --------------------------------------------------------------------------
# GEOMETRY
# --------------------------------------------------------------------------
def g(**kw):
    order = ["name", "type", "pos", "size", "euler", "material", "rgba",
             "contype", "conaffinity", "hfield", "group"]
    parts = [f'{k}="{kw[k]}"' for k in order if k in kw]
    parts += [f'{k}="{v}"' for k, v in kw.items() if k not in order]
    return "  <geom " + " ".join(parts) + "/>"


def f3(*v):
    return " ".join(f"{x:.4f}" for x in v)


class Build:
    def __init__(self):
        self.body = []
        self.n = 0
        self.be = 0.0
        self.cyl = 0

    def geom(self, gtype="box", be=1.0, **kw):
        kw["type"] = gtype
        kw.setdefault("contype", "0")
        kw.setdefault("conaffinity", "0")
        self.body.append(g(**kw))
        self.n += 1
        self.be += be
        if gtype in ("cylinder", "capsule"):
            self.cyl += 1

    def note(self, s):
        self.body.append(f"  <!-- {s} -->")


# ---- the emissive dressing ------------------------------------------------
NORTH_FACE = 3.268      # court-facing plane of the north (petrol) wall
SOUTH_FACE = -3.268     # court-facing plane of the south (rose) wall
GATE_FACE = 7.048       # court-facing plane of the east gate masses


def emissive(b, rng):
    # --- north wall curb: the shadow side's only legible feature ------------
    b.note("north (shadow) wall curb strip -- the near-band legibility anchor")
    x = -7.00
    i = 0
    while x < 6.95:
        ln = rng.uniform(1.05, 1.85)
        ln = min(ln, 7.02 - x)
        if ln < 0.16:
            break
        b.geom("box", name=f"{P}curb_n{i}",
               pos=f3(x + ln / 2, NORTH_FACE - 0.006, 0.0195),
               size=f3(ln / 2, 0.012, 0.0175),
               material=f"{P}emit")
        x += ln + rng.uniform(0.09, 0.22)
        i += 1

    # --- wall lamps.  Hood first so the amber always has a dark surround. ---
    def lamp(nm, px, py, pz, face, scale=1.0, mat=None):
        s = scale
        ny = -0.052 * face
        b.geom("box", name=f"{nm}_hood", pos=f3(px, py + ny * 0.55, pz + 0.052 * s),
               size=f3(0.055 * s, 0.030, 0.013 * s), material=f"{P}iron")
        b.geom("box", name=f"{nm}_back", pos=f3(px, py + ny * 0.30, pz),
               size=f3(0.048 * s, 0.016, 0.050 * s), material=f"{P}iron")
        b.geom("box", name=f"{nm}_lit", pos=f3(px, py + ny * 0.85, pz),
               size=f3(0.034 * s, 0.010, 0.038 * s), material=mat or f"{P}emit")

    b.note("north wall lamps -- just switched on")
    for k, px in enumerate(np.linspace(-6.0, 6.0, 5)):
        lamp(f"{P}lamp_n{k}", px + rng.uniform(-0.10, 0.10), NORTH_FACE, 0.108,
             face=1.0, scale=rng.uniform(0.90, 1.12))

    b.note("south wall lamps -- present but out-dominated by the sun")
    for k, px in enumerate(np.linspace(-5.2, 5.2, 3)):
        lamp(f"{P}lamp_s{k}", px + rng.uniform(-0.12, 0.12), SOUTH_FACE, 0.104,
             face=-1.0, scale=rng.uniform(0.88, 1.05), mat=f"{P}emit_warm")

    b.note("south wall curb -- short broken runs only, the sun owns this wall")
    for k, x0 in enumerate((-6.35, -2.60, 2.40)):
        ln = rng.uniform(0.55, 1.05)
        b.geom("box", name=f"{P}curb_s{k}", pos=f3(x0 + ln / 2, SOUTH_FACE + 0.006, 0.014),
               size=f3(ln / 2, 0.011, 0.0125), material=f"{P}emit_warm")

    # --- the gate: dark jambs flanking a blazing slot ----------------------
    b.note("east gate jamb lanterns -- they frame the light slot")
    for k, py in ((0, 0.63), (1, -0.63)):
        yy = py + (0.05 if py > 0 else -0.05)
        b.geom("box", name=f"{P}jamb{k}_brk", pos=f3(GATE_FACE - 0.030, yy, 0.298),
               size=f3(0.030, 0.011, 0.008), material=f"{P}iron")
        b.geom("box", name=f"{P}jamb{k}_hood", pos=f3(GATE_FACE - 0.062, yy, 0.322),
               size=f3(0.034, 0.034, 0.009), material=f"{P}iron")
        b.geom("box", name=f"{P}jamb{k}_lit", pos=f3(GATE_FACE - 0.062, yy, 0.272),
               size=f3(0.024, 0.024, 0.040), material=f"{P}emit")
        b.geom("box", name=f"{P}jamb{k}_pool", pos=f3(GATE_FACE - 0.062, yy, 0.0016),
               size=f3(0.085, 0.085, 0.0014), material=f"{P}emit_pool")

    b.note("gate threshold -- an inlaid light slot, low enough to read as set into")
    b.note("the stone; v6 stood four 22 mm loaves up and they dominated the gate")
    for k, py in enumerate(np.linspace(-0.40, 0.40, 3)):
        b.geom("box", name=f"{P}sill{k}", pos=f3(GATE_FACE + 0.015, py, 0.0068),
               size=f3(0.013, 0.115, 0.0055), material=f"{P}emit")

    # --- the beacon: the vanishing point of the hero vista ------------------
    b.note("THE BEACON -- terminus of the hero vista, dead centre on the horizon.")
    b.note("Lantern is EXPOSED between a dark collar and a dark hood: an iron cage")
    b.note("around it hides the emissive box completely and the beacon goes black.")
    bx, by = 8.62, 0.02
    b.geom("box", name=f"{P}bcn_base", pos=f3(bx, by, 0.052), size=f3(0.082, 0.082, 0.052),
           material=f"{P}far_c")
    b.geom("box", name=f"{P}bcn_mast", pos=f3(bx, by, 0.200), size=f3(0.034, 0.034, 0.150),
           material=f"{P}iron")
    b.geom("box", name=f"{P}bcn_collar", pos=f3(bx, by, 0.360), size=f3(0.070, 0.070, 0.014),
           material=f"{P}iron")
    b.geom("box", name=f"{P}bcn_lit", pos=f3(bx, by, 0.436), size=f3(0.058, 0.058, 0.062),
           material=f"{P}emit_hot")
    b.geom("box", name=f"{P}bcn_hood", pos=f3(bx, by, 0.512), size=f3(0.090, 0.090, 0.016),
           material=f"{P}iron")
    b.geom("box", name=f"{P}bcn_fin", pos=f3(bx, by, 0.548), size=f3(0.030, 0.030, 0.022),
           material=f"{P}iron")


# ---- the distant skyline ---------------------------------------------------
# Every mass sits at r >= 12.6 m.  shadowclip 1.8 * extent 6.5 = 11.7 m, so they
# fall outside the shadow frustum and cast nothing.  That is deliberate: at 11 deg
# sun elevation a 3 m mass throws a 15 m shadow, which inside the court would
# flatten the whole floor into shade and destroy the key light.
# The distant town, laid out as CLUSTERS rather than evenly spaced solos.
# (cluster azimuth deg, radius, n masses, has tower)  -- v2 spaced them evenly
# and the result read as a modern skyline of isolated slabs; a town is a low,
# horizontal, clumped band with the occasional tower breaking out of it.
#
# Radii start at 13 m.  shadowclip 1.8 * extent 6.5 = 11.7 m, so nothing here
# enters the shadow frustum.  That is deliberate: at 11 deg sun elevation a 3 m
# mass throws a 15 m shadow, and one of those laid across the court would
# flatten the floor into shade and destroy the key light entirely.
#
# Heights clear the walls: from the 0.12 m eye a 0.58 m wall at 3.35 m subtends
# 7.8 deg, so a mass at radius r needs h > 0.12 + 0.137 r to break the skyline.
# Azimuth +-8 deg carries nothing: that is the light slot, and it stays sky.
CLUSTERS = [
    (-63, 27.0, 3, True),
    (-41, 34.5, 2, False),
    (-24, 25.0, 3, True),
    (12, 36.0, 2, False),
    (33, 26.0, 3, True),
    (58, 32.0, 2, False),
    (88, 27.5, 3, True),
    (126, 35.0, 2, False),
    (155, 25.5, 3, True),
    (-172, 37.0, 2, False),
    (-133, 28.5, 3, True),
]
BANDMAT = [f"{P}far_a", f"{P}far_b", f"{P}far_c"]   # a = palest / furthest



def detail_mass(b, rng, tag, cx, cy, yaw, w, dp, h, mat, wmat, zbase=None, dense=1.0):
    """One built mass with a roofline, openings and clutter.

    Factored out because the mid tier (r 9-16 m) was still the original three-geom
    recipe -- plinth, box, cap -- while the skyline behind it had been given
    archetypes and windows.  The mid tier is the CLOSER of the two and therefore the
    larger in frame, so the blandest buildings in the scene were the ones the eye
    lands on first.  `dense` scales the window module: nearer masses want more, finer
    openings, because at 10 m the eye can resolve them.
    """
    hw, hd = w / 2, dp / 2
    zb = h * 0.055 if zbase is None else zbase
    E = f"0 0 {yaw:.4f}"

    b.geom("box", name=f"{tag}_p", pos=f3(cx, cy, zb),
           size=f3(hw * 1.10, hd * 1.12, zb), euler=E, material=f"{P}far_base")
    b.geom("box", name=f"{tag}_b", pos=f3(cx, cy, h / 2 + zb),
           size=f3(hw, hd, h / 2), euler=E, material=mat)
    top = h + zb

    if rng.random() < 0.70:                      # string course
        b.geom("box", name=f"{tag}_s", pos=f3(cx, cy, zb + h * rng.uniform(0.42, 0.62)),
               size=f3(hw * 1.025, hd * 1.035, h * 0.018), euler=E, material=mat)

    kind = rng.choice(["parapet", "stepped", "shed", "eaved"], p=[0.30, 0.30, 0.18, 0.22])
    if kind == "parapet":
        for sx in (-1, 1):
            b.geom("box", name=f"{tag}_pa{sx}",
                   pos=f3(cx - sx * math.sin(yaw) * hd * 0.94,
                          cy + sx * math.cos(yaw) * hd * 0.94, top + h * 0.05),
                   size=f3(hw * 1.01, hd * 0.06, h * 0.05), euler=E, material=mat)
        b.geom("box", name=f"{tag}_rf", pos=f3(cx, cy, top + h * 0.012),
               size=f3(hw * 0.97, hd * 0.92, h * 0.012), euler=E, material=mat)
        top += h * 0.10
    elif kind == "stepped":
        uw, ud = hw * rng.uniform(0.50, 0.72), hd * rng.uniform(0.58, 0.80)
        uh = h * rng.uniform(0.24, 0.40)
        ox = rng.uniform(-0.30, 0.30) * (hw - uw)
        b.geom("box", name=f"{tag}_u",
               pos=f3(cx + math.cos(yaw) * ox, cy + math.sin(yaw) * ox, top + uh / 2),
               size=f3(uw, ud, uh / 2), euler=E, material=mat)
        b.geom("box", name=f"{tag}_uc",
               pos=f3(cx + math.cos(yaw) * ox, cy + math.sin(yaw) * ox,
                      top + uh + h * 0.024),
               size=f3(uw * 1.10, ud * 1.12, h * 0.024), euler=E, material=mat)
        top += uh + h * 0.05
    elif kind == "shed":
        pitch = rng.uniform(0.13, 0.24) * rng.choice([-1, 1])
        b.geom("box", name=f"{tag}_sh", pos=f3(cx, cy, top + h * 0.05),
               size=f3(hw * 1.05, hd * 1.12, h * 0.020),
               euler=f"{pitch:.4f} 0 {yaw:.4f}", material=mat)
        top += h * 0.09
    else:
        b.geom("box", name=f"{tag}_c", pos=f3(cx, cy, top + h * 0.035),
               size=f3(hw * rng.uniform(1.02, 1.06), hd * rng.uniform(1.08, 1.16),
                       h * 0.035), euler=E, material=mat)
        top += h * 0.07

    # openings, all four faces
    rows = int(max(1, round(h / (rng.uniform(0.34, 0.48) / dense))))
    cols = int(max(2, round(w / (rng.uniform(0.95, 1.40) / dense))))
    wzc = h / rows * 0.30
    for fs, axis in ((-1, "long"), (1, "long"), (-1, "end"), (1, "end")):
        if axis == "long":
            nx, ny = -math.sin(yaw) * fs, math.cos(yaw) * fs
            half, depth, tx, ty = hw, hd, math.cos(yaw), math.sin(yaw)
        else:
            nx, ny = math.cos(yaw) * fs, math.sin(yaw) * fs
            half, depth, tx, ty = hd, hw, -math.sin(yaw), math.cos(yaw)
        nc = cols if axis == "long" else max(2, int(round(cols * hd / hw)))
        for rr in range(rows):
            zz = zb + h * (rr + 0.62) / rows
            if zz > zb + h - h * 0.05:
                continue
            for cc in range(nc):
                if rng.random() < 0.16:
                    continue
                ox = (cc - (nc - 1) / 2.0) * (2 * half / nc)
                b.geom("box", name=f"{tag}_g{axis}{fs}_{rr}_{cc}",
                       pos=f3(cx + tx * ox + nx * (depth - 0.012),
                              cy + ty * ox + ny * (depth - 0.012), zz),
                       size=f3(half / nc * 0.34, 0.030, wzc),
                       euler=E if axis == "long" else f"0 0 {yaw + math.pi / 2:.4f}",
                       material=wmat)

    for q in range(rng.integers(0, 3)):
        cw = hw * rng.uniform(0.08, 0.17)
        chh = h * rng.uniform(0.05, 0.13)
        ox, oy = rng.uniform(-0.72, 0.72) * hw, rng.uniform(-0.45, 0.45) * hd
        b.geom("box", name=f"{tag}_k{q}",
               pos=f3(cx + math.cos(yaw) * ox - math.sin(yaw) * oy,
                      cy + math.sin(yaw) * ox + math.cos(yaw) * oy, top + chh),
               size=f3(cw, cw * rng.uniform(0.7, 1.3), chh), euler=E, material=mat)
    return top


def skyline(b, rng):
    b.note("the distant town -- clustered masses, all r >= 13 m so none of them")
    b.note("enters the 11.7 m shadow frustum and lays a 15 m shadow on the court")
    win = []
    gi = 0
    for ci, (azd, rad, count, tower) in enumerate(CLUSTERS):
        a0 = np.deg2rad(azd)
        band = 0 if rad > 33.0 else (1 if rad > 27.0 else 2)
        mat = BANDMAT[band]
        for k in range(count):
            # spread the cluster ALONG the horizon, and stagger its depth so the
            # masses overlap and read as one settlement rather than a picket line
            spread = (k - (count - 1) / 2.0) * rng.uniform(0.050, 0.078)
            a = a0 + spread
            r = rad + rng.uniform(-2.6, 2.6)
            w = rng.uniform(5.0, 9.0)               # wide and low: a town, not towers
            dp = rng.uniform(3.0, 4.5)
            hmin = 0.12 + 0.137 * r                 # the wall-clearance line
            h = hmin * rng.uniform(1.02, 1.22)      # they only just break the wall line
            cx, cy = r * np.cos(a), r * np.sin(a)
            yaw = a + rng.uniform(-0.45, 0.45)
            hw, hd = w / 2, dp / 2
            # ARCHETYPE.  Every mass used to be the same three geoms -- plinth, box,
            # oversailing cap 1.09x/1.14x on all four sides.  An oversail on all four
            # sides is not a roof, it is a lid, and twenty-five identical lids is what
            # made the town read as cardboard.  Real rooflines differ, so pick one of
            # five and let the silhouette carry the variety the texture no longer can.
            kind = rng.choice(["parapet", "stepped", "shed", "wing", "eaved"],
                              p=[0.28, 0.24, 0.18, 0.16, 0.14])
            E = f"0 0 {yaw:.4f}"

            b.geom("box", name=f"{P}sky{gi}_p", pos=f3(cx, cy, h * 0.035),
                   size=f3(hw * 1.07, hd * 1.10, h * 0.035), euler=E,
                   material=f"{P}far_base")
            zb = h * 0.035
            b.geom("box", name=f"{P}sky{gi}_b", pos=f3(cx, cy, h / 2 + zb),
                   size=f3(hw, hd, h / 2), euler=E, material=mat)
            top = h + zb

            # a string course at roughly floor height: the horizontal signal that used
            # to live (wrongly) in the texture, now as geometry so it orients correctly
            if rng.random() < 0.62:
                zs = zb + h * rng.uniform(0.44, 0.60)
                b.geom("box", name=f"{P}sky{gi}_s", pos=f3(cx, cy, zs),
                       size=f3(hw * 1.022, hd * 1.03, h * 0.016), euler=E, material=mat)

            if kind == "parapet":
                # the wall carries past the roof on the two LONG sides only, and the
                # roof plane sits below it -- reads as a flat roof behind a parapet
                for sx in (-1, 1):
                    b.geom("box", name=f"{P}sky{gi}_pa{sx}",
                           pos=f3(cx - sx * math.sin(yaw) * hd * 0.94,
                                  cy + sx * math.cos(yaw) * hd * 0.94,
                                  top + h * 0.055),
                           size=f3(hw * 1.01, hd * 0.06, h * 0.055), euler=E, material=mat)
                b.geom("box", name=f"{P}sky{gi}_rf", pos=f3(cx, cy, top + h * 0.012),
                       size=f3(hw * 0.97, hd * 0.92, h * 0.012), euler=E, material=mat)
                top += h * 0.11

            elif kind == "stepped":
                # a smaller upper storey set well back: the classic massing break
                uw, ud = hw * rng.uniform(0.52, 0.72), hd * rng.uniform(0.60, 0.80)
                uh = h * rng.uniform(0.26, 0.42)
                ox = rng.uniform(-0.30, 0.30) * (hw - uw)
                b.geom("box", name=f"{P}sky{gi}_u",
                       pos=f3(cx + math.cos(yaw) * ox, cy + math.sin(yaw) * ox,
                              top + uh / 2),
                       size=f3(uw, ud, uh / 2), euler=E, material=mat)
                b.geom("box", name=f"{P}sky{gi}_uc",
                       pos=f3(cx + math.cos(yaw) * ox, cy + math.sin(yaw) * ox,
                              top + uh + h * 0.026),
                       size=f3(uw * 1.10, ud * 1.12, h * 0.026), euler=E, material=mat)
                top += uh + h * 0.05

            elif kind == "shed":
                # single-pitch roof, faked as a thin slab tilted about the long axis
                pitch = rng.uniform(0.13, 0.24) * rng.choice([-1, 1])
                b.geom("box", name=f"{P}sky{gi}_sh",
                       pos=f3(cx, cy, top + h * 0.055),
                       size=f3(hw * 1.05, hd * 1.12, h * 0.020),
                       euler=f"{pitch:.4f} 0 {yaw:.4f}", material=mat)
                top += h * 0.10

            elif kind == "wing":
                # an L: a lower wing pushed out to one side, so the plan is not a bar
                wl = hw * rng.uniform(0.40, 0.62)
                wh = h * rng.uniform(0.52, 0.74)
                sx = rng.choice([-1, 1])
                ox = sx * (hw + wl * 0.92)
                b.geom("box", name=f"{P}sky{gi}_w",
                       pos=f3(cx + math.cos(yaw) * ox, cy + math.sin(yaw) * ox,
                              zb + wh / 2),
                       size=f3(wl, hd * rng.uniform(0.72, 0.94), wh / 2),
                       euler=E, material=mat)
                b.geom("box", name=f"{P}sky{gi}_wc",
                       pos=f3(cx + math.cos(yaw) * ox, cy + math.sin(yaw) * ox,
                              zb + wh + h * 0.022),
                       size=f3(wl * 1.10, hd * 0.86, h * 0.022), euler=E, material=mat)

            else:  # "eaved" -- the old cap, but asymmetric and much shallower
                b.geom("box", name=f"{P}sky{gi}_c", pos=f3(cx, cy, top + h * 0.038),
                       size=f3(hw * rng.uniform(1.02, 1.06),
                               hd * rng.uniform(1.08, 1.16), h * 0.038),
                       euler=E, material=mat)
                top += h * 0.076

            # OPENINGS.  A regular grid of recessed dark boxes on the long faces.
            # Set 12 mm INTO the wall so the reveal catches the low sun on one cheek --
            # a flush black rectangle reads as a decal, a recessed one reads as a hole.
            wmat = f"{P}win_{'cba'[band]}"
            rows = int(max(1, round(h / rng.uniform(0.36, 0.52))))
            cols = int(max(2, round(w / rng.uniform(1.05, 1.55))))
            wzc = h / rows * 0.30
            # All four faces. Glazing only the long pair left every side-on mass in the
            # frame blank, which is worse than none at all -- the eye reads the blank
            # ones as a different, wrong kind of object.
            for fs, axis in ((-1, "long"), (1, "long"), (-1, "end"), (1, "end")):
                if axis == "long":
                    nx, ny = -math.sin(yaw) * fs, math.cos(yaw) * fs
                    half, depth = hw, hd
                else:
                    nx, ny = math.cos(yaw) * fs, math.sin(yaw) * fs
                    half, depth = hd, hw
                for rr in range(rows):
                    zz = zb + h * (rr + 0.62) / rows
                    if zz > top - h * 0.06:
                        continue
                    nc = cols if axis == "long" else max(2, int(round(cols * hd / hw)))
                    for cc in range(nc):
                        if rng.random() < 0.18:      # not every bay is glazed
                            continue
                        ox = (cc - (nc - 1) / 2.0) * (2 * half / nc)
                        if axis == "long":
                            tx, ty = math.cos(yaw), math.sin(yaw)
                        else:
                            tx, ty = -math.sin(yaw), math.cos(yaw)
                        px = cx + tx * ox + nx * (depth - 0.012)
                        py = cy + ty * ox + ny * (depth - 0.012)
                        b.geom("box", name=f"{P}sky{gi}_g{axis}{fs}_{rr}_{cc}",
                               pos=f3(px, py, zz),
                               size=f3(half / nc * 0.34, 0.030, wzc),
                               euler=E if axis == "long" else f"0 0 {yaw + math.pi/2:.4f}",
                               material=wmat)

            # roof clutter: at 25-37 m this is pure silhouette, and it is what stops a
            # roofline being a ruled line. Cheap -- one or two boxes per mass.
            for q in range(rng.integers(0, 3)):
                cw = hw * rng.uniform(0.07, 0.16)
                chh = h * rng.uniform(0.05, 0.13)
                ox = rng.uniform(-0.72, 0.72) * hw
                oy = rng.uniform(-0.45, 0.45) * hd
                b.geom("box", name=f"{P}sky{gi}_k{q}",
                       pos=f3(cx + math.cos(yaw) * ox - math.sin(yaw) * oy,
                              cy + math.sin(yaw) * ox + math.cos(yaw) * oy,
                              top + chh),
                       size=f3(cw, cw * rng.uniform(0.7, 1.3), chh), euler=E, material=mat)

            if band <= 1 and rng.random() < 0.55:
                win.append((cx, cy, h, hd))
            gi += 1
        if tower:
            a = a0 + rng.uniform(-0.05, 0.05)
            r = rad + rng.uniform(-1.4, 1.4)
            hmin = 0.12 + 0.137 * r
            th = hmin * rng.uniform(1.45, 1.85)
            cx, cy = r * np.cos(a), r * np.sin(a)
            yaw = a + rng.uniform(-0.4, 0.4)
            tw = rng.uniform(1.05, 1.70)
            b.geom("box", name=f"{P}twr{ci}", pos=f3(cx, cy, th / 2),
                   size=f3(tw, tw * 0.86, th / 2), euler=f"0 0 {yaw:.4f}", material=mat)
            b.geom("box", name=f"{P}twr{ci}_c", pos=f3(cx, cy, th + 0.13),
                   size=f3(tw * 1.24, tw * 1.08, 0.13), euler=f"0 0 {yaw:.4f}", material=mat)
            b.geom("box", name=f"{P}twr{ci}_f", pos=f3(cx, cy, th + 0.40),
                   size=f3(tw * 0.30, tw * 0.26, 0.14), euler=f"0 0 {yaw:.4f}", material=mat)
            win.append((cx, cy, th, tw))

    b.note("MID-GROUND, r = 9-16 m.  The bible section 8 asked for 12-18 masses at 9-15 m")
    b.note("and nobody built them: a radial histogram of the shipped scene found 228 geoms")
    b.note("at r 6-8, SIX at 8-12 and ZERO at 12-20, so the court sat alone in a flat")
    b.note("12-metre void with nothing between its wall at 3.35 m and a skyline at 20 m.")
    b.note("")
    b.note("RADII ARE A HARD CONSTRAINT, not a taste call, and the first cut of this tier")
    b.note("got it wrong in a way worth recording.  shadowclip 1.30 x extent 6.5 puts the")
    b.note("directional shadow frustum at +/-8.45 m about the model centre, so a mass is")
    b.note("only safely outside it once max(|x|,|y|) > 8.45 -- which at the worst azimuth")
    b.note("(45 deg) needs r >= 8.45*sqrt(2) = 11.95 m.  Placed at r = 9.6-15 the tier")
    b.note("""DID cast: a 3 m mass throws a 15.4 m shadow at 11 deg, and the measured""")
    b.note("""result on the hero vista was the entire near band flattened into uniform""")
    b.note("""shade -- contrast std 21.4 -> 9.7, mean 111 -> 91.  So the tall tier starts""")
    b.note("""at 12.5 m; the low tier that fills 8.8-11.5 m is confined to the ANTI-SOLAR""")
    b.note("""half of the horizon, where its shadow falls away from the court anyway.""")
    MID = [(-118, 13.4, 2.35), (-96, 15.2, 2.90), (-71, 12.8, 1.70), (-52, 14.6, 2.55),
           (-33, 12.6, 1.35), (-19, 16.1, 3.20), (24, 13.8, 2.10), (41, 12.9, 1.55),
           (63, 15.6, 3.05), (86, 13.1, 1.90), (112, 14.2, 2.65), (148, 12.7, 2.20),
           (166, 16.4, 3.40), (-156, 13.6, 2.45),
           # low tier, anti-solar half only (the sun bears 40 deg, so anything between
           # 140 and 300 deg throws its shadow away from the court)
           (152, 9.4, 0.72), (188, 10.6, 0.95), (221, 9.0, 0.58),
           (254, 11.2, 1.05), (286, 9.8, 0.80)]
    for k, (azd, rr, hh) in enumerate(MID):
        a = np.deg2rad(azd) + rng.uniform(-0.04, 0.04)
        r = rr + rng.uniform(-0.7, 0.7)
        cx, cy = r * np.cos(a), r * np.sin(a)
        yaw = a + rng.uniform(-0.55, 0.55)
        w, dp = rng.uniform(1.9, 4.2), rng.uniform(1.5, 3.0)
        m = f"{P}far_c"
        # a plinth wider than the mass: the critique found the distant masses terminating
        # in a clean flat cut with ground visible beneath and no contact darkening, so
        # they read as pasted on.  A darker, wider base course anchors them.
        # These are the CLOSEST masses in the scene, so they get the finest window
        # module (dense 1.35) -- at 10 m the eye resolves openings the skyline can
        # only imply.
        detail_mass(b, rng, f"{P}mid{k}", cx, cy, yaw, w, dp, hh, m,
                    f"{P}win_c", dense=1.35)
    b.body.append("")

    b.note("far landmark lights -- the only thing at 20 m that survives 320x240")
    rng.shuffle(win)
    for i, (cx, cy, h, hd) in enumerate(win[:8]):
        inward = np.arctan2(-cy, -cx)               # always face the court
        zz = h * rng.uniform(0.58, 0.88)
        px = cx + (hd * 1.04) * np.cos(inward)
        py = cy + (hd * 1.04) * np.sin(inward)
        b.geom("box", name=f"{P}fwin{i}", pos=f3(px, py, zz),
               size=f3(0.10, 0.10, 0.135), euler=f"0 0 {inward:.4f}",
               material=f"{P}emit_far")


# --------------------------------------------------------------------------
def main():
    print(f"model dir: {MODEL}")
    print("skybox:")
    build_sky()
    print("albedo:")
    build_distant()
    build_ridge_tex()
    print("hfields:")
    HF = {
        "n": (build_hfield("n", 4100, 96, 42, crest_bias=0.58, rough=1.05), 42, 96),
        "s": (build_hfield("s", 4200, 96, 42, crest_bias=0.54, rough=0.95), 42, 96),
        "e": (build_hfield("e", 4300, 42, 96, crest_bias=0.50, rough=0.70), 96, 42),
        # hf_w's profile saturated to a plateau and produced a hard horizontal top with
        # vertical cliffs -- a flat-topped iceberg mesa.  crest_bias moved off the rim and
        # rough cut, so the ridge line stops clipping flat.
        "w": (build_hfield("w", 4400, 42, 96, crest_bias=0.52, rough=0.88,
                           gain=0.74), 96, 42),
    }

    # ---------------- assets ----------------
    A = []
    A.append('<texture name="%ssky" type="skybox"' % P)
    A.append('         fileright="%ssky_right.png" fileleft="%ssky_left.png"' % (P, P))
    A.append('         fileup="%ssky_up.png"       filedown="%ssky_down.png"' % (P, P))
    A.append('         filefront="%ssky_front.png" fileback="%ssky_back.png"/>' % (P, P))
    A.append('<texture name="%stex_distant" type="2d" file="%sdistant.png"/>' % (P, P))
    A.append('<texture name="%stex_ridge"   type="2d" file="%sridge.png"/>' % (P, P))
    A.append("")
    # Three recession bands.  Authored aerial perspective: value up, chroma down,
    # shifted cool with distance.  Fog is unreachable in the viewer, so this is
    # the depth cue and it must survive with mjRND_FOG = 0.
    # AERIAL PERSPECTIVE IS DRIVEN BY `emission`, NOT BY TINT ALONE.
    # A pale tint alone does not recede: the sun still rakes the far masses and
    # drops their shadow sides to a dark saturated blue, which reads as NEAR.
    # emission adds a flat, light-independent floor to each band, so contrast
    # collapses with distance the way haze actually makes it collapse.  It is
    # also what makes the art bible's negative test (sun switched off) pass --
    # the depth banding is baked in and survives with no key light at all.
    # Base rgba drops as emission rises so the sunlit faces never clip.
    # WARM, AND MONOTONIC.  The first cut drove recession with `emission` over a COLD
    # blue-violet albedo (far_c 0.450 0.430 0.440 at 0.40, far_a 0.300 0.315 0.385 at
    # 1.55).  Two consequences the critique caught: the added light was cold, so a distant
    # mass could never take the sky's warmth and one band came out brighter AND cooler than
    # the sky behind it, reading as backlit glass; and the nearest band was so much darker
    # than the furthest that the ladder ran backwards, with near clusters going dark navy.
    # The mechanism is kept -- a flat additive term IS how haze collapses contrast, and it
    # is what makes the negative test pass with the sun off -- but the chromaticity now
    # tracks the horizon (1.00 0.845 0.585) and the three bands are solved so that
    # delivered luminance rises monotonically with distance and the SUNLIT face of the
    # furthest band still sits below the horizon sky.
    # WINDOWS.  The distant masses had no openings at all, which is why they read as
    # paper cutouts: nothing in the frame said how big they were.  Openings are the
    # cheapest scale cue there is.  They cannot live in the albedo (that map has to
    # stay direction-neutral, see build_distant), so they are geometry, and they get
    # their own per-band materials so aerial perspective still holds -- a hole 37 m
    # away is not as dark as one at 25 m.
    for nm, tint, emis in (("win_c", "0.118 0.104 0.094", "1.05"),
                           ("win_b", "0.104 0.092 0.086", "1.55"),
                           ("win_a", "0.092 0.083 0.079", "2.60")):
        A.append(f'<material name="{P}{nm}" rgba="{tint} 1" emission="{emis}"'
                 f' specular="0.10" shininess="0.14"/>')
    for nm, tint, emis, rp in (("far_c", "0.245 0.211 0.162", "1.80", "1.90 1.90"),
                               ("far_b", "0.222 0.193 0.155", "2.65", "1.70 1.70"),
                               ("far_a", "0.176 0.155 0.130", "4.40", "1.50 1.50")):
        A.append(f'<material name="{P}{nm}" rgba="{tint} 1" emission="{emis}"'
                 f' texrepeat="{rp}" texuniform="true" specular="0.04" shininess="0.03">')
        A.append(f'  <layer texture="{P}tex_distant" role="rgb"/>')
        A.append("</material>")
    # THE RANGES.  emission 2.45/2.65 over a blue-grey rgba landed them at roughly
    # (0.62 0.66 0.82) plus ambient -- flat, unlit, self-luminous slabs BRIGHTER than the
    # horizon sky and colder than it, with no form modelled by the sun at all.  Warm hue,
    # and the level dropped so they sit just under the haze band rather than punching
    # out of it.
    A.append(f'<material name="{P}range" rgba="0.166 0.145 0.120 1" emission="4.05"'
             f' texrepeat="0.10 0.10" texuniform="true" specular="0.02" shininess="0.02">')
    A.append(f'  <layer texture="{P}tex_ridge" role="rgb"/>')
    A.append("</material>")
    A.append(f'<material name="{P}range_far" rgba="0.158 0.140 0.120 1" emission="4.55"'
             f' texrepeat="0.08 0.08" texuniform="true" specular="0.02" shininess="0.02">')
    A.append(f'  <layer texture="{P}tex_ridge" role="rgb"/>')
    A.append("</material>")
    A.append(f'<material name="{P}far_base" rgba="0.196 0.166 0.136 1" emission="1.15"'
             f' texrepeat="2.4 2.4" texuniform="true" specular="0.03" shininess="0.03">')
    A.append(f'  <layer texture="{P}tex_distant" role="rgb"/>')
    A.append("</material>")
    A.append("")
    A.append(f'<material name="{P}iron" rgba="0.300 0.170 0.120 1" specular="0.30" shininess="0.35"/>')
    # Art bible colour 11 is rgba 1.00 0.84 0.56 at emission 0.95.  Rendered,
    # that lands at (242,203,136) BEFORE the scene light adds on top, which
    # reads as a white fluorescent tube, not a lamp just switched on.  Keeping
    # the bible's screen intent means dropping the base green/blue and the
    # emission until the strip renders as amber.
    A.append(f'<material name="{P}emit"      rgba="1.00 0.72 0.38 1" emission="0.84"/>')
    A.append(f'<material name="{P}emit_hot"  rgba="1.00 0.70 0.34 1" emission="0.96"/>')
    A.append(f'<material name="{P}emit_warm" rgba="1.00 0.66 0.30 1" emission="0.66"/>')
    A.append(f'<material name="{P}emit_far"  rgba="1.00 0.70 0.36 1" emission="0.90"/>')
    A.append(f'<material name="{P}emit_pool" rgba="1.00 0.60 0.26 1" emission="0.26"/>')
    A.append("")
    # islands: each windowed to zero at its own rim, so nothing has to line up
    for nm, sz in (("n", "40 16  13.5 1.0"), ("s", "40 16  12.2 1.0"),
                   ("e", "16 40   4.8 1.0"), ("w", "16 40  11.5 1.0")):
        txt, nrow, ncol = HF[nm]
        A.append(f'<hfield name="{P}hf_{nm}" nrow="{nrow}" ncol="{ncol}" size="{sz}"')
        A.append(f'        elevation="{txt}"/>')

    # ---------------- body ----------------
    b = Build()
    b.note("=== ATMOSPHERE ===  lights must be declared FIRST: only 7 render, first-N wins.")
    b.note("The three below are the art bible section 3 rig, verbatim and unmodified.")
    b.note("")
    b.note("DO NOT ADD A POSITIONAL LIGHT WITH castshadow=\"false\".  Measured on this")
    b.note("build: a spot or point light with castshadow=false drives TOTAL light to")
    b.note("zero on every geom outside its cone -- material `emission` included -- so")
    b.note("the whole far field renders pure black.  castshadow=\"true\" behaves; a")
    b.note("directional light with castshadow=false behaves.  The art bible's optional")
    b.note("east-gate spot was measured both ways and is NOT used: shadowless is broken,")
    b.note("and the shadow-casting version costs +2.3 ms of 3.9 ms at 320x240 for a pool")
    b.note("that is invisible against the emissive lamps already there.")
    b.body.append(
        '  <light name="aaa_sun" directional="true" pos="-6 -5 2" dir="-0.752 -0.631 -0.191"\n'
        '         diffuse="1.45 1.06 0.66" specular="0.38 0.32 0.22" castshadow="true"/>')
    b.body.append(
        '  <light name="aaa_skyfill" directional="true" pos="6 5 1.4" dir="0.70 0.59 -0.40"\n'
        '         diffuse="0.26 0.28 0.44" specular="0 0 0" castshadow="false"/>')
    b.body.append(
        '  <light name="aaa_bounce" directional="true" pos="-4 -3 0.05" dir="-0.60 -0.50 0.62"\n'
        '         diffuse="0.30 0.20 0.11" specular="0 0 0" castshadow="false"/>')
    b.body.append("")

    rng = np.random.default_rng(20260829)

    b.note("background ranges -- non-colliding islands, all beyond the shadow frustum")
    for nm, pos, mat in (("n", "0 52 -3.00", f"{P}range"),
                         ("s", "0 -52 -3.00", f"{P}range"),
                         ("e", "56 0 -3.00", f"{P}range_far"),
                         ("w", "-54 0 -3.00", f"{P}range")):
        b.geom("hfield", be=6.0, name=f"{P}range_{nm}", hfield=f"{P}hf_{nm}",
               pos=pos, material=mat)
    b.body.append("")

    skyline(b, rng)
    b.body.append("")
    emissive(b, rng)

    # ---------------- visual ----------------
    V = """<statistic extent="6.5" center="0 0 0.30"/>

<visual>
  <headlight diffuse="0 0 0" ambient="0.09 0.09 0.13" specular="0 0 0"/>
  <rgba haze="0.90 0.82 0.68 1" fog="0.86 0.78 0.66 1"/>
  <global azimuth="150" elevation="-8" offwidth="1920" offheight="1080"/>
  <!-- SHADOW RESOLUTION.  The uniform triangular sawteeth on the fountain's shadow are
       NOT the superimposed corner notches of concentric octagons, as the assembly report
       concluded, and they are not confined to the fountain: hiding the fountain simply
       moves the sawtooth onto whatever the next leading shadow edge is.  They are a
       shadow-map staircase on long silhouette edges lying nearly parallel to the light,
       and they scale exactly with texel size -- measured, four-way A/B at fixed camera:
           4096 / clip 1.8   ->  5.71 mm texel,  ~50 mm teeth   (shipped before)
           8192 / clip 1.8   ->  2.86 mm texel,  ~25 mm teeth
           8192 / clip 1.30  ->  2.06 mm texel,  ~18 mm teeth   <- shipped
          16384 / clip 1.30  ->  1.03 mm texel,  edge effectively clean
       Cost is flat (26.6 vs 26.6 ms at 1600x900: this machine is readback-bound), so the
       only real price is shadow-map memory, and 16384 would ask for ~800 MB.  clip 1.30 x
       extent 6.5 = 8.45 m still covers the court's own half-diagonal of 7.9 m.
       zfar 14 -> 17: the scenery AABB now spans 142 m and the worst-case eye-to-corner
       distance from inside the court is 106 m, against a 91 m far plane -- visible slices
       were being cut out of the west and north ranges.  17 x 6.5 = 110.5 m covers it.
       znear 0.003 -> 0.012 (78 mm): still 1.9x clear of the 148 mm of ground at the bottom
       of frame, and 2x clear of the 39 mm at which the duck's own jaw leaves the duck_eye
       frustum.  Raising it cuts the scene's segmentation-flip rate by a third. -->
  <quality shadowsize="8192" offsamples="8"/>
  <map znear="0.012" zfar="17" shadowclip="1.30" shadowscale="1.0"
       fogstart="0.9" fogend="3.2"/>
</visual>"""

    for fn, txt in ((f"aaa_{P[:-1]}_assets.xml", "\n".join(A)),
                    (f"aaa_{P[:-1]}_body.xml", "\n".join(b.body)),
                    (f"aaa_{P[:-1]}_visual.xml", V)):
        with open(os.path.join(MODEL, fn), "w") as fh:
            fh.write(txt + "\n")
        print(f"  wrote {fn}")

    print(f"\ngeoms={b.n}  BE={b.be:.0f}  cylinders={b.cyl}   "
          f"(budget: 140 geoms / 210 BE / 6 cyl)")


if __name__ == "__main__":
    main()
