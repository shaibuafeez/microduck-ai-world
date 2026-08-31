#!/usr/bin/env python3
"""THE WATER COURT - shared material library  (agent: "materials")

Writes, idempotently, into src/mjlab_microduck/robot/microduck/ :
    materials_tex_*.png            20 tileable 3-channel albedo maps
    aaa_materials_assets.xml       <texture>/<material> fragment (no wrappers)
    aaa_materials_body.xml         empty fragment (this agent spends 0 geoms)

Run with the sim venv:
    .venv-sim/bin/python scripts/aaa/materials.py


WHY THERE ARE NO NORMAL MAPS
----------------------------
The brief asked for normal maps.  This wheel ships only the classic OpenGL
renderer (`mujoco._render_filament` is absent), where `<layer role="normal">`,
`metallic` and `roughness` are provably inert - pixel maxdiff 0.0 against a
material without them.  The art bible bans them for that reason.

So every height field in this file is still built, and still differentiated
with a proper 3x3 Sobel operator - but the resulting surface normal is dotted
with the sun vector at *authoring* time and BAKED INTO THE ALBEDO.  That is the
only relief channel this renderer has.  Roughness is expressed through the two
levers that do move pixels: `specular` and `shininess`.

THE BAKE-DIRECTION RULE
-----------------------
Sun travels along (-0.752, -0.631, -0.191), so the vector TOWARD the sun is
L = (+0.752, +0.631, +0.191) (already unit).  Every baked highlight in this
file agrees with it.  Measured on this machine (scratchpad/mat/probe2.py):

    texture COLUMN index increases toward world +X
    texture ROW    index increases toward world -Y

hence for a height field h[row, col] the world-space normal is

    n = normalize(-dh/dcol, +dh/drow, 1)

The +dh/drow (not -) is the row-flip.  Getting this backwards is the exact
failure the bible calls out - walls turning to vertical fur.

COLOUR PIPELINE
---------------
MuJoCo's classic renderer treats textures as LINEAR (texture `colorspace`
resolves to LINEAR by default); it does no sRGB decode.  So a texel of 0.520 is
an albedo of 0.520.  The bible's sunlit swatches are exactly albedo x the
measured sun multiplier [1.605, 1.220, 0.900] - e.g. 0.520*1.605 = 0.835 = 0xD5
= the stated #D56835.  Therefore textures are written WITHOUT gamma encoding and
every one is renormalised so its mean equals its palette albedo exactly.
(The bible's own calibration PNGs were gamma-encoded to a mean of 170 against a
spec of 133; at that level the sun clips the red channel, which is the thing
pass-criterion 5 forbids.  Spec wins.)

Consequence for other agents: every textured material here carries
rgba="1 1 1 1" and holds its palette colour in the PNG.  Per-geom `rgba`
MULTIPLIES it (verified, scratchpad/mat/probe3.py), so the bible's
"+/-6% per-instance value tint" works by writing rgba="0.94 0.94 0.94 1" on
your geom.  Nothing double-tints.
"""

from __future__ import annotations

import os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(
    os.path.join(HERE, "..", "..", "src", "mjlab_microduck", "robot", "microduck")
)
PFX = "materials_tex_"

# Vector TOWARD the sun, world space. Unit. Azimuth 40 deg, elevation 11 deg.
SUN_L = np.array([0.752, 0.631, 0.191])

# ---------------------------------------------------------------------------
# palette (art bible section 2) - linear rgba, these are the delivered albedos
# ---------------------------------------------------------------------------
P_TERRACOTTA = np.array([0.520, 0.335, 0.232])   # 1  court flagstone
P_PLASTER    = np.array([0.620, 0.455, 0.290])   # 2  ochre plaster
P_STONE      = np.array([0.615, 0.567, 0.480])   # 3  bleached stone
P_DAMP       = np.array([0.620, 0.605, 0.595])   # 4  damp stone, bible literal
# DEVIATION, measured. Palette #4 at 0.620 sits 1.7x above the terracotta floor in
# G and 2.6x in B, so a puddle renders as a pale slab dropped on the pavement -
# exactly the "translucent ghost slab" the capability probe warned reflectance
# produces on light bases. The bible's own c2_B/c2_D calibration ran against a
# gamma-encoded floor of mean 170; against the spec-accurate floor (mean 133) the
# same relationship lands here. Delivered luminance is now 0.95x the dry floor:
# neutralised and damp, still nowhere near the black hole the bible rules out.
# Set P_WET = P_DAMP to restore the literal value.
P_WET        = np.array([0.400, 0.340, 0.325])
P_ROSE       = np.array([0.620, 0.388, 0.407])   # 6  Barragan rose (sun wall)
P_PETROL     = np.array([0.130, 0.220, 0.260])   # 7  deep petrol (shadow wall)
P_OLIVE      = np.array([0.230, 0.260, 0.150])   # 8  vegetation
P_IRON       = np.array([0.300, 0.170, 0.120])   # 9  iron oxide
P_LAPIS      = np.array([0.130, 0.200, 0.430])   # 10 the one saturated object
P_EMIT       = np.array([1.000, 0.840, 0.560])   # 11 amber emissive
P_TERRA      = np.array([0.400, 0.220, 0.160])   # deep terracotta (pots)

# extensions this agent adds (props/vegetation need them; all obey rule 1, R<=0.62)
P_WOOD       = np.array([0.355, 0.295, 0.235])   # weathered silvered timber
P_CANVAS     = np.array([0.505, 0.455, 0.360])   # unbleached sacking
P_PAINT      = np.array([0.340, 0.190, 0.145])   # red-lead paint over steel
P_SAND       = np.array([0.480, 0.400, 0.298])   # dust / spilled grain
P_RUIN       = np.array([0.560, 0.435, 0.300])   # plaster over exposed rubble
P_GLASS      = np.array([0.420, 0.500, 0.540])   # grimy lamp glass

# Aerial perspective target: cool, pale, and R capped so nothing clips (rule 1).
AERIAL = np.array([0.600, 0.680, 0.840])
# Bleached stone at >6 m. The bible's literal 0.77 0.74 0.70 would clip red
# (0.77*1.605 = 1.24); capped to 0.62 per pass-criterion 5, value still lifted.
P_STONE_FAR = 0.615 * np.array([0.995, 1.010, 1.060])


# ===========================================================================
# noise / relief library - everything below tiles seamlessly (FFT => periodic)
# ===========================================================================

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _radial(n: int) -> np.ndarray:
    f = np.fft.fftfreq(n)
    return np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)


def fnoise(n: int, exponent: float, seed: int) -> np.ndarray:
    """Seamless 1/f^exponent noise, unit std. High exponent = broad and smooth."""
    r = _rng(seed).normal(size=(n, n))
    f = _radial(n)
    f[0, 0] = 1e-6
    o = np.real(np.fft.ifft2(np.fft.fft2(r) * f ** (-exponent)))
    return (o - o.mean()) / (o.std() + 1e-9)


def bandnoise(n: int, cycles: float, seed: int, width: float = 0.75) -> np.ndarray:
    """Seamless noise concentrated at `cycles` repeats across the tile, unit std.

    Lets features be specified in real units: at a 0.90 m tile, cycles=30 is a
    3 cm feature. This is the anti-moire control - keep detail coarse.
    """
    r = _rng(seed).normal(size=(n, n))
    f = _radial(n) * n
    f[0, 0] = 1e-6
    o = np.real(np.fft.ifft2(np.fft.fft2(r) * np.exp(-(np.log(f / cycles) ** 2) / (2 * width * width))))
    return (o - o.mean()) / (o.std() + 1e-9)


def streaknoise(n: int, cycles: float, seed: int, stretch: float = 20.0,
                along: str = "col", width: float = 0.70) -> np.ndarray:
    """Seamless noise stretched into long parallel features, unit std.

    `along="col"` elongates features along the column axis (world +X);
    `along="row"` elongates them along the row axis (world -Y, i.e. downward on
    a vertical face). An isotropic blur cannot do this - it just makes blobs -
    so the anisotropy has to be applied in the frequency domain, by pushing the
    band away from the axis the feature runs along.
    """
    raw = _rng(seed).normal(size=(n, n))
    fy = np.fft.fftfreq(n)[:, None] * n
    fx = np.fft.fftfreq(n)[None, :] * n
    if along == "col":
        a = np.sqrt((fx * stretch) ** 2 + fy ** 2)
    else:
        a = np.sqrt(fx ** 2 + (fy * stretch) ** 2)
    a[0, 0] = 1e-6
    o = np.real(np.fft.ifft2(np.fft.fft2(raw) * np.exp(-(np.log(a / cycles) ** 2) / (2 * width ** 2))))
    return (o - o.mean()) / (o.std() + 1e-9)


def gblur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Periodic gaussian blur (used for cavity/AO extraction)."""
    n = a.shape[0]
    f = _radial(n)
    return np.real(np.fft.ifft2(np.fft.fft2(a) * np.exp(-2.0 * (np.pi * sigma) ** 2 * f ** 2)))


def sobel(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Periodic 3x3 Sobel. Returns (dh/dcol, dh/drow)."""
    def R(a, dy, dx):
        return np.roll(np.roll(a, dy, 0), dx, 1)
    gx = (R(h, -1, -1) + 2 * R(h, 0, -1) + R(h, 1, -1)
          - R(h, -1, 1) - 2 * R(h, 0, 1) - R(h, 1, 1)) / 8.0
    gy = (R(h, -1, -1) + 2 * R(h, -1, 0) + R(h, -1, 1)
          - R(h, 1, -1) - 2 * R(h, 1, 0) - R(h, 1, 1)) / 8.0
    return gx, gy


def relief(h: np.ndarray, slope: float, gain: float = 1.0, smooth: float = 0.0) -> np.ndarray:
    """Sun-matched baked shading from a height field. Mean stays ~1.0.

    Sobel -> tangent-space normal -> dot with the world sun vector, with the
    texture-row flip applied so the bake agrees with (-0.752, -0.631).
    Flat surface returns exactly 1.0, so this adds relief without shifting the
    material's average albedo.

    `smooth` low-passes the height field first, and matters more than anything
    else here: Sobel is a high-pass, so differentiating broadband noise dumps
    all its energy into the top octave, which is precisely the band that moires
    with no mipmapping. Relief must come from forms, not from grain.
    """
    if smooth > 0.0:
        h = gblur(h, smooth)
    gx, gy = sobel(h)
    nx = -gx * slope
    ny = gy * slope                    # row index runs toward -Y: flip
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + 1.0)
    lam = (nx * SUN_L[0] + ny * SUN_L[1] + SUN_L[2]) * inv
    return 1.0 + gain * (lam - SUN_L[2])


def cavity(h: np.ndarray, sigma: float, gain: float) -> np.ndarray:
    """Direction-neutral ambient occlusion: how far below the local mean a pixel
    sits. This is what made the bible's floor the best surface in any probe."""
    return np.clip((gblur(h, sigma) - h) * gain, 0.0, 1.0)


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def periodic_dist(a: np.ndarray, p: float, n: int) -> np.ndarray:
    d = np.abs(a - p) % n
    return np.minimum(d, n - d)


# ===========================================================================
# pattern generators
# ===========================================================================

def ashlar(n: int, ncourse: int, seed: int, wobble: float = 5.0,
           width_jit: float = 0.42) -> tuple[np.ndarray, np.ndarray]:
    """Irregular coursed paving. Returns (distance-to-joint px, stone id).

    Courses (rows) run across the tile with jittered heights; each course is
    broken into stones of jittered width whose cumulative widths sum exactly to
    n, so the pattern is toroidal. The sampling grid is displaced by very
    low-frequency noise, which makes every joint slightly wavy - straight
    machine-cut lines are the tell of programmer art, and hard thin lines
    alias badly with no mipmapping.
    """
    r = _rng(seed)
    rows, cols = np.mgrid[0:n, 0:n].astype(np.float32)
    rr = rows + bandnoise(n, 3.0, seed + 11) * wobble
    cc = cols + bandnoise(n, 3.0, seed + 12) * wobble

    ce = np.linspace(0.0, n, ncourse + 1)
    ce[1:-1] += r.normal(0.0, (n / ncourse) * 0.06, ncourse - 1)
    ce = np.sort(ce)
    ce[0], ce[-1] = 0.0, float(n)

    dv = np.full((n, n), 1e9, np.float32)
    for e in ce[:-1]:
        dv = np.minimum(dv, periodic_dist(rr, float(e), n))

    ci = np.clip(np.searchsorted(ce, np.mod(rr, n), side="right") - 1, 0, ncourse - 1)

    du = np.full((n, n), 1e9, np.float32)
    sid = np.zeros((n, n), np.int32)
    for c in range(ncourse):
        k = max(3, int(round(ncourse * r.uniform(0.8, 1.3))))
        w = 1.0 + r.uniform(-width_jit, width_jit, k)
        w = w / w.sum() * n
        pos = np.sort((np.concatenate([[0.0], np.cumsum(w)[:-1]]) + r.uniform(0, n)) % n)
        dcur = np.full((n, n), 1e9, np.float32)
        for p in pos:
            dcur = np.minimum(dcur, periodic_dist(cc, float(p), n))
        scur = np.mod(np.searchsorted(pos, np.mod(cc, n), side="right") - 1, k)
        m = ci == c
        du = np.where(m, dcur, du)
        sid = np.where(m, c * 149 + scur, sid)
    return np.minimum(du, dv), sid


def voronoi(n: int, cells: int, seed: int, jitter: float = 0.75):
    """Toroidal Worley. Returns (F2-F1 in px, cell id) - used for rubble and
    for glaze crazing."""
    r = _rng(seed)
    step = n / cells
    px, py = [], []
    for i in range(cells):
        for j in range(cells):
            px.append((i + 0.5 + (r.random() - 0.5) * jitter) * step)
            py.append((j + 0.5 + (r.random() - 0.5) * jitter) * step)
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float32)
    f1 = np.full((n, n), 1e9, np.float32)
    f2 = np.full((n, n), 1e9, np.float32)
    cid = np.zeros((n, n), np.int32)
    for idx in range(len(px)):
        dx = periodic_dist(xs, px[idx], n)
        dy = periodic_dist(ys, py[idx], n)
        d = np.sqrt(dx * dx + dy * dy)
        closer = d < f1
        f2 = np.where(closer, f1, np.minimum(f2, d))
        cid = np.where(closer, idx, cid)
        f1 = np.where(closer, d, f1)
    return f2 - f1, cid


def per_cell(cid: np.ndarray, seed: int, lo: float, hi: float) -> np.ndarray:
    """Random scalar per cell id, looked up per pixel."""
    tbl = _rng(seed).uniform(lo, hi, int(cid.max()) + 2)
    return tbl[cid]


# ===========================================================================
# output helper - exact palette match, then report
# ===========================================================================

_STATS: list[tuple[str, np.ndarray, float]] = []


def save(name: str, rgb: np.ndarray, target: np.ndarray | None, std: float | None = None) -> None:
    """Clip, trim contrast to `std`, force the mean to the palette albedo, write PNG.

    `std` is the 8-bit luminance standard deviation. The bible's anti-moire law
    is 13-20 for the ground and 4-8 for plaster; there is no mipmapping, so a
    texture over its ceiling degenerates into shimmer by the horizon. The trim
    scales every pixel's deviation from the material mean, so it lowers contrast
    without touching the spatial structure that gives the material its identity.
    """
    rgb = np.clip(rgb, 0.0, 1.0)
    if std is not None:
        mean = rgb.reshape(-1, 3).mean(0)
        k = 1.0
        for _ in range(8):
            trial = np.clip(mean + (rgb - mean) * k, 0.0, 1.0)
            cur = (trial * 255.0).mean(2).std()
            if abs(cur - std) < 0.15:
                break
            k *= std / max(cur, 1e-6)
        rgb = np.clip(mean + (rgb - mean) * k, 0.0, 1.0)
    if target is not None:
        for _ in range(6):                       # converges through the clip
            cur = rgb.reshape(-1, 3).mean(0)
            rgb = np.clip(rgb * (target / np.maximum(cur, 1e-6)), 0.0, 1.0)
    a = (rgb * 255.0 + 0.5).astype(np.uint8)
    path = os.path.join(MODEL_DIR, PFX + name + ".png")
    Image.fromarray(a).save(path, optimize=True)
    f = a.astype(np.float64)
    _STATS.append((name, f.reshape(-1, 3).mean(0), float(f.mean(2).std())))


# ===========================================================================
# 1. GROUND - the hero asset. Half of every frame is this texture.
# ===========================================================================

def _ground_fields(n: int):
    """Shared geometry for the dry and damp floor so the two register."""
    d, sid = ashlar(n, ncourse=7, seed=101, wobble=5.5, width_jit=0.44)

    jw = 7.5                                     # joint half width, px (~6.6 mm)
    g = smoothstep(d / jw)                       # 0 in the groove, 1 on the stone

    # height: stone body, gently crowned, chewed back at the arrises
    h = g.copy()
    h += 0.22 * bandnoise(n, 11.0, 111, 0.5) * g  # slow surface undulation
    h += 0.07 * bandnoise(n, 40.0, 112, 0.45) * g # tooling / pick marks
    chip = np.clip(bandnoise(n, 22.0, 113) * 1.25 - 0.30, 0.0, 1.0)
    h -= chip * np.clip(1.0 - d / (jw * 1.9), 0.0, 1.0) * 0.40      # broken corners
    h = np.clip(h, 0.0, 1.3)
    return d, sid, g, h, jw


def make_ground(n: int = 1024) -> None:
    d, sid, g, h, jw = _ground_fields(n)

    # --- value -------------------------------------------------------------
    ao = 0.60 + 0.40 * g                          # direction-neutral joint AO
    ao *= 1.0 - 0.26 * cavity(h, 4.0, 2.0)        # deepen chips and pits
    shoulder = relief(h, slope=12.0, gain=0.46, smooth=1.1)   # sun-matched arris highlight

    # joint sand: wind-blown grit half-fills the grooves, so they are not a
    # uniform black line. This is the difference between "tiled" and "paved".
    sand = np.clip(bandnoise(n, 9.0, 121) * 0.75 + 0.42, 0.0, 1.0) * (1.0 - g)
    # foot polish: broad low-frequency burnished tracks
    polish = fnoise(n, 2.3, 131) * 0.055
    # iron leach and damp bloom, both very low frequency
    leach = np.clip(fnoise(n, 2.0, 141) * 0.5 + 0.5, 0, 1)
    grain = bandnoise(n, 190.0, 151) * 0.022

    v = ao * (1.0 + polish) * shoulder + sand * 0.30 + grain
    v *= 1.0 + 0.048 * per_cell(sid, 161, -1.0, 1.0)          # per-stone value

    # a couple of spalled stones - lighter, rougher, dark rim
    spall = (per_cell(sid, 171, 0.0, 1.0) > 0.88).astype(np.float64)
    v *= 1.0 + spall * (0.10 + 0.07 * bandnoise(n, 45.0, 172, 0.35))

    # --- colour ------------------------------------------------------------
    fired = np.array([0.470, 0.278, 0.196])       # over-fired, redder
    pale = np.array([0.575, 0.405, 0.296])        # under-fired, ochre
    t = per_cell(sid, 181, 0.0, 1.0)[..., None]
    base = fired[None, None, :] * (1 - t) + pale[None, None, :] * t

    rgb = base * v[..., None]
    rgb *= (1.0 - 0.16 * (leach * sand)[..., None] * np.array([0.10, 0.30, 0.45]))  # grey grit
    rgb *= 1.0 + 0.05 * (leach - 0.5)[..., None] * np.array([1.0, -0.4, -1.0])      # iron stain
    save("ground", rgb, P_TERRACOTTA, std=17.5)

    # --- damp twin ---------------------------------------------------------
    # Same joints, same stones, so a puddle geom laid over the floor reads as
    # the SAME pavement gone wet. Water fills and blurs the grooves, kills the
    # micro relief and desaturates hard toward Damp Stone, so joint contrast is
    # deliberately low - which also makes any UV misregistration invisible.
    gw = smoothstep(d / (jw * 2.4))
    vw = (0.86 + 0.14 * gw)
    vw *= 1.0 - 0.12 * cavity(h, 6.0, 1.6)
    vw *= relief(h, slope=13.0, gain=0.09, smooth=1.6)                     # water flattens it
    vw *= 1.0 + 0.028 * per_cell(sid, 161, -1.0, 1.0)
    vw += bandnoise(n, 130.0, 191, 0.35) * 0.008
    film = np.clip(fnoise(n, 2.1, 193) * 0.5 + 0.5, 0, 1)      # thicker/thinner water
    wet = base * 0.52 + P_WET[None, None, :] * 0.48            # still visibly the same stone
    wet = wet * (1.0 - 0.09 * film[..., None]) + 0.09 * film[..., None] * np.array([0.44, 0.46, 0.50])
    save("ground_damp", wet * vw[..., None], P_WET, std=8.0)


# ===========================================================================
# 2. PLASTER - low frequency ONLY, per the bible. No vertical detail, ever.
# ===========================================================================

def _plaster_rgb(n: int, seed: int, base: np.ndarray, amp: float = 1.0) -> np.ndarray:
    wash = fnoise(n, 2.5, seed) * 0.052 * amp          # lime wash mottle
    wash += fnoise(n, 1.7, seed + 1) * 0.026 * amp
    patch = smoothstep(np.abs(bandnoise(n, 2.2, seed + 2)) * 0.9)   # old repairs
    wash += (patch - 0.5) * 0.030 * amp
    grain = bandnoise(n, 260.0, seed + 3) * 0.011 * amp

    # soft float-coat undulation, differentiated and lit by the real sun so the
    # surface has form instead of being a flat field
    h = fnoise(n, 2.6, seed + 4)
    v = 1.0 + wash + grain
    v *= relief(h, slope=2.2, gain=0.30 * amp)

    cool = np.clip(fnoise(n, 2.2, seed + 5) * 0.5 + 0.5, 0, 1)      # lime bloom
    tint = np.array([1.0, 1.0, 1.0]) + (cool - 0.5)[..., None] * np.array([-0.06, 0.0, 0.10]) * amp
    return base[None, None, :] * v[..., None] * tint


def make_plaster(n: int = 1024) -> None:
    save("plaster", _plaster_rgb(n, 201, P_PLASTER), P_PLASTER, std=6.5)

    # authored aerial perspective: value up, chroma down, shifted cool.
    # Reduced contrast too - distance eats contrast before it eats colour.
    m = n // 2
    mid_t = 0.90 * P_PLASTER + 0.10 * AERIAL
    far_t = 0.78 * P_PLASTER + 0.22 * AERIAL
    save("plaster_mid", _plaster_rgb(m, 201, mid_t, amp=0.72), mid_t, std=5.0)
    save("plaster_far", _plaster_rgb(m, 201, far_t, amp=0.46), far_t, std=3.4)


# ===========================================================================
# 3. STONE - bleached limestone. Capitals, plinths, copings, fountain rim.
# ===========================================================================

def _stone_rgb(n: int, base: np.ndarray, amp: float = 1.0) -> np.ndarray:
    """Worked limestone: mostly smooth dressed face, sparse deep vugs, a few
    hairline cracks, broad weathering drift. The smoothness is the point - a
    stonemason's face is flat, and the eye reads the rare defect, not an even
    sprinkle of them."""
    seed = 301
    # dominant read at every distance: slow bedding drift in the block
    form = gblur(fnoise(n, 2.7, seed) * 1.0 + bandnoise(n, 6.0, seed + 1, 0.45) * 0.35, 2.5)

    # Defect density is the thing that separates stone from lichen. Every defect
    # below is thresholded HARD (>=1.7 sigma) and then gated by a broad mask, so
    # damage gathers in a few places and most of the face stays clean.
    vug = smoothstep((bandnoise(n, 18.0, seed + 2, 0.30) - 1.80) / 0.35)
    vug *= smoothstep((fnoise(n, 2.5, seed + 9) * 0.5 + 0.5 - 0.64) / 0.14)
    # No crack network. At texrepeat 1.8 one tile is 0.55 m, so a capital shows
    # half a tile - a meandering crack system at that scale reads as lichen, not
    # as stone. Dressed limestone is smooth; the drift and the sugary grain carry
    # it, and the rare pit is the only defect the eye should find.
    crack = np.zeros_like(form)
    # chipped arrises: short bites where the block has been knocked
    bite = smoothstep((bandnoise(n, 20.0, seed + 3, 0.32) - 1.60) / 0.35)
    bite *= smoothstep((fnoise(n, 2.4, seed + 8) * 0.5 + 0.5 - 0.60) / 0.16)

    h = form - vug * 1.4 - crack * 0.40 - bite * 1.0
    ao = 1.0 - 0.44 * cavity(h, 3.5, 1.7) * amp
    lit = relief(h, slope=5.0, gain=0.45 * amp, smooth=1.3)
    tooth = bandnoise(n, 90.0, seed + 5, 0.45) * 0.016 * amp     # sugary grain
    tooth += bandnoise(n, 22.0, seed + 11, 0.50) * 0.020 * amp   # broad tonal drift
    weather = fnoise(n, 2.5, seed + 6) * 0.055 * amp             # exposure bleaching

    v = ao * lit * (1.0 + weather) + tooth
    dirt = np.clip(vug * 0.9 + crack * 0.45 + bite * 0.5, 0, 1) * amp
    warm = np.clip(fnoise(n, 2.1, seed + 7) * 0.5 + 0.5, 0, 1)
    rgb = base[None, None, :] * v[..., None]
    rgb *= 1.0 - dirt[..., None] * np.array([0.20, 0.24, 0.30])
    rgb *= 1.0 + (warm - 0.5)[..., None] * np.array([0.08, 0.03, -0.06]) * amp
    return rgb


def make_stone(n: int = 1024) -> None:
    save("stone", _stone_rgb(n, P_STONE), P_STONE, std=11.0)
    save("stone_far", _stone_rgb(n // 2, P_STONE_FAR, amp=0.40), P_STONE_FAR, std=4.5)


# ===========================================================================
# 4. ROSE - the SUN wall. One blazing plane; grime must rise from its base.
#    Authored with texuniform="false" texrepeat="14 1": v spans the wall height
#    exactly once, u repeats 14x along its 14.2 m run.
# ===========================================================================

def _wall_v(n: int) -> np.ndarray:
    """0 at the wall BASE, 1 at its top.

    MEASURED: the texture row axis runs down a vertical face - row 0 is the top
    of the geom, not the bottom. Without this flip the splash zone renders under
    the coping and the drip stain sits on the ground, which is exactly backwards.
    """
    return 1.0 - (np.arange(n)[:, None] / (n - 1.0)) * np.ones((1, n))


def make_rose(n: int = 1024) -> None:
    seed = 401
    vv = _wall_v(n)
    ragged = bandnoise(n, 4.0, seed, 0.45) * 0.085 + bandnoise(n, 11.0, seed + 1, 0.4) * 0.030

    splash = 1.0 - smoothstep((vv + ragged - 0.02) / 0.30)      # rain splash zone
    drip = smoothstep((vv + ragged * 0.6 - 0.82) / 0.20)        # dirt under coping
    bleach = smoothstep((vv - 0.45) / 0.55)                     # sun-bleached top

    h = gblur(fnoise(n, 2.7, seed + 2), 2.0) * 1.0
    h += bandnoise(n, 22.0, seed + 3, 0.35) * 0.18
    v = relief(h, slope=2.4, gain=0.32, smooth=1.2)
    v *= 1.0 + fnoise(n, 2.4, seed + 4) * 0.045                 # render mottle
    v += bandnoise(n, 240.0, seed + 5) * 0.010

    bloom = np.clip(fnoise(n, 2.3, seed + 6) * 0.5 + 0.5, 0, 1)  # lime efflorescence
    v *= 1.0 + 0.08 * bleach - 0.20 * splash - 0.13 * drip

    rgb = P_ROSE[None, None, :] * v[..., None]
    # splash and drip desaturate toward grey dirt; bleaching pushes it pinker
    grime = np.clip(splash * 0.9 + drip * 0.7, 0, 1)[..., None]
    rgb = rgb * (1 - 0.34 * grime) + grime * 0.34 * np.array([0.36, 0.31, 0.30]) * v[..., None]
    rgb *= 1.0 + (bloom - 0.5)[..., None] * np.array([0.05, 0.07, 0.09])
    save("rose", rgb, P_ROSE, std=11.0)


# ===========================================================================
# 5. PETROL - the SHADOW wall. Salt bloom is the only thing that will read in
#    near-black shade, so it is doing the heavy lifting for pass-criterion 2
#    ("< 2% of pixels below luminance 25").
# ===========================================================================

def make_petrol(n: int = 1024) -> None:
    seed = 501
    vv = _wall_v(n)
    rising = 1.0 - smoothstep((vv + bandnoise(n, 6.0, seed) * 0.14 - 0.02) / 0.46)

    salt = np.clip(fnoise(n, 2.4, seed + 1) * 0.5 + 0.5, 0, 1)
    salt = smoothstep((salt - 0.42) / 0.30) * (0.35 + 0.65 * rising)
    salt = np.clip(salt + bandnoise(n, 20.0, seed + 2) * 0.10, 0, 1)

    h = fnoise(n, 2.5, seed + 3) + bandnoise(n, 26.0, seed + 4) * 0.30
    h += salt * 0.5                                             # crust stands proud
    v = relief(h, slope=3.4, gain=0.40)
    v *= 1.0 + fnoise(n, 2.2, seed + 5) * 0.06
    v += bandnoise(n, 200.0, seed + 6) * 0.012

    rgb = P_PETROL[None, None, :] * v[..., None]
    # efflorescent crust: pale, slightly warm, up to ~2.4x the base value
    crust = np.array([0.315, 0.345, 0.340])
    rgb = rgb * (1 - salt[..., None] * 0.85) + salt[..., None] * 0.85 * crust[None, None, :] * v[..., None]
    save("petrol", rgb, P_PETROL, std=10.0)


# ===========================================================================
# 6. TERRA - thrown terracotta. Ridges run AROUND the pot, so they are banded
#    in v and their shading is up/down only: direction-neutral in XY, which is
#    the only bake that stays honest on a cylinder.
# ===========================================================================

def make_terra(n: int = 512) -> None:
    seed = 601
    # MEASURED (scratchpad/mat/uvprobe.py): on BOTH cylinders and boxes, and for
    # texuniform either way, the texture ROW axis runs along the geom's HEIGHT
    # (increasing downward) and the COLUMN axis runs horizontally - around the
    # circumference of a cylinder. So banding in rows is what puts the throw
    # ridges around the pot.
    rows = np.arange(n)[:, None] * np.ones((1, n))
    wob = bandnoise(n, 3.0, seed, 0.45) * 7.0 + bandnoise(n, 9.0, seed + 1, 0.4) * 2.5
    t = ((rows + wob) * (13.0 / n)) % 1.0                      # 13 throw ridges
    # crisp four-step profile, not a sine: the capability probe found hard bands
    # read as genuine relief where smooth gradients read as painted stripes
    prof = np.where(t < 0.10, 1.0, np.where(t < 0.46, 0.55,
                    np.where(t < 0.62, -0.85, 0.05)))
    h = gblur(prof, 1.4) * 0.55 + bandnoise(n, 30.0, seed + 2, 0.35) * 0.12

    gx, gy = sobel(gblur(h, 1.0))
    up = np.clip(-gy * 11.0, -1, 1)                            # along the pot's height
    v = 1.0 + up * 0.13                                        # direction-neutral in XY
    v *= 1.0 - 0.32 * cavity(h, 3.0, 2.0)
    v *= 1.0 + fnoise(n, 2.3, seed + 3) * 0.07                 # clay body variation
    v += bandnoise(n, 120.0, seed + 4, 0.35) * 0.010

    cloud = np.clip(fnoise(n, 2.2, seed + 5) * 0.5 + 0.5, 0, 1)   # kiln fire-clouding
    cloud = smoothstep((cloud - 0.50) / 0.34)
    lime = np.clip(fnoise(n, 2.6, seed + 6) * 0.5 + 0.5, 0, 1)
    lime = smoothstep((lime - 0.62) / 0.22) * smoothstep((rows / n - 0.5) / 0.5)

    rgb = P_TERRA[None, None, :] * v[..., None]
    rgb *= 1.0 - cloud[..., None] * np.array([0.26, 0.30, 0.28])      # smoke shadow
    rgb = rgb * (1 - lime[..., None] * 0.45) + lime[..., None] * 0.45 * np.array([0.46, 0.44, 0.40])
    save("terra", rgb, P_TERRA, std=13.0)


# ===========================================================================
# 7. OLIVE - vegetation. Silhouette does the work; this only stops the leaves
#    reading as flat cut-outs.
# ===========================================================================

def make_olive(n: int = 512) -> None:
    seed = 701
    h = bandnoise(n, 26.0, seed) + bandnoise(n, 60.0, seed + 1) * 0.5
    v = relief(h, slope=4.5, gain=0.55)
    v *= 1.0 + fnoise(n, 2.2, seed + 2) * 0.10
    v *= 1.0 - 0.25 * cavity(h, 5.0, 1.4)

    # a scatter of leaves turned to catch the low sun - pushed olive/khaki,
    # never toward the mint-aqua or saturated yellow reserved for the duck
    flash = np.clip(bandnoise(n, 30.0, seed + 3) * 1.4 - 0.60, 0, 1)
    dust = np.clip(fnoise(n, 2.1, seed + 4) * 0.5 + 0.5, 0, 1)

    rgb = P_OLIVE[None, None, :] * v[..., None]
    rgb = rgb * (1 - flash[..., None] * 0.55) + flash[..., None] * 0.55 * np.array([0.360, 0.355, 0.180])
    rgb *= 1.0 + (dust - 0.5)[..., None] * np.array([0.16, 0.12, 0.10])      # court dust
    save("olive", rgb, P_OLIVE, std=14.0)


# ===========================================================================
# 8. IRON - rusted ironwork. Rust streaks + bright wear on the edges.
# ===========================================================================

def make_iron(n: int = 1024) -> None:
    """Wrought iron gone to rust. The read is: broad blooms of orange oxide over
    dark scale, lifting in discrete plates, with the sound metal still showing
    through in the flat areas. Big forms first, grain last and quiet."""
    seed = 801
    # scale plates: real oxide lifts in irregular flakes, not in noise
    edge, cid = voronoi(n, 9, seed, jitter=0.95)
    plate = smoothstep(edge / 9.0)                              # 0 at a plate edge
    lift = per_cell(cid, seed + 1, 0.0, 1.0)
    # flaking is LOCAL - it starts where water sits and spreads. Gating the
    # per-cell lift by a broad mask is what stops this reading as lizard skin.
    lifted = smoothstep((lift - 0.62) / 0.20)
    lifted *= smoothstep((fnoise(n, 2.4, seed + 7) * 0.5 + 0.5 - 0.50) / 0.22)

    # rust blooms: broad, soft, clustered - the dominant colour event
    bloom = np.clip(fnoise(n, 2.5, seed + 2) * 0.5 + 0.5, 0, 1)
    rust = smoothstep((bloom - 0.44) / 0.26)
    rust = np.clip(rust + lifted * 0.70, 0, 1)                  # flakes are always rusty

    pit = smoothstep((bandnoise(n, 24.0, seed + 3, 0.30) - 2.15) / 0.35)
    pit *= smoothstep((fnoise(n, 2.4, seed + 8) * 0.5 + 0.5 - 0.48) / 0.22)

    h = gblur(fnoise(n, 2.6, seed + 4), 3.0) * 0.8              # the forged form
    h += lifted * plate * 0.45 - (1 - plate) * lifted * 0.45    # proud flakes, sunk edges
    h -= pit * 1.3
    ao = 1.0 - 0.34 * cavity(h, 4.0, 1.5)
    lit = relief(h, slope=6.5, gain=0.50, smooth=1.4)
    v = ao * lit * (1.0 + fnoise(n, 2.3, seed + 5) * 0.10)
    v += bandnoise(n, 130.0, seed + 6, 0.35) * 0.012

    rgb = P_IRON[None, None, :] * v[..., None]
    rust_c = np.array([0.430, 0.205, 0.105])
    rgb = rgb * (1 - rust[..., None] * 0.80) + rust[..., None] * 0.80 * rust_c[None, None, :] * v[..., None]
    # bright wear on the proud edges, rubbed back to bare metal. This is what
    # the specular 0.30 / shininess 0.35 actually has to catch.
    wear = smoothstep((relief(h, slope=6.5, gain=1.0, smooth=1.4) - 1.045) / 0.030)
    wear *= (1 - pit) * (1 - rust * 0.7)
    rgb = rgb * (1 - wear[..., None] * 0.55) + wear[..., None] * 0.55 * np.array([0.44, 0.42, 0.41])
    save("iron", rgb, P_IRON, std=17.0)


# ===========================================================================
# 9. PAINT - red-lead painted steel, chipping back to rust. (extension)
# ===========================================================================

def make_paint(n: int = 1024) -> None:
    """Red-lead over steel, failing. Paint does not erode evenly - it loses whole
    flakes with hard edges, and the loss clusters where water sits. So the chip
    mask is a Voronoi cell set gated by a broad wetness mask, not a threshold on
    noise: that difference is the whole material."""
    seed = 901
    edge, cid = voronoi(n, 17, seed, jitter=0.95)
    gone = per_cell(cid, seed + 1, 0.0, 1.0)
    cluster = smoothstep((fnoise(n, 2.4, seed + 2) * 0.5 + 0.5 - 0.55) / 0.16)
    chip = smoothstep((gone - 0.62) / 0.05) * cluster           # crisp flake edges
    chip = smoothstep((gblur(chip, 0.9) - 0.42) / 0.22)         # keep the edge hard

    h = -chip * 1.2 + gblur(fnoise(n, 2.6, seed + 4), 3.0) * 0.35   # film + panel form
    ao = 1.0 - 0.36 * cavity(h, 3.0, 2.0)
    lit = relief(h, slope=6.0, gain=0.40, smooth=1.2)
    v = ao * lit * (1.0 + fnoise(n, 2.4, seed + 5) * 0.075)
    v += bandnoise(n, 140.0, seed + 6, 0.35) * 0.008

    chalk = np.clip(fnoise(n, 2.2, seed + 7) * 0.5 + 0.5, 0, 1)     # UV-chalked paint
    paint = P_PAINT[None, None, :] * (1.0 + (chalk - 0.5)[..., None] * np.array([0.16, 0.18, 0.22]))
    steel = np.array([0.270, 0.190, 0.150])                         # rusty substrate
    # rust weeps a little way out from under the paint edge
    weep = np.clip(gblur(chip, 3.0) * 1.8 - chip, 0, 1)
    rgb = (paint * (1 - chip[..., None]) + steel[None, None, :] * chip[..., None]) * v[..., None]
    rgb = rgb * (1 - weep[..., None] * 0.45) + weep[..., None] * 0.45 * \
        np.array([0.330, 0.170, 0.095]) * v[..., None]
    save("paint", rgb, P_PAINT, std=16.0)


# ===========================================================================
# 10. LAPIS - the ONE saturated object in 16x16 m. Glaze crazing.
# ===========================================================================

def make_lapis(n: int = 512) -> None:
    seed = 1001
    edge, cid = voronoi(n, 26, seed, jitter=0.9)               # crackle network
    craze = 1.0 - smoothstep(edge / 2.6)
    fine, _ = voronoi(n, 52, seed + 1, jitter=0.9)
    craze = np.clip(craze + (1.0 - smoothstep(fine / 1.8)) * 0.45, 0, 1)

    pool = fnoise(n, 2.5, seed + 2)                            # glaze thickness
    h = pool * 0.8 - craze * 0.9
    v = relief(h, slope=5.0, gain=0.45)
    v *= 1.0 - 0.30 * cavity(h, 3.0, 2.0)
    v *= 1.0 + pool * 0.09
    pin = np.clip(bandnoise(n, 70.0, seed + 3) * 1.5 - 0.95, 0, 1)   # glaze pinholes
    v *= 1.0 - pin * 0.35

    # hue drifts blue -> violet where the glaze pools thick. Never toward mint.
    hue = np.clip(pool * 0.5 + 0.5, 0, 1)[..., None]
    deep = np.array([0.105, 0.150, 0.430])
    light = np.array([0.170, 0.255, 0.445])
    rgb = (deep * (1 - hue) + light * hue) * v[..., None]
    rgb *= 1.0 - craze[..., None] * np.array([0.30, 0.34, 0.22])     # dirt in the crackle
    save("lapis", rgb, P_LAPIS, std=16.0)


# ===========================================================================
# 11. WOOD - weathered, silvered timber. Crates, ladder, cart, broom.
# ===========================================================================

def make_wood(n: int = 1024) -> None:
    """Weathered sawn boards. Five things carry it: the board gaps, cupping
    across each board, coarse cathedral grain, proud latewood raised by
    weathering, and a few rust-bled nails. The grain is deliberately coarse -
    fine grain is invisible on a 5 cm crate slat and shimmers at distance."""
    seed = 1101
    rows, cols = np.mgrid[0:n, 0:n].astype(np.float32)
    nplank = 5
    pw = n / nplank
    pid = np.floor(rows / pw).astype(np.int32)
    tp = (rows % pw) / pw                                       # 0..1 across a board
    gap = 1.0 - smoothstep(np.minimum(rows % pw, pw - (rows % pw)) / 5.0)
    cup = -np.cos(2 * np.pi * tp) * 0.5 + 0.5                   # boards cup hollow

    # cathedral grain: long lines running the length of the board
    grain = streaknoise(n, 40.0, seed, stretch=22.0, along="col", width=0.80)

    # per-board offset so grain does not run continuously across the gaps
    off = _rng(seed + 5).uniform(0, n, nplank + 1)[pid].astype(np.int32)
    idx = (cols.astype(np.int32) + off) % n
    ridx = rows.astype(np.int32)
    grain = grain[ridx, idx]
    late = smoothstep((np.abs(grain) - 1.05) / 0.75)            # latewood bands

    split = np.clip(1.0 - np.abs(bandnoise(n, 3.0, seed + 7, 0.4)) * 4.0, 0, 1) ** 2
    split *= smoothstep((fnoise(n, 2.4, seed + 8) * 0.5 + 0.5 - 0.55) / 0.20)

    h = late * 0.7 - gap * 4.0 - split * 1.8 - cup * 0.9 + grain * 0.30
    ao = 1.0 - 0.50 * cavity(h, 4.5, 1.3)
    lit = relief(h, slope=6.0, gain=0.46, smooth=1.5)           # raised weathered grain
    v = ao * lit
    v *= 1.0 + per_cell(pid, seed + 9, -0.085, 0.085)           # each board differs
    v *= 1.0 + fnoise(n, 2.3, seed + 10) * 0.07
    v += bandnoise(n, 130.0, seed + 11, 0.35) * 0.008

    silver = np.clip(fnoise(n, 2.1, seed + 12) * 0.5 + 0.5, 0, 1)   # UV-silvered face
    rgb = P_WOOD[None, None, :] * v[..., None]
    rgb *= 1.0 - late[..., None] * np.array([0.20, 0.24, 0.26])     # dark latewood
    rgb *= 1.0 + (silver - 0.5)[..., None] * np.array([-0.12, -0.05, 0.12])
    # nail heads with a rust bleed weeping downslope from them
    nail = smoothstep((bandnoise(n, 10.0, seed + 13, 0.26) - 2.35) / 0.22)
    bleed = np.clip(gblur(nail, 7.0) * 2.6, 0, 1)
    rgb = rgb * (1 - bleed[..., None] * 0.42) + \
        bleed[..., None] * 0.42 * np.array([0.300, 0.165, 0.105]) * v[..., None]
    rgb = rgb * (1 - nail[..., None] * 0.7) + nail[..., None] * 0.7 * \
        np.array([0.190, 0.150, 0.130]) * v[..., None]
    save("wood", rgb, P_WOOD, std=16.0)


# ===========================================================================
# 12. CANVAS - sacking / awning. Weave baked with a real Sobel normal.
# ===========================================================================

def make_canvas(n: int = 512) -> None:
    seed = 1201
    rows, cols = np.mgrid[0:n, 0:n].astype(np.float32)
    per = n / 30.0                                             # 30 threads per tile
    u = 2 * np.pi * cols / per
    v_ = 2 * np.pi * rows / per
    # plain weave: warp over weft alternating, so the height field is the max of
    # two out-of-phase thread bundles. Sobel of this is a real woven normal.
    warp = np.cos(u) * (0.5 + 0.5 * np.cos(v_))
    weft = np.cos(v_) * (0.5 + 0.5 * np.cos(u + np.pi))
    h = np.maximum(warp, weft) * 1.0
    h += bandnoise(n, 5.0, seed, 0.45) * 0.55                  # slack, sag and folds
    slub = smoothstep((bandnoise(n, 13.0, seed + 1, 0.32) - 0.95) / 0.40)
    h += slub * 0.7                                            # thick slubs in the yarn

    ao = 1.0 - 0.50 * cavity(h, 2.4, 1.6)
    lit = relief(h, slope=4.2, gain=0.58, smooth=0.6)
    v = ao * lit * (1.0 + fnoise(n, 2.4, seed + 2) * 0.09)

    # water staining with the classic darker tide rim
    st = np.clip(fnoise(n, 2.6, seed + 3) * 0.5 + 0.5, 0, 1)
    stain = smoothstep((st - 0.52) / 0.14)
    rim = np.clip(1.0 - np.abs(st - 0.52) / 0.035, 0, 1)
    dirt = smoothstep((rows / n - 0.60) / 0.34)                # grubby at the foot

    rgb = P_CANVAS[None, None, :] * v[..., None]
    rgb *= 1.0 - stain[..., None] * np.array([0.17, 0.21, 0.25])
    rgb *= 1.0 - rim[..., None] * np.array([0.24, 0.29, 0.34])
    rgb *= 1.0 - dirt[..., None] * np.array([0.22, 0.27, 0.33])
    # one faded indigo stripe pair - the universal grain-sack marking
    t = (rows / n) % 1.0
    band = (np.clip(1.0 - np.abs(t - 0.30) / 0.030, 0, 1) +
            np.clip(1.0 - np.abs(t - 0.38) / 0.018, 0, 1))
    band = np.clip(band * (0.55 + 0.45 * np.clip(bandnoise(n, 8.0, seed + 4, 0.4), -1, 1)), 0, 1)
    rgb = rgb * (1 - band[..., None] * 0.50) + band[..., None] * 0.50 * \
        np.array([0.215, 0.220, 0.275]) * v[..., None]
    save("canvas", rgb, P_CANVAS, std=12.0)


# ===========================================================================
# 13. SAND - drift, dust, spilled grain.
# ===========================================================================

def make_sand(n: int = 512) -> None:
    seed = 1301
    h = bandnoise(n, 5.0, seed) * 1.0 + bandnoise(n, 16.0, seed + 1) * 0.4
    ripple = np.sin(2 * np.pi * (np.arange(n)[:, None] / (n / 9.0)) + bandnoise(n, 4.0, seed + 2) * 2.2)
    h += ripple * 0.22
    v = relief(h, slope=3.0, gain=0.34)
    v *= 1.0 - 0.22 * cavity(h, 4.0, 1.4)
    v *= 1.0 + fnoise(n, 2.2, seed + 3) * 0.06
    grit = np.clip(bandnoise(n, 120.0, seed + 4) * 1.5 - 0.75, 0, 1)
    v *= 1.0 + grit * 0.10
    rgb = P_SAND[None, None, :] * v[..., None]
    rgb *= 1.0 + (np.clip(fnoise(n, 2.4, seed + 5) * 0.5 + 0.5, 0, 1) - 0.5)[..., None] * \
        np.array([0.09, 0.05, -0.04])
    save("sand", rgb, P_SAND, std=13.0)


# ===========================================================================
# 14. RUIN - render fallen off a rubble core. For the west ruin stumps.
# ===========================================================================

def make_ruin(n: int = 1024) -> None:
    seed = 1401
    edge, cid = voronoi(n, 6, seed, jitter=0.95)               # rubble stones, ~16 cm
    joint = smoothstep(edge / 16.0)
    hs = joint + bandnoise(n, 18.0, seed + 1, 0.4) * 0.25 * joint

    # the surviving render, torn away in patches with a ragged, undercut edge
    skin = np.clip(fnoise(n, 2.6, seed + 2) * 0.5 + 0.5, 0, 1)
    skin = skin + bandnoise(n, 16.0, seed + 6, 0.4) * 0.055     # ragged tear line
    skin = smoothstep((skin - 0.46) / 0.32)          # soft, crumbling tear
    h = hs * (1 - skin) + (1.45 + bandnoise(n, 40.0, seed + 3, 0.35) * 0.10) * skin

    ao = 1.0 - 0.32 * cavity(h, 5.0, 1.4)
    lit = relief(h, slope=7.0, gain=0.46, smooth=1.4)
    v = ao * lit * (1.0 + fnoise(n, 2.3, seed + 4) * 0.08)
    v *= 1.0 + (1 - skin) * per_cell(cid, seed + 5, -0.10, 0.10)

    rubble = np.array([0.505, 0.415, 0.315])                   # grey-brown core stone
    render = P_PLASTER * 1.02
    rgb = (render[None, None, :] * skin[..., None] +
           rubble[None, None, :] * (1 - skin[..., None])) * v[..., None]
    rgb *= 1.0 - (1 - joint)[..., None] * (1 - skin)[..., None] * np.array([0.10, 0.11, 0.13])
    save("ruin", rgb, P_RUIN, std=18.0)


# ===========================================================================
# 15. GLASS - grimy lamp glazing. Alpha comes from material rgba, not the PNG
#     (texture alpha gives no cut-out in this renderer).
# ===========================================================================

def make_glass(n: int = 512) -> None:
    seed = 1501
    # Glass is a smooth material; its whole character is the few things ON it.
    h = bandnoise(n, 5.0, seed, 0.45) * 0.6                    # cylinder-drawn ripple
    v = relief(h, slope=1.8, gain=0.26, smooth=1.0)
    dust = np.clip(fnoise(n, 2.5, seed + 1) * 0.5 + 0.5, 0, 1)
    # rain streaks: long runs down the pane, soft and sparse
    streak = np.abs(streaknoise(n, 14.0, seed + 2, stretch=16.0, along="row", width=0.6))
    streak = smoothstep((streak - 1.10) / 0.55)
    spot = smoothstep((bandnoise(n, 22.0, seed + 3, 0.30) - 2.05) / 0.30)   # splash spots
    spot *= smoothstep((fnoise(n, 2.5, seed + 4) * 0.5 + 0.5 - 0.52) / 0.20)
    v *= 1.0 + (dust - 0.5) * 0.16 + streak * 0.20 + spot * 0.40
    rgb = P_GLASS[None, None, :] * v[..., None]
    rgb *= 1.0 + (dust - 0.5)[..., None] * np.array([0.12, 0.05, -0.05])
    save("glass", rgb, P_GLASS, std=9.0)


# ===========================================================================
# 16. LAMP LENS - emissive. Hot core, cooler rim, a little soot.
#     texuniform="false" texrepeat="1 1" => one lens per face.
# ===========================================================================

def make_lamp(n: int = 256) -> None:
    seed = 1601
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1.0)
    r = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.55
    core = np.clip(1.0 - r ** 1.7, 0, 1)
    v = 0.42 + 0.58 * core
    soot = np.clip(bandnoise(n, 7.0, seed) * 0.9 + 0.25, 0, 1) * smoothstep((r - 0.45) / 0.55)
    v *= 1.0 - soot * 0.45
    v *= 1.0 + bandnoise(n, 30.0, seed + 1) * 0.05
    rgb = P_EMIT[None, None, :] * v[..., None]
    rgb *= 1.0 + (1 - core)[..., None] * np.array([-0.05, 0.02, 0.10])   # cooler rim
    save("lamp", rgb, None)


# ===========================================================================
# XML
# ===========================================================================

def _tex(name: str) -> str:
    return f'<texture name="{PFX}{name}" type="2d" file="{PFX}{name}.png"/>'


def _mat(name: str, tex: str | None, *, rgba: str = "1 1 1 1", spec: float, shin: float,
         refl: float = 0.0, emission: float = 0.0, texrepeat: str = "1 1",
         texuniform: str = "true") -> str:
    a = [f'name="{name}"']
    if tex:
        a.append(f'texture="{PFX}{tex}"')
        a.append(f'texuniform="{texuniform}"')
        a.append(f'texrepeat="{texrepeat}"')
    a.append(f'rgba="{rgba}"')
    a.append(f'specular="{spec:g}"')
    a.append(f'shininess="{shin:g}"')
    if refl:
        a.append(f'reflectance="{refl:g}"')
    if emission:
        a.append(f'emission="{emission:g}"')
    return "<material " + " ".join(a) + "/>"


# (material, texture, texrepeat, texuniform, specular, shininess, reflectance,
#  emission, rgba, one-line note)
MATERIALS = [
    # ---- bible section 4, the named library ----------------------------------
    ("aaa_ground",      "ground",      "1.11 1.11", "true",  0.10, 0.08, 0.00, 0.00, "1 1 1 1",
     "court flagstone, HERO. one 1024px tile per 0.90 m => 12.9 cm stones"),
    ("aaa_ground_wet",  "ground_damp", "1.11 1.11", "true",  0.50, 0.75, 0.22, 0.00, "1 1 1 1",
     "puddles, rill, basin. registers with aaa_ground. keep geoms 0.8 mm thin"),
    ("aaa_plaster",     "plaster",     "0.625 0.625", "true", 0.06, 0.04, 0.00, 0.00, "1 1 1 1",
     "walls and column shafts, 0-3 m"),
    ("aaa_plaster_mid", "plaster_mid", "0.625 0.625", "true", 0.05, 0.04, 0.00, 0.00, "1 1 1 1",
     "same, 3-6 m. aerial perspective baked in"),
    ("aaa_plaster_far", "plaster_far", "0.625 0.625", "true", 0.04, 0.03, 0.00, 0.00, "1 1 1 1",
     "same, >6 m"),
    ("aaa_stone",       "stone",       "1.8 1.8",   "true",  0.10, 0.06, 0.00, 0.00, "1 1 1 1",
     "capitals, plinths, copings, fountain rim"),
    ("aaa_stone_far",   "stone_far",   "1.8 1.8",   "true",  0.06, 0.04, 0.00, 0.00, "1 1 1 1",
     "same, >6 m"),
    ("aaa_rose",        "rose",        "14 1",      "false", 0.07, 0.05, 0.00, 0.00, "1 1 1 1",
     "the SUN wall ONLY. texuniform=false: v spans the wall height exactly once"),
    ("aaa_petrol",      "petrol",      "14 1",      "false", 0.14, 0.10, 0.00, 0.00, "1 1 1 1",
     "the SHADOW wall ONLY. salt bloom rises from the base"),
    ("aaa_terra",       "terra",       "3.33 3.33", "true",  0.10, 0.06, 0.00, 0.00, "1 1 1 1",
     "pots, amphorae. throw ridges band around the form"),
    ("aaa_olive",       "olive",       "4 4",       "true",  0.12, 0.08, 0.00, 0.00, "1 1 1 1",
     "all vegetation"),
    ("aaa_iron",        "iron",        "3.33 3.33", "true",  0.30, 0.35, 0.00, 0.00, "1 1 1 1",
     "raw rusted ironwork, brackets, rings"),
    ("aaa_lapis",       "lapis",       "4 4",       "true",  0.45, 0.55, 0.05, 0.00, "1 1 1 1",
     "THE ONE saturated object. use on exactly one prop, in shade"),
    ("aaa_emit",        None,          "1 1",       "true",  0.00, 0.00, 0.00, 0.95,
     "1 0.84 0.56 1", "lamps, curb strips, beacon"),
    # ---- extensions this agent adds -----------------------------------------
    ("aaa_emit_low",    None,          "1 1",       "true",  0.00, 0.00, 0.00, 0.45,
     "1 0.84 0.56 1", "long curb runs that would otherwise blow out"),
    ("aaa_emit_lamp",   "lamp",        "1 1",       "false", 0.00, 0.00, 0.00, 0.95, "1 1 1 1",
     "lamp lens with a hot core. one lens per geom face"),
    ("aaa_wood",        "wood",        "1.43 1.43", "true",  0.08, 0.06, 0.00, 0.00, "1 1 1 1",
     "weathered timber: crates, ladder, cart, broom"),
    ("aaa_canvas",      "canvas",      "4.5 4.5",   "true",  0.05, 0.04, 0.00, 0.00, "1 1 1 1",
     "sacking and awnings. woven relief baked from a Sobel normal"),
    ("aaa_paint",       "paint",       "3.33 3.33", "true",  0.34, 0.42, 0.00, 0.00, "1 1 1 1",
     "red-lead painted steel chipping back to rust"),
    ("aaa_sand",        "sand",        "1.67 1.67", "true",  0.06, 0.05, 0.00, 0.00, "1 1 1 1",
     "drifted dust, spilled grain"),
    ("aaa_ruin",        "ruin",        "1 1",       "true",  0.09, 0.06, 0.00, 0.00, "1 1 1 1",
     "render fallen off a rubble core. the west ruin stumps"),
    ("aaa_glass",       "glass",       "2 2",       "true",  0.85, 0.90, 0.06, 0.00,
     "1 1 1 0.34", "grimy glazing. alpha lives in rgba, never in the PNG"),
]

# Also publish every material under the agent-prefixed name required by the
# output contract, so a composer can use either convention. Set False if some
# other agent turns out to declare the aaa_* names itself (repeated names are a
# hard compile error in MuJoCo).
EMIT_PREFIXED_ALIASES = True


def write_xml() -> None:
    lines = [
        "<!-- aaa_materials_assets.xml - generated by scripts/aaa/materials.py.",
        "     THE WATER COURT shared material library. Do not hand-edit.",
        "     Textures are LINEAR and carry their palette albedo as their mean, so",
        "     every material is rgba=\"1 1 1 1\"; per-geom rgba MULTIPLIES it, which",
        "     is how you get the bible's +/-6% per-instance tint. -->",
        "",
    ]
    for name in sorted({m[1] for m in MATERIALS if m[1]}):
        lines.append(_tex(name))
    lines.append("")
    # ONE name per material.  This file used to emit the whole library TWICE -- once as
    # aaa_X and once as a byte-identical materials_X alias -- which is 44 <material>
    # elements for 22 materials, doubles nmat for nothing, and violates the output
    # contract's "<yourname>_ prefix" rule with the aaa_* half.  Only the contract-
    # prefixed set ships.  Anything no geom references is dropped again by assemble.py's
    # pruner, so a spare in this library costs nothing downstream.
    for (mname, tex, trep, tuni, spec, shin, refl, emis, rgba, note) in MATERIALS:
        lines.append(f"<!-- {note} -->")
        lines.append(_mat("materials_" + mname[len("aaa_"):], tex, rgba=rgba, spec=spec,
                          shin=shin, refl=refl, emission=emis, texrepeat=trep,
                          texuniform=tuni))
    with open(os.path.join(MODEL_DIR, "aaa_materials_assets.xml"), "w") as f:
        f.write("\n".join(lines) + "\n")

    body = (
        "<!-- aaa_materials_body.xml - generated by scripts/aaa/materials.py.\n"
        "     The materials agent spends 0 of the 1250 scenery geoms: this library is\n"
        "     assets only. The swatch board used to review it lives in the throwaway\n"
        "     test scene, not in the shipped world, so nothing floats outside the\n"
        "     court walls. Intentionally empty. -->\n"
    )
    with open(os.path.join(MODEL_DIR, "aaa_materials_body.xml"), "w") as f:
        f.write(body)


# ===========================================================================

def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    make_ground()
    make_plaster()
    make_stone()
    make_rose()
    make_petrol()
    make_terra()
    make_olive()
    make_iron()
    make_paint()
    make_lapis()
    make_wood()
    make_canvas()
    make_sand()
    make_ruin()
    make_glass()
    make_lamp()
    write_xml()

    print(f"{'texture':<16} {'mean rgb (0-255)':>22}   {'lum std':>7}   {'kB':>6}")
    total = 0
    for name, mean, std in _STATS:
        kb = os.path.getsize(os.path.join(MODEL_DIR, PFX + name + ".png")) / 1024
        total += kb
        print(f"{name:<16} {str(mean.round(1)):>22}   {std:7.1f}   {kb:6.0f}")
    print(f"{'TOTAL':<16} {'':>22}   {'':>7}   {total:6.0f} kB")
    print(f"\nwrote {len(_STATS)} PNGs + aaa_materials_assets.xml + aaa_materials_body.xml")
    print(f"into {MODEL_DIR}")


if __name__ == "__main__":
    main()
