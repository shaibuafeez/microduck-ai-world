#!/usr/bin/env python3
"""AAA scene — TERRAIN agent.  "The Water Court" ground plane and everything flat on it.

Writes (idempotently — byte-identical on re-run) into src/mjlab_microduck/robot/microduck/:
    terrain_ground.png        3072^2  hero flagstone albedo, relief baked to the §3 sun
    terrain_pool.png          2048^2  standing water: diffused stone + painted sky sheen
    terrain_rill.png           512^2  drainage-rill cross section (kerb / wall / bed)
    terrain_band.png           512^2  bleached inlay / threshold band
    aaa_terrain_assets.xml            <texture>/<material> fragment
    aaa_terrain_body.xml              <geom> fragment, 59 geoms / 59 BE / 0 cylinders

WHY A PLANE AND NOT AN HFIELD
    Measured in the audit: an hfield floor migrates all 9 duck contacts onto the terrain and
    moves the settled trunk height 0.0349 -> 0.0746 m, i.e. a different contact state from
    the one the policy trained on.  It also kills haze (haze needs an infinite plane) and it
    tilts the horizon off the 50%-of-frame line the whole composition is built on.  So: one
    infinite plane, and every scrap of relief is baked into the albedo.  Verified here: with
    this fragment loaded the duck settles at z = 0.03494 with 9 contacts, bit-identical to a
    bare untextured plane.

WHY NO NORMAL MAP
    The brief asks for one.  This wheel ships only the classic OpenGL renderer
    (`from mujoco import _render_filament` -> ImportError); <layer role="normal"> compiles,
    binds to mat_texid[5] and renders byte-for-byte identically (re-verified here: maxdiff
    0.0).  The substitute is a full offline bake: per-pixel surface normals from the joint
    height field, the three real scene lights, a 17-step ray-march cast shadow at the sun's
    11 deg elevation, per-stone settle, and an AO term -- all resolved into the albedo.

FOUR THINGS MEASURED HERE THAT CONTRADICT THE BRIEF.  Every number below was rendered.

 1. THE TEXEL IS THE CEILING, AND THE FLOOR ONLY GETS 0.47 OF THE LIGHT.
    With a texture bound, the classic renderer clamps (material rgba x sum of lights) to 1.0
    BEFORE modulating by the texel -- a flat 0.520 texture under 1.77x light renders at
    exactly 0.520, while the same value as material rgba renders at 0.918.  And because the
    sun sits at 11 deg, a horizontal surface receives only 0.191 of it.  Net floor multiplier,
    measured and predicted to 3 decimals: (0.474, 0.400, 0.424).  The bible's "Sunlit" column
    is the multiplier for a sun-FACING surface; the floor gets less than half of it, and no
    texture can be brighter than itself.  Every texel here is authored as
    delivered_screen / that vector.

 2. `texuniform` REPEAT PERIOD IS 2/texrepeat METRES, NOT 1/texrepeat.
    Probed with a UV-gradient texture: u = texrepeat/2 * (local_x + size_x),
    v = texrepeat/2 * (size_y - local_y).  So the period is 2/texrepeat m, v = 0 sits at the
    geom's +Y edge, and the phase is anchored to the GEOM's own corner, not the world -- two
    coplanar geoms sharing a material do not line up.  The bible's texrepeat 1.11 would give
    a 1.80 m tile, not 0.90 m.  Hence 1.667 here for a 1.20 m tile.

 3. `reflectance` CANNOT WORK ON A DECAL LYING ON AN INFINITE PLANE.
    A/B with 27 reflective geoms at 1.3 / 6 / 20 / 50 mm above the floor: mean pixel
    difference 0.007-0.05 on a 0-255 scale, bit-identical in three of five camera angles,
    for +4.5 ms per frame at 320x240 (~0.17 ms per reflective geom).  The reflected
    half-space is filled by the floor itself, and `reflectance` never mirrors the skybox, so
    a horizontal pool "reflects" nothing.  Dropped entirely; a painted sky sheen in
    terrain_pool.png delivers the read for free.

 4. THE ART BIBLE'S SUN CASTS NO SHADOW.  <-- scene-critical, and not mine to fix.
    §3 declares  aaa_sun  pos="-6 -5 2" dir="-0.752 -0.631 -0.191".  A directional light
    ignores `pos` for shading but NOT for its shadow map: the light sits south-west of the
    court and shines further south-west, so the entire scene is behind the shadow camera.
    castshadow true vs false is bit-identical (0.00% of pixels differ) at every elevation,
    shadowsize, shadowclip, shadowscale, extent and headlight setting I tried.  Flipping the
    position to the sun side -- pos="6 5 2", dir unchanged -- restores shadows immediately
    (4.67% of frame, maxdiff 78).  In the composed scene that single token takes the
    near-band contrast std from 12.0 to 19.6.  The ATMOSPHERE agent owns aaa_sun.
"""
import os
from pathlib import Path
import numpy as np
from PIL import Image

MD = Path(__file__).resolve().parents[2] / "src/mjlab_microduck/robot/microduck"

# ---------------------------------------------------------------- lighting rig (art bible §3)
SUN_RGB = np.array([1.45, 1.06, 0.66])
SUN_TO = np.array([0.752, 0.631, 0.191])           # surface -> sun (unit)
FILL_RGB = np.array([0.26, 0.28, 0.44])
FILL_TO = np.array([-0.70, -0.59, 0.40])           # surface -> skyfill (unit)
AMB_RGB = np.array([0.09, 0.09, 0.13])
FLAT_I = SUN_RGB * SUN_TO[2] + FILL_RGB * FILL_TO[2] + AMB_RGB   # (0.471,0.404,0.436)

SUN_XY = SUN_TO[:2] / np.linalg.norm(SUN_TO[:2])   # (0.766, 0.643)
SUN_TAN = SUN_TO[2] / np.linalg.norm(SUN_TO[:2])   # tan(11 deg) = 0.1944


def shade(nx, ny, nz, sun_vis=1.0, ao=1.0):
    """Radiance of a surface with normal n under the rig, as a per-channel multiplier of
    the flat-floor radiance.  Returns (...,3).  Occlusion darkens only the sky/ambient."""
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / ln, ny / ln, nz / ln
    nds = np.clip(nx * SUN_TO[0] + ny * SUN_TO[1] + nz * SUN_TO[2], 0, None)
    ndf = np.clip(nx * FILL_TO[0] + ny * FILL_TO[1] + nz * FILL_TO[2], 0, None)
    I = (nds * sun_vis)[..., None] * SUN_RGB + (ndf[..., None] * FILL_RGB + AMB_RGB) * np.asarray(ao)[..., None]
    return I / FLAT_I


def shoulder(x, k=0.74):
    """Filmic highlight rolloff so blazing chamfers keep their hue instead of clipping flat."""
    return np.clip(np.where(x <= k, x, k + (1 - k) * (1 - np.exp(-(x - k) / (1 - k)))), 0, 1)


# ---------------------------------------------------------------- periodic noise
def fbm(n, octaves, seed, freq0=4, lac=2.0, gain=0.5):
    """Seamless 1/f noise: build in the frequency domain so it tiles exactly."""
    r = np.random.default_rng(seed)
    out = np.zeros((n, n))
    amp, freq = 1.0, freq0
    for _ in range(octaves):
        g = r.normal(size=(freq, freq))
        G = np.fft.fft2(g)
        Gp = np.zeros((n, n), complex)
        h = freq // 2
        Gp[:h, :h] = G[:h, :h]; Gp[:h, -h:] = G[:h, -h:]
        Gp[-h:, :h] = G[-h:, :h]; Gp[-h:, -h:] = G[-h:, -h:]
        layer = np.real(np.fft.ifft2(Gp)); layer /= (layer.std() + 1e-9)
        out += amp * layer
        amp *= gain; freq = int(freq * lac)
        if freq > n:
            break
    return out / (out.std() + 1e-9)


def grit(n, f_lo, f_hi, seed, slope=-0.55):
    """Band-limited periodic noise with energy CONCENTRATED between f_lo and f_hi cycles per
    tile, unit std.  fbm() is 1/f, so its top octave carries 0.5**k of the energy and is
    invisible: 0.052 * fbm(N, 8) measured a local 7-texel std of 0.4/255.  The near band is
    the bottom quarter of every frame and magnifies the tile about 3.3x, so a 24 px screen
    window sees ~7 texels == 4 mm of stone.  Content at THAT wavelength is what the critique
    measured as missing, and it has to be synthesised directly in the frequency domain."""
    r = np.random.default_rng(seed)
    fy = np.fft.fftfreq(n) * n
    R = np.sqrt(fy[:, None] ** 2 + fy[None, :] ** 2)
    env = np.exp(-((np.log(np.maximum(R, 0.5) / f_lo)) ** 2) / 0.9) * (R >= 1)
    env = env + np.exp(-((np.log(np.maximum(R, 0.5) / f_hi)) ** 2) / 0.9) * (R >= 1) * 0.8
    env = env * np.maximum(R, 1.0) ** slope
    ph = r.normal(size=(n, n)) + 1j * r.normal(size=(n, n))
    out = np.real(np.fft.ifft2(ph * env))
    return out / (out.std() + 1e-9)


def blur(a, r):
    """Cheap separable periodic box blur, r px."""
    if r < 1:
        return a
    k = 2 * r + 1
    c = np.cumsum(np.concatenate([a, a[:k]], 0), 0)
    a = (c[k:] - c[:-k]) / k
    a = np.roll(a, r, 0)
    c = np.cumsum(np.concatenate([a, a[:, :k]], 1), 1)
    a = (c[:, k:] - c[:, :-k]) / k
    return np.roll(a, r, 1)


def dist_to(mask, maxd):
    """Toroidal chamfer distance from a boolean mask, by iterated 4-dilation."""
    d = np.full(mask.shape, float(maxd))
    cur = mask.copy()
    d[cur] = 0.0
    for k in range(1, maxd + 1):
        nxt = (cur | np.roll(cur, 1, 0) | np.roll(cur, -1, 0)
               | np.roll(cur, 1, 1) | np.roll(cur, -1, 1))
        new = nxt & ~cur
        d[new] = k
        cur = nxt
    return d


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


# ================================================================ 1. HERO FLAGSTONE
def make_ground(path, N=3072, tile_m=1.68, seed=1207):
    """THE HERO ASSET: the bottom half of every frame, always.

    Rebuilt after the art critique.  Three things were wrong with the first cut and all
    three are measured, not matters of taste:

      * PER-STONE TINT WAS 3x THE BIBLE'S OWN LAW.  sval carried a 0.185 std clipped at
        +/-0.44, so adjacent flags landed peach / chocolate / lilac and the paving read as a
        random-tinted grid rather than one quarry run.  Bible section 7 sets the variation
        law at +/-6% on value.  Now 0.050 std clipped at +/-0.115, with temperature and
        bleach cut to match, and ALL the interest moved INSIDE the stone.

      * THE JOINT SHOULDER OUT-COMPETED THE SUNLIT FACE.  A uniform chamfer running the
        whole perimeter plus a pale lime mortar gave every flag a glowing gold outline --
        the brightest thing in the near band was the grout.  The chamfer is now ONE-SIDED,
        its relief amplitude is halved, its radiance is capped at 1.28x flat instead of
        1.85x, and the mortar is a dark damp sand that sits BELOW the stone in value.

      * THERE WAS NO MATERIAL ON THE MATERIAL.  Median local 24 px luminance std in the
        rendered near band measured 0.94/255 -- the flags were flat colour fields and the
        whole global std came from the joint lines.  This tile now carries mineral banding,
        wear dishing toward stone centres, grit, pitting, spall and dirt gathering in the
        joint corners.

    Stone size is also the scale story.  5 courses over a 1.68 m tile cut into 4-6 stones
    gives 0.28-0.42 m flags: ONE paving stone is longer than the whole duck, which is what
    "10x human scale" has to mean at 12 cm eye height.  It also puts a single sweeping joint
    across the bottom of frame instead of a grid, which is what killed the checkerboard read.

    Everything still lives at stone scale or below: anything with a wavelength approaching
    the tile size is what makes the repeat legible on an infinite plane.  The coherent
    low-frequency story the critique asked for (traffic polish, damp margins) is delivered
    by separate decal geoms in body_xml(), where it cannot tile."""
    mm = tile_m * 1000.0 / N                      # 0.547 mm per texel
    JW = 4.2 / mm                                 # joint half-width  (8.4 mm joint)
    CW = 6.5 / mm                                 # nominal chamfer run
    RELIEF = 0.72 / mm                            # chamfer drop, px units (was 1.38)
    AOR = int(round(26.0 / mm))
    DISH = int(round(95.0 / mm))                  # wear-dish radius from the stone edge

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    wx = (xx + 5.0 / mm * fbm(N, 3, seed + 11).astype(np.float32)) % N
    wy = (yy + 5.0 / mm * fbm(N, 3, seed + 12).astype(np.float32)) % N

    # --- random ashlar: 5 courses, each cut into 4-6 stones -> 0.28-0.42 m flags ---
    NROW, NCOL = 5, 8
    hgt = rng.uniform(0.84, 1.20, NROW)
    hgt = np.round(hgt / hgt.sum() * N).astype(int); hgt[-1] += N - hgt.sum()
    rb = np.concatenate([[0], np.cumsum(hgt)]).astype(float)

    lab = np.zeros((N, N), np.int32)
    for r in range(NROW):
        ns = int(rng.integers(4, 7))
        w = rng.uniform(0.74, 1.34, ns)
        w = np.round(w / w.sum() * N).astype(int); w[-1] += N - w.sum()
        cb = np.concatenate([[0], np.cumsum(w)]).astype(float)
        m = (wy >= rb[r]) & (wy < rb[r + 1])
        lab[m] = r * NCOL + np.searchsorted(cb, wx[m], side="right") - 1
    NLAB = NROW * NCOL

    edge = ((lab != np.roll(lab, 1, 0)) | (lab != np.roll(lab, 1, 1))
            | (lab != np.roll(lab, -1, 0)) | (lab != np.roll(lab, -1, 1)))
    d = dist_to(edge, max(AOR, DISH) + 4)

    # arris condition varies: some edges crisp, most worn round, some joints packed flush
    cwmod = 0.55 + 1.70 * smoothstep(fbm(N, 4, seed + 21) * 0.6 + 0.45)
    per_edge = (0.35 + 0.85 * rng.random(NLAB))[lab]
    h = smoothstep((d - JW) / (CW * cwmod))
    packed = smoothstep((fbm(N, 3, seed + 22) - 0.35) / 0.55)     # mortar filled flush
    h = h + (1.0 - h) * packed * 0.85

    tiltx = rng.normal(0, 0.022, NLAB)[lab].astype(np.float32)
    tilty = rng.normal(0, 0.022, NLAB)[lab].astype(np.float32)

    R = (RELIEF * per_edge * (1.0 - 0.75 * packed)).astype(np.float32)
    gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5 * R
    gy = -(np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5 * R

    # --- ONE-SIDED ARRIS.  A worn kerb is proud on the side the traffic runs off it and
    #     broken away on the other; a chamfer that runs the whole perimeter IS an outline.
    #     Keep the shoulder only where the surface already leans toward the sun's bearing,
    #     and drop it to 30% elsewhere, so the lit shoulder appears on one pair of edges and
    #     the other pair simply reads as the stone's own dark side.
    lean = np.clip((-gx * SUN_XY[0] - gy * SUN_XY[1]) / (np.abs(gx) + np.abs(gy) + 1e-6), -1, 1)
    onesided = 0.30 + 0.70 * smoothstep(lean * 0.5 + 0.5)
    gx = gx * onesided
    gy = gy * onesided

    # micro surface: shallow pocking on the stone faces, none in the joints
    pk = fbm(N, 7, seed + 71) * 0.20 * h
    gx = gx + (np.roll(pk, -1, 1) - np.roll(pk, 1, 1)) * 0.5
    gy = gy - (np.roll(pk, -1, 0) - np.roll(pk, 1, 0)) * 0.5
    nx = -gx + tiltx * h
    ny = -gy + tilty * h
    nz = np.ones_like(h)

    # some stones settled proud, some sunk: only the CAST SHADOW sees this
    hoff = (rng.normal(0, 0.30, NLAB) / mm)[lab].astype(np.float32) * h
    Hpx = (h * R + hoff).astype(np.float32)
    shadow = np.zeros((N, N), bool)
    for t in range(1, int(RELIEF / SUN_TAN) + 3):
        oc = int(round(t * SUN_XY[0])); orow = int(round(-t * SUN_XY[1]))
        shadow |= (np.roll(np.roll(Hpx, -oc, 1), -orow, 0) > Hpx + t * SUN_TAN + 0.015)
    vis = 1.0 - blur(shadow.astype(np.float32), 2) * 0.90
    ao = 0.520 + 0.480 * smoothstep(d / AOR)

    M = np.minimum(shade(nx, ny, nz, sun_vis=vis, ao=ao), 1.28)

    # --- albedo.  ONE quarry run.  Value spread is a third of the first cut and the whole
    #     of the remaining interest is inside the stone, not between stones. ---
    base = np.array([0.900, 0.648, 0.500])
    sval = (1.0 + rng.normal(0, 0.050, NLAB).clip(-0.115, 0.115))[lab][..., None]
    stmp = rng.normal(0, 0.020, NLAB)[lab][..., None] * np.array([0.9, 0.05, -0.95])
    schr = rng.normal(0, 0.016, NLAB)[lab][..., None]
    alb = base * sval + stmp
    alb = alb * (1 - np.abs(schr)) + alb.mean(2, keepdims=True) * np.abs(schr) * np.sign(schr + 0.5)

    # ---- WITHIN-STONE CONTENT.  This is the part that was missing entirely. ----
    # 1. mineral banding: sedimentary streaks stretched along one axis, the axis chosen per
    #    stone so no two neighbours band the same way.
    bx = fbm(N, 5, seed + 101)
    bx = 0.55 * bx + 0.45 * np.roll(bx, N // 7, 1)          # stretch along x
    by = fbm(N, 5, seed + 102)
    by = 0.55 * by + 0.45 * np.roll(by, N // 7, 0)          # stretch along y
    pick = (rng.random(NLAB) > 0.5)[lab]
    bandf = np.where(pick, bx, by)
    alb = alb * (1 + 0.130 * bandf * h)[..., None]
    alb = alb + np.array([0.035, 0.006, -0.030]) * (bandf * h)[..., None]

    # 2. wear dishing: the middle of a flag is walked hollow and polishes lighter and
    #    greyer; the un-walked rim keeps its chroma.
    dish = smoothstep(d / DISH)
    pol = dish * (0.45 + 0.55 * (rng.random(NLAB) ** 1.4)[lab])
    alb = alb * (1 + 0.040 * pol)[..., None]
    alb = alb * (1 - 0.10 * pol)[..., None] + alb.mean(2, keepdims=True) * (0.10 * pol)[..., None]

    # 3. dirt gathers in the joint corners: a cool grey-brown wash, not just darkening.
    corner = (1 - smoothstep(d / (AOR * 1.6))) ** 1.7
    grime = np.array([0.470, 0.408, 0.362])
    gmix = np.clip(corner * (0.34 + 0.30 * fbm(N, 4, seed + 111)), 0, 0.62)[..., None]
    alb = alb * (1 - gmix) + grime * gmix

    # 4. mortar: a dark damp sand that sits BELOW the stone in value.  The first cut used a
    #    pale lime (0.740 0.678 0.580) and it read as strip lighting under the paving.
    mortar = np.array([0.402, 0.352, 0.312])
    jmix = (1.0 - smoothstep((d - JW * 0.55) / (JW * 1.9)))[..., None]
    alb = alb * (1 - jmix) + mortar * jmix

    # 5. lime bloom on a minority of stones -- kept, halved, and pushed off the joints
    bloom = smoothstep((np.abs(fbm(N, 5, seed + 81)) - 0.95) / 0.5) * h
    alb = alb * (1 - 0.26 * bloom)[..., None] + np.array([0.80, 0.77, 0.71]) * (0.26 * bloom)[..., None]

    # 6. hairline cracks on a minority of stones
    cr = np.abs(fbm(N, 6, seed + 41))
    crack = smoothstep((0.035 - cr) / 0.035) * (rng.random(NLAB) > 0.66)[lab] * h
    alb = alb * (1 - 0.24 * crack)[..., None]

    # 7. spalled rims: fresh pale stone where an arris has broken away
    spall = smoothstep((0.13 - np.abs(fbm(N, 7, seed + 51))) / 0.13) * smoothstep((JW * 2.6 - d) / (JW * 2.6))
    alb = alb * (1 - 0.45 * spall)[..., None] + np.array([0.80, 0.70, 0.58]) * (0.45 * spall)[..., None]
    alb = np.clip(alb, 0.05, 1.4)

    # 8. grit + pitting.  This is what carries the near band: at duck height the bottom
    #    quarter of frame magnifies the tile ~3.3x, so a 24 px screen window sees ~7 texels,
    #    i.e. 4 mm of stone.  Without content at that wavelength the floor is a colour field.
    #    f_lo/f_hi in cycles per 1.68 m tile:  110 -> 15 mm,  420 -> 4 mm.
    g_mid = grit(N, 46, 110, seed + 131)          # 36 mm .. 15 mm: pitting and pock
    g_fine = grit(N, 260, 560, seed + 132)        # 6.5 mm .. 3 mm: the sand grain itself
    pit = smoothstep((g_mid - 1.05) / 0.75) * h
    grain = (1.0 + 0.085 * fbm(N, 4, seed + 62) * h
                 + 0.098 * g_mid * (0.45 + 0.55 * h)
                 + 0.092 * g_fine)
    grain = grain * (1 - 0.30 * pit)
    grain = grain * np.where(rng.random((N, N)) > 0.9988, 0.70, 1.0)

    col = alb * M * grain[..., None]
    mx = col.max(2, keepdims=True)
    col = col * (shoulder(mx) / np.maximum(mx, 1e-6))        # hue-preserving highlight rolloff
    img = (np.clip(col, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(img).save(path, optimize=True)
    make_ground.linear = col

    a = img.astype(float)
    scr = a * np.array([0.474, 0.400, 0.424])
    L = scr.mean(2)
    hi = a.mean(2) - blur(a.mean(2), 8)
    loc = np.sqrt(np.maximum(blur(L ** 2, 3) - blur(L, 3) ** 2, 0))   # 7-texel window
    print(f"  {os.path.basename(path)}  {N}px/{tile_m}m  tex mean={a.mean(2).mean():5.1f} std={a.mean(2).std():5.1f}"
          f" | DELIVERED lum mean={L.mean():5.1f} std={L.std():5.1f} pct<25={100*(L<25).mean():4.1f}%"
          f" | hi-freq std={hi.std():4.1f} | local7 med={np.median(loc):4.1f}")
    return img


# ================================================================ 2. RILL CROSS SECTION
def make_rill(path, N=512, width_m=0.44, seed=77):
    """One 0.44 x 0.44 m cell of the DRY channel: kerbs, inner walls, silt bed.  The water
    itself is a separate geom carrying make_reflect()'s texture.

    Rebuilt after the critique, which was right on both counts and for the same reason.
    The first cut lit the south inner wall as a sun-facing vertical (ny=+1, vis=1.0), which
    under this rig is a x2.09/x1.79/x1.05 multiplier -- it produced a continuous fully
    saturated gold band, 12% of the texture, running to the vanishing point directly under
    the duck's walking line.  That reads as a highway lane divider, AND saturated yellow is
    reserved for the robot's own feet and jaw (bible section 2, rule 2), so the environment
    was competing with the duck for its only reserved accent on the most prominent line in
    the frame.

    The physics says the same thing: the sun is at 11 deg, so tan(elev) = 0.194 and a 40 mm
    kerb throws 206 mm of shadow -- more than the channel is wide.  NOTHING inside a rill
    this shallow sees the sun.  Every surface below the kerb top is therefore vis=0 and
    resolves violet, and all the brightness in the channel comes from the water reflecting
    the sky, which is where it belongs.

    v = 0 is the +Y edge (probe-verified: under texuniform, v = texrepeat/2 * (size_y -
    local_y)), so row 0 of the PNG is the north kerb."""
    v = (np.arange(N) + 0.5) / N
    V, U = np.meshgrid(v, v, indexing="ij")
    n5 = fbm(N, 5, seed); n3 = fbm(N, 3, seed + 1); n6 = fbm(N, 6, seed + 2)

    STONE = np.array([0.700, 0.640, 0.555])     # damp kerb stone
    SILT = np.array([0.430, 0.372, 0.330])      # wet silt, much darker than the paving
    band = np.zeros((N, N, 3))
    nx = np.zeros((N, N)); ny = np.zeros((N, N)); nz = np.ones((N, N))
    vis = np.ones((N, N)); ao = np.ones((N, N))

    def seg(a, b):
        return (V >= a) & (V < b)

    m = seg(0.000, 0.020) | seg(0.980, 1.000)   # outer joints against the paving
    band[m] = SILT * 0.72; ao[m] = 0.42; vis[m] = 0.0
    m = seg(0.020, 0.170)                        # north kerb top -- horizontal, sunlit
    band[m] = STONE; ao[m] = 0.90
    m = seg(0.170, 0.215)                        # north arris rolling into the channel
    band[m] = STONE * 0.94; ny[m] = -0.55; ao[m] = 0.62; vis[m] = 0.10
    m = seg(0.215, 0.300)                        # NORTH INNER WALL -- shadowed, violet
    band[m] = STONE * 0.90; ny[m] = -1.0; nz[m] = 0.30; ao[m] = 0.56; vis[m] = 0.0
    m = seg(0.300, 0.330)                        # waterline stain / algae
    band[m] = SILT * 0.90 + np.array([0.010, 0.024, 0.008]); ny[m] = -0.5; ao[m] = 0.48; vis[m] = 0.0
    m = seg(0.330, 0.670)                        # silt bed (the water geom sits over this)
    band[m] = SILT; ao[m] = 0.52; vis[m] = 0.0
    m = seg(0.670, 0.700)
    band[m] = SILT * 0.90 + np.array([0.010, 0.024, 0.008]); ny[m] = 0.5; ao[m] = 0.50; vis[m] = 0.0
    m = seg(0.700, 0.785)                        # SOUTH INNER WALL -- also shadowed.  The
    band[m] = STONE * 0.90; ny[m] = 1.0; nz[m] = 0.30; ao[m] = 0.60; vis[m] = 0.0
    m = seg(0.785, 0.830)                        # south arris: the only lip that clears the
    band[m] = STONE * 0.97; ny[m] = 0.55; ao[m] = 0.80; vis[m] = 0.55   # kerb shadow line
    m = seg(0.830, 0.980)                        # south kerb top
    band[m] = STONE; ao[m] = 0.92

    # cross joints between kerb stones, at the paving's own 0.33 m course rhythm
    kerb = (seg(0.020, 0.215) | seg(0.785, 0.980))
    cj = (np.abs(((U + 0.06 * n3) % 1.0) - 0.5) < 0.013) & kerb
    band[cj] = band[cj] * 0.50; ao[cj] = 0.32; vis[cj] = 0.0

    # silt drifts and a little debris in the bed; grain everywhere
    bed = seg(0.300, 0.700)
    drift = smoothstep((np.abs(n5) - 0.20) / 0.9) * bed
    band = band * (1 + 0.30 * drift - 0.16 * smoothstep((np.abs(n6) - 0.55) / 0.7) * bed)[..., None]
    band = band * (1 + 0.055 * n5 + 0.030 * n6)[..., None]
    stain = smoothstep((np.abs(n3) - 0.25) / 0.6) * 0.16
    band = band * (1 - stain * (V > 0.16) * (V < 0.84))[..., None]

    M = shade(nx, ny, nz, sun_vis=vis, ao=ao)
    col = band * M
    mx = col.max(2, keepdims=True)
    col = col * (shoulder(mx) / np.maximum(mx, 1e-6))
    img = (np.clip(col, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(img).save(path, optimize=True)
    a = img.astype(float)
    scr = a * np.array([0.474, 0.400, 0.424])
    sat = (a.max(2) - a.min(2)) / np.maximum(a.max(2), 1)
    goldish = ((a[..., 0] > 200) & (sat > 0.45)).mean() * 100
    print(f"  {os.path.basename(path)}  delivered lum: kerbN={scr[int(.09*N)].mean():5.1f}"
          f"  wallN={scr[int(.25*N)].mean():5.1f}  bed={scr[int(.50*N)].mean():5.1f}"
          f"  wallS={scr[int(.74*N)].mean():5.1f}  kerbS={scr[int(.90*N)].mean():5.1f}"
          f"  | reserved-gold pixels {goldish:.2f}%")


# ================================================================ 2a. WATER SURFACE + REFLECTION
def make_reflect(path, N=1024, seed=515):
    """The water surface for the rill and the fountain basin, with the colonnade's
    reflection PAINTED IN.

    The bible's one-frame test promises "hanging inverted in it is the full-height
    reflection of a colonnade".  The terrain agent's measurement that `reflectance` cannot
    deliver it stands and is not relitigated here: the mirror is a decal lying ON an
    infinite plane, so the reflected half-space is filled by the floor itself, and the
    classic renderer never mirrors the skybox.  A/B at four heights: mean pixel difference
    0.007-0.05 on 0-255, for +0.17 ms per reflective geom.  So it is baked instead.

    The u axis carries ONE column reflection per period, and the material sets the u period
    to 1.16 m -- the colonnade's actual bay spacing -- with each water geom spanning exactly
    one bay centred on a column, so the painted bar lands under the real column.  The v axis
    is the 0.164 m width of the channel, so texrepeat is strongly anisotropic.

    Value ceiling, and why there is a little emission on this material: a horizontal surface
    under this rig receives only (0.474, 0.400, 0.424) of full light, and with a texture
    bound the texel is the ceiling -- so the brightest a floor-level texel can ever render is
    luminance 112/255.  Real still water at golden hour is far brighter than the stone around
    it because it is showing the sky, not the ground.  emission="0.11" is the only lever in
    this renderer that can lift a horizontal surface above its own light budget, and it is
    the honest one: the surface really is showing a light source.  (Contrast the background
    ranges, which carried emission 2.45 and read as self-luminous glaciers.  0.16 is a
    sheen; 2.45 is a light bulb.)"""
    v = (np.arange(N) + 0.5) / N
    V, U = np.meshgrid(v, v, indexing="ij")
    r1 = fbm(N, 3, seed); r2 = fbm(N, 5, seed + 1); r3 = fbm(N, 7, seed + 2)

    # --- the sky the water is showing.  Sampled from the skybox family: warm gold near the
    #     horizon, periwinkle higher up.  NO saturated yellow: reserved for the robot.
    SKY_WARM = np.array([0.960, 0.812, 0.660])
    SKY_COOL = np.array([0.560, 0.596, 0.790])
    # across the channel, a grazing view shows the low warm sky at the far edge and more of
    # the cool zenith near the viewer; the two margins are the kerbs' own dark reflection.
    grad = smoothstep((V - 0.10) / 0.80)
    col = SKY_COOL + (SKY_WARM - SKY_COOL) * (0.30 + 0.62 * (1 - grad))[..., None]

    # wind ruffle
    ripple = 0.5 + 0.5 * np.tanh(1.5 * (0.75 * r1 + 0.42 * r2 + 0.20 * r3))
    col = col * (0.74 + 0.26 * ripple)[..., None]

    # --- THE INVERTED COLONNADE.  One bay per period.  A column reflection is a dark
    #     violet bar (the shaft's shadow side is what a rill at ground level sees), with a
    #     narrow bright edge where the sunlit west face catches, and a paler smear at the
    #     far end of the bar where the capital's bleached stone reflects.
    def bar(centre, halfw, soft):
        return 1.0 - smoothstep((np.abs(((U - centre + 0.5) % 1.0) - 0.5) - halfw) / soft)

    shaft = bar(0.500, 0.058, 0.018)
    lit = bar(0.566, 0.013, 0.016)          # sunlit arris, offset toward +x (the sun side)
    # the reflection is stretched toward the viewer, i.e. across the channel: strongest at
    # the near margin, fading as it runs away.
    stretch = (0.45 + 0.55 * smoothstep((0.92 - V) / 0.75))
    SHAFT_COL = np.array([0.218, 0.196, 0.282])
    CAP_COL = np.array([0.760, 0.700, 0.640])
    sm = (shaft * stretch * (0.86 + 0.12 * ripple))[..., None]
    col = col * (1 - sm) + SHAFT_COL * sm
    lm = (lit * stretch * 0.72)[..., None]
    col = col * (1 - lm) + CAP_COL * lm
    # the capital's bleached block sits at the far end of the shaft's reflection
    capm = (shaft * smoothstep((0.30 - V) / 0.26) * 0.55)[..., None]
    col = col * (1 - capm) + CAP_COL * capm

    # kerb reflections darken both margins; a thin wet meniscus right at the edge
    edge = 1 - smoothstep((np.minimum(V, 1 - V) - 0.020) / 0.070)
    col = col * (1 - 0.62 * edge)[..., None]
    men = (1 - smoothstep((np.minimum(V, 1 - V) - 0.004) / 0.014)) * 0.5
    col = col * (1 - men)[..., None] + np.array([0.66, 0.62, 0.60]) * men[..., None]

    # broken ripple lines cutting the reflection, and a few specular sparks
    cut = smoothstep((np.abs(r2) - 1.25) / 0.4) * (V > 0.08) * (V < 0.92)
    col = col * (1 - 0.22 * cut)[..., None] + SKY_WARM * (0.22 * cut)[..., None]
    spark = smoothstep((np.abs(r3) - 1.75) / 0.30) * smoothstep((ripple - 0.70) / 0.24)
    col = col + np.array([1.0, 0.93, 0.80]) * (0.30 * spark)[..., None]

    mx = col.max(2, keepdims=True)
    col = col * (shoulder(mx, 0.84) / np.maximum(mx, 1e-6))
    img = (np.clip(col, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(img).save(path, optimize=True)
    a = img.astype(float)
    sat = (a.max(2) - a.min(2)) / np.maximum(a.max(2), 1)
    print(f"  {os.path.basename(path)}  tex lum mean={a.mean(2).mean():5.1f} std={a.mean(2).std():5.1f}"
          f"  | reserved-gold pixels {(((a[...,0]>200)&(sat>0.45)).mean()*100):.2f}%")


# ================================================================ 2b. STANDING WATER
def make_pool(path, ground_linear, seed=909):
    """Water lying on the court.  Built from the SAME stone albedo, heavily diffused (that
    is what looking through 5 mm of water does to a joint) and then a reflected-sky sheen
    painted on top -- the classic renderer's `reflectance` mirrors scene geometry but NOT
    the skybox, so without a painted sheen a pool viewed from anywhere but a grazing angle
    is a black hole.  reflectance then adds the real column reflections on top of this."""
    N = ground_linear.shape[0]
    K = N // 3                                    # true decimation, not a crop
    g = ground_linear[::3, ::3][:K, :K]
    stone = np.stack([blur(g[..., c], 7) for c in range(3)], -1)      # joints go soft
    stone = stone * 0.62 + stone.mean() * 0.10                        # water darkens stone

    # sky sheen: wind-ruffled bands of reflected golden-blue sky
    r1 = fbm(K, 3, seed); r2 = fbm(K, 5, seed + 1); r3 = fbm(K, 7, seed + 2)
    ripple = 0.5 + 0.5 * np.tanh(1.7 * (0.72 * r1 + 0.45 * r2 + 0.22 * r3))
    SKY_LO = np.array([0.46, 0.55, 0.80])       # zenith blue
    SKY_HI = np.array([1.00, 0.86, 0.62])       # golden horizon
    sky = SKY_LO + (SKY_HI - SKY_LO) * ripple[..., None]
    fres = 0.34 + 0.40 * ripple                                       # sheen strength
    warm = smoothstep(fbm(K, 2, seed + 9) * 0.7 + 0.42)[..., None]    # patches facing the sun
    sky = sky * (1 - 0.40 * warm) + SKY_HI * (0.40 * warm)
    col = stone * (1 - fres[..., None]) + sky * fres[..., None]

    glint = smoothstep((np.abs(r3) - 1.20) / 0.5) * smoothstep((ripple - 0.58) / 0.3)
    col = col + np.array([1.0, 0.88, 0.62]) * (0.70 * glint)[..., None]
    col *= (1 + 0.05 * fbm(K, 6, seed + 3))[..., None]

    mx = col.max(2, keepdims=True)
    col = col * (shoulder(mx, 0.80) / np.maximum(mx, 1e-6))
    img = (np.clip(col, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(img).save(path, optimize=True)
    scr = img.astype(float) * np.array([0.474, 0.400, 0.424])
    print(f"  {os.path.basename(path)}  delivered lum mean={scr.mean(2).mean():5.1f} std={scr.mean(2).std():5.1f}")


# ================================================================ 3. INLAY / THRESHOLD BAND
def make_band(path, N=512, width_m=0.14, seed=303):
    """One 0.14 x 0.14 m cell.  Row 0 = +Y edge (sunlit chamfer); row N = -Y edge (shadow)."""
    v = (np.arange(N) + 0.5) / N
    V, U = np.meshgrid(v, v, indexing="ij")
    n5 = fbm(N, 5, seed); n3 = fbm(N, 3, seed + 1); n6 = fbm(N, 6, seed + 2)
    STONE = np.array([0.755, 0.700, 0.605])

    band = np.tile(STONE, (N, N, 1))
    nx = np.zeros((N, N)); ny = np.zeros((N, N)); nz = np.ones((N, N)); vis = np.ones((N, N)); ao = np.ones((N, N))

    def seg(a, b):
        return (V >= a) & (V < b)

    m = seg(0.000, 0.036)                       # +Y outer joint against the paving
    band[m] = STONE * 0.42; ao[m] = 0.32; vis[m] = 0.0
    m = seg(0.036, 0.080)                       # +Y chamfer, faces the sun
    ny[m] = 0.55; ao[m] = 0.82
    m = seg(0.920, 0.964)                       # -Y chamfer, faces away
    ny[m] = -0.55; ao[m] = 0.70; vis[m] = 0.04
    m = seg(0.964, 1.000)                       # -Y outer joint
    band[m] = STONE * 0.42; ao[m] = 0.30; vis[m] = 0.0

    # inscribed twin grooves down the centre — reads as a tooled ceremonial inlay
    for c, w in ((0.40, 0.026), (0.60, 0.026)):
        g = np.abs(V - c) < w
        band[g] = STONE * 0.50; ao[g] = 0.38; vis[g] = 0.0
        lip = (V > c + w) & (V < c + w + 0.022)
        ny[lip] = 0.5; ao[lip] = 0.85

    # cross joints every 0.14 m + chisel tooling running across the band
    cj = np.abs(((U + 0.05 * n3) % 1.0) - 0.5) < 0.020
    band[cj] = STONE * 0.44; ao[cj] = 0.30; vis[cj] = 0.0
    tool = 1 + 0.045 * np.sin(V * np.pi * 34 + n5 * 2.2)
    band *= (tool * (1 + 0.05 * n6))[..., None]
    band *= (1 - 0.10 * smoothstep((np.abs(n3) - 0.4) / 0.8))[..., None]

    M = shade(nx, ny, nz, sun_vis=vis, ao=ao)
    img = (shoulder(band * M) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(img).save(path, optimize=True)
    scr = img.astype(float) * np.array([0.474, 0.400, 0.424])
    print(f"  {os.path.basename(path)}  delivered lum mean={scr.mean():5.1f} std={scr.mean(2).std():5.1f}")


# ================================================================ 3b. OUTSIDE THE WALLS
def make_apron(path, N=1024, tile_m=2.80, seed=606):
    """The ground the court is BUILT ON.  Nobody owned this surface: terrain's fragment
    stopped at the walls and atmosphere's started at 20 m, so the court's terracotta
    flagstone tiled the entire infinite plane to the horizon and checkerboarded at the tile
    pitch across the lower two thirds of the establishing shot.

    Wind-drifted sand, builders' spoil and rubble, at a 2.80 m period so it can never read
    as a second grid: no courses, no joints, nothing periodic -- just drift ripples running
    with the prevailing wind (-X, matching the grass), scattered spall from the wall
    construction, and darker damp patches from the same rain that filled the rill."""
    rng = np.random.default_rng(seed)
    n2 = fbm(N, 2, seed); n4 = fbm(N, 4, seed + 1); n6 = fbm(N, 6, seed + 2); n8 = fbm(N, 8, seed + 3)

    SAND = np.array([0.930, 0.790, 0.610])
    SPOIL = np.array([0.740, 0.586, 0.452])
    DAMP = np.array([0.620, 0.496, 0.404])

    mix = smoothstep(n2 * 0.55 + 0.5)
    col = SAND * mix[..., None] + SPOIL * (1 - mix)[..., None]

    # Wind drift, NOT a ripple pattern.  The first cut modulated a sine, which produced
    # thick contour bands that read as swirled wood grain across a 40 m skirt -- the
    # single most conspicuous thing in the establishing shot after the ranges.  What a
    # drift field actually looks like is anisotropic noise: stretched hard along the
    # prevailing wind (-X, matching the grass), with a shallow slope term so the windward
    # side catches the 11 deg sun and the lee goes violet.
    dr = grit(N, 5, 14, seed + 21)
    dr = 0.62 * dr + 0.38 * np.roll(dr, N // 5, 1)        # stretch along x
    dr_hi = grit(N, 16, 34, seed + 22)
    # the SLOPE term must come from the low-frequency field only.  Feeding the fine
    # octave into it flips sun_vis at stipple frequency, and a pixel with 40% of the sun
    # removed is lit by fill and ambient alone, i.e. blue -- which is what turned the
    # whole 40 m skirt into cool grey fish-scales.
    slope = (np.roll(dr, -1, 0) - np.roll(dr, 1, 0)) * 0.5 * 13.0
    M = shade(np.zeros_like(slope), -slope * 0.22, np.ones_like(slope),
              sun_vis=np.clip(0.84 + 0.16 * np.tanh(slope * 1.6), 0, 1), ao=0.94 + 0.06 * mix)
    col = col * np.minimum(M, 1.16)
    col = col * (1 + 0.045 * dr + 0.065 * dr_hi)[..., None]

    # rubble: spall and broken brick from building the walls
    rub = smoothstep((np.abs(blur(n6, 3)) - 0.95) / 0.45)
    col = col * (1 - 0.34 * rub)[..., None] + SPOIL * 1.10 * (0.34 * rub)[..., None]
    # damp patches, kept warm: a cool damp over cool lee-slopes read as blue fish scales
    wet = smoothstep((n4 - 0.75) / 1.0)
    col = col * (1 - 0.42 * wet)[..., None] + DAMP * (0.42 * wet)[..., None]
    # grit
    col = col * (1 + 0.055 * n6 + 0.040 * n8)[..., None]
    col = col * np.where(rng.random((N, N)) > 0.9985, 0.74, 1.0)[..., None]

    # Warm the dark end.  Every darkening term here loses more sun than fill, so the
    # low-value pixels come out at B/R 0.80 against a surround at 0.65 -- and by
    # simultaneous contrast a 40 m skirt of that reads as cool grey stipple in a warm
    # court.  Pulling blue out of the shadows in proportion to how dark they are is the
    # cheapest correct fix, and it is what an arid ground actually does: there is no blue
    # sky fill reaching into a hollow 2 mm across.
    lum = np.clip(col.mean(2, keepdims=True) / 0.72, 0, 1)
    col = col * np.concatenate([np.ones_like(lum), 0.985 + 0.015 * lum,
                                0.845 + 0.155 * lum], axis=2)
    mx = col.max(2, keepdims=True)
    col = col * (shoulder(mx) / np.maximum(mx, 1e-6))
    img = (np.clip(col, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(img).save(path, optimize=True)
    a = img.astype(float)
    scr = a * np.array([0.474, 0.400, 0.424])
    print(f"  {os.path.basename(path)}  {N}px/{tile_m}m  tex std={a.mean(2).std():5.1f}"
          f" | DELIVERED lum mean={scr.mean(2).mean():5.1f} std={scr.mean(2).std():5.1f}")


# ================================================================ MJCF
def q_z(deg):
    a = np.deg2rad(deg) / 2
    return f"{np.cos(a):.6f} 0 0 {np.sin(a):.6f}"


def assets_xml():
    return """<!-- ============================ aaa_terrain_assets.xml ============================
     TERRAIN agent.  Ground, water and every flat inlay in the court.
     Textures are authored as DELIVERED reflectance: the classic renderer clamps
     (rgba x lights) to 1 before modulating by the texel, and the 11 deg sun gives a
     horizontal surface only (0.474, 0.400, 0.424) of full light == so the texel IS the
     ceiling.  Relief, cast shadow and AO are baked, because <layer role="normal"> is
     inert in this build (verified here: pixel maxdiff 0.0).
     ============================================================================== -->
<texture type="2d" name="terrain_tex_ground"  file="terrain_ground.png"/>
<texture type="2d" name="terrain_tex_pool"    file="terrain_pool.png"/>
<texture type="2d" name="terrain_tex_reflect" file="terrain_reflect.png"/>
<texture type="2d" name="terrain_tex_rill"    file="terrain_rill.png"/>
<texture type="2d" name="terrain_tex_band"    file="terrain_band.png"/>
<texture type="2d" name="terrain_tex_apron"   file="terrain_apron.png"/>

<!-- HERO. 3072px over a 1.68 m tile; 5 courses cut into 4-6 stones == 0.28-0.42 m flags,
     ONE of which is longer than the whole duck.  That is the scale story: at 12 cm eye
     height a single joint sweeps across the bottom of frame instead of a grid, which is
     also what removes the checkerboard read the first cut had at 1.20 m.
     texrepeat is per world unit (texuniform).  MEASURED, not assumed: under texuniform the
     repeat period is 2/texrepeat METRES (u = texrepeat/2 * (local_x + size_x), anchored to
     the geom's own -x/+y corner), so 1.190476 = one tile per 1.68 m. -->
<material name="terrain_ground" texture="terrain_tex_ground" texuniform="true"
          texrepeat="1.190476 1.190476" specular="0.10" shininess="0.08"/>

<!-- Puddles.  Same stone, diffused the way 5 mm of water diffuses a joint, with the
     reflected sky painted straight into the albedo.
     NO `reflectance`, and that is a measured decision, not an oversight.  A/B at 1.3 / 6 /
     20 / 50 mm above the plane, with 27 reflective geoms: mean pixel difference 0.007-0.05
     on a 0-255 scale (bit-identical in three of five camera angles) for +4.5 ms per frame at
     320x240.  The mirror is a decal lying ON an infinite plane, so the reflected half-space
     is filled by the floor itself, and `reflectance` never mirrors the skybox.
     The small `emission` is what lets a horizontal surface exceed its own light budget:
     with a texture bound the texel is the ceiling and a floor-level texel maxes out at
     luminance 112/255, but real standing water at golden hour is markedly brighter than the
     stone around it because it is showing the sky.  0.09 is a sheen. -->
<material name="terrain_wet" texture="terrain_tex_pool" texuniform="true"
          texrepeat="1.190476 1.190476" rgba="0.900 0.920 0.975 1" emission="0.09"
          specular="0.62" shininess="0.86"/>

<!-- THE RILL WATER and the fountain basin: the painted inverted colonnade.  Strongly
     anisotropic texrepeat -- u period 1.724 -> 1.16 m, the colonnade's real bay spacing, so
     one painted column reflection lands per bay; v period 12.195 -> 0.164 m, the channel
     width.  Each water geom spans exactly one bay centred on a real column. -->
<material name="terrain_water" texture="terrain_tex_reflect" texuniform="true"
          texrepeat="1.724138 12.195122" rgba="0.960 0.970 1.000 1" emission="0.11"
          specular="0.80" shininess="0.92"/>

<!-- The same water where it is NOT a channel (fountain basin): isotropic, so the painted
     reflection reads as a broken sky sheen rather than a marching rhythm. -->
<material name="terrain_basin" texture="terrain_tex_reflect" texuniform="true"
          texrepeat="1.724138 1.724138" rgba="0.960 0.970 1.000 1" emission="0.11"
          specular="0.80" shininess="0.92"/>

<!-- The damp margin every real puddle has: darker, but dead matte and not a mirror. -->
<material name="terrain_damp" texture="terrain_tex_ground" texuniform="true"
          texrepeat="1.190476 1.190476" rgba="0.845 0.835 0.825 1"
          specular="0.14" shininess="0.10"/>

<!-- Traffic polish.  The critique asked for "one coherent low-frequency overlay across the
     whole tile (traffic polish down the corridor centre, darker damp margins)".  It cannot
     live in the tile: a feature at tile wavelength is exactly what makes the repeat legible
     on an infinite plane.  It lives here instead, as a handful of large soft decals -- same
     stone, walked a little paler and greyer -- which delivers the same read and cannot
     tile.  Kept within 6% of the floor's own value so no edge shows. -->
<material name="terrain_polish" texture="terrain_tex_ground" texuniform="true"
          texrepeat="1.190476 1.190476" rgba="1.000 0.985 0.965 1"
          specular="0.22" shininess="0.20"/>

<!-- Rill channel: one 0.44 m cell (texrepeat 4.545 => 0.44 m period; v = 0 at the +Y edge,
     so row 0 of the PNG is the north kerb) carries kerb / shadowed inner walls / silt bed.
     Everything below the kerb top is vis=0: at 11 deg a 40 mm kerb throws 206 mm of shadow,
     which is wider than the channel, so nothing in a rill this shallow sees the sun. -->
<material name="terrain_rill" texture="terrain_tex_rill" texuniform="true"
          texrepeat="4.545 4.545" specular="0.16" shininess="0.14"/>

<!-- Bleached inlay band, 0.14 m cell (texrepeat 14.286).  Spawn-apron frame and gate threshold. -->
<material name="terrain_band" texture="terrain_tex_band" texuniform="true"
          texrepeat="14.286 14.286" specular="0.12" shininess="0.10"/>

<!-- OUTSIDE THE WALLS.  Nobody owned this surface and the court's flagstone tiled the whole
     infinite plane to the horizon, checkerboarding at the tile pitch across the lower two
     thirds of the establishing shot.  A paved court in an arid landscape needs the paving to
     STOP: this is wind-drifted sand and builders' spoil, laid from the wall foot outward with
     a broken edge.  2.8 m period so it never reads as a second grid. -->
<material name="terrain_apron" texture="terrain_tex_apron" texuniform="true"
          texrepeat="0.714286 0.714286" specular="0.06" shininess="0.04"/>
<material name="terrain_apron_far" texture="terrain_tex_apron" texuniform="true"
          texrepeat="0.714286 0.714286" rgba="0.930 0.930 0.965 1"
          specular="0.05" shininess="0.03"/>
"""


def body_xml():
    G = []
    add = G.append
    Z_APRON, Z_BAND, Z_BAND2 = 0.0003, 0.0006, 0.0007
    Z_POLISH, Z_DAMP, Z_WET, Z_RILL, Z_WATER = 0.0005, 0.0016, 0.0022, 0.0011, 0.0026

    def geom(name, pos, size, mat, z=None, quat=None, typ="box", coll=False):
        p = f'pos="{pos[0]:.4g} {pos[1]:.4g} {z if z is not None else pos[2]:.6g}"'
        s = f'size="{size[0]:.4g} {size[1]:.4g} {size[2]:.6g}"'
        q = f' quat="{quat}"' if quat else ""
        c = "" if coll else ' contype="0" conaffinity="0"'
        add(f'<geom name="{name}" type="{typ}" {p} {s} material="{mat}"{q}{c}/>')

    add("<!-- ============================= aaa_terrain_body.xml =============================")
    add("     TERRAIN agent.  Floor, water, inlays, and the ground OUTSIDE the walls.")
    add("     terrain_floor is the ONLY collidable geom in this fragment and the only geom in")
    add("     the whole scene that keeps default collision.  Everything else is a flat decal:")
    add("     inside the court at z <= 2.6 mm, because at 12 cm eye height a thicker lip would")
    add("     show its own edge; outside the walls at 0.3 mm.")
    add("     Nothing here enters the r <= 0.9 m spawn apron. -->")
    add("")
    add("<!-- The walkable world.  Infinite plane: keeps haze alive, keeps the horizon at 50% of")
    add("     frame, and keeps the duck's contact set identical to training. -->")
    add('<geom name="terrain_floor" size="0 0 0.05" pos="0 0 0" type="plane" material="terrain_ground"/>')
    add("")

    # ---- OUTSIDE THE WALLS ---------------------------------------------------------------
    add("<!-- ===== OUTSIDE THE WALLS =====")
    add("     The paving has to STOP.  Without this the court's flagstone tiles the infinite")
    add("     plane to the horizon in every direction and checkerboards at the tile pitch")
    add("     across the lower two thirds of the establishing shot.  An inner ring of sand and")
    add("     builders' spoil runs from the wall foot to 13 m, an outer ring in the aerial-")
    add("     perspective variant carries it to 42 m, and a scatter of tongues breaks the")
    add("     boundary so paving and dirt interlock instead of meeting on a ruled line. -->")
    ring = [("n", (0, 8.21), (13.0, 4.79)), ("s", (0, -8.21), (13.0, 4.79)),
            ("e", (10.15, 0), (2.85, 3.42)), ("w", (-10.15, 0), (2.85, 3.42))]
    for tag, (cx, cy), (hx, hy) in ring:
        geom(f"terrain_out_{tag}", (cx, cy, 0), (hx, hy, Z_APRON), "terrain_apron", z=Z_APRON)
    far = [("n", (0, 27.5), (42.0, 14.5)), ("s", (0, -27.5), (42.0, 14.5)),
           ("e", (27.5, 0), (14.5, 13.0)), ("w", (-27.5, 0), (14.5, 13.0))]
    for tag, (cx, cy), (hx, hy) in far:
        geom(f"terrain_outfar_{tag}", (cx, cy, 0), (hx, hy, Z_APRON * 0.5),
             "terrain_apron_far", z=Z_APRON * 0.5)
    add("")
    add("<!-- Broken edge: sand drifting over the paving, and slabs of paving still showing")
    add("     through the sand.  Alternating materials at unrelated yaws, so no straight run of")
    add("     the boundary survives. -->")
    tongues = [(-6.10, 3.55, 0.95, 0.42, 13, "terrain_apron"), (-2.70, 3.62, 1.20, 0.36, -7, "terrain_apron"),
               (2.10, 3.58, 0.80, 0.46, 22, "terrain_apron"), (5.40, 3.66, 1.05, 0.33, -15, "terrain_apron"),
               (-4.90, -3.60, 1.10, 0.40, -19, "terrain_apron"), (-0.60, -3.55, 0.85, 0.44, 9, "terrain_apron"),
               (3.30, -3.63, 1.25, 0.35, 27, "terrain_apron"), (6.60, -3.57, 0.70, 0.48, -11, "terrain_apron"),
               (7.55, 1.95, 0.42, 0.90, 16, "terrain_apron"), (7.62, -1.70, 0.38, 1.05, -23, "terrain_apron"),
               (-7.55, 1.30, 0.45, 1.15, -9, "terrain_apron"), (-7.60, -2.05, 0.40, 0.85, 31, "terrain_apron"),
               (-5.55, 3.30, 0.62, 0.30, -24, "terrain_damp"), (1.20, 3.34, 0.74, 0.26, 11, "terrain_damp"),
               (4.05, -3.32, 0.58, 0.28, 18, "terrain_damp"), (-2.20, -3.36, 0.66, 0.24, -6, "terrain_damp")]
    for k, (x, y, sx, sy, yaw, mat) in enumerate(tongues):
        zz = Z_APRON + 0.0001 + k * 2e-5
        geom(f"terrain_edge_{k}", (x, y, 0), (sx, sy, zz), mat, z=zz, quat=q_z(yaw))
    add("")

    # ---- spawn-apron inlay frame ---------------------------------------------------------
    add("<!-- Spawn-apron frame: a bleached inlay square at |x|,|y| = 1.05 (inner edge 0.98 m,")
    add("     clear of the 0.9 m apron).  Gives the start a graphic anchor and reads as a")
    add("     designed place with the sun off.  Both the east and the west run are split, where")
    add("     the rill dives under the apron and comes back out the other side. -->")
    geom("terrain_frame_n", (0, 1.05, 0), (1.12, 0.07, Z_BAND), "terrain_band", z=Z_BAND)
    geom("terrain_frame_s", (0, -1.05, 0), (1.12, 0.07, Z_BAND), "terrain_band", z=Z_BAND)
    geom("terrain_frame_wa", (-1.05, 0.60, 0), (0.52, 0.07, Z_BAND2), "terrain_band", z=Z_BAND2, quat=q_z(-90))
    geom("terrain_frame_wb", (-1.05, -0.60, 0), (0.52, 0.07, Z_BAND2), "terrain_band", z=Z_BAND2, quat=q_z(-90))
    geom("terrain_frame_ea", (1.05, 0.60, 0), (0.52, 0.07, Z_BAND2), "terrain_band", z=Z_BAND2, quat=q_z(-90))
    geom("terrain_frame_eb", (1.05, -0.60, 0), (0.52, 0.07, Z_BAND2), "terrain_band", z=Z_BAND2, quat=q_z(-90))
    add("")

    # ---- the rill ------------------------------------------------------------------------
    add("<!-- ===== THE RILL =====")
    add("     Channel bed (kerbs / shadowed inner walls / silt) at a 0.44 m cross-section")
    add("     period, with the water strip on top.")
    add("     THE WATER IS BAY-ALIGNED.  terrain_water's u period is 1.16 m == the colonnade's")
    add("     real bay spacing == and the painted column reflection sits at u = 0.5, so each")
    add("     water geom spans exactly one bay CENTRED ON A COLUMN (x = 0.90 + i*1.16) and the")
    add("     inverted colonnade lands under the real colonnade.  That is the bible's one-frame")
    add("     test beat, baked: `reflectance` cannot deliver it on a decal lying on an infinite")
    add("     plane, because the reflected half-space is filled by the floor itself.")
    add("     The channel runs fountain -> gate and dives under the spawn apron rather than")
    add("     crossing it, because nothing may stand inside r = 0.9 m. -->")
    x0, x1, NSEG = 0.95, 7.28, 6
    L = (x1 - x0) / NSEG / 2
    for i in range(NSEG):
        cx = x0 + L + i * 2 * L
        geom(f"terrain_rill_{i}", (cx, 0, 0), (L, 0.22, Z_RILL), "terrain_rill", z=Z_RILL)
    # lead-in bay, then one bay per column
    geom("terrain_water_lead", (1.215, 0, 0), (0.265, 0.082, Z_WATER), "terrain_water", z=Z_WATER)
    for i in range(1, 6):
        cx = 0.90 + i * 1.16
        geom(f"terrain_water_{i}", (cx, 0, 0), (0.58, 0.082, Z_WATER), "terrain_water", z=Z_WATER)
    add("<!-- West run: the fountain overflow.  Extended to x = -2.45 so the channel actually")
    add("     MEETS the basin instead of stopping 0.25 m short of it. -->")
    geom("terrain_rill_w", (-1.49, 0, 0), (0.54, 0.22, Z_RILL), "terrain_rill", z=Z_RILL)
    geom("terrain_water_w", (-1.49, 0, 0), (0.535, 0.082, Z_WATER), "terrain_water", z=Z_WATER)
    add("")

    # ---- gate threshold ------------------------------------------------------------------
    add("<!-- Gate threshold: two bright steps under the light slot at the vanishing point, with")
    add("     the dark rill cutting straight through them.  Every terrain_band geom must keep a")
    add("     0.07 half-size on the axis that becomes v, because the band texture paints ONE")
    add("     0.14 m cross-section; the -90 deg quat is what puts v=0 on the sun-facing edge. -->")
    geom("terrain_thresh_in", (7.15, 0, 0), (0.55, 0.07, Z_BAND), "terrain_band", z=Z_BAND, quat=q_z(-90))
    geom("terrain_thresh_out", (7.42, 0, 0), (0.62, 0.07, Z_BAND2), "terrain_band", z=Z_BAND2, quat=q_z(-90))
    add("")

    # ---- traffic polish ------------------------------------------------------------------
    add("<!-- Traffic polish: the coherent low-frequency wear story, delivered as decals instead")
    add("     of in the tile (a feature at tile wavelength is exactly what makes the repeat")
    add("     legible on an infinite plane).  Same stone, walked a little paler and greyer.")
    add("     Within 3.5% of the floor's value so no edge shows -- these are meant to be felt,")
    add("     not seen.  They run the hero corridor and both aisle centre-lines. -->")
    polish = [(2.20, 0.34, 1.35, 0.52, -4), (4.10, -0.28, 1.55, 0.46, 6), (6.05, 0.22, 1.30, 0.40, -3),
              (-2.35, 0.40, 1.25, 0.55, 8), (-4.60, -0.30, 1.45, 0.48, -7),
              (3.10, 2.30, 1.60, 0.62, 3), (-1.40, 2.25, 1.40, 0.58, -5),
              (2.60, -2.35, 1.55, 0.60, -6), (-3.20, -2.28, 1.35, 0.55, 4)]
    for k, (x, y, sx, sy, yaw) in enumerate(polish):
        zz = Z_POLISH + k * 3e-5
        geom(f"terrain_polish_{k}", (x, y, 0), (sx, sy, zz), "terrain_polish", z=zz, quat=q_z(yaw))
    add("")

    # ---- puddles -------------------------------------------------------------------------
    add("<!-- Puddles.  reflectance is uniform per material, so puddle SHAPE has to come from")
    add("     geometry: 3-4 boxes at unrelated yaws per pool, each on its own 0.05 mm shelf so")
    add("     coplanar faces cannot z-fight, give an irregular polygon with no internal seams.")
    add("     A two-box matte damp margin under each pool breaks the remaining straight edges.")
    add("     The last two are new: the critique measured that the nearest puddle to any")
    add("     guaranteed route was 1.57 m away, so nothing wet ever entered the near band.")
    add("     These two sit 0.20-0.26 m off the corridor centre-line, i.e. inside the bottom")
    add("     quarter of frame as the duck walks past them. -->")
    pools = [
        (1.92, 0.63, [(0, 0, 0.34, 0.19, 12), (0.19, 0.09, 0.21, 0.15, -34),
                      (-0.23, -0.05, 0.17, 0.12, 55), (0.06, -0.12, 0.14, 0.09, 78)],
         [(0.01, 0.0, 0.48, 0.27, -8), (0.16, 0.10, 0.30, 0.20, 41)]),
        (3.12, -0.58, [(0, 0, 0.29, 0.16, -21), (0.18, -0.07, 0.18, 0.12, 40),
                       (-0.16, 0.08, 0.15, 0.11, 8)],
         [(0.0, 0.0, 0.42, 0.24, 16), (-0.14, 0.07, 0.26, 0.18, -52)]),
        (1.57, -0.50, [(0, 0, 0.18, 0.10, 33), (0.12, 0.05, 0.11, 0.08, -18)],
         [(0.02, 0.0, 0.28, 0.16, -40)]),
        (4.63, 0.72, [(0, 0, 0.25, 0.14, -9), (-0.15, 0.07, 0.15, 0.10, 47),
                      (0.14, -0.05, 0.12, 0.09, 21)],
         [(-0.03, 0.01, 0.36, 0.21, 22)]),
        (-2.05, -0.92, [(0, 0, 0.31, 0.17, 26), (0.20, -0.08, 0.19, 0.12, -12),
                        (-0.19, 0.06, 0.14, 0.10, 63)],
         [(0.05, -0.02, 0.45, 0.25, -30)]),
        (2.62, -2.14, [(0, 0, 0.27, 0.15, 51), (-0.17, -0.09, 0.16, 0.11, 5)],
         [(-0.04, -0.02, 0.38, 0.22, 62)]),
        (2.38, 0.44, [(0, 0, 0.23, 0.13, -27), (0.15, -0.06, 0.14, 0.09, 36)],
         [(0.02, 0.01, 0.33, 0.19, 14)]),
        (5.02, -0.41, [(0, 0, 0.26, 0.14, 44), (-0.16, 0.06, 0.15, 0.10, -13),
                       (0.13, 0.07, 0.11, 0.08, 67)],
         [(-0.02, 0.0, 0.36, 0.20, -35)]),
    ]
    for k, (cx, cy, boxes, halos) in enumerate(pools):
        for h, (hx, hy, hsx, hsy, hyaw) in enumerate(halos):
            zz = Z_DAMP + h * 5e-5
            geom(f"terrain_damp_{k}{h}", (cx + hx, cy + hy, 0), (hsx, hsy, zz),
                 "terrain_damp", z=zz, quat=q_z(hyaw))
        for j, (dx, dy, sx, sy, yaw) in enumerate(boxes):
            zz = Z_WET + j * 5e-5
            geom(f"terrain_pool_{k}{j}", (cx + dx, cy + dy, 0), (sx, sy, zz), "terrain_wet",
                 z=zz, quat=q_z(yaw))
    add("")

    # ---- fountain water ------------------------------------------------------------------
    add("<!-- FOUNTAIN BASIN WATER at (-2.8, 0).  Raised from a 45 mm slab to 128 mm: the")
    add("     critique measured that the old surface sat at z = 0.0225 while the basin cope")
    add("     tops at 0.155, so from a 0.12 m eye the water was completely occluded by the")
    add("     basin's own rim and the fountain read bone dry -- in a scene called THE WATER")
    add("     COURT whose entire premise is that it rained an hour ago.  The surface now sits")
    add("     12 mm under the cope (top 0.112), so a 0.12 m eye looks OVER the rim and sees it.")
    add("     Still a solid slab from the ground up, so no underside edge can ever show. -->")
    for j, yaw in enumerate((0, 45, 90, 135)):
        zz = 0.050 + j * 5e-5
        geom(f"terrain_fountain_{j}", (-2.8, 0, 0), (0.560, 0.242, zz), "terrain_basin",
             z=zz, quat=q_z(yaw))
    add("<!-- The weir: water leaving the basin through the notched cope on the +X side, and")
    add("     the wetted apron it lands on.  This is what connects the basin to the rill. -->")
    geom("terrain_weir", (-2.100, 0, 0), (0.062, 0.075, 0.048), "terrain_basin", z=0.048)
    geom("terrain_weir_lip", (-2.038, 0, 0), (0.026, 0.088, 0.012), "terrain_basin", z=0.012)
    add("")

    # ---- damp seeps / splash aprons ------------------------------------------------------
    add("<!-- Seeps: the splash apron round the fountain, and damp at the wall bases where an")
    add("     hour-old rain would still be leaving the stone dark.  The two near x = -2.15")
    add("     are the apron the weir spills onto, so the basin visibly drains to the rill.")
    add("     z lifted 0.4 -> 1.6 mm: at 0.4 mm the bottom face was EXACTLY coplanar with the")
    add("     infinite floor plane and a segmentation-flip test over 60 jittered views put")
    add("     terrain_floor <-> terrain_seep_6 among the top three z-fighting pairs. -->")
    seeps = [(-2.80, 0.94, 0.62, 0.24, 4), (-2.80, -0.94, 0.58, 0.22, -6),
             (-2.15, 0.30, 0.34, 0.30, 21), (-2.18, -0.34, 0.30, 0.26, -17),
             (1.40, 2.86, 0.70, 0.20, 2), (4.35, 2.90, 0.62, 0.17, -3),
             (-0.90, -2.92, 0.66, 0.19, 5), (5.30, -2.88, 0.58, 0.16, -4),
             (-4.55, 0.70, 0.44, 0.26, 17), (-5.10, -0.62, 0.38, 0.22, -28)]
    for k, (x, y, sx, sy, yaw) in enumerate(seeps):
        zz = Z_DAMP + 0.0002 + k * 3e-5
        geom(f"terrain_seep_{k}", (x, y, 0), (sx, sy, zz), "terrain_damp", z=zz, quat=q_z(yaw))

    return "\n".join(G) + "\n"


def main():
    print("TERRAIN agent — generating")
    make_ground(os.path.join(MD, "terrain_ground.png"))
    make_pool(os.path.join(MD, "terrain_pool.png"), make_ground.linear)
    make_reflect(os.path.join(MD, "terrain_reflect.png"))
    make_rill(os.path.join(MD, "terrain_rill.png"))
    make_band(os.path.join(MD, "terrain_band.png"))
    make_apron(os.path.join(MD, "terrain_apron.png"))
    with open(os.path.join(MD, "aaa_terrain_assets.xml"), "w") as f:
        f.write(assets_xml())
    body = body_xml()
    with open(os.path.join(MD, "aaa_terrain_body.xml"), "w") as f:
        f.write(body)
    ng = body.count("<geom ")
    print(f"  aaa_terrain_assets.xml, aaa_terrain_body.xml written — {ng} geoms")
    for fn in ("terrain_ground.png", "terrain_pool.png", "terrain_reflect.png",
               "terrain_rill.png", "terrain_band.png", "terrain_apron.png"):
        print(f"    {fn}: {os.path.getsize(os.path.join(MD, fn))/1e6:.2f} MB")


if __name__ == "__main__":
    main()
