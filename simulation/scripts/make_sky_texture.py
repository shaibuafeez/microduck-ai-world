#!/usr/bin/env python3
"""Generate a seamless procedural sky cubemap for scene_obstacles.xml.

Six faces, sampled from one periodic 3D noise volume, so clouds are continuous
across every face edge with no seam work. A hazy mountain range is baked into
the horizon: distant terrain as texture costs nothing at runtime, where the
same silhouette as geometry would be thousands of collidable triangles.

The sun direction matches the scene's key light, so cloud lighting and the
robot's shadow agree.

    python3 scripts/make_sky_texture.py
"""
import numpy as np
from PIL import Image

S = 768                      # face resolution
V = 96                       # noise volume side
OUT = "src/mjlab_microduck/robot/microduck/sky_{}.png"

SUN = np.array([0.45, -0.55, 0.75])
SUN /= np.linalg.norm(SUN)

ZENITH  = np.array([0.16, 0.34, 0.62])
HORIZON = np.array([0.78, 0.85, 0.92])
HAZE    = np.array([0.82, 0.86, 0.90])
GROUND  = np.array([0.40, 0.40, 0.42])
ROCK    = np.array([0.15, 0.18, 0.26])   # cool blue-grey; distance shifts it to HAZE


def periodic_volume(seed: int) -> np.ndarray:
    """1/f noise on a V^3 torus. Periodic, so sampling never hits a seam."""
    r = np.random.default_rng(seed).normal(size=(V, V, V))
    F = np.fft.fftn(r)
    k = np.fft.fftfreq(V)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    f = np.sqrt(kx**2 + ky**2 + kz**2)
    f[0, 0, 0] = 1e-6
    out = np.real(np.fft.ifftn(F * f**-1.7))
    return (out - out.mean()) / (out.std() + 1e-9)


def sample(vol: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Trilinear sample of a periodic volume at float coords p[...,3]."""
    p = p % V
    i0 = np.floor(p).astype(np.int32)
    fr = p - i0
    i1 = (i0 + 1) % V
    x0, y0, z0 = i0[..., 0], i0[..., 1], i0[..., 2]
    x1, y1, z1 = i1[..., 0], i1[..., 1], i1[..., 2]
    fx, fy, fz = fr[..., 0], fr[..., 1], fr[..., 2]
    c00 = vol[x0, y0, z0] * (1 - fx) + vol[x1, y0, z0] * fx
    c01 = vol[x0, y0, z1] * (1 - fx) + vol[x1, y0, z1] * fx
    c10 = vol[x0, y1, z0] * (1 - fx) + vol[x1, y1, z0] * fx
    c11 = vol[x0, y1, z1] * (1 - fx) + vol[x1, y1, z1] * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


def fbm(vol: np.ndarray, d: np.ndarray, octaves: int = 5) -> np.ndarray:
    total = np.zeros(d.shape[:-1])
    amp, freq = 1.0, 22.0
    norm = 0.0
    for _ in range(octaves):
        total += amp * sample(vol, d * freq)
        norm += amp
        amp *= 0.5
        freq *= 2.05
    return total / norm


# Ridge line: mountain top elevation as a function of azimuth. Periodic in
# azimuth by construction (sum of harmonics), so the range closes on itself.
_rng = np.random.default_rng(21)


def _harmonics(seed: int, n_max: int = 18):
    r = np.random.default_rng(seed)
    return [(r.uniform(0.5, 1.0) / n**0.55, n, r.uniform(0, 2 * np.pi)) for n in range(1, n_max)]


_RANGES = [
    # (harmonics, base elevation, amplitude, haze mix, rock brightness)
    # Peaks reach ~0.45 elevation so they read as mountains, not a kerb. Haze
    # rises with distance — that gradient is what sells depth, not the outline.
    (_harmonics(21), 0.105, 0.150, 0.56, 1.20),   # far
    (_harmonics(37), 0.065, 0.110, 0.34, 0.88),   # mid
    (_harmonics(53), 0.030, 0.080, 0.13, 0.58),   # near
]


def ridge(az: np.ndarray, H, base: float, amp: float) -> np.ndarray:
    """Ridged silhouette. Summing plain sines gives dunes; taking 1-|sin| per
    octave puts a cusp wherever the term crosses zero, which is what makes a
    skyline read as peaks instead of hills."""
    r = sum(a * (1.0 - np.abs(np.sin(n * az + ph))) for a, n, ph in H)
    lo, hi = 0.55, 2.35                      # empirical range of the sum
    r = np.clip((r - lo) / (hi - lo), 0, 1)
    return base + amp * r**1.25              # exponent sharpens the peaks


FACES = {                     # face -> direction from (u, v) in [-1, 1]
    # Convention verified empirically against MuJoCo's renderer (solid-colour
    # probe): "right" is the -X face and "left" is +X, which is the reverse of
    # what the names suggest. Getting this backwards puts a hard seam in the sky.
    "right": lambda u, v: np.stack([-np.ones_like(u), -u, v], -1),   # -X
    "left":  lambda u, v: np.stack([np.ones_like(u), u, v], -1),     # +X
    "front": lambda u, v: np.stack([-u, np.ones_like(u), v], -1),    # +Y
    "back":  lambda u, v: np.stack([u, -np.ones_like(u), v], -1),    # -Y
    "up":    lambda u, v: np.stack([u, -v, np.ones_like(u)], -1),    # +Z
    "down":  lambda u, v: np.stack([u, v, -np.ones_like(u)], -1),    # -Z
}


def main() -> None:
    clouds = periodic_volume(3)
    detail = periodic_volume(9)

    ax = np.linspace(-1, 1, S)
    u = ax[None, :].repeat(S, 0)
    v = ax[::-1, None].repeat(S, 1)          # row 0 is the top of the image

    for name, fn in FACES.items():
        d = fn(u, v).astype(np.float64)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        e = d[..., 2]                        # elevation, -1 (down) .. +1 (up)
        az = np.arctan2(d[..., 1], d[..., 0])

        # --- sky gradient + sun -------------------------------------------
        t = np.clip(e, 0, 1) ** 0.55
        col = HORIZON[None, None] * (1 - t[..., None]) + ZENITH[None, None] * t[..., None]

        cos_sun = np.clip((d * SUN).sum(-1), -1, 1)
        col += (np.clip(cos_sun, 0, 1) ** 220)[..., None] * np.array([2.2, 2.0, 1.7])
        col += (np.clip(cos_sun, 0, 1) ** 8)[..., None] * np.array([0.30, 0.24, 0.15])

        # --- clouds --------------------------------------------------------
        n = fbm(clouds, d) + 0.35 * fbm(detail, d * 2.3, 3)
        cover = np.clip((n - 0.02) * 2.4, 0, 1)
        cover *= np.clip((e - 0.015) * 7.0, 0, 1)          # thin out at the horizon
        shade = np.clip(0.55 + 0.65 * fbm(detail, d * 3.1, 3), 0.35, 1.15)
        cloud_col = np.stack([shade, shade * 0.995, shade * 0.985], -1) * np.array([1.02, 1.01, 1.0])
        col = col * (1 - cover[..., None]) + cloud_col * cover[..., None]

        # --- mountains, drawn far to near ------------------------------
        # A flat fill under a ridge line reads as cardboard. Each range gets
        # slope shading from the sun, noise gullies, and a snowline.
        for H, base, amp, hazemix, dark in _RANGES:
            top = ridge(az, H, base, amp)
            band = (e < top) & (e > -0.05)
            if not band.any():
                continue

            # how far below this range's crest we are, 0 at the ridge
            drop = np.clip((top - e) / (amp * 0.9 + 1e-6), 0, 1)

            # Slope shading: d(top)/d(az) is the silhouette's local tilt, which
            # stands in for which way the face turns relative to the sun.
            eps = 0.004
            daz = (ridge(az + eps, H, base, amp) - ridge(az - eps, H, base, amp)) / (2 * eps)
            slope = np.tanh(daz * 1.9)
            sun_az = np.arctan2(SUN[1], SUN[0])
            facing = np.cos(az - sun_az)
            lit = np.clip(0.62 + 0.42 * (-slope) * np.sign(facing) + 0.18 * facing, 0.30, 1.30)

            # Rock detail: 3D noise gives gullies and crags inside the body.
            crag = fbm(detail, d * 1.7, 4)
            gully = np.clip(0.80 + 0.55 * crag, 0.45, 1.35)

            body = ROCK[None, None] * (dark * (0.55 + 0.85 * drop[..., None])) \
                   * lit[..., None] * gully[..., None]

            # Snow above a per-range snowline, thinning on steep faces.
            snowline = top - amp * 0.42
            snow = np.clip((e - snowline) / (amp * 0.34 + 1e-6), 0, 1)
            snow *= np.clip(0.35 + 0.9 * (1.0 - np.abs(slope)), 0, 1)
            snow *= np.clip(0.45 + 0.75 * crag, 0, 1)
            snow_col = np.array([0.96, 0.97, 1.0])[None, None] * np.clip(lit, 0.55, 1.2)[..., None]
            body = body * (1 - snow[..., None]) + snow_col * snow[..., None]

            mount = body * (1 - hazemix) + HAZE[None, None] * hazemix
            col = np.where(band[..., None], mount, col)

        # --- below the horizon ---------------------------------------------
        # The ground plane hides most of this; keep it hazy rather than dark so
        # the seam at the horizon stays invisible.
        below = np.clip(-e * 9, 0, 1)
        floorcol = ROCK * 0.58 * (1 - 0.13) + HAZE * 0.13
        col = col * (1 - below[..., None]) + floorcol[None, None] * below[..., None]

        img = (np.clip(col, 0, 1) ** (1 / 1.35) * 255).astype(np.uint8)
        Image.fromarray(img).save(OUT.format(name))
        print("wrote", OUT.format(name))


if __name__ == "__main__":
    main()
