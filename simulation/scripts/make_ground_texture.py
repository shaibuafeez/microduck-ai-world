#!/usr/bin/env python3
"""Generate the seamless concrete ground texture used by scene_obstacles.xml.

Why FFT and not tiled Perlin: filtering white noise in the frequency domain is
inherently periodic, so the result tiles with no seam and no blending tricks.
Re-run this to change the ground; MuJoCo reads the PNG at scene load.

    python3 scripts/make_ground_texture.py
"""
import numpy as np
from PIL import Image

N = 1024
OUT = "src/mjlab_microduck/robot/microduck/ground_concrete.png"


def fnoise(exponent: int | float, seed: int) -> np.ndarray:
    """Seamless 1/f^exponent noise. Higher exponent = broader, smoother features."""
    r = np.random.default_rng(seed).normal(size=(N, N))
    F = np.fft.fft2(r)
    fy = np.fft.fftfreq(N)[:, None]
    fx = np.fft.fftfreq(N)[None, :]
    f = np.sqrt(fx**2 + fy**2)
    f[0, 0] = 1e-6                      # avoid dividing the DC term by zero
    F *= f ** (-exponent)
    out = np.real(np.fft.ifft2(F))
    return (out - out.mean()) / (out.std() + 1e-9)


def main() -> None:
    h = 0.62 * fnoise(1.9, 1) + 0.28 * fnoise(1.2, 2) + 0.10 * fnoise(0.35, 3)
    h = (h - h.min()) / (h.max() - h.min())

    v = 0.38 + 0.19 * h                 # keep the value range narrow — concrete is flat
    rgb = np.stack([v * 1.015, v * 1.000, v * 0.972], axis=-1)   # faint warm cast

    speck = np.random.default_rng(5).random((N, N))
    rgb[speck > 0.9985] *= 0.72         # sparse dark aggregate

    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
