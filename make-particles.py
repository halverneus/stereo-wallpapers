#!/usr/bin/env python3
"""Seamlessly looping side-by-side stereo particle wallpaper for Sparky Stereo OS.

Writes raw RGB frames (3840x1080, 30 fps, left eye on the left half) to stdout;
build.sh encodes them. Public domain (CC0 1.0), see LICENSE.

Design:
  * Every particle sits at a fixed depth behind the screen. Disparity d (px at 1920 width)
    ranges from +4 (nearest) to +40 (farthest); left eye at x - d/2, right eye at x + d/2,
    so the right copy is always further right (uncrossed, never pops out).
  * Motion is a closed Lissajous-style orbit around a home position whose frequencies are
    whole numbers of cycles per clip, plus a twinkle with whole cycles. Position, brightness,
    everything is periodic in t with period = clip length, so frame N == frame 0 and the
    loop is seamless with no wrapping or pop-in.
  * Near particles: larger, brighter, bigger/faster orbits. Far: small, dim, slow.
Needs Python 3 and numpy.
"""
import sys
import numpy as np

EYE_W, H = 1920, 1080
FPS, SECONDS = 30, 12
N_FRAMES = FPS * SECONDS
N_PART = 420
D_NEAR, D_FAR = 4.0, 40.0
SEED = 7

rng = np.random.default_rng(SEED)

# depth parameter s: 0 = nearest, 1 = farthest. Bias slightly toward far (more small
# background specks, fewer big foreground motes) like a real volume.
s = rng.random(N_PART) ** 0.7
disp = D_NEAR + (D_FAR - D_NEAR) * s
near = 1.0 - s  # 1 = nearest

radius = 1.0 + 3.6 * near ** 1.6          # gaussian sigma-ish, px
bright = 0.30 + 0.75 * near ** 1.2
amp = 12 + 110 * near ** 1.5               # orbit amplitude, px

# Whole-number cycles per clip => periodic. Mostly 1, near particles sometimes 2.
fx = rng.integers(1, 2 + (near > 0.6), N_PART)
fy = rng.integers(1, 2 + (near > 0.6), N_PART)
px = rng.random(N_PART) * 2 * np.pi
py = rng.random(N_PART) * 2 * np.pi
ax = amp * (0.6 + 0.4 * rng.random(N_PART))
ay = amp * (0.4 + 0.4 * rng.random(N_PART))
# secondary harmonic for less mechanical paths
ax2 = ax * 0.25 * rng.random(N_PART)
ay2 = ay * 0.25 * rng.random(N_PART)
px2 = rng.random(N_PART) * 2 * np.pi
py2 = rng.random(N_PART) * 2 * np.pi

hx = rng.random(N_PART) * (EYE_W + 120) - 60
hy = rng.random(N_PART) * (H + 120) - 60

tw_f = rng.integers(1, 4, N_PART)          # twinkle cycles per clip
tw_p = rng.random(N_PART) * 2 * np.pi
tw_a = 0.25 * rng.random(N_PART)

# Low-saturation tint: pale warm white to pale cool white
warm = rng.random(N_PART)
tint = np.stack([0.95 + 0.05 * warm, np.full(N_PART, 0.95), 0.90 + 0.10 * (1 - warm)], axis=1)

# Background: near-black with a very faint cool vertical gradient
yy = np.linspace(0, 1, H)[:, None, None]
bg = (np.array([0.010, 0.012, 0.020]) + yy * np.array([0.012, 0.014, 0.024])).astype(np.float32)
bg = np.broadcast_to(bg, (H, EYE_W, 3))

order = np.argsort(-disp)  # draw far first (additive anyway, but harmless)


def splat(img, x, y, r, col):
    R = int(np.ceil(r * 3.2))
    x0, y0 = int(np.floor(x)) - R, int(np.floor(y)) - R
    x1, y1 = x0 + 2 * R + 2, y0 + 2 * R + 2
    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x1, EYE_W), min(y1, H)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    gx = np.arange(cx0, cx1) + 0.5 - x
    gy = np.arange(cy0, cy1) + 0.5 - y
    # soft core + faint halo
    d2 = gy[:, None] ** 2 + gx[None, :] ** 2
    g = np.exp(-d2 / (2 * r * r)) + 0.12 * np.exp(-d2 / (2 * (2.6 * r) ** 2))
    img[cy0:cy1, cx0:cx1] += g[:, :, None] * col


def frame(i):
    t = 2 * np.pi * i / N_FRAMES
    x = hx + ax * np.sin(fx * t + px) + ax2 * np.sin(2 * fx * t + px2)
    y = hy + ay * np.sin(fy * t + py) + ay2 * np.sin(2 * fy * t + py2)
    b = bright * (1 - tw_a + tw_a * np.cos(tw_f * t + tw_p))
    L = np.array(bg, dtype=np.float32)
    Rimg = np.array(bg, dtype=np.float32)
    for k in order:
        col = tint[k] * b[k]
        splat(L, x[k] - disp[k] / 2, y[k], radius[k], col)
        splat(Rimg, x[k] + disp[k] / 2, y[k], radius[k], col)
    out = np.concatenate([L, Rimg], axis=1)
    # gentle tone map to avoid hard clipping where motes overlap
    out = 1 - np.exp(-out * 1.4)
    out = out / (1 - np.exp(-1.4))
    # Dither (+/- half a step, fixed per frame index so a rebuild is identical) rather than plain
    # rounding: the near-black gradient bands visibly otherwise.
    dither = np.random.default_rng(1000 + i).random(out.shape[:2])[:, :, None]
    return (np.clip(out, 0, 1) ** (1 / 1.1) * 255 + dither).clip(0, 255).astype(np.uint8)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--info":
        print(f"disparity {disp.min():.1f}..{disp.max():.1f} px, {N_FRAMES} frames")
        sys.exit()
    outp = sys.stdout.buffer
    for i in range(N_FRAMES):
        outp.write(frame(i).tobytes())
