#!/usr/bin/env python3
"""Seamlessly looping stereoscopic 'shifting waves' wallpaper (side-by-side, left eye left).

Depth map D(x,y,t) in pixels of disparity at 1920 width, 4 (crest, near) .. 40 (trough, far).
A surface point at texture coordinate u appears at u - D/2 in the left eye and u + D/2 in the
right eye, so disparity (xR - xL) = D >= 0 (always behind the screen).
Each eye is built by inverse mapping: for output pixel x, solve u -/+ D(u)/2 = x by fixed-point
iteration and evaluate the procedural texture at u.  All temporal phases use integer cycles per
clip, so frame N == frame 0 and the clip loops seamlessly.

Writes raw RGB frames (3840x1080, 30 fps) to stdout; build.sh encodes them.
Usage: make-waves.py [--frames N] | make-waves.py --png-frame K OUT.png (needs pillow)
Needs Python 3 and numpy. Public domain (CC0 1.0), see LICENSE.
"""
import sys
import numpy as np

W, H, FPS, SECS = 1920, 1080, 30, 12
NF = FPS * SECS
DMIN, DMAX = 4.0, 40.0
TAU = 2 * np.pi

# (amplitude, kx cycles across width, ky cycles across height, temporal cycles per clip, phase)
WAVES = [
    (1.00, 1.3, 0.55, 1, 0.0),
    (0.70, -0.8, 1.10, 2, 1.7),
    (0.45, 2.1, -0.70, 3, 4.1),
    (0.30, 0.4, 1.90, -2, 2.6),
]
ASUM = sum(abs(w[0]) for w in WAVES)

ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
yn = ys / H


def wave(u, t):
    """Returns h in [-1,1] (+1 = crest = nearest) and dh/du, dh/dy (per pixel)."""
    h = np.zeros_like(u); hx = np.zeros_like(u); hy = np.zeros_like(u)
    for a, kx, ky, n, ph in WAVES:
        arg = TAU * (kx * u / W + ky * yn - n * t / NF) + ph
        h += a * np.sin(arg)
        c = a * np.cos(arg)
        hx += c * (TAU * kx / W)
        hy += c * (TAU * ky / H)
    return h / ASUM, hx / ASUM, hy / ASUM


def disparity(h):
    return DMIN + (DMAX - DMIN) * (1.0 - h) * 0.5


# Static tileable fine texture (soft speckle) for the eyes to fuse on.
rng = np.random.default_rng(7)
NT = 512
noise = rng.random((NT, NT)).astype(np.float32)
for _ in range(2):  # small box blur, wraps -> tileable, kills pixel shimmer
    noise = sum(np.roll(np.roll(noise, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9
noise = (noise - noise.mean()) / noise.std()


def sample_noise(u, v):
    x0 = np.floor(u); y0 = np.floor(v)
    fx = u - x0; fy = v - y0
    x0 = x0.astype(np.int64) % NT; y0 = y0.astype(np.int64) % NT
    x1 = (x0 + 1) % NT; y1 = (y0 + 1) % NT
    return ((noise[y0, x0] * (1 - fx) + noise[y0, x1] * fx) * (1 - fy)
            + (noise[y1, x0] * (1 - fx) + noise[y1, x1] * fx) * fy)


def grid(c, spacing, width):
    """Anti-aliased line profile: 1 on line centres, ~width px wide, smooth falloff."""
    d = np.abs(((c / spacing + 0.5) % 1.0) - 0.5) * spacing
    return np.clip(1.0 - (d - width * 0.5) / 1.2, 0.0, 1.0)


LIGHT = np.array([-0.45, -0.55, 0.70], np.float32); LIGHT /= np.linalg.norm(LIGHT)
vign = (1.0 - 0.35 * (((xs / W - 0.5) * 1.6) ** 2 + ((ys / H - 0.5) * 1.9) ** 2)).clip(0.4, 1)


def eye(t, sign):
    """sign=-1 for left eye (point drawn at u - D/2), +1 for right eye."""
    u = xs.copy()
    for _ in range(4):
        h, _, _ = wave(u, t)
        u = xs - sign * disparity(h) * 0.5
    h, hx, hy = wave(u, t)
    # Height field for shading: crests up toward viewer. Scale slopes to read as gentle swells.
    s = 140.0
    nx, ny, nz = -hx * s, -hy * s, np.ones_like(hx)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    lam = (nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]) * inv
    # Specular-ish sheen
    hz = LIGHT + np.array([0, 0, 1], np.float32); hz /= np.linalg.norm(hz)
    spec = np.clip((nx * hz[0] + ny * hz[1] + nz * hz[2]) * inv, 0, 1) ** 40
    # Grid drifts slightly with the surface in v (sheet stretches as it swells)
    v = ys + 10.0 * h
    # Vertical-line spacing (128) >> max disparity (40) so the eyes can't pair the wrong lines.
    g = np.maximum(grid(u, 128.0, 2.2), grid(v, 96.0, 2.2))
    g2 = grid(v + 48, 96.0, 1.6) * 0.35
    tex = sample_noise(u * 0.9, v * 0.9)
    depth_fade = 0.75 + 0.25 * (h * 0.5 + 0.5)  # nearer slightly brighter
    base = (0.10 + 0.20 * lam + 0.035 * tex) * depth_fade
    lines = (0.42 * g + 0.18 * g2) * (0.55 + 0.45 * lam) * depth_fade
    lum = (base + lines) * vign + 0.22 * spec
    col = np.stack([lum * 0.86, lum * 0.94, lum * 1.05], -1)  # pale blue-grey, low saturation
    return col


def frame(i):
    t = float(i)
    l = eye(t, -1); r = eye(t, +1)
    img = np.concatenate([l, r], 1)
    return (np.clip(img, 0, 1) ** (1 / 1.1) * 255 + 0.5).astype(np.uint8)


def main():
    args = sys.argv[1:]
    if "--png-frame" in args:
        from PIL import Image
        k = int(args[args.index("--png-frame") + 1]); out = args[args.index("--png-frame") + 2]
        Image.fromarray(frame(k)).save(out); return
    n = int(args[args.index("--frames") + 1]) if "--frames" in args else NF
    out = sys.stdout.buffer
    for i in range(n):
        out.write(frame(i).tobytes())
        if i % 30 == 0:
            print(f"frame {i}/{n}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
