"""A tiny NumPy point/Gaussian-splat rasteriser.

Not a radiance-field renderer -- it is a z-buffered disc splatter used for:
* the mock-scene generator (turn a coloured point cloud into RGB + depth),
* ``preview_render.png`` / ``preview_orbit.mp4`` when gsplat isn't available,
* a cheap "does the cloud explain the images" PSNR proxy for the fallback trainer.

It is fully vectorised (no Python per-point loop): points are sorted near-first,
expanded by a square disc kernel, and the first write to each pixel wins
(= nearest surface).  ~10 ms for 1e6 points at radius 2 on CPU.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .geometry import Camera


def rasterize_points(pts_world: np.ndarray, colors01: np.ndarray, camera: Camera,
                     point_radius_px: int = 2,
                     background=(8, 10, 14),
                     opacity: Optional[np.ndarray] = None,
                     max_points: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splat coloured points into ``camera``.

    Returns ``(rgb uint8 (H,W,3), depth float32 (H,W) NaN=empty, hit_count int32)``.
    ``opacity`` (N,) in [0,1]: points with opacity < 0.02 are skipped; otherwise
    opacity currently only gates visibility (no true alpha blend -- this renderer
    is for previews, not final imagery).
    """
    H, W = int(camera.intr.height), int(camera.intr.width)
    bg = np.array(background, dtype=np.float32)
    rgb = np.tile(bg, (H, W, 1)).astype(np.float32)
    depth = np.full((H, W), np.nan, dtype=np.float32)
    count = np.zeros((H, W), dtype=np.int32)

    pts_world = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
    if len(pts_world) == 0:
        return rgb.astype(np.uint8), depth, count
    colors01 = np.asarray(colors01, dtype=np.float32).reshape(-1, 3)

    if max_points and len(pts_world) > max_points:
        sel = np.random.default_rng(0).choice(len(pts_world), max_points, replace=False)
        pts_world = pts_world[sel]
        colors01 = colors01[sel]
        if opacity is not None:
            opacity = np.asarray(opacity).reshape(-1)[sel]

    uv, z, valid = camera.project(pts_world, clip=False)
    valid &= np.isfinite(z) & (z > 1e-3)
    if opacity is not None:
        valid &= np.asarray(opacity, np.float32).reshape(-1) >= 0.02
    u = uv[:, 0]; v = uv[:, 1]
    r = int(max(0, point_radius_px))
    valid &= (u > -r - 1) & (u < W + r) & (v > -r - 1) & (v < H + r)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return rgb.astype(np.uint8), depth, count

    order = idx[np.argsort(z[idx])]                  # near first
    cu = np.round(u[order]).astype(np.int64)
    cv = np.round(v[order]).astype(np.int64)
    cz = z[order].astype(np.float32)
    cc = np.clip(colors01[order], 0, 1)

    offs = np.arange(-r, r + 1)
    oy, ox = np.meshgrid(offs, offs, indexing="ij")
    ox = ox.reshape(-1); oy = oy.reshape(-1)
    K = len(ox)

    px = (cu[:, None] + ox[None, :]).reshape(-1)
    py = (cv[:, None] + oy[None, :]).reshape(-1)
    pcolor = np.repeat(cc, K, axis=0)
    pz = np.repeat(cz, K)

    inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px = px[inb]; py = py[inb]; pcolor = pcolor[inb]; pz = pz[inb]
    if len(px) == 0:
        return rgb.astype(np.uint8), depth, count

    flat = py * W + px
    # near-first order is preserved; first occurrence per pixel = nearest point
    uniq, first = np.unique(flat, return_index=True)
    fy = uniq // W
    fx = uniq % W
    rgb[fy, fx] = pcolor[first] * 255.0
    depth[fy, fx] = pz[first]
    np.add.at(count, (py, px), 1)

    return np.clip(rgb, 0, 255).astype(np.uint8), depth, count


def psnr(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """PSNR in dB between two uint8 / float images (optionally masked)."""
    a = a.astype(np.float64); b = b.astype(np.float64)
    if a.max() > 1.5:
        a = a / 255.0
    if b.max() > 1.5:
        b = b / 255.0
    if mask is not None:
        mask = mask.astype(bool)
        if mask.ndim == 2 and a.ndim == 3:
            mask = np.repeat(mask[:, :, None], a.shape[2], axis=2)
        a = a[mask]; b = b[mask]
    if a.size == 0:
        return float("nan")
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(-10.0 * np.log10(mse))
