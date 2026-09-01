"""Step 8 (part 2) - renderable preview of the trained scene.

Writes ``preview_render.png`` (one orbit still) and, when possible,
``preview_orbit.mp4`` (a short orbiting fly-around).  Plus a bonus
``preview_confidence.png`` that tints low-confidence Gaussians red so the
occlusion-flagging is visible at a glance for the demo.

Uses the CPU NumPy splatter (:mod:`phase3_reconstruction.render`) so a preview is
always produced even without gsplat / a GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..config import PreviewConfig
from ..gaussian_scene import GaussianScene
from ..geometry import Camera, Intrinsics, look_at
from ..logging_utils import get_logger
from ..render import rasterize_points

log = get_logger(__name__)


@dataclass
class PreviewPaths:
    still: str
    video: Optional[str]
    confidence_still: Optional[str]


def _orbit_camera(centroid, radius, az_deg, elev_deg, intr) -> Camera:
    az = math.radians(az_deg)
    el = math.radians(elev_deg)
    offset = np.array([radius * math.cos(el) * math.cos(az),
                       radius * math.cos(el) * math.sin(az),
                       radius * math.sin(el)])
    eye = centroid + offset
    R = look_at(eye, centroid, up=np.array([0.0, 0.0, 1.0]))
    t = -R @ eye
    return Camera(frame_id=f"orbit_{az_deg:.0f}", R=R, t=t, intr=intr)


def render_preview(scene: GaussianScene, out_dir: str, cfg: PreviewConfig,
                   seed: int = 0) -> PreviewPaths:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not cfg.enabled or len(scene) == 0:
        log.warning("preview disabled or empty scene - skipping preview render")
        return PreviewPaths(still="", video=None, confidence_still=None)

    W, H = cfg.width, cfg.height
    hfov = math.radians(60.0)
    fx = 0.5 * W / math.tan(hfov / 2)
    intr = Intrinsics(fx=fx, fy=fx, cx=W / 2, cy=H / 2, width=W, height=H)

    centroid = scene.centroid()
    ext = scene.extent()
    radius = float(cfg.radius_scale * max(np.linalg.norm(ext), 1.0))

    means = scene.means
    colors = scene.colors
    opac = scene.opacity

    # low-confidence tint layer
    low = scene.observation_count <= 2
    conf_colors = colors.copy()
    conf_colors[low] = 0.65 * conf_colors[low] + 0.35 * np.array([1.0, 0.15, 0.15])

    frames = []
    still_idx = cfg.n_orbit_frames // 4
    still_path = out / "preview_render.png"
    conf_path = out / "preview_confidence.png"

    for i in range(cfg.n_orbit_frames):
        az = 360.0 * i / max(cfg.n_orbit_frames, 1)
        cam = _orbit_camera(centroid, radius, az, cfg.elevation_deg, intr)
        rgb, _, _ = rasterize_points(means, colors, cam,
                                     point_radius_px=cfg.point_radius_px,
                                     background=tuple(cfg.background),
                                     opacity=opac, max_points=cfg.max_render_points)
        frames.append(rgb)
        if i == still_idx:
            cv2.imwrite(str(still_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            rgb_c, _, _ = rasterize_points(means, conf_colors, cam,
                                           point_radius_px=cfg.point_radius_px,
                                           background=tuple(cfg.background),
                                           opacity=opac,
                                           max_points=cfg.max_render_points)
            cv2.imwrite(str(conf_path), cv2.cvtColor(rgb_c, cv2.COLOR_RGB2BGR))

    if not still_path.exists():                       # n_orbit_frames tiny
        cv2.imwrite(str(still_path), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))

    video_path = None
    if cfg.make_video and len(frames) > 1:
        video_path = _write_video(frames, out, cfg.fps)

    log.info("preview: %s%s", still_path,
             f" + {video_path}" if video_path else " (no video)")
    return PreviewPaths(still=str(still_path), video=video_path,
                        confidence_still=str(conf_path))


def _write_video(frames, out: Path, fps: int) -> Optional[str]:
    H, W = frames[0].shape[:2]
    mp4_path = out / "preview_orbit.mp4"
    for fourcc_name in ("mp4v", "avc1", "MJPG"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        ext = ".mp4" if fourcc_name in ("mp4v", "avc1") else ".avi"
        path = out / f"preview_orbit{ext}"
        vw = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
        if vw.isOpened():
            for fr in frames:
                vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            vw.release()
            if path.exists() and path.stat().st_size > 1000:
                return str(path)
    # last resort: animated GIF via Pillow
    try:
        from PIL import Image
        gif_path = out / "preview_orbit.gif"
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / max(fps, 1)), loop=0, optimize=True)
        log.warning("no working video codec - wrote %s instead of mp4", gif_path)
        return str(gif_path)
    except Exception as e:                            # noqa: BLE001
        log.warning("could not write any preview video (%s)", e)
        return None
