"""Camera model + projective geometry helpers.

Coordinate / camera convention (fixed for the whole of Phase 3, matches COLMAP
and OpenCV so Phase 2 -> Phase 3 -> Phase 4 stay consistent):

* **World frame**  : right-handed, metric (metres). Established by Phase 2. We do
  NOT silently re-center or re-scale it -- see :class:`SimilarityTransform`.
* **Camera frame** : ``x_cam = R @ x_world + t``  (world -> camera, "extrinsic").
  Camera looks down **+Z**, **+X** right, **+Y** down (OpenCV pinhole).
* **Image**        : ``u = fx * x/z + cx`` , ``v = fy * y/z + cy`` , origin top-left.
* Camera centre in world coords: ``C = -R.T @ t``.

Every ``R`` we accept is expected to be a proper rotation (orthonormal,
det = +1); :func:`Camera.from_lists` re-orthonormalises defensively and warns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .logging_utils import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Intrinsics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def scaled(self, sx: float, sy: float) -> "Intrinsics":
        """Return intrinsics for an image resized by (sx, sy)."""
        return Intrinsics(
            fx=self.fx * sx, fy=self.fy * sy,
            cx=self.cx * sx, cy=self.cy * sy,
            width=max(1, int(round(self.width * sx))),
            height=max(1, int(round(self.height * sy))),
        )


# --------------------------------------------------------------------------- #
# Rotation utilities
# --------------------------------------------------------------------------- #
def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Nearest proper rotation to ``R`` (SVD projection onto SO(3))."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:            # reflection -> flip least-significant axis
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    """World->camera rotation for a camera at ``eye`` looking at ``target``
    using the OpenCV convention (+Z forward, +Y down)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, dtype=np.float64)
    z = target - eye
    z /= np.linalg.norm(z) + 1e-12
    x = np.cross(z, up)
    nx = np.linalg.norm(x)
    if nx < 1e-8:                        # looking straight along `up`
        up = np.array([0.0, 1.0, 0.0])
        x = np.cross(z, up)
        nx = np.linalg.norm(x)
    x /= nx
    y = np.cross(z, x)                   # +Y down
    return np.stack([x, y, z], axis=0)   # rows are camera axes in world coords


def rotation_to_euler_zyx(R: np.ndarray) -> Tuple[float, float, float]:
    """Return (roll, pitch, yaw) in radians (Tait-Bryan, X-Y-Z intrinsic).
    Only used to synthesise plausible IMU values for the mock generator."""
    R = np.asarray(R, dtype=np.float64)
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = np.arcsin(sy)
    if abs(sy) < 0.99999:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:                               # gimbal lock
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
@dataclass
class Camera:
    """A single posed pinhole camera."""

    frame_id: str
    R: np.ndarray            # (3,3) world -> camera
    t: np.ndarray            # (3,)  world -> camera translation
    intr: Intrinsics
    confidence: float = 1.0

    # -- constructors ------------------------------------------------------- #
    @classmethod
    def from_lists(cls, frame_id: str, R, t, intr: Intrinsics,
                   confidence: float = 1.0) -> "Camera":
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        t = np.asarray(t, dtype=np.float64).reshape(3)
        if not np.isfinite(R).all() or not np.isfinite(t).all():
            raise ValueError(f"frame {frame_id}: non-finite pose")
        err = np.abs(R @ R.T - np.eye(3)).max()
        if err > 1e-4:
            log.warning("frame %s: R not orthonormal (err=%.2e), re-projecting to SO(3)",
                        frame_id, err)
            R = orthonormalize(R)
        return cls(frame_id=frame_id, R=R, t=t, intr=intr, confidence=float(confidence))

    @classmethod
    def from_center(cls, frame_id: str, center: np.ndarray, R: np.ndarray,
                    intr: Intrinsics, confidence: float = 1.0) -> "Camera":
        center = np.asarray(center, dtype=np.float64).reshape(3)
        t = -R @ center
        return cls.from_lists(frame_id, R, t, intr, confidence)

    # -- properties ------------------------------------------------------- #
    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def forward(self) -> np.ndarray:
        """Viewing direction in world coords (camera +Z axis)."""
        return self.R.T @ np.array([0.0, 0.0, 1.0])

    # -- projection ------------------------------------------------------- #
    def project(self, pts_world: np.ndarray, clip: bool = True
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project world points.

        Returns ``(uv, z_cam, valid)`` where

        * ``uv``     : (N,2) pixel coords (float)
        * ``z_cam``  : (N,)  depth along the optical axis (metres)
        * ``valid``  : (N,)  bool, in front of camera AND (if ``clip``) inside image
        """
        pts_world = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
        pc = pts_world @ self.R.T + self.t          # (N,3) camera coords
        z = pc[:, 2]
        eps = 1e-6
        safe_z = np.where(np.abs(z) < eps, eps, z)
        u = self.intr.fx * pc[:, 0] / safe_z + self.intr.cx
        v = self.intr.fy * pc[:, 1] / safe_z + self.intr.cy
        uv = np.stack([u, v], axis=1)
        valid = z > eps
        if clip:
            valid &= (u >= 0) & (u <= self.intr.width - 1) & \
                     (v >= 0) & (v <= self.intr.height - 1)
        return uv, z, valid

    def backproject(self, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Lift pixels + metric depth to world points. ``uv`` (N,2), ``depth`` (N,)."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        depth = np.asarray(depth, dtype=np.float64).reshape(-1)
        x = (uv[:, 0] - self.intr.cx) / self.intr.fx * depth
        y = (uv[:, 1] - self.intr.cy) / self.intr.fy * depth
        pc = np.stack([x, y, depth], axis=1)        # camera coords
        return pc @ self.R + (-self.R.T @ self.t)   # (pc - t) @ R^-T  == pc@R + C

    def backproject_depth_map(self, depth_map: np.ndarray, stride: int = 1,
                              mask: np.ndarray = None
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """Dense back-projection of a full depth map.

        Returns ``(pts_world (M,3), pix (M,2) int)`` for every sampled pixel with
        finite positive depth and ``mask != True``.
        """
        h, w = depth_map.shape[:2]
        ys = np.arange(0, h, stride)
        xs = np.arange(0, w, stride)
        gx, gy = np.meshgrid(xs, ys)
        gx = gx.reshape(-1)
        gy = gy.reshape(-1)
        d = depth_map[gy, gx].astype(np.float64)
        keep = np.isfinite(d) & (d > 0)
        if mask is not None:
            keep &= ~mask[gy, gx].astype(bool)
        gx, gy, d = gx[keep], gy[keep], d[keep]
        uv = np.stack([gx.astype(np.float64), gy.astype(np.float64)], axis=1)
        pts = self.backproject(uv, d)
        return pts, np.stack([gx, gy], axis=1)


# --------------------------------------------------------------------------- #
# Similarity transform (train-stability normalisation that we must be able to
# invert exactly before export -- Section 4 of the brief).
# --------------------------------------------------------------------------- #
@dataclass
class SimilarityTransform:
    """``x' = scale * R @ x + translation`` , with an exact inverse.

    Phase 3 optionally maps the Phase-2 world into a unit-ish box centred on the
    origin for numerical stability during Gaussian training, then maps every
    exported coordinate back. ``identity()`` is a no-op used when normalisation
    is disabled.
    """

    scale: float = 1.0
    rotation: np.ndarray = None          # (3,3), default identity
    translation: np.ndarray = None       # (3,), default zero

    def __post_init__(self):
        if self.rotation is None:
            self.rotation = np.eye(3)
        if self.translation is None:
            self.translation = np.zeros(3)
        self.rotation = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
        self.translation = np.asarray(self.translation, dtype=np.float64).reshape(3)

    @classmethod
    def identity(cls) -> "SimilarityTransform":
        return cls(1.0, np.eye(3), np.zeros(3))

    @classmethod
    def normalizing(cls, points: np.ndarray, target_radius: float = 1.0
                    ) -> "SimilarityTransform":
        """Build a transform that centres ``points`` at the origin and scales so
        the median distance to the centre is ``target_radius``."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        c = np.median(pts, axis=0)
        r = np.median(np.linalg.norm(pts - c, axis=1))
        r = float(r) if r > 1e-9 else 1.0
        s = target_radius / r
        return cls(scale=s, rotation=np.eye(3), translation=-s * c)

    @property
    def is_identity(self) -> bool:
        return (abs(self.scale - 1.0) < 1e-12
                and np.allclose(self.rotation, np.eye(3))
                and np.allclose(self.translation, 0.0))

    def apply(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        return self.scale * (pts @ self.rotation.T) + self.translation

    def inverse(self) -> "SimilarityTransform":
        inv_R = self.rotation.T
        inv_s = 1.0 / self.scale
        inv_t = -inv_s * (inv_R @ self.translation)
        return SimilarityTransform(inv_s, inv_R, inv_t)

    def apply_inverse(self, pts: np.ndarray) -> np.ndarray:
        return self.inverse().apply(pts)

    def transform_camera(self, cam: Camera) -> Camera:
        """Return the camera expressed in the transformed world frame.

        Derivation.  With ``w' = s R_s w + t_s`` we have
        ``w = (1/s) R_s^T (w' - t_s)`` and ``x_cam = R w + t``.  Pixel coords are
        invariant to a positive scaling of ``x_cam``, so multiply through by
        ``s``::

            s * x_cam = (R R_s^T) w' - (R R_s^T) t_s + s t

        i.e. ``R' = R R_s^T`` , ``t' = s t - R' t_s`` .  Projecting a transformed
        point with ``(R', t')`` reproduces the original pixel exactly; metric
        depths come out multiplied by ``s`` (transformed units), which is what we
        want during training and undo on export.
        """
        Rn = cam.R @ self.rotation.T
        tn = self.scale * cam.t - Rn @ self.translation
        return Camera(frame_id=cam.frame_id, R=Rn, t=tn, intr=cam.intr,
                      confidence=cam.confidence)

    def to_dict(self) -> dict:
        return {
            "scale": float(self.scale),
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SimilarityTransform":
        return cls(scale=float(d.get("scale", 1.0)),
                   rotation=np.asarray(d.get("rotation", np.eye(3).tolist())),
                   translation=np.asarray(d.get("translation", [0, 0, 0])))


def bilinear_sample(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinearly sample ``img`` (H,W) or (H,W,C) at float pixel coords ``uv`` (N,2).
    Coords are clamped to the image border."""
    img = np.asarray(img)
    h, w = img.shape[:2]
    u = np.clip(uv[:, 0], 0, w - 1.0)
    v = np.clip(uv[:, 1], 0, h - 1.0)
    x0 = np.floor(u).astype(int); x1 = np.minimum(x0 + 1, w - 1)
    y0 = np.floor(v).astype(int); y1 = np.minimum(y0 + 1, h - 1)
    # fractional parts (0 at / past the last row/col, so weights still sum to 1
    # and we don't get an all-zero result on the image border)
    fx = u - x0
    fy = v - y0
    wa = (1 - fx) * (1 - fy)
    wb = fx * (1 - fy)
    wc = (1 - fx) * fy
    wd = fx * fy
    if img.ndim == 2:
        out = (wa * img[y0, x0] + wb * img[y0, x1]
               + wc * img[y1, x0] + wd * img[y1, x1])
    else:
        w4 = np.stack([wa, wb, wc, wd], axis=1)[:, :, None]
        stacked = np.stack([img[y0, x0], img[y0, x1], img[y1, x0], img[y1, x1]], axis=1)
        out = (w4 * stacked).sum(axis=1)
    return out
