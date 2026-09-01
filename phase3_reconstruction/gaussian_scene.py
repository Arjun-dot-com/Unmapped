"""The in-memory representation of a 3D Gaussian scene, shared by init / training
/ confidence / export so those stages never disagree on layout.

Conventions
-----------
* ``means``  (N,3) float64 - Gaussian centres, **Phase-2 metric world frame**.
* ``colors`` (N,3) float32 in [0,1] - base (DC) RGB colour.  (Full spherical
  harmonics live only inside the gsplat checkpoint; the flattened ``.ply`` keeps
  the view-independent term, which is what Phase 4 meshing wants.)
* ``opacity`` (N,) float32 in [0,1].
* ``scales`` (N,3) float32 - **linear** std-dev per axis, in metres.
* ``quats``  (N,4) float32 - rotation wxyz, normalised.
* ``observation_count`` (N,) int32 - how many cameras actually see this Gaussian
  (``-1`` until the confidence stage fills it).
* ``confidence`` (N,) float32 in [0,1] - combined observation / view-diversity
  score; low values mark **under-observed** (near-occluded) regions.  We flag,
  never hallucinate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np

from .geometry import SimilarityTransform


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 -> wxyz."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


@dataclass
class GaussianScene:
    means: np.ndarray
    colors: np.ndarray
    opacity: np.ndarray
    scales: np.ndarray
    quats: np.ndarray
    observation_count: np.ndarray = None
    confidence: np.ndarray = None
    meta: Dict = field(default_factory=dict)

    # ---- construction helpers ---------------------------------------- #
    def __post_init__(self):
        n = len(self.means)
        self.means = np.asarray(self.means, dtype=np.float64).reshape(n, 3)
        self.colors = np.clip(np.asarray(self.colors, dtype=np.float32).reshape(n, 3), 0, 1)
        self.opacity = np.clip(np.asarray(self.opacity, dtype=np.float32).reshape(n), 0, 1)
        self.scales = np.asarray(self.scales, dtype=np.float32).reshape(n, 3)
        self.quats = np.asarray(self.quats, dtype=np.float32).reshape(n, 4)
        norm = np.linalg.norm(self.quats, axis=1, keepdims=True)
        self.quats = self.quats / np.where(norm > 1e-9, norm, 1.0)
        if self.observation_count is None:
            self.observation_count = np.full(n, -1, dtype=np.int32)
        else:
            self.observation_count = np.asarray(self.observation_count, dtype=np.int32).reshape(n)
        if self.confidence is None:
            self.confidence = np.full(n, np.nan, dtype=np.float32)
        else:
            self.confidence = np.asarray(self.confidence, dtype=np.float32).reshape(n)

    def __len__(self) -> int:
        return len(self.means)

    @classmethod
    def from_points(cls, xyz: np.ndarray, rgb01: np.ndarray, scale_m: np.ndarray,
                    opacity: float = 0.1) -> "GaussianScene":
        n = len(xyz)
        scale_m = np.asarray(scale_m, dtype=np.float32)
        if scale_m.ndim == 1:
            scale_m = np.repeat(scale_m[:, None], 3, axis=1)
        quats = np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1))
        return cls(means=xyz, colors=rgb01,
                   opacity=np.full(n, opacity, np.float32),
                   scales=scale_m, quats=quats)

    def subset(self, idx) -> "GaussianScene":
        idx = np.asarray(idx)
        return GaussianScene(
            means=self.means[idx], colors=self.colors[idx], opacity=self.opacity[idx],
            scales=self.scales[idx], quats=self.quats[idx],
            observation_count=self.observation_count[idx],
            confidence=self.confidence[idx], meta=dict(self.meta))

    def apply_similarity(self, sim: SimilarityTransform) -> "GaussianScene":
        """Return the scene mapped through ``sim`` (means, scales, orientation)."""
        if sim.is_identity:
            return self
        means = sim.apply(self.means)
        scales = self.scales * float(sim.scale)
        qR = _mat_to_quat(sim.rotation).astype(np.float32)
        quats = _quat_mul(np.tile(qR, (len(self), 1)), self.quats).astype(np.float32)
        return GaussianScene(means=means, colors=self.colors, opacity=self.opacity,
                             scales=scales, quats=quats,
                             observation_count=self.observation_count,
                             confidence=self.confidence, meta=dict(self.meta))

    # ---- (de)serialisation ---------------------------------------- #
    def to_ply_columns(self, include_confidence: bool = True) -> Dict[str, np.ndarray]:
        cols = {
            "x": self.means[:, 0].astype(np.float32),
            "y": self.means[:, 1].astype(np.float32),
            "z": self.means[:, 2].astype(np.float32),
            "red": np.clip(self.colors[:, 0] * 255, 0, 255).astype(np.uint8),
            "green": np.clip(self.colors[:, 1] * 255, 0, 255).astype(np.uint8),
            "blue": np.clip(self.colors[:, 2] * 255, 0, 255).astype(np.uint8),
            "opacity": self.opacity.astype(np.float32),
            "scale_x": self.scales[:, 0].astype(np.float32),
            "scale_y": self.scales[:, 1].astype(np.float32),
            "scale_z": self.scales[:, 2].astype(np.float32),
        }
        if include_confidence:
            cols["observation_count"] = np.maximum(self.observation_count, 0).astype(np.uint16)
            conf = np.where(np.isfinite(self.confidence), self.confidence, 0.0)
            cols["confidence"] = conf.astype(np.float32)
        return cols

    def save_npz(self, path: str) -> None:
        np.savez_compressed(
            path, means=self.means, colors=self.colors, opacity=self.opacity,
            scales=self.scales, quats=self.quats,
            observation_count=self.observation_count, confidence=self.confidence,
            meta=json.dumps(self.meta))

    @classmethod
    def load_npz(cls, path: str) -> "GaussianScene":
        d = np.load(path, allow_pickle=False)
        meta = {}
        if "meta" in d:
            try:
                meta = json.loads(str(d["meta"]))
            except Exception:                        # noqa: BLE001
                meta = {}
        return cls(means=d["means"], colors=d["colors"], opacity=d["opacity"],
                   scales=d["scales"], quats=d["quats"],
                   observation_count=d.get("observation_count"),
                   confidence=d.get("confidence"), meta=meta)

    def extent(self) -> np.ndarray:
        if not len(self):
            return np.zeros(3)
        lo = np.percentile(self.means, 1, axis=0)
        hi = np.percentile(self.means, 99, axis=0)
        return hi - lo

    def centroid(self) -> np.ndarray:
        return np.median(self.means, axis=0) if len(self) else np.zeros(3)
