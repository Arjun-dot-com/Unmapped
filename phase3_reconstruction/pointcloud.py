"""Small NumPy/SciPy point-cloud helpers (voxel downsample, outlier removal,
nearest-neighbour spacing).  Pure-Python fallback for the bits of Open3D we use,
so the core pipeline has no Open3D dependency.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:                                   # pragma: no cover
    cKDTree = None


def voxel_downsample(xyz: np.ndarray, voxel: float,
                     attrs: Optional[Dict[str, np.ndarray]] = None,
                     reduce: str = "mean") -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Grid-average points into ``voxel``-sized cells.

    Returns ``(xyz_ds, attrs_ds)``.  ``attrs`` values are averaged per cell
    (``reduce='mean'``) or taken from the first point (``reduce='first'``).
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    attrs = attrs or {}
    if len(xyz) == 0 or voxel <= 0:
        return xyz, {k: np.asarray(v) for k, v in attrs.items()}

    keys = np.floor((xyz - xyz.min(axis=0)) / voxel).astype(np.int64)
    # hashable cell id
    key_view = np.ascontiguousarray(keys).view(
        np.dtype((np.void, keys.dtype.itemsize * 3)))
    _, inv, counts = np.unique(key_view, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    n_cells = counts.shape[0]

    sums = np.zeros((n_cells, 3))
    np.add.at(sums, inv, xyz)
    xyz_ds = sums / counts[:, None]

    out_attrs: Dict[str, np.ndarray] = {}
    for k, v in attrs.items():
        v = np.asarray(v)
        if reduce == "first" or not np.issubdtype(v.dtype, np.number):
            first = np.zeros(n_cells, dtype=np.int64)
            seen = np.zeros(n_cells, dtype=bool)
            for i, c in enumerate(inv):
                if not seen[c]:
                    seen[c] = True
                    first[c] = i
            out_attrs[k] = v[first]
        else:
            if v.ndim == 1:
                s = np.zeros(n_cells, dtype=np.float64)
                np.add.at(s, inv, v.astype(np.float64))
                out_attrs[k] = (s / counts).astype(v.dtype)
            else:
                s = np.zeros((n_cells, v.shape[1]), dtype=np.float64)
                np.add.at(s, inv, v.astype(np.float64))
                out_attrs[k] = (s / counts[:, None]).astype(v.dtype)
    out_attrs["_count"] = counts.astype(np.int32)
    return xyz_ds, out_attrs


def statistical_outlier_mask(xyz: np.ndarray, k: int = 12, std_ratio: float = 2.0
                             ) -> np.ndarray:
    """Return a boolean *keep* mask (True = inlier).  A point is an outlier if its
    mean distance to its ``k`` nearest neighbours is > global_mean + std_ratio*std.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    n = len(xyz)
    if n <= k + 1 or cKDTree is None:
        return np.ones(n, dtype=bool)
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=k + 1)
    mean_d = d[:, 1:].mean(axis=1)
    thr = mean_d.mean() + std_ratio * mean_d.std()
    return mean_d <= thr


def nn_spacing(xyz: np.ndarray, k: int = 4, sample: int = 20000) -> np.ndarray:
    """Per-point distance to its nearest neighbour (used to size initial Gaussians).
    Approximated on a random subset for speed, then broadcast via the tree."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    n = len(xyz)
    if n < 2:
        return np.full(n, 0.05)
    if cKDTree is None:
        return np.full(n, max(1e-3, np.linalg.norm(xyz.std(axis=0)) / max(n ** (1 / 3), 1)))
    tree = cKDTree(xyz)
    kk = min(k + 1, n)
    d, _ = tree.query(xyz, k=kk)
    return d[:, 1:].mean(axis=1) if kk > 1 else np.full(n, 0.05)
