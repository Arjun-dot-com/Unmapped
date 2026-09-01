"""Minimal, dependency-free PLY reader/writer for point clouds.

We deliberately do **not** depend on Open3D for the core path: Phase 4 wants a
plain ``.ply`` with ``x y z``, ``red green blue`` and our extra per-point
attributes (``opacity``, ``observation_count``, ``confidence``).  This module
reads ``ascii`` and ``binary_little_endian`` PLYs (enough for COLMAP / Phase 2
output) and writes ``binary_little_endian`` (compact, exact).

If Open3D *is* installed it is only used opportunistically elsewhere; nothing
here needs it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

_PLY_TO_NP = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}
_NP_TO_PLY = {
    "i1": "char", "u1": "uchar", "i2": "short", "u2": "ushort",
    "i4": "int", "u4": "uint", "f4": "float", "f8": "double",
}


@dataclass
class PlyData:
    """Column-oriented vertex data. ``columns[name] -> (N,) ndarray``."""

    columns: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return 0 if not self.columns else len(next(iter(self.columns.values())))

    @property
    def names(self) -> List[str]:
        return list(self.columns.keys())

    def has(self, *names: str) -> bool:
        return all(n in self.columns for n in names)

    def xyz(self) -> np.ndarray:
        if not self.has("x", "y", "z"):
            raise KeyError("PLY has no x/y/z columns")
        return np.stack([self.columns["x"], self.columns["y"],
                         self.columns["z"]], axis=1).astype(np.float64)

    def rgb(self, default: Tuple[int, int, int] = (180, 180, 180)) -> np.ndarray:
        """(N,3) uint8. Falls back to a constant grey if colour is absent."""
        if self.has("red", "green", "blue"):
            return np.stack([self.columns["red"], self.columns["green"],
                             self.columns["blue"]], axis=1).astype(np.uint8)
        return np.tile(np.array(default, dtype=np.uint8), (self.count, 1))


def _parse_header(f) -> Tuple[str, int, List[Tuple[str, str]]]:
    line = f.readline().decode("ascii", "replace").strip()
    if line != "ply":
        raise ValueError("not a PLY file (missing 'ply' magic)")
    fmt = None
    count = None
    props: List[Tuple[str, str]] = []
    in_vertex = False
    while True:
        raw = f.readline()
        if not raw:
            raise ValueError("unexpected EOF in PLY header")
        line = raw.decode("ascii", "replace").strip()
        if line.startswith("format"):
            fmt = line.split()[1]
        elif line.startswith("element"):
            _, name, n = line.split()[:3]
            in_vertex = name == "vertex"
            if in_vertex:
                count = int(n)
        elif line.startswith("property") and in_vertex:
            parts = line.split()
            if parts[1] == "list":
                raise ValueError("list properties on 'vertex' are not supported")
            props.append((parts[2], parts[1]))          # (name, type)
        elif line == "end_header":
            break
    if fmt is None or count is None:
        raise ValueError("PLY header missing 'format' or 'vertex' element")
    return fmt, count, props


def read_ply(path: str) -> PlyData:
    with open(path, "rb") as f:
        fmt, count, props = _parse_header(f)
        names = [p[0] for p in props]
        if fmt == "ascii":
            data = np.loadtxt(f, dtype=np.float64, ndmin=2) if count else np.zeros((0, len(props)))
            if data.shape[0] != count:
                raise ValueError(f"PLY says {count} verts, found {data.shape[0]}")
            cols = {n: data[:, i] for i, n in enumerate(names)}
        elif fmt in ("binary_little_endian", "binary_big_endian"):
            endian = "<" if fmt.endswith("little_endian") else ">"
            dtype = np.dtype([(n, endian + _PLY_TO_NP[t]) for (n, t) in props])
            buf = f.read(dtype.itemsize * count)
            if len(buf) < dtype.itemsize * count:
                raise ValueError("PLY body shorter than header promises")
            arr = np.frombuffer(buf, dtype=dtype, count=count)
            cols = {n: np.array(arr[n]) for n in names}
        else:
            raise ValueError(f"unsupported PLY format '{fmt}'")
    return PlyData(columns=cols)


def write_ply(path: str, columns: Dict[str, np.ndarray],
              comments: List[str] = None) -> None:
    """Write ``binary_little_endian``. Column dtype -> PLY type is inferred; cast
    your integers/uint8 before calling if you care about the on-disk type."""
    if not columns:
        raise ValueError("write_ply: no columns")
    names = list(columns.keys())
    n = len(columns[names[0]])
    fields = []
    arrays = []
    for name in names:
        a = np.asarray(columns[name])
        if a.ndim != 1 or len(a) != n:
            raise ValueError(f"column '{name}' must be 1-D length {n}")
        kind = a.dtype.str[1:]                       # e.g. 'f8', 'u1'
        if kind not in _NP_TO_PLY:
            a = a.astype(np.float32)
            kind = "f4"
        fields.append((name, "<" + kind))
        arrays.append(a.astype("<" + kind))
    struct = np.empty(n, dtype=np.dtype(fields))
    for (name, _), a in zip(fields, arrays):
        struct[name] = a

    header = ["ply", "format binary_little_endian 1.0"]
    for c in (comments or []):
        for cl in str(c).splitlines():
            header.append(f"comment {cl}")
    header.append(f"element vertex {n}")
    for name, dt in fields:
        header.append(f"property {_NP_TO_PLY[dt[1:]]} {name}")
    header.append("end_header\n")
    with open(path, "wb") as f:
        f.write(("\n".join(header)).encode("ascii"))
        f.write(struct.tobytes())


# --------------------------------------------------------------------------- #
# Convenience wrappers used by the pipeline
# --------------------------------------------------------------------------- #
def read_point_cloud(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(xyz (N,3) float64, rgb (N,3) uint8)``."""
    ply = read_ply(path)
    return ply.xyz(), ply.rgb()


def write_point_cloud(path: str, xyz: np.ndarray, rgb: np.ndarray = None,
                      extra: Dict[str, np.ndarray] = None,
                      comments: List[str] = None) -> None:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    cols: Dict[str, np.ndarray] = {
        "x": xyz[:, 0].astype(np.float32),
        "y": xyz[:, 1].astype(np.float32),
        "z": xyz[:, 2].astype(np.float32),
    }
    if rgb is not None:
        rgb = np.asarray(rgb).reshape(-1, 3)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        cols["red"], cols["green"], cols["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    for k, v in (extra or {}).items():
        cols[k] = np.asarray(v).reshape(-1)
    write_ply(path, cols, comments=comments)
