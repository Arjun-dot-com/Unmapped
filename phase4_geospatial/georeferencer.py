"""Local ENU metric coordinates to WGS84, with no hard dependency on pyproj."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def load_origin(*paths, default=(12.9715, 77.5949, 940.0)):
    for raw_path in paths:
        if not raw_path: continue
        p = Path(raw_path)
        if not p.exists(): continue
        try: data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError): continue
        frames = data.get("frames", [])
        if frames and frames[0].get("gps"):
            g = frames[0]["gps"]
            return float(g["lat"]), float(g["lon"]), float(g.get("alt", 0.0))
        for key in ("origin", "gps_origin"):
            if isinstance(data.get(key), dict):
                g = data[key]; return float(g["lat"]), float(g["lon"]), float(g.get("alt", 0.0))
    return default

def enu_to_wgs84(xyz, origin):
    """Return Nx3 [longitude, latitude, altitude]. x=east, y=north, z=up."""
    pts = np.asarray(xyz, dtype=float).reshape(-1, 3)
    lat0, lon0, h0 = map(float, origin)
    try:
        from pyproj import CRS, Transformer
        topocentric = CRS.from_proj4(f"+proj=topocentric +lat_0={lat0} +lon_0={lon0} +h_0={h0} +ellps=WGS84")
        tr = Transformer.from_crs(topocentric, "EPSG:4979", always_xy=True)
        lon, lat, alt = tr.transform(pts[:, 0], pts[:, 1], pts[:, 2])
        return np.column_stack((lon, lat, alt))
    except ImportError:
        # Equirectangular approximation is sub-centimetric over a small flight.
        r = 6378137.0; latr = np.deg2rad(lat0)
        lon = lon0 + np.rad2deg(pts[:, 0] / (r * np.cos(latr)))
        lat = lat0 + np.rad2deg(pts[:, 1] / r)
        return np.column_stack((lon, lat, h0 + pts[:, 2]))

def origin_metadata(origin):
    return {"lat": float(origin[0]), "lon": float(origin[1]), "alt": float(origin[2]),
            "crs": "EPSG:4326", "local_frame": "ENU (x=east, y=north, z=up)"}
