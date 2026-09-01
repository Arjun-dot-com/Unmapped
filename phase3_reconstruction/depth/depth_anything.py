"""Step 2 - monocular depth estimation.

We wrap **Depth Anything V2** (a pretrained, scene-generalising monocular depth
network) behind a tiny :class:`DepthPrediction` interface.  The rest of Phase 3
only ever sees a relative, affine-invariant *inverse-depth* map per frame -- it
does not care which backend produced it.  That map is turned into real metres by
:mod:`phase3_reconstruction.fusion.depth_scale_fusion`.

Backends
--------
* ``DepthAnythingV2Predictor`` - the real thing, via HuggingFace ``transformers``
  (``depth-anything/Depth-Anything-V2-{Small,Base,Large}-hf``).  Needs ``torch``.
* ``MockDepthPredictor`` - a dependency-free stand-in so the pipeline (and CI)
  runs on any machine:
    - ``synthetic_gt``  : read the mock scene's ground-truth depth and degrade it
      into an affine-invariant, noisy prior (used by ``--mock``).  This keeps the
      end-to-end demo meaningful without bundling 100s of MB of weights.
    - ``image_heuristic``: a crude image-only prior (vertical gradient + shading).
      Used only when a real model is unavailable on *real* data; quality is poor
      and it says so loudly.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..config import DepthConfig
from ..exceptions import Phase3DependencyError
from ..logging_utils import get_logger

log = get_logger(__name__)

_HF_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


@dataclass
class DepthPrediction:
    """Output of a monocular depth backend for ONE frame.

    * ``relative`` : (H,W) float32.  If ``kind == 'inverse_depth'`` larger values
      are nearer (disparity-like, what Depth Anything V2 emits); if
      ``kind == 'depth'`` larger values are farther.  Either way the map is only
      correct up to an unknown per-frame affine transform -> that is exactly what
      the fusion step solves for.
    * ``kind``     : 'inverse_depth' | 'depth'
    * ``valid``    : (H,W) bool, pixels with a usable prediction.
    """

    frame_id: str
    relative: np.ndarray
    kind: str = "inverse_depth"
    valid: Optional[np.ndarray] = None
    backend: str = "unknown"

    def __post_init__(self):
        self.relative = np.asarray(self.relative, dtype=np.float32)
        if self.valid is None:
            self.valid = np.isfinite(self.relative)
        self.valid = np.asarray(self.valid, dtype=bool)


# --------------------------------------------------------------------------- #
def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# --------------------------------------------------------------------------- #
class DepthAnythingV2Predictor:
    """Real Depth Anything V2 inference (needs ``torch`` + ``transformers``)."""

    def __init__(self, cfg: DepthConfig):
        if not (_module_available("torch") and _module_available("transformers")):
            raise Phase3DependencyError(
                "Depth Anything V2 needs `torch` and `transformers`. "
                "Install with:  pip install \"torch>=2.0\" transformers   "
                "(or run with --mock / depth.backend=mock).")
        try:
            import torch                                       # noqa: F401
            from transformers import (AutoImageProcessor,
                                      AutoModelForDepthEstimation)
        except Exception as e:                                 # noqa: BLE001
            raise Phase3DependencyError(
                f"failed to import the Depth Anything V2 stack: {e}") from e

        self._torch = torch
        model_id = cfg.checkpoint or _HF_IDS.get(cfg.model_size, _HF_IDS["small"])
        dev = cfg.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = dev
        log.info("loading Depth Anything V2 '%s' on %s", model_id, dev)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(dev).eval()
        self.backend = f"depth_anything_v2:{cfg.model_size}"
        self.kind = "inverse_depth"       # DA-V2 (non-metric) emits disparity-like output

    def predict(self, rgb: np.ndarray, frame_id: str = "") -> DepthPrediction:
        torch = self._torch
        h, w = rgb.shape[:2]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            pred = self.model(**inputs).predicted_depth          # (1, h', w')
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
        ).squeeze().float().cpu().numpy()
        return DepthPrediction(frame_id=frame_id, relative=pred,
                               kind=self.kind, backend=self.backend)


# --------------------------------------------------------------------------- #
class MockDepthPredictor:
    """Dependency-free depth prior. See module docstring for the two modes."""

    def __init__(self, cfg: DepthConfig, gt_depth_dir: Optional[str] = None):
        self.cfg = cfg
        self.gt_depth_dir = Path(gt_depth_dir) if gt_depth_dir else None
        self.mode = "synthetic_gt" if (cfg.mock_use_gt_depth and self.gt_depth_dir
                                       and self.gt_depth_dir.is_dir()) else "image_heuristic"
        self.kind = "inverse_depth"
        self.backend = f"mock:{self.mode}"
        self._rng = np.random.default_rng(0)
        if self.mode == "image_heuristic":
            log.warning("MockDepthPredictor is running in 'image_heuristic' mode - "
                        "this is a PLACEHOLDER prior (vertical gradient + shading), "
                        "not a real depth estimate. Install torch+transformers or "
                        "use --mock for a meaningful result.")
        else:
            log.info("MockDepthPredictor: using synthetic GT depth from %s as the "
                     "'monocular' prior (noise_std=%.3f).", self.gt_depth_dir,
                     cfg.mock_noise_std)

    # -- helpers ------------------------------------------------------- #
    def _gt_path(self, frame_id: str) -> Optional[Path]:
        if not self.gt_depth_dir:
            return None
        for stem in (f"frame_{frame_id}", f"frame_{str(frame_id).zfill(6)}", str(frame_id)):
            for ext in (".npy", ".npz"):
                p = self.gt_depth_dir / f"{stem}{ext}"
                if p.is_file():
                    return p
        return None

    def _from_gt(self, frame_id: str, hw) -> Optional[DepthPrediction]:
        p = self._gt_path(frame_id)
        if p is None:
            return None
        arr = np.load(p)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[arr.files[0]]
        depth = np.asarray(arr, dtype=np.float32)
        if depth.shape != tuple(hw):
            depth = cv2.resize(depth, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
        valid = np.isfinite(depth) & (depth > 0)
        inv = np.zeros_like(depth)
        inv[valid] = 1.0 / depth[valid]
        # Degrade into an affine-invariant, noisy prior (mimic DA-V2):
        std = float(self.cfg.mock_noise_std)
        inv = inv * (1.0 + self._rng.normal(0.0, std, size=inv.shape).astype(np.float32))
        inv += self._rng.normal(0.0, std * 0.1 * (inv.max() + 1e-6), size=inv.shape).astype(np.float32)
        if self.cfg.mock_smooth_px and self.cfg.mock_smooth_px > 0:
            inv = cv2.GaussianBlur(inv, (0, 0), float(self.cfg.mock_smooth_px))
        # arbitrary affine: normalise to ~[0,1] then apply a random gain/bias
        lo, hi = np.percentile(inv[valid], [2, 98]) if valid.any() else (0.0, 1.0)
        inv = (inv - lo) / max(hi - lo, 1e-6)
        gain = float(self._rng.uniform(0.7, 1.4))
        bias = float(self._rng.uniform(-0.15, 0.15))
        inv = inv * gain + bias
        return DepthPrediction(frame_id=frame_id, relative=inv, kind="inverse_depth",
                               valid=valid, backend=self.backend)

    def _from_image(self, frame_id: str, rgb: np.ndarray) -> DepthPrediction:
        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        # vertical prior: image top -> far (small inverse depth), bottom -> near
        vy = np.linspace(0.15, 1.0, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
        # shading prior: brighter often = nearer / lit surface
        shade = cv2.GaussianBlur(gray, (0, 0), 3.0)
        # central bias (drone usually frames the subject centrally)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx / w) - 0.5) ** 2 + ((yy / h) - 0.5) ** 2)
        centre = 1.0 - np.clip(r / 0.7, 0, 1)
        inv = 0.55 * vy + 0.30 * shade + 0.15 * centre
        inv = cv2.GaussianBlur(inv, (0, 0), 2.0)
        inv = (inv - inv.min()) / max(inv.max() - inv.min(), 1e-6)
        return DepthPrediction(frame_id=frame_id, relative=inv.astype(np.float32),
                               kind="inverse_depth", backend=self.backend)

    # -- api --------------------------------------------------------- #
    def predict(self, rgb: np.ndarray, frame_id: str = "") -> DepthPrediction:
        hw = rgb.shape[:2]
        if self.mode == "synthetic_gt":
            out = self._from_gt(frame_id, hw)
            if out is not None:
                return out
            log.warning("frame %s: no GT depth file found, using image heuristic", frame_id)
        return self._from_image(frame_id, rgb)


# --------------------------------------------------------------------------- #
def create_depth_predictor(cfg: DepthConfig, gt_depth_dir: Optional[str] = None,
                           mock: bool = False):
    """Backend selection.

    * ``cfg.backend == 'mock'``  -> always the mock predictor.
    * ``cfg.backend == 'depth_anything_v2'`` -> real model, hard error if missing.
    * ``cfg.backend == 'auto'``  -> real model if importable, otherwise mock
      (with a warning).  ``mock=True`` (the --mock flag) forces the mock path.
    """
    backend = cfg.backend
    if backend == "mock" or mock:
        return MockDepthPredictor(cfg, gt_depth_dir=gt_depth_dir)
    if backend == "depth_anything_v2":
        return DepthAnythingV2Predictor(cfg)          # raises if unavailable
    # auto
    if _module_available("torch") and _module_available("transformers"):
        try:
            return DepthAnythingV2Predictor(cfg)
        except Phase3DependencyError as e:
            log.warning("Depth Anything V2 unavailable (%s); falling back to mock prior", e)
    else:
        log.warning("torch/transformers not installed; using mock depth prior. "
                    "Install the 'depth' extra for real monocular depth.")
    return MockDepthPredictor(cfg, gt_depth_dir=gt_depth_dir)
