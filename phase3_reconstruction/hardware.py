"""Cached, quiet probe for optional GPU / torch / gsplat availability.

``import torch`` can be slow and, in a mismatched environment, spew C-level
warnings to stderr.  We do it at most once, with stderr+warnings suppressed, and
cache the result so every call site (hardware string, trainer dispatch) shares it.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import platform
import warnings
from dataclasses import dataclass
from typing import Optional

_PROBE: Optional["TorchProbe"] = None


@dataclass
class TorchProbe:
    torch_importable: bool = False
    cuda_available: bool = False
    gsplat_importable: bool = False
    device_name: str = ""
    vram_gb: float = 0.0
    cuda_version: str = ""
    error: str = ""


def probe_torch(force: bool = False) -> TorchProbe:
    global _PROBE
    if _PROBE is not None and not force:
        return _PROBE
    p = TorchProbe()
    if importlib.util.find_spec("torch") is None:
        _PROBE = p
        return p
    try:
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            os.environ.setdefault("PYTORCH_DISABLE_NUMPY", "0")
            import torch
            p.torch_importable = True
            try:
                p.cuda_available = bool(torch.cuda.is_available())
                if p.cuda_available:
                    props = torch.cuda.get_device_properties(0)
                    p.device_name = props.name
                    p.vram_gb = props.total_memory / (1024 ** 3)
                    p.cuda_version = torch.version.cuda or ""
            except Exception as e:                    # noqa: BLE001
                p.error = f"cuda probe failed: {e}"
        p.gsplat_importable = importlib.util.find_spec("gsplat") is not None
    except Exception as e:                            # noqa: BLE001
        p.error = f"torch import failed: {e}"
    _PROBE = p
    return p


def hardware_string() -> str:
    p = probe_torch()
    if p.cuda_available:
        v = f", CUDA {p.cuda_version}" if p.cuda_version else ""
        return f"NVIDIA {p.device_name}, {p.vram_gb:.0f}GB VRAM{v}"
    cpu = platform.processor() or platform.machine() or "unknown CPU"
    cores = os.cpu_count() or "?"
    tail = " (torch present, no CUDA)" if p.torch_importable else " (no torch)"
    return f"CPU only - {cpu} ({cores} logical cores), {platform.system()} {platform.release()}{tail}"


def gsplat_trainable() -> bool:
    p = probe_torch()
    return p.torch_importable and p.cuda_available and p.gsplat_importable
