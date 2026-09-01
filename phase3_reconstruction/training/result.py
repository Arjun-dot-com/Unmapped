from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..gaussian_scene import GaussianScene
from ..geometry import SimilarityTransform


@dataclass
class TrainResult:
    scene: GaussianScene
    backend: str
    iterations_run: int = 0
    train_time_seconds: float = 0.0
    final_loss: float = float("nan")
    final_psnr_db: float = float("nan")
    psnr_history: List[float] = field(default_factory=list)
    native_export_path: Optional[str] = None
    # The similarity transform the trainer ACTUALLY applied to the world before
    # optimising (and inverted before returning ``scene``).  Identity for the
    # NumPy fallback, which needs no normalisation.  Recorded so export / native
    # checkpoint metadata is accurate.
    sim_used: SimilarityTransform = field(default_factory=SimilarityTransform.identity)
    extra: Dict = field(default_factory=dict)
