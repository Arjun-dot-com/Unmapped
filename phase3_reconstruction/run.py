"""Step 10 - single CLI entrypoint for the whole Phase 3 pipeline.

Real upstream data:
    python -m phase3_reconstruction.run \
        --frames-dir ./phase1_ingestion/output \
        --poses-dir  ./phase2_pose/output \
        --out        ./phase3_reconstruction/output \
        --config     configs/default.yaml

Standalone toy scene (no GPU, no external weights, generates its own input):
    python -m phase3_reconstruction.run --mock

Config overrides without editing YAML:
    ... --set training.iterations=3000 --set depth.model_size=base
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from . import __version__
from .config import load_config
from .exceptions import Phase3Error
from .logging_utils import get_logger, setup_logging

log = get_logger(__name__)

_DEFAULT_MOCK_DIR = "phase3_reconstruction/mock_data"
_DEFAULT_OUT = "phase3_reconstruction/output"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="phase3-reconstruct",
        description="Phase 3 - single-pass drone video -> metric 3D Gaussian scene",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--frames-dir", help="Phase 1 output dir (frames/, masks/, manifest.json)")
    p.add_argument("--poses-dir", help="Phase 2 output dir (poses.json, sparse_cloud.ply)")
    p.add_argument("--out", default=_DEFAULT_OUT, help="output directory")
    p.add_argument("--config", default=None,
                   help="YAML config (default: configs/default.yaml, or "
                        "configs/mock_fast.yaml with --mock)")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="section.key=value", help="override a config value (repeatable)")
    p.add_argument("--mock", action="store_true",
                   help="generate a synthetic toy scene and run entirely on it "
                        "(no GPU / no external models required)")
    p.add_argument("--mock-dir", default=_DEFAULT_MOCK_DIR,
                   help="where to put / find the synthetic scene")
    p.add_argument("--regen-mock", action="store_true",
                   help="force-regenerate the synthetic scene even if it exists")
    p.add_argument("--max-frames", type=int, default=None,
                   help="cap number of frames (overrides config.max_frames)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--version", action="version", version=f"phase3_reconstruction {__version__}")
    return p


def _ensure_mock(mock_dir: str, regen: bool) -> "tuple[str, str]":
    """Generate the synthetic scene if needed; return (frames_dir, poses_dir)."""
    from tools.generate_mock_input import generate_mock_scene  # local import on purpose

    md = Path(mock_dir)
    p1 = md / "phase1_ingestion" / "output"
    p2 = md / "phase2_pose" / "output"
    if regen or not (p1 / "manifest.json").is_file() or not (p2 / "poses.json").is_file():
        log.info("generating synthetic mock scene under %s ...", md)
        generate_mock_scene(str(md))
    else:
        log.info("using existing mock scene under %s (pass --regen-mock to rebuild)", md)
    return str(p1), str(p2)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    try:
        if args.mock:
            cfg_path = args.config or str(Path(__file__).parent.parent / "configs" / "mock_fast.yaml")
            # make sure `tools` is importable when run as a module from repo root
            repo_root = Path(__file__).resolve().parent.parent
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            frames_dir, poses_dir = _ensure_mock(args.mock_dir, args.regen_mock)
        else:
            if not args.frames_dir or not args.poses_dir:
                raise Phase3Error("need --frames-dir and --poses-dir (or use --mock)")
            cfg_path = args.config or str(Path(__file__).parent.parent / "configs" / "default.yaml")
            frames_dir, poses_dir = args.frames_dir, args.poses_dir

        overrides = list(args.overrides)
        if args.max_frames is not None:
            overrides.append(f"max_frames={args.max_frames}")
        cfg = load_config(cfg_path, overrides=overrides)
        log.info("phase3_reconstruction %s | config: %s", __version__, cfg_path)

        from .pipeline import run_pipeline
        result = run_pipeline(frames_dir, poses_dir, args.out, cfg, mock=args.mock)

        log.info("DONE. Outputs:")
        for k in ("splat_scene_ply", "splat_scene_native", "preview_render",
                  "preview_orbit", "preview_confidence", "training_meta"):
            if result.get(k):
                log.info("  %-22s %s", k, result[k])
        return 0

    except Phase3Error as e:
        log.error("Phase 3 stopped: %s", e)
        return 2
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130
    except Exception as e:                            # noqa: BLE001
        log.error("unexpected failure: %s", e)
        log.debug("%s", traceback.format_exc())
        if args.log_level == "DEBUG":
            raise
        log.error("(re-run with --log-level DEBUG for the full traceback)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
