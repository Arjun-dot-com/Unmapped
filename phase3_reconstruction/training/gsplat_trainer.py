"""Real 3D Gaussian Splatting training via `gsplat` (CUDA required).

This follows the documented gsplat pattern: a ``ParameterDict`` of Gaussian
attributes, one Adam optimiser per attribute, ``gsplat.rasterization`` for the
forward pass, and ``gsplat.strategy.DefaultStrategy`` for adaptive densification
/ pruning.  Photometric loss is ``(1-lambda)*L1 + lambda*(1-SSIM)`` computed only
over non-dynamic pixels (Phase-1 masks) so moving objects never supervise the
static field.

NOTE: this module cannot be exercised in a CPU-only / no-gsplat environment.  It
is written to the gsplat >= 1.0 API and kept deliberately small; treat the
densification hyper-parameters in ``configs/default.yaml`` as the tuning surface.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from ..config import TrainingConfig
from ..gaussian_scene import GaussianScene
from ..geometry import SimilarityTransform
from ..logging_utils import get_logger
from .result import TrainResult

log = get_logger(__name__)


def _rgb_to_sh0(rgb: "np.ndarray"):
    # inverse of the SH DC evaluation used by 3DGS (C0 = 0.28209479177387814)
    return (np.asarray(rgb, np.float32) - 0.5) / 0.28209479177387814


def _build_ssim(torch):
    def _gaussian_window(win_size=11, sigma=1.5, channels=3, device="cuda"):
        coords = torch.arange(win_size, dtype=torch.float32, device=device) - win_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = (g / g.sum())
        w2d = g[:, None] @ g[None, :]
        return w2d.expand(channels, 1, win_size, win_size).contiguous()

    def ssim(x, y):
        # x,y : (1,3,H,W) in [0,1]
        c = x.shape[1]
        win = _gaussian_window(channels=c, device=x.device)
        pad = win.shape[-1] // 2
        mu_x = torch.nn.functional.conv2d(x, win, padding=pad, groups=c)
        mu_y = torch.nn.functional.conv2d(y, win, padding=pad, groups=c)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sig_x = torch.nn.functional.conv2d(x * x, win, padding=pad, groups=c) - mu_x2
        sig_y = torch.nn.functional.conv2d(y * y, win, padding=pad, groups=c) - mu_y2
        sig_xy = torch.nn.functional.conv2d(x * y, win, padding=pad, groups=c) - mu_xy
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        s = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / \
            ((mu_x2 + mu_y2 + c1) * (sig_x + sig_y + c2))
        return s.mean()
    return ssim


def train_gsplat(init_scene: GaussianScene, dataset, masks, cfg: TrainingConfig,
                 sim: SimilarityTransform, out_dir, seed: int = 0) -> TrainResult:
    import torch
    from gsplat import rasterization
    try:
        from gsplat.strategy import DefaultStrategy
        have_strategy = True
    except Exception:                                # noqa: BLE001
        have_strategy = False

    device = "cuda"
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.perf_counter()

    # ---- normalise the world for numerical stability -------------- #
    train_scene0 = init_scene.apply_similarity(sim) if not sim.is_identity else init_scene
    cams = [sim.transform_camera(fr.camera) if not sim.is_identity else fr.camera
            for fr in dataset.frames]

    N = len(train_scene0)
    means = torch.tensor(train_scene0.means, dtype=torch.float32, device=device)
    scales = torch.log(torch.tensor(np.maximum(train_scene0.scales, 1e-4),
                                    dtype=torch.float32, device=device))
    quats = torch.tensor(train_scene0.quats, dtype=torch.float32, device=device)
    opac = torch.logit(torch.tensor(np.clip(train_scene0.opacity, 1e-3, 1 - 1e-3),
                                    dtype=torch.float32, device=device))
    sh_dim = (cfg.sh_degree + 1) ** 2
    sh = torch.zeros(N, sh_dim, 3, dtype=torch.float32, device=device)
    sh[:, 0, :] = torch.tensor(_rgb_to_sh0(train_scene0.colors), device=device)

    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opac),
        "sh0": torch.nn.Parameter(sh[:, :1, :].contiguous()),
        "shN": torch.nn.Parameter(sh[:, 1:, :].contiguous()),
    }).to(device)

    optimizers = {
        "means": torch.optim.Adam([params["means"]], lr=cfg.lr_means),
        "scales": torch.optim.Adam([params["scales"]], lr=cfg.lr_scales),
        "quats": torch.optim.Adam([params["quats"]], lr=cfg.lr_quats),
        "opacities": torch.optim.Adam([params["opacities"]], lr=cfg.lr_opacities),
        "sh0": torch.optim.Adam([params["sh0"]], lr=cfg.lr_sh0),
        "shN": torch.optim.Adam([params["shN"]], lr=cfg.lr_shN),
    }

    strategy = None
    strat_state = None
    if have_strategy:
        strategy = DefaultStrategy(
            verbose=False,
            refine_start_iter=cfg.densify_start_iter,
            refine_stop_iter=cfg.densify_stop_iter,
            refine_every=cfg.densify_every,
            reset_every=cfg.reset_opacity_every,
            prune_opa=cfg.prune_opacity_thresh,
            grow_grad2d=cfg.densify_grad_thresh,
        )
        strat_state = strategy.initialize_state(scene_scale=1.0)

    ssim = _build_ssim(torch)
    ds = cfg.train_image_downscale

    # ---- pre-load training tensors ---------------------------- #
    views = []
    for fr, cam in zip(dataset.frames, cams):
        img = fr.load_image(downscale=ds).astype(np.float32) / 255.0
        H, W = img.shape[:2]
        dm = masks.get(fr.frame_id) if masks is not None else None
        if dm is None:
            keep = np.ones((H, W), np.float32)
        else:
            if dm.shape != (H, W):
                import cv2
                dm = cv2.resize(dm.astype(np.uint8), (W, H),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
            keep = (~dm).astype(np.float32)
        K = cam.intr.K.copy()
        if ds > 1:
            K[:2, :] /= ds
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = cam.R
        viewmat[:3, 3] = cam.t
        views.append({
            "rgb": torch.tensor(img, device=device).permute(2, 0, 1)[None],
            "keep": torch.tensor(keep, device=device)[None, None],
            "K": torch.tensor(K, dtype=torch.float32, device=device)[None],
            "viewmat": torch.tensor(viewmat, device=device)[None],
            "H": H, "W": W,
        })

    lam = cfg.ssim_lambda
    psnr_hist = []
    order = np.random.permutation(len(views))
    last_loss = float("nan")

    for step in range(cfg.iterations):
        v = views[int(order[step % len(order)])]
        if step % len(order) == 0 and step > 0:
            order = np.random.permutation(len(views))

        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        renders, alphas, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]).squeeze(-1),
            colors=colors,
            viewmats=v["viewmat"],
            Ks=v["K"],
            width=v["W"], height=v["H"],
            sh_degree=cfg.sh_degree,
            packed=False,
            near_plane=0.01, far_plane=1e4,
        )
        pred = renders[..., :3].permute(0, 3, 1, 2).clamp(0, 1)
        gt = v["rgb"]
        keep = v["keep"]

        l1 = (torch.abs(pred - gt) * keep).sum() / keep.sum().clamp_min(1.0) / 3.0
        s = ssim(pred * keep, gt * keep)
        loss = (1 - lam) * l1 + lam * (1 - s)

        if strategy is not None:
            strategy.step_pre_backward(params, optimizers, strat_state, step, info)
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers.values():
            opt.step()
        if strategy is not None:
            strategy.step_post_backward(params, optimizers, strat_state, step, info)

        last_loss = float(loss.detach().cpu())
        if step % max(1, cfg.eval_every) == 0 or step == cfg.iterations - 1:
            with torch.no_grad():
                mse = (((pred - gt) ** 2) * keep).sum() / keep.sum().clamp_min(1.0) / 3.0
                p = float(-10.0 * torch.log10(mse.clamp_min(1e-10)).cpu())
            psnr_hist.append(p)
            log.info("iter %5d/%d | loss %.4f | PSNR %.2f dB | N=%d",
                     step, cfg.iterations, last_loss, p, params["means"].shape[0])

    dt = time.perf_counter() - t0

    # ---- pull params back, undo normalisation ---------------- #
    with torch.no_grad():
        final = GaussianScene(
            means=params["means"].cpu().numpy(),
            colors=(params["sh0"].squeeze(1).cpu().numpy() * 0.28209479177387814 + 0.5),
            opacity=torch.sigmoid(params["opacities"]).squeeze(-1).cpu().numpy(),
            scales=torch.exp(params["scales"]).cpu().numpy(),
            quats=params["quats"].cpu().numpy(),
            meta=dict(init_scene.meta),
        )
    if not sim.is_identity:
        final = final.apply_similarity(sim.inverse())
    final.meta.update({"trainer": "gsplat", "radiance_optimised": True,
                       "sh_degree": cfg.sh_degree})

    # ---- native checkpoint ------------------------------- #
    native_dir = Path(out_dir)
    native_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = native_dir / "checkpoint.pt"
    import torch as _t
    _t.save({
        "splats": {k: params[k].detach().cpu() for k in params},
        "sh_degree": cfg.sh_degree,
        "sim_transform": sim.to_dict(),
        "config": cfg.__dict__,
        "note": "gsplat-native params: means, scales(log), quats, "
                "opacities(logit), sh0, shN. World = Phase-2 metric AFTER "
                "sim_transform; apply its inverse to return to Phase-2 frame.",
    }, ckpt_path)

    return TrainResult(
        scene=final, backend="gsplat", iterations_run=cfg.iterations,
        train_time_seconds=dt, final_loss=last_loss,
        final_psnr_db=(psnr_hist[-1] if psnr_hist else float("nan")),
        psnr_history=psnr_hist, native_export_path=str(ckpt_path),
        sim_used=sim,
        extra={"gsplat_strategy": "DefaultStrategy" if have_strategy else "none"},
    )
