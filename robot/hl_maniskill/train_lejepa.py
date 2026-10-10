"""Train the single-latent LeJEPA comparison."""

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from hl_maniskill.data import TemporalDataset, image_norm_params, load_episode_lens
from hl_maniskill.models import LatentPredictor
from hl_maniskill.utils import default_device, load_json, make_run_dir, save_json
from hl_maniskill.models import LeJEPAEncoder, covariance_loss, predictive_loss, variance_loss


def _move(batch: dict, device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

def _loss_terms(z_t, z_t1, pred_z, weights: dict):
    """Encoder-only loss; deliberately has no invariance input or term."""
    z_batch = torch.cat([z_t, z_t1], dim=0)
    terms = {}
    if float(weights.get("pred", 0.0)) > 0:
        terms["pred"] = predictive_loss(pred_z, z_t1)
    if float(weights.get("variance", 0.0)) > 0:
        terms["variance"] = variance_loss(z_batch)
    if float(weights.get("covariance", 0.0)) > 0:
        terms["covariance"] = covariance_loss(z_batch)
    total = torch.zeros((), device=z_t.device)
    for key, value in terms.items():
        total = total + float(weights.get(key, 0.0)) * value
    return total, terms


def train_lejepa(
    cfg,
    *,
    seed: int | None = None,
    tag: str | None = None,
) -> Path:
    mt, rep = cfg.experiment, cfg.rep
    task = str(cfg.data.task)
    seed = int(mt.get("train_seed", 0) if seed is None else seed)
    torch.manual_seed(seed)
    device = torch.device(default_device(rep.get("device", None)))
    data_root, exp_root = Path(cfg.paths.data_root), Path(cfg.paths.exp_root)
    mean, std = image_norm_params(rep.encoder.get("image_norm", "none"))

    root = data_root / f"{task}_{mt.tag}"
    split = load_json(root / "split.json")
    lens = load_episode_lens(root)
    episodes = sorted(map(int, split["train"]))
    train = TemporalDataset(
        root / "episodes", episodes, lens, int(rep.image_size), mean, std,
    )
    batch_size = int(rep.batch_size)
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=int(rep.get("num_workers", 0)),
    )
    if not train_loader:
        raise RuntimeError("representation dataset is smaller than one batch")
    steps = len(train_loader)

    baseline_cfg = rep.get("lejepa", {}) or {}
    projection_hidden = int(baseline_cfg.get("projection_hidden", 512))
    projection_dim = int(baseline_cfg.get("projection_dim", 128))
    normalize = bool(baseline_cfg.get("normalize", rep.get("normalize", False)))
    encoder = LeJEPAEncoder(
        backbone=str(rep.encoder.backbone),
        pretrained=bool(rep.encoder.get("pretrained", False)),
        projection_hidden=projection_hidden,
        projection_dim=projection_dim,
        normalize=normalize,
    ).to(device)
    predictor = LatentPredictor(
        projection_dim, int(mt.require_action_dim), hidden=int(rep.predictor.hidden),
    ).to(device)
    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=float(rep.lr), weight_decay=float(rep.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(rep.epochs))
    # Reuse the reconstruction-free weights.
    configured_weights = baseline_cfg.get("loss_weights", rep.loss_weights)
    weights = {
        key: float(value) for key, value in dict(configured_weights).items()
        if key in {"pred", "variance", "covariance"}
    }

    condition = tag or f"{rep.tag}_lejepa"
    run_dir = make_run_dir(exp_root, "representation", f"{condition}_seed{seed}")
    metrics = {
        "task": task, "train_episodes": episodes,
        "objective": "predictive+anti_collapse", "matched_pairs_used": False,
        "train": [],
    }
    for epoch in range(int(rep.epochs)):
        encoder.train(); predictor.train()
        totals = {}
        bar = tqdm(train_loader, total=steps, desc=f"encoder-only {condition} seed={seed} epoch={epoch}")
        for step, raw in enumerate(bar):
            batch = _move(raw, device)
            joined = torch.cat([batch["x_t"], batch["x_t1"]], dim=0)
            z_t, z_t1 = encoder(joined).chunk(2, dim=0)
            pred_z = predictor(z_t, batch["a_t"])
            loss, terms = _loss_terms(z_t, z_t1, pred_z, weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, float(rep.get("grad_clip", 10.0)))
            optimizer.step()
            values = {"total": float(loss.detach())}
            values.update({key: float(value.detach()) for key, value in terms.items()})
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + value
            bar.set_postfix({key: f"{value / (step + 1):.4f}" for key, value in totals.items()})
        scheduler.step()
        metrics["train"].append(
            {"epoch": epoch, **{key: value / steps for key, value in totals.items()}},
        )

    torch.save({
        "state_dict": encoder.state_dict(),
        "encoder_cfg": {
            "architecture": "single_latent",
            "backbone": str(rep.encoder.backbone),
            "pooled_dim": int(encoder.feat_dim),
            "projection_hidden": projection_hidden,
            "projection_dim": projection_dim,
            "normalize": normalize,
            "image_size": int(rep.image_size),
            "image_norm": str(rep.encoder.get("image_norm", "none")),
        },
        "objective": {
            "predictive": True, "anti_collapse": True,
            "matched_pair_invariance": False, "weights": weights,
        },
        "task": task, "seed": seed, "epoch": int(rep.epochs) - 1,
    }, run_dir / "encoder.pt")
    torch.save({"state_dict": predictor.state_dict()}, run_dir / "predictor.pt")

    save_json(run_dir / "metrics.json", metrics)
    save_json(run_dir / "run_manifest.json", {
        "stage": "lejepa_representation",
        "architecture": f"resnet_pooled_single_projection_{projection_dim}",
        "task": task, "seed": seed,
        "batch_size": batch_size, "steps_per_epoch": steps,
        "matched_pairs_used": False,
        "matched_pair_invariance": False, "h_l_decomposition": False,
        "loss_weights": weights,
    })
    return run_dir


# Same frozen-encoder behavior-cloning protocol for every representation.
