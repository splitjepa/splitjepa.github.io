"""Train the SplitJEPA representation."""

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from hl_maniskill.data import TemporalDataset, image_norm_params, load_episode_lens
from hl_maniskill.models import LatentPredictor
from hl_maniskill.utils import default_device, load_json, make_run_dir, save_json
from hl_maniskill.data import PairDataset, make_split_pair_arrays
from hl_maniskill.models import VisualEncoder, compute_rep_loss

def _move(batch: dict, device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def train_splitjepa(
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
    train_episodes = sorted(map(int, split["train"]))
    pred_ds = TemporalDataset(
        root / "episodes", train_episodes, lens, int(rep.image_size), mean, std,
    )
    pair_ds = PairDataset(
        root / "episodes", make_split_pair_arrays(root, "train"),
        int(rep.image_size), mean, std,
    )
    batch_size = int(rep.batch_size)
    loader_args = dict(batch_size=batch_size, shuffle=True, drop_last=True,
                       num_workers=int(rep.get("num_workers", 0)))
    pred_loader = DataLoader(pred_ds, **loader_args)
    pair_loader = DataLoader(pair_ds, **loader_args)
    if not pred_loader or not pair_loader:
        raise RuntimeError("representation dataset is smaller than one batch")
    steps = len(pred_loader)

    encoder = VisualEncoder(
        backbone=rep.encoder.backbone,
        pretrained=bool(rep.encoder.get("pretrained", False)),
        zh_dim=int(rep.zh_dim), zl_dim=int(rep.zl_dim),
        pool=rep.encoder.get("pool", "avg"),
        normalize=bool(rep.get("normalize", False)),
    ).to(device)
    z_dim = int(rep.zh_dim) + int(rep.zl_dim)
    predictor = LatentPredictor(z_dim, int(mt.require_action_dim), hidden=int(rep.predictor.hidden)).to(device)
    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(rep.lr), weight_decay=float(rep.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(rep.epochs))
    weights = dict(rep.loss_weights)

    condition = tag or rep.tag
    run_dir = make_run_dir(exp_root, "representation", f"{condition}_seed{seed}")
    metrics = {"task": task, "train_episodes": train_episodes, "train": []}
    for epoch in range(int(rep.epochs)):
        encoder.train(); predictor.train()
        totals: dict[str, float] = {}
        pair_iter = iter(pair_loader)
        bar = tqdm(pred_loader, total=steps, desc=f"mt-rep {condition} seed={seed} epoch={epoch}")
        for step, pred_batch in enumerate(bar):
            try:
                pair_batch = next(pair_iter)
            except StopIteration:
                pair_iter = iter(pair_loader)
                pair_batch = next(pair_iter)
            p, q = _move(pred_batch, device), _move(pair_batch, device)
            x_t, x_t1, a_t = p["x_t"], p["x_t1"], p["a_t"]
            x_a, x_b = q["x_a"], q["x_b"]
            batch_size = x_t.shape[0]
            zh, zl = encoder(torch.cat([x_t, x_t1, x_a, x_b], dim=0))
            zh_t, zh_t1, zh_a, zh_b = torch.split(zh, batch_size)
            zl_t, zl_t1, zl_a, zl_b = torch.split(zl, batch_size)
            z_t, z_t1 = torch.cat([zh_t, zl_t], -1), torch.cat([zh_t1, zl_t1], -1)
            z_a, z_b = torch.cat([zh_a, zl_a], -1), torch.cat([zh_b, zl_b], -1)
            pred_z = predictor(z_t, a_t) if weights.get("pred", 0.0) > 0 else None
            loss, per = compute_rep_loss(
                z_t=z_t, z_t1=z_t1, zh_a=zh_a, zh_b=zh_b,
                pred=pred_z, weights=weights,
                batch_z=torch.cat([z_t, z_t1, z_a, z_b]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, float(rep.get("grad_clip", 10.0)))
            optimizer.step()
            for key, value in per.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            bar.set_postfix({key: f"{value / (step + 1):.4f}" for key, value in totals.items()})
        scheduler.step()
        metrics["train"].append({"epoch": epoch, **{key: value / steps for key, value in totals.items()}})

    torch.save({
        "state_dict": encoder.state_dict(),
        "encoder_cfg": {
            "backbone": rep.encoder.backbone,
            "zh_dim": int(rep.zh_dim), "zl_dim": int(rep.zl_dim),
            "normalize": bool(rep.get("normalize", False)),
            "image_size": int(rep.image_size),
            "image_norm": str(rep.encoder.get("image_norm", "none")),
        },
        "task": task, "seed": seed, "epoch": int(rep.epochs) - 1,
    }, run_dir / "encoder.pt")
    torch.save({"state_dict": predictor.state_dict()}, run_dir / "predictor.pt")
    save_json(run_dir / "metrics.json", metrics)
    save_json(run_dir / "run_manifest.json", {
        "stage": "representation", "task": task, "seed": seed,
        "batch_size": batch_size, "steps_per_epoch": steps,
    })
    return run_dir


# Single-latent LeJEPA baseline: no matched-pair input or invariance term.
