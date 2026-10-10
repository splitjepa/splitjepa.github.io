"""Train the common policy MLP on a frozen representation."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from hl_maniskill.data import BcDataset, load_episode_lens
from hl_maniskill.policy import (
    ActionNorm, MLPPolicy, apply_preprocess, encode_policy_input,
    input_pipeline, make_backend,
)
from hl_maniskill.utils import default_device, load_json, make_run_dir, save_json

def _action_norm(dataset: BcDataset, device) -> ActionNorm:
    actions = torch.stack([dataset[i]["a_t"] for i in range(len(dataset))], dim=0).to(device)
    return ActionNorm.from_actions(actions)


def _encode(backend, batch, device, stack: int):
    images = batch["x_t"].to(device)
    x = apply_preprocess(images, backend.preprocess)
    return encode_policy_input(backend, x) if stack > 1 else backend.encode(x)


@torch.no_grad()
def _evaluate_task(policy, backend, norm, loader, device, stack) -> float:
    policy.eval(); backend.module.eval()
    total = count = 0
    for batch in loader:
        z = _encode(backend, batch, device, stack)
        target = norm.normalize(batch["a_t"].to(device))
        pred = policy(z)
        total += float(F.mse_loss(pred, target, reduction="sum"))
        count += target.numel()
    return total / max(count, 1)


def train_policy(
    cfg,
    rep_ckpt: str,
    *,
    variant: str | None = None,
    seed: int | None = None,
    tag: str | None = None,
) -> Path:
    mt, bc = cfg.experiment, cfg.bc
    task = str(cfg.data.task)
    seed = int(bc.get("seed", 0) if seed is None else seed)
    variant = variant or str(mt.get("primary_variant", "ours_zhzl"))
    torch.manual_seed(seed)
    device = torch.device(default_device(bc.get("device", None)))
    data_root, exp_root = Path(cfg.paths.data_root), Path(cfg.paths.exp_root)
    stack = max(1, int(bc.get("stack_frames", 1)))

    root = data_root / f"{task}_{mt.tag}"
    split, lens = load_json(root / "split.json"), load_episode_lens(root)
    train_episodes = sorted(map(int, split["train"]))
    kwargs = dict(size=int(bc.get("dataset_image_size", 128)), stack_frames=stack)
    train_data = BcDataset(root / "episodes", train_episodes, lens, **kwargs)
    val_data = BcDataset(root / "episodes", split["val"], lens, **kwargs)
    batch_size = int(bc.batch_size)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=int(bc.get("num_workers", 0)),
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    steps = len(train_loader)

    backend = make_backend(variant, device, rep_ckpt=rep_ckpt)
    # The visual representation is frozen for every method; only the policy head is optimized.
    expected_dim = int(cfg.rep.zh_dim) + int(cfg.rep.zl_dim)
    if backend.feature_dim != expected_dim:
        raise ValueError(
            f"{variant} exposes {backend.feature_dim} features; expected {expected_dim} "
            "so every method uses the same policy MLP"
        )
    feat_dim = backend.feature_dim * stack
    action_dim = int(mt.require_action_dim)
    # Backends are instantiated above and consume different numbers of randomdraws. 
    # Reset here so every method starts from the exact same policy MLP.
    torch.manual_seed(seed)
    policy = MLPPolicy(feat_dim, action_dim, hidden=tuple(bc.policy_hidden)).to(device)
    norm = _action_norm(train_data, device)

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=float(bc.lr),
        weight_decay=float(bc.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(bc.epochs))

    condition = tag or bc.tag
    run_dir = make_run_dir(exp_root, "bc", f"{variant}_{condition}_seed{seed}")
    metrics = {"task": task, "train": [], "val": []}
    best_val = float("inf")
    for epoch in range(int(bc.epochs)):
        policy.train()
        backend.module.eval()
        total = 0.0
        for batch in tqdm(train_loader, total=steps, desc=f"mt-bc {condition} seed={seed} epoch={epoch}"):
            z = _encode(backend, batch, device, stack)
            target = norm.normalize(batch["a_t"].to(device))
            loss = F.mse_loss(policy(z), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        scheduler.step()
        train_mse = total / max(steps, 1)
        val_mse = _evaluate_task(policy, backend, norm, val_loader, device, stack)
        metrics["train"].append({"epoch": epoch, "mse": train_mse})
        metrics["val"].append({"epoch": epoch, "mse": val_mse})
        print(f"[bc] epoch={epoch} val_mse={val_mse:.6f}")
        if val_mse < best_val:
            best_val = val_mse
            checkpoint = {
                "policy": policy.state_dict(),
                "task": task, "seed": seed,
                "variant": variant, "rep_ckpt": str(rep_ckpt),
                "input_pipeline": input_pipeline(variant),
                "feature_dim": feat_dim, "action_dim": action_dim,
                "policy_architecture": [feat_dim, *map(int, bc.policy_hidden), action_dim],
                "action_norm": {"mean": norm.mean.cpu().tolist(), "std": norm.std.cpu().tolist()},
                "preprocess": backend.preprocess,
                "bc_cfg": {"stack_frames": stack,
                           "hidden": list(bc.policy_hidden), "data_tag": str(mt.tag),
                           },
                "val_mse": val_mse,
            }
            torch.save(checkpoint, run_dir / "policy.pt")
    save_json(run_dir / "metrics.json", metrics)
    save_json(run_dir / "run_manifest.json", {
        "stage": "behavior_cloning", "task": task,
        "train_episodes": train_episodes, "seed": seed, "variant": variant,
        "input_pipeline": input_pipeline(variant),
        "batch_size": batch_size, "visual_encoder_frozen": True,
    })
    return run_dir
