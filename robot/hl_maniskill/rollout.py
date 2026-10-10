"""Evaluate a frozen-representation policy in closed loop."""

from pathlib import Path
import numpy as np
import torch
from hl_maniskill.envs import get_task_spec, make_env, obs_image
from hl_maniskill.policy import (
    ActionNorm, MLPPolicy, apply_preprocess, encode_policy_input, make_backend,
)
from hl_maniskill.utils import default_device, make_run_dir, save_json

def _bool(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().reshape(-1)[0])
    return bool(np.asarray(value).reshape(-1)[0])


def _image(obs) -> torch.Tensor:
    arr = obs_image(obs)
    return torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)


def _load_models(checkpoint: str, device):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    bc_cfg = ck["bc_cfg"]
    backend = make_backend(ck["variant"], device, rep_ckpt=ck["rep_ckpt"])
    hidden = tuple(bc_cfg["hidden"])
    policy = MLPPolicy(
        int(ck["feature_dim"]), int(ck["action_dim"]), hidden=hidden,
    ).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval(); backend.module.eval()
    value = ck["action_norm"]
    norm = ActionNorm(
        torch.tensor(value["mean"], device=device),
        torch.tensor(value["std"], device=device),
    )
    return ck, backend, policy, norm


@torch.no_grad()
def rollout_one(env, task: str, policy, backend, norm, ck, seed: int, max_steps: int, device) -> dict:
    spec = get_task_spec(task)
    obs, info = env.reset(seed=seed)
    obj0, goal0 = spec.object_position(env), spec.goal_position(env, obs)
    qpos0 = obs["agent"]["qpos"].detach().cpu().numpy().reshape(-1)
    min_distance = spec.distance(obj0, goal0)
    interacted = False
    stack = int(ck["bc_cfg"]["stack_frames"])
    buffer = [_image(obs).to(device)] * stack
    success = False
    for step in range(max_steps):
        x = _image(obs).to(device)
        buffer.append(x); buffer.pop(0)
        frames = torch.stack(buffer, dim=1) if stack > 1 else x
        frames = apply_preprocess(frames, backend.preprocess)
        z = encode_policy_input(backend, frames) if stack > 1 else backend.encode(frames)
        action = norm.denormalize(policy(z))
        obs, _, terminated, truncated, info = env.step(action)
        success = _bool(info.get("success", False))
        obj, goal = spec.object_position(env), spec.goal_position(env, obs)
        tcp = obs["extra"]["tcp_pose"].detach().cpu().numpy().reshape(-1)[:3]
        min_distance = min(min_distance, spec.distance(obj, goal))
        if task == "PickCube-v1":
            interacted |= _bool(info.get("is_grasped", False))
        else:
            interacted |= bool(np.linalg.norm(tcp - obj) < 0.045)
        if success or _bool(terminated) or _bool(truncated):
            break
    return {
        "seed": seed, "steps": step + 1,
        "initial_object": np.asarray(obj0, dtype=float).reshape(-1).tolist(),
        "initial_goal": np.asarray(goal0, dtype=float).reshape(-1).tolist(),
        "initial_qpos": np.asarray(qpos0, dtype=float).reshape(-1).tolist(),
        "interaction": interacted,
        "min_distance": float(min_distance),
    }


def evaluate_checkpoint(cfg, checkpoint: str, tag: str | None = None) -> Path:
    device = torch.device(default_device(cfg.eval.get("device", None)))
    ck, backend, policy, norm = _load_models(checkpoint, device)
    run_dir = make_run_dir(Path(cfg.paths.exp_root), "rollout", tag or Path(checkpoint).parent.name)
    output = {
        "checkpoint": str(checkpoint),
        "method": str(ck["variant"]),
        "seed": int(ck["seed"]),
        "task": str(ck["task"]),
        "variants": {},
    }
    task = output["task"]
    for variant in list(cfg.eval.variants):
        env = make_env(task, variant=variant, control_mode=str(cfg.experiment.require_control_mode), image_size=int(cfg.rep.image_size))
        try:
            max_steps = int(cfg.eval.get("max_steps", 0)) or 50
            episodes = [
                rollout_one(env, task, policy, backend, norm, ck,
                            int(cfg.eval.base_seed) + i, max_steps, device)
                for i in range(int(cfg.eval.n_episodes))
            ]
        finally:
            env.close()
        output["variants"][variant] = {
            "sr_10cm": float(np.mean([ep["min_distance"] <= 0.10 for ep in episodes])),
            "interaction_rate": float(np.mean([ep["interaction"] for ep in episodes])),
            "min_distance_mean": float(np.mean([ep["min_distance"] for ep in episodes])),
            "episodes": episodes,
        }
        print(f"[rollout] {task} {variant}: SR@10cm={output['variants'][variant]['sr_10cm']:.3f}")
    id_by_seed = {int(ep["seed"]): ep for ep in output["variants"]["id"]["episodes"]}
    audit = {}
    for variant, result in output["variants"].items():
        if variant == "id":
            continue
        errors = []
        for ep in result["episodes"]:
            ref = id_by_seed[int(ep["seed"])]
            for field in ("initial_object", "initial_goal", "initial_qpos"):
                errors.append(float(np.max(np.abs(
                    np.asarray(ep[field], dtype=float) - np.asarray(ref[field], dtype=float)
                ))))
        audit[variant] = max(errors, default=0.0)
    output["paired_initial_state_audit"] = audit
    worst = max(audit.values(), default=0.0)
    if bool(cfg.eval.get("require_paired_initial_state", False)) and worst > float(
        cfg.eval.get("paired_initial_state_atol", 1e-6)
    ):
        raise RuntimeError(f"{task}: OOD intervention changed physical initial state (max error={worst})")
    save_json(run_dir / "rollout.json", output)
    return run_dir
