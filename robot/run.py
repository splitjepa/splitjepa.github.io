#!/usr/bin/env python3
"""Run one SplitJEPA-vs-LeJEPA robot seed."""

from __future__ import annotations
import argparse
from pathlib import Path
from hl_maniskill.config import load_config
from hl_maniskill.train_lejepa import train_lejepa
from hl_maniskill.train_policy import train_policy
from hl_maniskill.train_splitjepa import train_splitjepa


METHODS = ("ours_zh", "ours_zhzl", "lejepa")


def checkpoint(run_dir: Path) -> str:
    path = run_dir / "encoder.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def newest_policy(run_dir: Path) -> str:
    path = run_dir / "policy.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", choices=("representation", "policy", "rollout", "all"), default="all")
    parser.add_argument("--ours-checkpoint", default=None)
    parser.add_argument("--lejepa-checkpoint", default=None)
    parser.add_argument("--policies", nargs="*", default=[])
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config([args.config], overrides=args.set, root=".")
    ours = args.ours_checkpoint
    lejepa = args.lejepa_checkpoint
    if args.stage in ("representation", "all"):
        ours = checkpoint(train_splitjepa(
            cfg, seed=args.seed, tag=f"{cfg.rep.tag}_ours",
        ))
        lejepa = checkpoint(train_lejepa(
            cfg, seed=args.seed, tag=f"{cfg.rep.tag}_lejepa",
        ))
        print(f"OURS_CHECKPOINT={ours}")
        print(f"LEJEPA_CHECKPOINT={lejepa}")

    policy_paths = list(args.policies)
    if args.stage in ("policy", "all"):
        if not ours or not lejepa:
            parser.error("policy stage requires --ours-checkpoint and --lejepa-checkpoint")
        for method in METHODS:
            rep = lejepa if method == "lejepa" else ours
            run = train_policy(
                cfg, rep, variant=method, seed=args.seed,
                tag=f"{cfg.bc.tag}_{method}",
            )
            policy_paths.append(newest_policy(run))
            print(f"POLICY={policy_paths[-1]}")

    if args.stage in ("rollout", "all"):
        if not policy_paths:
            parser.error("rollout stage requires --policies or policies produced in this run")
        from hl_maniskill.rollout import evaluate_checkpoint

        for policy in policy_paths:
            run = evaluate_checkpoint(cfg, policy, tag=f"paper_seed{args.seed}_{Path(policy).parent.name}")
            print(f"ROLLOUT={run / 'rollout.json'}")


if __name__ == "__main__":
    main()
