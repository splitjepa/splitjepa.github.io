"""Run the paper's core synthetic comparison: LeJEPA vs SplitJEPA."""

import argparse
import json
import os

import torch

from lejepa import Encoder, encode, train_lejepa
from metrics import evaluate
from splitjepa import apply_head, fit_splitjepa
from world import gaussian, make_observation_map


CONFIG = {
    "d_invariant": 4,
    "d_variant": 4,
    "coupling_layers": 4,
    "coupling_strength": 0.5,
    "rho": 0.5,
    "steps": 10_000,
    "batch_size": 256,
    "lr": 3e-3,
    "sigreg_weight": 1e-6,
    "sigreg_slices": 128,
    "sigreg_knots": 17,
    "num_zca": 30_000,
    "num_pairs": 30_000,
    "num_probe_train": 15_000,
    "num_probe_test": 15_000,
    "zca_eps": 1e-5,
}


def run(seed, device):
    cfg = CONFIG
    d = cfg["d_invariant"] + cfg["d_variant"]
    torch.manual_seed(seed)
    observe = make_observation_map(
        d, seed, device, cfg["coupling_layers"], cfg["coupling_strength"]
    )
    torch.manual_seed(seed + 77_777)
    encoder = Encoder(d, cfg["coupling_layers"]).to(device)
    train_lejepa(encoder, observe, cfg, device)

    # Both methods use the same frozen encoder and ZCA transform.
    mean, whitening, rotation = fit_splitjepa(encoder, observe, cfg, device)
    z_train = gaussian(cfg["num_probe_train"], d, device)
    z_test = gaussian(cfg["num_probe_test"], d, device)
    h_train = encode(encoder, observe, z_train)
    h_test = encode(encoder, observe, z_test)
    lejepa_train = apply_head(h_train, mean, whitening)
    lejepa_test = apply_head(h_test, mean, whitening)

    return {
        "seed": seed,
        "lejepa": evaluate(
            z_train, lejepa_train, z_test, lejepa_test, cfg["d_invariant"]
        ),
        "splitjepa": evaluate(
            z_train, lejepa_train @ rotation,
            z_test, lejepa_test @ rotation,
            cfg["d_invariant"],
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; use --device cpu.")

    result = run(args.seed, torch.device(args.device))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"seed={args.seed}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

