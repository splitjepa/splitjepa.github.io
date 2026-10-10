"""Frozen representation adapters and the common policy MLP."""

import torch
import torch.nn as nn

from hl_maniskill.data import image_norm_params
from hl_maniskill.models import LeJEPAEncoder, VisualEncoder

class MLPPolicy(nn.Module):
    """representation -> MLP -> action. Same architecture for ALL baselines."""

    def __init__(self, in_dim: int, out_dim: int, hidden: tuple[int, ...] = (512, 256)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ActionNorm:
    """Fixed action normalization statistics."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    @classmethod
    def from_actions(cls, actions: torch.Tensor) -> "ActionNorm":
        mean = actions.mean(dim=0)
        std = actions.std(dim=0).clamp_min(1e-6)
        return cls(mean, std)

    def normalize(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.mean.to(a.device)) / self.std.to(a.device)

    def denormalize(self, a: torch.Tensor) -> torch.Tensor:
        return a * self.std.to(a.device) + self.mean.to(a.device)

PIPELINES = {
    "lejepa": "image -> single reconstruction-free latent -> policy",
    "ours_zh": "image -> [r_H, zeros] -> policy",
    "ours_zhzl": "image -> [r_H, r_L] -> policy",
}


def input_pipeline(variant: str) -> str:
    return PIPELINES[variant]


class Backend:
    def __init__(self, module: nn.Module, feature_dim: int, preprocess: dict):
        self.module = module
        self.feature_dim = int(feature_dim)
        self.preprocess = preprocess

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.module(images)


def _normalize(images: torch.Tensor, size: int, mean, std) -> torch.Tensor:
    if images.shape[-2:] != (size, size):
        images = nn.functional.interpolate(images, (size, size), mode="bilinear", align_corners=False)
    if mean is not None:
        mu = torch.as_tensor(mean, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        sigma = torch.as_tensor(std, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        images = (images - mu) / sigma
    return images


def apply_preprocess(images: torch.Tensor, preprocess: dict) -> torch.Tensor:
    """Preprocess one image or a temporal stack without changing its layout."""
    if images.ndim == 5:
        batch, steps = images.shape[:2]
        flat = _normalize(images.flatten(0, 1), int(preprocess["size"]),
                          preprocess.get("mean"), preprocess.get("std"))
        return flat.view(batch, steps, *flat.shape[1:])
    return _normalize(images, int(preprocess["size"]),
                      preprocess.get("mean"), preprocess.get("std"))


def encode_policy_input(backend: Backend, frames: torch.Tensor) -> torch.Tensor:
    if frames.ndim == 4:
        frames = frames.unsqueeze(1)
    return torch.cat([backend.module(frames[:, i]) for i in range(frames.shape[1])], dim=-1)


def make_backend(variant: str, device, rep_ckpt: str | None = None) -> Backend:
    """Load a frozen LeJEPA, r_H, or [r_H, r_L] encoder."""
    if variant not in PIPELINES:
        raise ValueError(f"unsupported paper variant: {variant}")
    if rep_ckpt is None:
        raise ValueError(f"{variant} requires a representation checkpoint")

    checkpoint = torch.load(rep_ckpt, map_location="cpu", weights_only=False)
    enc = checkpoint["encoder_cfg"]
    if variant == "lejepa":
        encoder = LeJEPAEncoder(
            backbone=enc["backbone"],
            projection_hidden=int(enc.get("projection_hidden", 512)),
            projection_dim=int(enc.get("projection_dim", 128)),
            normalize=bool(enc.get("normalize", False)),
        )
        encoder.load_state_dict(checkpoint["state_dict"])
        module = _SingleLatent(encoder)
        feature_dim = int(enc.get("projection_dim", 128))
    else:
        encoder = VisualEncoder(
            backbone=enc["backbone"], zh_dim=int(enc["zh_dim"]),
            zl_dim=int(enc["zl_dim"]), normalize=bool(enc.get("normalize", False)),
        )
        encoder.load_state_dict(checkpoint["state_dict"])
        include_zl = variant == "ours_zhzl"
        module = _SplitLatent(encoder, include_zl=include_zl)
        # Every policy receives the same tensor width. For r_H-only, the r_L
        # slots are fixed zeros: identical MLP shape, no additional information.
        feature_dim = int(enc["zh_dim"]) + int(enc["zl_dim"])

    module.to(device).eval()
    module.requires_grad_(False)
    mean, std = image_norm_params(enc.get("image_norm", "none"))
    preprocess = {"size": int(enc["image_size"]), "mean": mean, "std": std}
    return Backend(module, feature_dim, preprocess)


class _SingleLatent(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, images):
        return self.encoder(images)


class _SplitLatent(nn.Module):
    """Expose head outputs only; shared trunk features never leave the encoder."""
    def __init__(self, encoder, include_zl: bool):
        super().__init__()
        self.encoder = encoder
        self.include_zl = include_zl

    def forward(self, images):
        zh, zl = self.encoder(images)
        return torch.cat([zh, zl], dim=-1) if self.include_zl else torch.cat([zh, torch.zeros_like(zl)], dim=-1)
