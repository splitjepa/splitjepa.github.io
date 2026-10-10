"""LeJEPA encoder and reconstruction-free training objective."""

import math
import torch
import torch.nn.functional as F
from torch import nn
from world import ou_pairs


class Encoder(nn.Module):
    """Learned inverse couplings."""
    def __init__(self, d, layers):
        super().__init__()
        self.half = d // 2
        self.weights = nn.ParameterList(
            nn.Parameter(torch.randn(self.half, self.half) / math.sqrt(self.half))
            for _ in range(layers)
        )

    def forward(self, x):
        h = x
        for index, weight in reversed(list(enumerate(self.weights))):
            left, right = h[:, : self.half], h[:, self.half :]
            if index % 2 == 0:
                right = right - torch.tanh(left @ weight)
            else:
                left = left - torch.tanh(right @ weight)
            h = torch.cat([left, right], dim=1)
        return h


class SIGReg(nn.Module):
    """Sliced characteristic-function regularizer toward N(0, I)."""
    def __init__(self, knots, slices, t_max=3.0):
        super().__init__()
        self.slices = slices
        t = torch.linspace(0.0, t_max, knots)
        dt = t_max / (knots - 1)
        quadrature = torch.full((knots,), 2.0 * dt)
        quadrature[[0, -1]] = dt
        self.register_buffer("t", t)
        self.register_buffer("target", torch.exp(-0.5 * t.square()))
        self.register_buffer("weights", quadrature * torch.exp(-0.5 * t.square()))

    def forward(self, views):
        h = views.flatten(0, 1)
        directions = F.normalize(
            torch.randn(h.shape[1], self.slices, device=h.device), dim=0
        )
        projected = (h @ directions).unsqueeze(-1) * self.t
        error = (
            (projected.cos().mean(0) - self.target).square()
            + projected.sin().mean(0).square()
        )
        return (error @ self.weights).mean() * len(h)


def train_lejepa(encoder, observe, cfg, device):
    regularizer = SIGReg(cfg["sigreg_knots"], cfg["sigreg_slices"]).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=cfg["lr"])
    d = cfg["d_invariant"] + cfg["d_variant"]

    for step in range(cfg["steps"]):
        halfway = max(cfg["steps"] // 2, 1)
        progress = max(step - halfway, 0) / max(cfg["steps"] - halfway, 1)
        lr = cfg["lr"] if step < halfway else cfg["lr"] * (
            1 + math.cos(math.pi * progress)
        ) / 2
        for group in optimizer.param_groups:
            group["lr"] = lr

        z1, z2 = ou_pairs(cfg["batch_size"], d, cfg["rho"], device)
        h = encoder(observe(torch.cat([z1, z2]))).reshape(
            2, cfg["batch_size"], d
        )
        alignment = (h.mean(0) - h).square().mean()
        sigreg = regularizer(h)
        loss = (
            (1 - cfg["sigreg_weight"]) * alignment
            + cfg["sigreg_weight"] * sigreg
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0 or step == cfg["steps"] - 1:
            print(f"step {step:5d}  loss={loss.item():.4e}", flush=True)


@torch.no_grad()
def encode(encoder, observe, z, chunk=10_000):
    encoder.eval()
    return torch.cat(
        [encoder(observe(z[start : start + chunk]))
         for start in range(0, len(z), chunk)]
    )

