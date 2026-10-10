"""Synthetic Gaussian world and nonlinear observation process."""

import math
import torch


def gaussian(n, d, device):
    return torch.randn(n, d, device=device)


def ou_pairs(n, d, rho, device):
    """Stationary transition z+ = rho*z + sqrt(1-rho^2)*epsilon."""
    z = gaussian(n, d, device)
    eps = gaussian(n, d, device)
    return z, rho * z + math.sqrt(1.0 - rho**2) * eps


def matched_pairs(n, d_invariant, d_variant, device):
    """Two views with shared invariant factors and independent variants."""
    invariant = gaussian(n, d_invariant, device)
    variant_1 = gaussian(n, d_variant, device)
    variant_2 = gaussian(n, d_variant, device)
    return (
        torch.cat([invariant, variant_1], dim=1),
        torch.cat([invariant, variant_2], dim=1),
    )


def make_observation_map(d, seed, device, layers, strength):
    """Alternating additive tanh couplings; weights stay simulator-private."""
    generator = torch.Generator(device=device).manual_seed(seed)
    weights = []
    for _ in range(layers):
        matrix = torch.randn(d // 2, d // 2, generator=generator, device=device)
        orthogonal, _ = torch.linalg.qr(matrix)
        weights.append(strength * orthogonal)

    def observe(z):
        x = z
        for index, weight in enumerate(weights):
            left, right = x[:, : d // 2], x[:, d // 2 :]
            if index % 2 == 0:
                right = right + torch.tanh(left @ weight)
            else:
                left = left + torch.tanh(right @ weight)
            x = torch.cat([left, right], dim=1)
        return x

    return observe

