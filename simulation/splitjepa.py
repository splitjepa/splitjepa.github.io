"""SplitJEPA: recover invariant and variant blocks from a frozen encoder."""

import torch
from lejepa import encode
from world import gaussian, matched_pairs


def apply_head(h, mean, whitening, rotation=None):
    normalized = (h - mean) @ whitening
    return normalized if rotation is None else normalized @ rotation


def fit_splitjepa(encoder, observe, cfg, device):
    """Fit ZCA and the matched-pair spectral rotation without latent labels."""
    d_invariant = cfg["d_invariant"]
    d = d_invariant + cfg["d_variant"]

    # Normalize the frozen representation.
    h = encode(encoder, observe, gaussian(cfg["num_zca"], d, device))
    mean = h.mean(0, keepdim=True)
    centered = h - mean
    covariance = centered.T @ centered / (len(centered) - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    whitening = (
        eigenvectors * eigenvalues.clamp_min(cfg["zca_eps"]).rsqrt()
    ) @ eigenvectors.T

    # Pair differences vary only in the variant block. 
    z1, z2 = matched_pairs(
        cfg["num_pairs"], d_invariant, cfg["d_variant"], device
    )
    h1 = apply_head(encode(encoder, observe, z1), mean, whitening)
    h2 = apply_head(encode(encoder, observe, z2), mean, whitening)
    delta = h1 - h2
    delta = delta - delta.mean(0, keepdim=True)
    difference_covariance = delta.T @ delta / (len(delta) - 1)
    _, rotation = torch.linalg.eigh(difference_covariance)
    return mean, whitening, rotation

