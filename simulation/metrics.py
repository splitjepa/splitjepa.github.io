"""Held-out metrics."""

import torch

def _with_intercept(x):
    return torch.cat([x, torch.ones(len(x), 1, device=x.device)], dim=1)


def _linear_r2(x_train, y_train, x_test, y_test):
    weight = torch.linalg.lstsq(_with_intercept(x_train), y_train).solution
    prediction = _with_intercept(x_test) @ weight
    residual = (y_test - prediction).square().sum()
    total = (y_test - y_test.mean(0, keepdim=True)).square().sum()
    return (1 - residual / total).item()


def evaluate(z_train, h_train, z_test, h_test, d_invariant):
    """Evaluate the empirical map h = z A^T + b on disjoint samples."""
    solution = torch.linalg.lstsq(_with_intercept(z_train), h_train).solution
    a_hat, intercept = solution[:-1].T, solution[-1]
    prediction = z_test @ a_hat.T + intercept
    global_r2 = 1 - (h_test - prediction).square().sum() / (
        h_test - h_test.mean(0, keepdim=True)
    ).square().sum()

    a12 = a_hat[:d_invariant, d_invariant:]
    a21 = a_hat[d_invariant:, :d_invariant]
    cross_energy = a12.square().sum() + a21.square().sum()
    return {
        "global_r2": global_r2.item(),
        "invariant_r2": _linear_r2(
            h_train[:, :d_invariant], z_train[:, :d_invariant],
            h_test[:, :d_invariant], z_test[:, :d_invariant],
        ),
        "variant_r2": _linear_r2(
            h_train[:, d_invariant:], z_train[:, d_invariant:],
            h_test[:, d_invariant:], z_test[:, d_invariant:],
        ),
        "a12_fro": torch.linalg.norm(a12, "fro").item(),
        "a21_fro": torch.linalg.norm(a21, "fro").item(),
        "block_purity": (1 - cross_energy / a_hat.square().sum()).item(),
    }

