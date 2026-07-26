from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import minimize

from .state import TensorState, flatten, scale, unflatten, weighted_sum


@dataclass
class ResidualDiagnostics:
    average_pair_cosine: float
    conflict_rate: float
    residual_norm_mean: float
    residual_norm_std: float
    aggregate_alignment: list[float]


def _unit_vectors(states: list[TensorState], eps: float = 1e-12):
    vectors = [flatten(state) for state in states]
    norms = torch.tensor([vector.norm().item() for vector in vectors], dtype=torch.float64)
    units = [vector / max(norm, eps) for vector, norm in zip(vectors, norms.tolist())]
    return vectors, units, norms


def residual_diagnostics(
    residuals: list[TensorState], weights: list[float]
) -> ResidualDiagnostics:
    _, units, norms = _unit_vectors(residuals)
    gram = torch.stack(
        [torch.stack([torch.dot(left, right) for right in units]) for left in units]
    ).double()
    off_diagonal = gram[~torch.eye(len(units), dtype=torch.bool)]
    aggregate = sum(weight * unit for weight, unit in zip(weights, units))
    aggregate = aggregate / aggregate.norm().clamp_min(1e-12)
    return ResidualDiagnostics(
        average_pair_cosine=float(off_diagonal.mean()) if off_diagonal.numel() else 1.0,
        conflict_rate=float((off_diagonal < 0).double().mean()) if off_diagonal.numel() else 0.0,
        residual_norm_mean=float(norms.mean()),
        residual_norm_std=float(norms.std(unbiased=False)),
        aggregate_alignment=[float(torch.dot(aggregate, unit)) for unit in units],
    )


def average_residual(residuals: list[TensorState], weights: list[float]) -> TensorState:
    return weighted_sum(residuals, weights)


def tail_reweighted_residual(
    residuals: list[TensorState],
    weights: list[float],
    client_ids: list[int],
    tail_ids: set[int],
    multiplier: float,
) -> TensorState:
    adjusted = [
        weight * (multiplier if client_id in tail_ids else 1.0)
        for weight, client_id in zip(weights, client_ids)
    ]
    return weighted_sum(residuals, adjusted)


def sfat_update(
    updates: list[TensorState],
    losses: list[float],
    top_k: int,
    multiplier: float,
    use_slack: bool,
):
    """LoRA adaptation of the official SFAT client-wise slack aggregation."""
    if not use_slack or top_k <= 0:
        weights = [1.0 / len(updates)] * len(updates)
        return weighted_sum(updates, weights), weights
    top_k = min(top_k, len(updates))
    selected = set(np.argsort(np.asarray(losses))[-top_k:].tolist())
    raw = [multiplier if index in selected else 1.0 for index in range(len(updates))]
    total = sum(raw)
    weights = [value / total for value in raw]
    return weighted_sum(updates, weights), weights


def qfedavg_update(
    updates: list[TensorState],
    losses: list[float],
    learning_rate: float,
    q: float,
):
    """q-FedAvg dynamic update, following the authors' reference implementation."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    deltas = []
    hs = []
    for update, loss in zip(updates, losses):
        safe_loss = max(float(loss), 1e-10)
        gradient_norm_sq = float(
            sum(value.float().pow(2).sum() for value in update.values())
            / (learning_rate**2)
        )
        deltas.append(scale(update, safe_loss**q / learning_rate))
        hs.append(
            q * safe_loss ** (q - 1.0) * gradient_norm_sq
            + safe_loss**q / learning_rate
        )
    denominator = max(sum(hs), 1e-12)
    combined = scale(
        weighted_sum(deltas, [1.0] * len(deltas)),
        len(deltas) / denominator,
    )
    return combined, {
        "losses": [float(value) for value in losses],
        "hs": hs,
        "denominator": denominator,
    }


def pcgrad_residual(
    residuals: list[TensorState], weights: list[float], seed: int
) -> TensorState:
    template = residuals[0]
    vectors = [flatten(state).clone() for state in residuals]
    generator = torch.Generator().manual_seed(seed)
    projected = []
    for index, original in enumerate(vectors):
        current = original.clone()
        order = torch.randperm(len(vectors), generator=generator).tolist()
        for other_index in order:
            if other_index == index:
                continue
            other = vectors[other_index]
            dot = torch.dot(current, other)
            if dot < 0:
                current = current - dot / other.pow(2).sum().clamp_min(1e-12) * other
        projected.append(current)
    combined = sum(weight * vector for weight, vector in zip(weights, projected))
    return unflatten(combined, template)


def fedrda_residual(
    residuals: list[TensorState],
    weights: list[float],
    client_ids: list[int],
    tail_ids: set[int],
    rho: float,
    kappa: float,
):
    template = residuals[0]
    _, units, norms = _unit_vectors(residuals)
    gram = np.asarray(
        [[float(torch.dot(left, right)) for right in units] for left in units],
        dtype=np.float64,
    )
    gram = (gram + gram.T) / 2
    weighted_direction = sum(weight * unit for weight, unit in zip(weights, units))
    direction_norm = float(weighted_direction.norm())
    if direction_norm < 1e-12:
        return unflatten(torch.zeros_like(units[0]), template), {
            "success": True,
            "message": "zero mean direction",
            "aligned_norm": 0.0,
            "mean_residual_norm": float(sum(w * n for w, n in zip(weights, norms))),
        }
    target_coeff = np.asarray(weights, dtype=np.float64) / direction_norm
    tail_positions = [i for i, client_id in enumerate(client_ids) if client_id in tail_ids]
    count = len(residuals)
    tail_count = len(tail_positions)

    def objective(x):
        diff = x[:count] - target_coeff
        return float(diff @ gram @ diff + rho * x[count:].sum())

    def gradient(x):
        grad = np.empty_like(x)
        grad[:count] = 2 * gram @ (x[:count] - target_coeff)
        grad[count:] = rho
        return grad

    def constraints(x):
        coefficients = x[:count]
        alignments = gram @ coefficients
        tail_slack = x[count:]
        values = [*list(alignments + kappa), 1.0 - float(coefficients @ gram @ coefficients)]
        values.extend(
            alignments[position] + tail_slack[j]
            for j, position in enumerate(tail_positions)
        )
        return np.asarray(values)

    initial = np.concatenate(
        [
            target_coeff,
            np.maximum(-(gram @ target_coeff)[tail_positions], 0.0),
        ]
    )
    result = minimize(
        objective,
        initial,
        jac=gradient,
        constraints={"type": "ineq", "fun": constraints},
        bounds=[(None, None)] * count + [(0.0, None)] * tail_count,
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-9},
    )
    if not result.success or np.min(constraints(result.x)) < -1e-5:
        coefficients = np.zeros(count, dtype=np.float64)
        success = False
        message = f"QP fallback to zero: {result.message}"
    else:
        coefficients = result.x[:count]
        success = True
        message = str(result.message)
    aligned = sum(float(coefficient) * unit for coefficient, unit in zip(coefficients, units))
    mean_norm = float(sum(weight * norm for weight, norm in zip(weights, norms.tolist())))
    aligned_update = unflatten(aligned * mean_norm, template)
    return aligned_update, {
        "success": success,
        "message": message,
        "aligned_norm": float(aligned.norm()),
        "mean_residual_norm": mean_norm,
        "min_alignment": float(np.min(gram @ coefficients)),
        "tail_min_alignment": (
            float(np.min((gram @ coefficients)[tail_positions])) if tail_positions else None
        ),
    }
