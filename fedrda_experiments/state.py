from __future__ import annotations

from collections import OrderedDict

import torch


TensorState = OrderedDict[str, torch.Tensor]


def trainable_state(model, device: str = "cpu") -> TensorState:
    return OrderedDict(
        (name, param.detach().to(device=device, dtype=torch.float32).clone())
        for name, param in model.named_parameters()
        if param.requires_grad
    )


@torch.no_grad()
def load_trainable_state(model, state: TensorState) -> None:
    params = dict(model.named_parameters())
    for name, value in state.items():
        if name not in params:
            raise KeyError(f"Missing trainable parameter: {name}")
        if params[name].shape != value.shape:
            raise ValueError(f"Shape mismatch for {name}: {params[name].shape} != {value.shape}")
        params[name].copy_(value.to(device=params[name].device, dtype=params[name].dtype))


def subtract(left: TensorState, right: TensorState) -> TensorState:
    return OrderedDict((name, left[name] - right[name]) for name in left)


def add(base: TensorState, *updates: TensorState) -> TensorState:
    return OrderedDict(
        (name, base[name] + sum(update[name] for update in updates))
        for name in base
    )


def scale(state: TensorState, factor: float) -> TensorState:
    return OrderedDict((name, value * factor) for name, value in state.items())


def weighted_sum(states: list[TensorState], weights: list[float]) -> TensorState:
    if len(states) != len(weights) or not states:
        raise ValueError("states and weights must be non-empty and have equal length")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    normalized = [float(weight) / total for weight in weights]
    return OrderedDict(
        (
            name,
            sum(weight * state[name] for state, weight in zip(states, normalized)),
        )
        for name in states[0]
    )


def flatten(state: TensorState) -> torch.Tensor:
    return torch.cat([value.reshape(-1).float() for value in state.values()])


def unflatten(vector: torch.Tensor, template: TensorState) -> TensorState:
    output: TensorState = OrderedDict()
    offset = 0
    for name, value in template.items():
        size = value.numel()
        output[name] = vector[offset : offset + size].reshape_as(value).clone()
        offset += size
    if offset != vector.numel():
        raise ValueError("Vector length does not match template")
    return output
