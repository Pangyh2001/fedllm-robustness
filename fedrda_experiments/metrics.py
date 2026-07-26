from __future__ import annotations

import math

import numpy as np


def summarize_clients(client_metrics: list[dict], metric_key: str, tail_ratio: float):
    values = np.asarray([metrics[metric_key] for metrics in client_metrics], dtype=np.float64)
    sizes = np.asarray([metrics["num_examples"] for metrics in client_metrics], dtype=np.float64)
    tail_count = max(1, math.ceil(len(values) * tail_ratio))
    return {
        "client_macro": float(values.mean()),
        "sample_weighted": float(np.average(values, weights=sizes)),
        "worst_client": float(values.min()),
        "bottom_tail": float(np.sort(values)[:tail_count].mean()),
        "client_std": float(values.std()),
    }


def summarize_conditional_asr(client_metrics: list[dict], tail_ratio: float):
    values = np.asarray(
        [metrics["conditional_asr"] for metrics in client_metrics], dtype=np.float64
    )
    finite = np.isfinite(values)
    clean_correct = sum(metrics["clean_correct"] for metrics in client_metrics)
    attacked_clean_correct = sum(
        metrics["attacked_clean_correct"] for metrics in client_metrics
    )
    finite_values = values[finite]
    tail_count = max(1, math.ceil(len(finite_values) * tail_ratio))
    return {
        "client_macro": float(finite_values.mean()) if len(finite_values) else float("nan"),
        "pooled": (
            1.0 - attacked_clean_correct / clean_correct
            if clean_correct > 0
            else float("nan")
        ),
        "worst_client": float(finite_values.max()) if len(finite_values) else float("nan"),
        "top_tail": (
            float(np.sort(finite_values)[-tail_count:].mean())
            if len(finite_values)
            else float("nan")
        ),
        "client_std": float(finite_values.std()) if len(finite_values) else float("nan"),
        "num_clients_with_clean_correct": int(finite.sum()),
    }


def select_tail_clients(vulnerability_ema: dict[int, float], tail_ratio: float) -> set[int]:
    tail_count = max(1, math.ceil(len(vulnerability_ema) * tail_ratio))
    ordered = sorted(vulnerability_ema, key=vulnerability_ema.get, reverse=True)
    return set(ordered[:tail_count])
