from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    name: str = "agnews"
    cache_dir: str | None = None
    max_train_samples: int | None = None
    max_test_samples: int | None = None
    max_length: int = 256
    num_clients: int = 10
    dirichlet_alpha: float = 0.1
    partition_mode: str = "label_skew_equal"
    val_fraction: float = 0.1
    min_client_samples: int = 64
    min_client_test_samples: int = 100


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen2.5-3B-Instruct"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    dtype: str = "bfloat16"


@dataclass
class AttackConfig:
    epsilon: float = 0.03
    eval_epsilon: float | None = None
    epsilon_mode: str = "relative_rms"
    train_steps: int = 3
    eval_steps: list[int] = field(default_factory=lambda: [1, 10, 20])
    eval_restarts: int = 1
    step_size: float | None = None
    random_start: bool = True


@dataclass
class FederatedConfig:
    algorithm: str = "fedrda"
    rounds: int = 50
    local_epochs: int = 1
    max_train_batches: int | None = 50
    batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 2e-4
    client_fraction: float = 1.0
    adv_weight: float = 0.3
    clean_consistency_weight: float = 0.1
    calfat_tau: float = 0.1
    sfat_top_k: int = 1
    sfat_multiplier: float = 1.4
    qfed_q: float = 1.0
    warmup_rounds: int = 5
    residual_weight: float = 1.0
    residual_norm_cap: float = 1.0
    residual_ema: float = 0.0
    vulnerability_ema: float = 0.9
    tail_ratio: float = 0.2
    tail_reweight: float = 2.0
    tail_refresh_every: int = 1
    tail_eval_steps: int = 1
    tail_eval_batches: int | None = None
    risk_temperature: float = 1.0
    risk_weight_cap: float = 3.0
    qp_rho: float = 10.0
    qp_kappa: float = 0.1
    evaluate_every: int = 1
    max_eval_batches: int | None = None
    final_eval_batches: int | None = None


@dataclass
class RunConfig:
    seed: int = 42
    output_dir: str = "outputs_refactor"
    run_name: str = "agnews_fedrda"
    device: str = "cuda"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    valid = {f.name for f in dataclasses.fields(instance)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"Unknown config keys for {type(instance).__name__}: {sorted(unknown)}")
    for key, value in values.items():
        current = getattr(instance, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"{key} must be a mapping")
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> RunConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _merge_dataclass(RunConfig(), raw)


def parse_args() -> tuple[RunConfig, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--algorithm")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--residual-weight", type=float)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--warmup-rounds", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--eval-restarts", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--dirichlet-alpha", type=float)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--clean-consistency-weight", type=float)
    parser.add_argument("--tail-ratio", type=float)
    parser.add_argument("--qfed-q", type=float)
    parser.add_argument("--qp-kappa", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.algorithm:
        cfg.federated.algorithm = args.algorithm
    if args.model_name_or_path:
        cfg.model.name_or_path = args.model_name_or_path
    if args.seed is not None:
        cfg.seed = args.seed
    if args.run_name:
        cfg.run_name = args.run_name
    if args.residual_weight is not None:
        cfg.federated.residual_weight = args.residual_weight
    if args.rounds is not None:
        cfg.federated.rounds = args.rounds
    if args.warmup_rounds is not None:
        cfg.federated.warmup_rounds = args.warmup_rounds
    if args.max_train_batches is not None:
        cfg.federated.max_train_batches = args.max_train_batches
    if args.eval_restarts is not None:
        cfg.attack.eval_restarts = args.eval_restarts
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.dirichlet_alpha is not None:
        cfg.data.dirichlet_alpha = args.dirichlet_alpha
    if args.lora_rank is not None:
        cfg.model.lora_rank = args.lora_rank
    if args.clean_consistency_weight is not None:
        cfg.federated.clean_consistency_weight = args.clean_consistency_weight
    if args.tail_ratio is not None:
        cfg.federated.tail_ratio = args.tail_ratio
    if args.qfed_q is not None:
        cfg.federated.qfed_q = args.qfed_q
    if args.qp_kappa is not None:
        cfg.federated.qp_kappa = args.qp_kappa
    return cfg, args.resume
