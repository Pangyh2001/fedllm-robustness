from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from fedrda_experiments.config import load_config
from fedrda_experiments.data import dataset_spec, load_federated_data
from fedrda_experiments.metrics import summarize_clients, summarize_conditional_asr
from fedrda_experiments.modeling import load_model_and_tokenizer
from fedrda_experiments.state import load_trainable_state
from fedrda_experiments.training import evaluate_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one trained checkpoint over an embedding-PGD epsilon sweep."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--epsilons", type=float, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--samples-per-client", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def client_subset(dataset, limit: int, seed: int):
    if limit < 0 or limit >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), size=limit, replace=False)).tolist()
    return Subset(dataset, indices)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg.model.name_or_path = args.model_name_or_path
    cfg.device = args.device
    cfg.attack.eval_restarts = args.restarts

    set_seed(args.seed)
    num_labels = dataset_spec(cfg.data.name)["num_labels"]
    model, tokenizer = load_model_and_tokenizer(
        cfg.model, num_labels=num_labels, device=args.device
    )
    clients, split_metadata = load_federated_data(cfg.data, tokenizer, args.seed)

    checkpoint_path = Path(args.run_dir) / "latest_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    load_trainable_state(model, checkpoint["global_state"])

    report = {
        "run_dir": str(Path(args.run_dir)),
        "checkpoint_round": int(checkpoint["round"]),
        "epsilon_mode": cfg.attack.epsilon_mode,
        "steps": args.steps,
        "restarts": args.restarts,
        "batch_size": args.batch_size,
        "samples_per_client": args.samples_per_client,
        "seed": args.seed,
        "data_partition": split_metadata,
        "results": {},
    }

    for epsilon in args.epsilons:
        # Reusing the same RNG seed makes random starts comparable across budgets.
        set_seed(args.seed + 10_000)
        cfg.attack.epsilon = float(epsilon)
        started = time.perf_counter()
        client_results = []
        for client in clients:
            subset = client_subset(
                client.test,
                args.samples_per_client,
                args.seed * 1000 + client.client_id,
            )
            metrics = evaluate_dataset(
                model,
                subset,
                args.device,
                args.batch_size,
                attack_cfg=cfg.attack,
                attack_steps=args.steps,
                restarts=args.restarts,
                max_batches=None,
            )
            client_results.append({"client_id": client.client_id, **metrics})
            print(
                f"epsilon={epsilon:g} client={client.client_id} "
                f"clean={metrics['clean_accuracy']:.4f} "
                f"robust={metrics['robust_accuracy']:.4f}",
                flush=True,
            )

        epsilon_result = {
            "clients": client_results,
            "clean": summarize_clients(
                client_results, "clean_accuracy", cfg.federated.tail_ratio
            ),
            "robust": summarize_clients(
                client_results, "robust_accuracy", cfg.federated.tail_ratio
            ),
            "conditional_asr": summarize_conditional_asr(
                client_results, cfg.federated.tail_ratio
            ),
            "elapsed_sec": time.perf_counter() - started,
        }
        report["results"][f"{epsilon:g}"] = epsilon_result
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True),
            encoding="utf-8",
        )
        print(
            f"epsilon={epsilon:g} macro_robust="
            f"{epsilon_result['robust']['client_macro']:.4f} "
            f"pooled_asr={epsilon_result['conditional_asr']['pooled']:.4f} "
            f"time={epsilon_result['elapsed_sec']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
