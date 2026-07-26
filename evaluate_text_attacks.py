from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fedrda_experiments.config import load_config
from fedrda_experiments.data import dataset_spec, load_federated_data
from fedrda_experiments.metrics import summarize_clients


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["bert_attack", "deepwordbug"],
        choices=["bert_attack", "deepwordbug"],
    )
    parser.add_argument("--num-examples-per-client", type=int, default=-1)
    parser.add_argument("--query-budget", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_trained_model(cfg, run_dir: Path):
    num_labels = dataset_spec(cfg.data.name)["num_labels"]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[cfg.model.dtype]
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name_or_path,
        num_labels=num_labels,
        dtype=dtype,
        pad_token_id=tokenizer.pad_token_id,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, run_dir / "final_model")
    model.cuda().eval()
    return model, tokenizer


def main():
    args = parse_args()
    try:
        import textattack
        from textattack.attack_recipes import BERTAttackLi2020, DeepWordBugGao2018
        from textattack.attack_results import SuccessfulAttackResult
        from textattack.datasets import Dataset
        from textattack.models.wrappers import HuggingFaceModelWrapper
    except ImportError as error:
        raise SystemExit(
            "TextAttack is optional. Install it with: pip install textattack"
        ) from error

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir)
    model, tokenizer = load_trained_model(cfg, run_dir)
    clients, _ = load_federated_data(cfg.data, tokenizer, cfg.seed)
    wrapper = HuggingFaceModelWrapper(model, tokenizer)
    recipes = {
        "bert_attack": BERTAttackLi2020,
        "deepwordbug": DeepWordBugGao2018,
    }
    output = {}
    for attack_name in args.attacks:
        attack = recipes[attack_name].build(wrapper)
        client_results = []
        for client in clients:
            pairs = list(zip(client.test.texts, client.test.labels))
            if args.num_examples_per_client > 0:
                rng = np.random.default_rng(args.seed + client.client_id)
                chosen = rng.choice(
                    len(pairs),
                    size=min(args.num_examples_per_client, len(pairs)),
                    replace=False,
                )
                pairs = [pairs[int(index)] for index in chosen]
            dataset = Dataset(pairs)
            attack_args = textattack.AttackArgs(
                num_examples=len(pairs),
                query_budget=args.query_budget,
                random_seed=args.seed + client.client_id,
                disable_stdout=True,
                silent=True,
            )
            results = list(textattack.Attacker(attack, dataset, attack_args).attack_dataset())
            successful = sum(
                isinstance(result, SuccessfulAttackResult) for result in results
            )
            skipped = sum(result.__class__.__name__ == "SkippedAttackResult" for result in results)
            initially_correct = len(results) - skipped
            robust_correct = initially_correct - successful
            client_results.append(
                {
                    "client_id": client.client_id,
                    "num_examples": len(results),
                    "initially_correct": initially_correct,
                    "successful_attacks": successful,
                    "robust_accuracy": robust_correct / max(len(results), 1),
                    "conditional_asr": successful / max(initially_correct, 1),
                }
            )
        output[attack_name] = {
            "clients": client_results,
            "robust": summarize_clients(
                client_results, "robust_accuracy", cfg.federated.tail_ratio
            ),
            "conditional_asr": {
                "pooled": sum(item["successful_attacks"] for item in client_results)
                / max(sum(item["initially_correct"] for item in client_results), 1),
                "client_macro": float(
                    np.mean([item["conditional_asr"] for item in client_results])
                ),
            },
        }
    path = run_dir / "text_attack_metrics.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
