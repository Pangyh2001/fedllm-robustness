from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .aggregation import (
    average_residual,
    fedrda_residual,
    qfedavg_update,
    residual_diagnostics,
    sfat_update,
)
from .config import RunConfig
from .data import dataset_spec, load_federated_data
from .metrics import select_tail_clients, summarize_clients, summarize_conditional_asr
from .modeling import load_model_and_tokenizer
from .state import add, load_trainable_state, scale, trainable_state, weighted_sum
from .training import evaluate_dataset, evaluate_objective_loss, local_update


SUPPORTED_ALGORITHMS = {
    "fedavg",
    "fedpgd",
    "calfat",
    "sfat",
    "qfedavg_eat",
    "fedrda",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=True)
    os.replace(temp, path)


def _append_jsonl(path: Path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=True) + "\n")


class ExperimentRunner:
    def __init__(self, cfg: RunConfig, resume: bool = False):
        if cfg.federated.algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm {cfg.federated.algorithm}; "
                f"choose from {sorted(SUPPORTED_ALGORITHMS)}"
            )
        self.cfg = cfg
        self.device = cfg.device
        self.run_dir = (
            Path(cfg.output_dir)
            / f"{cfg.run_name}__{cfg.federated.algorithm}__seed{cfg.seed}"
        )
        self.resume = resume
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.run_dir / "latest_checkpoint.pt"
        self.metrics_path = self.run_dir / "round_metrics.jsonl"

    def _prepare(self):
        set_seed(self.cfg.seed)
        num_labels = dataset_spec(self.cfg.data.name)["num_labels"]
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.cfg.model, num_labels=num_labels, device=self.device
        )
        self.clients, split_metadata = load_federated_data(
            self.cfg.data, self.tokenizer, self.cfg.seed
        )
        _write_json(self.run_dir / "resolved_config.json", dataclasses.asdict(self.cfg))
        _write_json(self.run_dir / "data_split.json", split_metadata)
        self.global_state = trainable_state(self.model)
        self.residual_ema = {}
        self.vulnerability_ema = {}
        self.start_round = 0
        if self.resume:
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            self.global_state = checkpoint["global_state"]
            self.residual_ema = checkpoint["residual_ema"]
            self.vulnerability_ema = checkpoint["vulnerability_ema"]
            self.start_round = checkpoint["round"]
            load_trainable_state(self.model, self.global_state)
        elif self.metrics_path.exists():
            raise FileExistsError(
                f"{self.metrics_path} exists. Use a new run_name or pass --resume."
            )

    def _save_checkpoint(self, completed_round: int):
        temp = self.checkpoint_path.with_suffix(".tmp")
        torch.save(
            {
                "round": completed_round,
                "global_state": self.global_state,
                "residual_ema": self.residual_ema,
                "vulnerability_ema": self.vulnerability_ema,
            },
            temp,
        )
        os.replace(temp, self.checkpoint_path)

    def _validation_vulnerability(self):
        scores = {}
        raw_metrics = {}
        for client in self.clients:
            metrics = evaluate_dataset(
                self.model,
                client.validation,
                self.device,
                self.cfg.federated.eval_batch_size,
                attack_cfg=self.cfg.attack,
                attack_steps=self.cfg.attack.train_steps,
                restarts=1,
                max_batches=self.cfg.federated.max_eval_batches,
            )
            score = metrics["clean_accuracy"] - metrics["robust_accuracy"]
            previous = self.vulnerability_ema.get(client.client_id)
            ema = (
                score
                if previous is None
                else self.cfg.federated.vulnerability_ema * previous
                + (1.0 - self.cfg.federated.vulnerability_ema) * score
            )
            self.vulnerability_ema[client.client_id] = ema
            scores[client.client_id] = score
            raw_metrics[client.client_id] = metrics
        return scores, raw_metrics

    def _evaluate_test(self, final_evaluation: bool):
        all_results = {}
        steps_list = sorted(set(self.cfg.attack.eval_steps))
        if not final_evaluation:
            steps_list = [min(steps_list)]
        for steps in steps_list:
            clients = []
            for client in self.clients:
                clients.append(
                    {
                        "client_id": client.client_id,
                        **evaluate_dataset(
                            self.model,
                            client.test,
                            self.device,
                            self.cfg.federated.eval_batch_size,
                            attack_cfg=self.cfg.attack,
                            attack_steps=steps,
                            restarts=(
                                self.cfg.attack.eval_restarts
                                if final_evaluation and steps == max(steps_list)
                                else 1
                            ),
                            max_batches=(
                                None
                                if final_evaluation
                                else self.cfg.federated.max_eval_batches
                            ),
                        ),
                    }
                )
            all_results[f"pgd_{steps}"] = {
                "clients": clients,
                "clean": summarize_clients(
                    clients, "clean_accuracy", self.cfg.federated.tail_ratio
                ),
                "robust": summarize_clients(
                    clients, "robust_accuracy", self.cfg.federated.tail_ratio
                ),
                "conditional_asr": summarize_conditional_asr(
                    clients, self.cfg.federated.tail_ratio
                ),
            }
        return all_results

    def _selected_clients(self, round_index: int):
        count = max(
            1,
            math.ceil(len(self.clients) * self.cfg.federated.client_fraction),
        )
        rng = random.Random(self.cfg.seed + round_index)
        return rng.sample(self.clients, count)

    def run(self):
        self._prepare()
        cfg = self.cfg.federated
        for round_index in range(self.start_round, cfg.rounds):
            round_started = time.perf_counter()
            load_trainable_state(self.model, self.global_state)
            if cfg.algorithm == "fedrda":
                raw_vulnerability, validation_metrics = self._validation_vulnerability()
                tail_ids = select_tail_clients(
                    self.vulnerability_ema, self.cfg.federated.tail_ratio
                )
            else:
                raw_vulnerability, validation_metrics, tail_ids = {}, {}, set()
            selected = self._selected_clients(round_index)
            sample_counts = [len(client.train) for client in selected]
            weight_total = sum(sample_counts)
            weights = [count / weight_total for count in sample_counts]
            clean_updates = []
            robust_updates = []
            residuals = []
            single_updates = []
            qfed_losses = []
            local_metrics = []
            for client in selected:
                branch_seed = self.cfg.seed * 100_000 + round_index * 1_000 + client.client_id
                clean_update = None
                robust_update = None
                client_record = {"client_id": client.client_id}
                if cfg.algorithm == "fedrda":
                    clean_update, metrics = local_update(
                        self.model,
                        client.train,
                        self.global_state,
                        "clean",
                        cfg,
                        self.cfg.attack,
                        self.device,
                        branch_seed,
                    )
                    clean_updates.append(clean_update)
                    client_record["clean_branch"] = metrics
                    robust_update, metrics = local_update(
                        self.model,
                        client.train,
                        self.global_state,
                        "eat",
                        cfg,
                        self.cfg.attack,
                        self.device,
                        branch_seed,
                    )
                    robust_updates.append(robust_update)
                    client_record["robust_branch"] = metrics
                else:
                    objective = {
                        "fedavg": "clean",
                        "fedpgd": "pgd",
                        "calfat": "calfat",
                        "sfat": "pgd",
                        "qfedavg_eat": "eat",
                    }[cfg.algorithm]
                    if cfg.algorithm == "qfedavg_eat":
                        qloss = evaluate_objective_loss(
                            self.model,
                            client.train,
                            self.device,
                            cfg.eval_batch_size,
                            self.cfg.attack,
                            cfg.max_eval_batches,
                        )
                        qfed_losses.append(qloss)
                        client_record["broadcast_adversarial_loss"] = qloss
                    class_counts = np.bincount(
                        client.train.labels,
                        minlength=dataset_spec(self.cfg.data.name)["num_labels"],
                    ).tolist()
                    robust_update, metrics = local_update(
                        self.model,
                        client.train,
                        self.global_state,
                        objective,
                        cfg,
                        self.cfg.attack,
                        self.device,
                        branch_seed,
                        class_counts=class_counts,
                    )
                    single_updates.append(robust_update)
                    client_record["local_update"] = metrics
                if cfg.algorithm == "fedrda":
                    residual = {
                        name: robust_update[name] - clean_update[name]
                        for name in clean_update
                    }
                    beta = cfg.residual_ema
                    if client.client_id in self.residual_ema and beta > 0:
                        residual = {
                            name: beta * self.residual_ema[client.client_id][name]
                            + (1 - beta) * value
                            for name, value in residual.items()
                        }
                    self.residual_ema[client.client_id] = residual
                    residuals.append(residual)
                local_metrics.append(client_record)

            aggregation_metrics = {}
            if cfg.algorithm == "sfat":
                losses = [record["local_update"]["loss"] for record in local_metrics]
                global_update, sfat_weights = sfat_update(
                    single_updates,
                    losses,
                    cfg.sfat_top_k,
                    cfg.sfat_multiplier,
                    use_slack=round_index > 0,
                )
                aggregation_metrics.update(
                    {
                        "aggregator": "sfat",
                        "client_losses": losses,
                        "weights": sfat_weights,
                    }
                )
            elif cfg.algorithm == "qfedavg_eat":
                global_update, qfed_metrics = qfedavg_update(
                    single_updates,
                    qfed_losses,
                    cfg.learning_rate,
                    cfg.qfed_q,
                )
                aggregation_metrics.update(
                    {"aggregator": "qfedavg", **qfed_metrics}
                )
            elif cfg.algorithm != "fedrda":
                global_update = weighted_sum(single_updates, weights)
                aggregation_metrics["aggregator"] = "sample_weighted_average"
            else:
                clean_average = weighted_sum(clean_updates, weights)
                diagnostics = residual_diagnostics(residuals, weights)
                aggregation_metrics["residual_diagnostics"] = dataclasses.asdict(diagnostics)
                client_ids = [client.client_id for client in selected]
                if round_index < cfg.warmup_rounds:
                    robust_correction = average_residual(residuals, weights)
                    aggregation_metrics["aggregator"] = "average_residual"
                else:
                    robust_correction, qp_metrics = fedrda_residual(
                        residuals,
                        weights,
                        client_ids,
                        tail_ids,
                        cfg.qp_rho,
                        cfg.qp_kappa,
                    )
                    aggregation_metrics["aggregator"] = "tail_constrained_residual"
                    aggregation_metrics["qp"] = qp_metrics
                global_update = add(
                    clean_average, scale(robust_correction, cfg.residual_weight)
                )
            self.global_state = add(self.global_state, global_update)
            load_trainable_state(self.model, self.global_state)

            test_metrics = None
            if (
                (round_index + 1) % cfg.evaluate_every == 0
                or round_index + 1 == cfg.rounds
            ):
                test_metrics = self._evaluate_test(
                    final_evaluation=(round_index + 1 == cfg.rounds)
                )
            record = {
                "round": round_index + 1,
                "algorithm": cfg.algorithm,
                "selected_client_ids": [client.client_id for client in selected],
                "tail_client_ids": sorted(tail_ids),
                "raw_vulnerability": raw_vulnerability,
                "vulnerability_ema": self.vulnerability_ema,
                "validation": validation_metrics,
                "local": local_metrics,
                "aggregation": aggregation_metrics,
                "test": test_metrics,
                "round_time_sec": time.perf_counter() - round_started,
            }
            _append_jsonl(self.metrics_path, record)
            self._save_checkpoint(round_index + 1)
            print(
                f"round={round_index + 1}/{cfg.rounds} "
                f"algorithm={cfg.algorithm} tail={sorted(tail_ids)} "
                f"time={record['round_time_sec']:.1f}s",
                flush=True,
            )
        final_dir = self.run_dir / "final_model"
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        if self.metrics_path.exists():
            last_record = json.loads(self.metrics_path.read_text(encoding="utf-8").splitlines()[-1])
            _write_json(
                self.run_dir / "summary.json",
                {
                    "run_dir": str(self.run_dir),
                    "algorithm": cfg.algorithm,
                    "seed": self.cfg.seed,
                    "final_test": last_record.get("test"),
                },
            )
        return self.run_dir
