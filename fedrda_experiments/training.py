from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .attacks import EmbeddingPGD
from .config import AttackConfig, FederatedConfig
from .state import TensorState, load_trainable_state, subtract, trainable_state


def _batch_to_device(batch, device):
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "labels"}
    }


def local_update(
    model,
    dataset,
    initial_state: TensorState,
    objective: str,
    fed_cfg: FederatedConfig,
    attack_cfg: AttackConfig,
    device: str,
    seed: int,
    class_counts: list[int] | None = None,
):
    if objective not in {"clean", "pgd", "eat", "calfat"}:
        raise ValueError(f"Unsupported local objective: {objective}")
    load_trainable_state(model, initial_state)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=fed_cfg.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=fed_cfg.learning_rate,
    )
    attack = None
    if objective != "clean":
        attack = EmbeddingPGD(
            model,
            epsilon=attack_cfg.epsilon,
            steps=attack_cfg.train_steps,
            epsilon_mode=attack_cfg.epsilon_mode,
            step_size=attack_cfg.step_size,
            random_start=attack_cfg.random_start,
        )
    losses = []
    started = time.perf_counter()
    model.train()
    for _ in range(fed_cfg.local_epochs):
        for raw_batch in loader:
            batch = _batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            if objective == "clean":
                outputs = model(**batch)
                loss = outputs.loss
            else:
                logit_bias = None
                loss_type = "ce"
                if objective == "calfat":
                    if class_counts is None:
                        raise ValueError("CalFAT requires client class counts")
                    counts = torch.tensor(
                        class_counts,
                        device=batch["labels"].device,
                        dtype=torch.float32,
                    )
                    logit_bias = fed_cfg.calfat_tau * torch.log(counts + 1e-7)
                    loss_type = "kl"
                adversarial_embeddings = attack.generate(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["labels"],
                    loss_type=loss_type,
                    logit_bias=logit_bias,
                )
                model.train()
                robust_outputs = model(
                    inputs_embeds=adversarial_embeddings,
                    attention_mask=batch["attention_mask"],
                )
                robust_logits = robust_outputs.logits.float()
                if objective == "pgd":
                    loss = F.cross_entropy(robust_logits, batch["labels"])
                elif objective == "calfat":
                    loss = F.cross_entropy(
                        robust_logits + logit_bias, batch["labels"]
                    )
                else:
                    clean_outputs = model(**batch)
                    robust_loss = F.cross_entropy(robust_logits, batch["labels"])
                    consistency = F.kl_div(
                        F.log_softmax(robust_logits, dim=-1),
                        F.softmax(clean_outputs.logits.float().detach(), dim=-1),
                        reduction="batchmean",
                    )
                    loss = (
                        (1.0 - fed_cfg.adv_weight) * clean_outputs.loss
                        + fed_cfg.adv_weight * robust_loss
                        + fed_cfg.clean_consistency_weight * consistency
                    )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            losses.append(float(loss.detach()))
    final_state = trainable_state(model)
    update = subtract(final_state, initial_state)
    elapsed = time.perf_counter() - started
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0
    )
    del optimizer
    return update, {
        "objective": objective,
        "loss": float(sum(losses) / max(len(losses), 1)),
        "num_batches": len(loader) * fed_cfg.local_epochs,
        "elapsed_sec": elapsed,
        "peak_gpu_memory_mb": peak_memory,
    }


def evaluate_objective_loss(
    model,
    dataset,
    device: str,
    batch_size: int,
    attack_cfg: AttackConfig,
    max_batches: int | None = None,
):
    """Evaluate adversarial loss at the broadcast global state for q-FedAvg."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    attack = EmbeddingPGD(
        model,
        epsilon=attack_cfg.epsilon,
        steps=attack_cfg.train_steps,
        epsilon_mode=attack_cfg.epsilon_mode,
        step_size=attack_cfg.step_size,
        random_start=attack_cfg.random_start,
    )
    losses = []
    model.eval()
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _batch_to_device(raw_batch, device)
        with torch.enable_grad():
            embeddings = attack.generate(
                batch["input_ids"], batch["attention_mask"], batch["labels"]
            )
        with torch.no_grad():
            logits = model(
                inputs_embeds=embeddings,
                attention_mask=batch["attention_mask"],
            ).logits.float()
            losses.extend(
                F.cross_entropy(logits, batch["labels"], reduction="none")
                .cpu()
                .tolist()
            )
    if not losses:
        raise RuntimeError("Objective loss evaluation produced zero examples")
    return float(sum(losses) / len(losses))


def evaluate_dataset(
    model,
    dataset,
    device: str,
    batch_size: int,
    attack_cfg: AttackConfig | None = None,
    attack_steps: int | None = None,
    restarts: int = 1,
    max_batches: int | None = None,
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    clean_correct = 0
    robust_correct = 0
    attacked_clean_correct = 0
    total = 0
    attack = None
    if attack_steps is not None:
        if attack_cfg is None:
            raise ValueError("attack_cfg is required for adversarial evaluation")
        attack = EmbeddingPGD(
            model,
            epsilon=attack_cfg.epsilon,
            steps=attack_steps,
            epsilon_mode=attack_cfg.epsilon_mode,
            step_size=attack_cfg.step_size,
            random_start=attack_cfg.random_start,
        )
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _batch_to_device(raw_batch, device)
        with torch.no_grad():
            clean_logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits.float()
        clean_predictions = clean_logits.argmax(dim=-1)
        clean_mask = clean_predictions.eq(batch["labels"])
        clean_correct += int(clean_mask.sum())
        total += batch["labels"].numel()
        if attack is None:
            continue
        worst_loss = torch.full(
            (batch["labels"].size(0),), -torch.inf, device=batch["labels"].device
        )
        worst_predictions = clean_predictions
        for _ in range(restarts):
            with torch.enable_grad():
                adversarial_embeddings = attack.generate(
                    batch["input_ids"], batch["attention_mask"], batch["labels"]
                )
            with torch.no_grad():
                logits = model(
                    inputs_embeds=adversarial_embeddings,
                    attention_mask=batch["attention_mask"],
                ).logits.float()
                losses = F.cross_entropy(logits, batch["labels"], reduction="none")
            replace = losses > worst_loss
            worst_loss = torch.where(replace, losses, worst_loss)
            worst_predictions = torch.where(replace, logits.argmax(dim=-1), worst_predictions)
        robust_correct += int(worst_predictions.eq(batch["labels"]).sum())
        attacked_clean_correct += int(
            (worst_predictions.eq(batch["labels"]) & clean_mask).sum()
        )
    if total == 0:
        raise RuntimeError("Evaluation dataset produced zero examples")
    result = {
        "num_examples": total,
        "clean_correct": clean_correct,
        "clean_accuracy": clean_correct / total,
    }
    if attack is not None:
        result.update(
            {
                "robust_accuracy": robust_correct / total,
                "robust_correct": robust_correct,
                "attacked_clean_correct": attacked_clean_correct,
                "conditional_asr": (
                    1.0 - attacked_clean_correct / clean_correct
                    if clean_correct > 0
                    else float("nan")
                ),
                "robustness_gap": (clean_correct - robust_correct) / total,
            }
        )
    return result
