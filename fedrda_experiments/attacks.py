from __future__ import annotations

import torch
import torch.nn.functional as F


class EmbeddingPGD:
    def __init__(
        self,
        model,
        epsilon: float,
        steps: int,
        epsilon_mode: str = "relative_rms",
        step_size: float | None = None,
        random_start: bool = True,
    ):
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if epsilon_mode not in {"absolute", "relative_rms"}:
            raise ValueError("epsilon_mode must be absolute or relative_rms")
        self.model = model
        self.epsilon = epsilon
        self.steps = steps
        self.epsilon_mode = epsilon_mode
        self.step_size = step_size
        self.random_start = random_start

    def _budget(self, clean_embeddings, attention_mask):
        if self.epsilon_mode == "absolute":
            return torch.full(
                (clean_embeddings.size(0), 1, 1),
                self.epsilon,
                device=clean_embeddings.device,
                dtype=clean_embeddings.dtype,
            )
        valid = attention_mask.unsqueeze(-1).to(clean_embeddings.dtype)
        denom = valid.sum(dim=(1, 2), keepdim=True).clamp_min(1) * clean_embeddings.size(-1)
        rms = ((clean_embeddings.float().pow(2) * valid).sum(dim=(1, 2), keepdim=True) / denom).sqrt()
        return (self.epsilon * rms).to(clean_embeddings.dtype)

    def generate(
        self,
        input_ids,
        attention_mask,
        labels,
        loss_type: str = "ce",
        logit_bias: torch.Tensor | None = None,
    ):
        if loss_type not in {"ce", "kl"}:
            raise ValueError("loss_type must be ce or kl")
        embed_layer = self.model.get_input_embeddings()
        clean = embed_layer(input_ids).detach()
        budget = self._budget(clean, attention_mask)
        mask = attention_mask.unsqueeze(-1).to(clean.dtype)
        if self.random_start and self.steps > 1:
            delta = torch.empty_like(clean).uniform_(-1, 1) * budget * mask
        else:
            delta = torch.zeros_like(clean)
        alpha = self.step_size
        if alpha is None:
            alpha = 1.25 if self.steps == 1 else 2.0 / self.steps

        was_training = self.model.training
        self.model.eval()
        clean_probabilities = None
        if loss_type == "kl":
            with torch.no_grad():
                clean_logits = self.model(
                    inputs_embeds=clean,
                    attention_mask=attention_mask,
                ).logits.float()
                if logit_bias is not None:
                    clean_logits = clean_logits + logit_bias
                clean_probabilities = F.softmax(clean_logits, dim=-1)
        for _ in range(self.steps):
            delta.requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            outputs = self.model(
                inputs_embeds=clean + delta,
                attention_mask=attention_mask,
            )
            logits = outputs.logits.float()
            if logit_bias is not None:
                logits = logits + logit_bias
            if loss_type == "ce":
                loss = F.cross_entropy(logits, labels)
            else:
                loss = F.kl_div(
                    F.log_softmax(logits, dim=-1),
                    clean_probabilities,
                    reduction="batchmean",
                )
            grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
            delta = (delta.detach() + alpha * budget * grad.sign()).clamp(-budget, budget)
            delta = delta * mask
        self.model.zero_grad(set_to_none=True)
        self.model.train(was_training)
        return (clean + delta).detach()
