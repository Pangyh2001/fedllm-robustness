from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import ModelConfig


def load_model_and_tokenizer(cfg: ModelConfig, num_labels: int, device: str):
    tokenizer = AutoTokenizer.from_pretrained(cfg.name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Causal sequence-classification heads pool the final token. When an
    # embedding attack passes inputs_embeds, Gemma cannot infer pad positions,
    # so left padding guarantees that the final position is a real token in
    # both clean and adversarial forwards.
    tokenizer.padding_side = "left"
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[cfg.dtype]
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.name_or_path,
        num_labels=num_labels,
        dtype=dtype,
        pad_token_id=tokenizer.pad_token_id,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.to(device)
    model.print_trainable_parameters()
    return model, tokenizer
