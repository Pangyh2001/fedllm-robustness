from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from .config import DataConfig


DATASETS = {
    "agnews": {
        "hub_name": "ag_news",
        "local_name": "agnews",
        "text_field": "text",
        "num_labels": 4,
    },
    "dbpedia14": {
        "hub_name": "dbpedia_14",
        "local_name": "dbpedia14",
        "text_field": "content",
        "num_labels": 14,
    },
}


def dataset_spec(name: str):
    if name not in DATASETS:
        raise ValueError(f"Unsupported dataset {name}; choose from {sorted(DATASETS)}")
    return DATASETS[name]


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int, example_ids=None):
        self.texts = list(texts)
        self.labels = [int(label) for label in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.example_ids = list(example_ids) if example_ids is not None else list(range(len(self.texts)))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
            "example_id": torch.tensor(self.example_ids[index], dtype=torch.long),
        }


@dataclass
class ClientData:
    client_id: int
    train: TextDataset
    validation: TextDataset
    test: TextDataset


def _balanced_subset(texts, labels, limit: int | None, rng: np.random.Generator):
    labels = np.asarray(labels, dtype=np.int64)
    if limit is None or limit >= len(labels):
        indices = np.arange(len(labels))
    else:
        classes = np.unique(labels)
        per_class = limit // len(classes)
        chosen = []
        for label in classes:
            label_indices = np.flatnonzero(labels == label)
            rng.shuffle(label_indices)
            chosen.extend(label_indices[:per_class])
        remaining = limit - len(chosen)
        if remaining:
            pool = np.setdiff1d(np.arange(len(labels)), np.asarray(chosen), assume_unique=False)
            chosen.extend(rng.choice(pool, size=remaining, replace=False))
        indices = np.asarray(chosen)
        rng.shuffle(indices)
    return [texts[i] for i in indices], labels[indices].tolist()


def shared_dirichlet_partitions(
    train_labels,
    test_labels,
    num_clients: int,
    alpha: float,
    min_samples: int,
    rng: np.random.Generator,
    max_attempts: int = 500,
):
    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    classes = np.unique(np.concatenate([train_labels, test_labels]))
    for _ in range(max_attempts):
        train_parts = [[] for _ in range(num_clients)]
        test_parts = [[] for _ in range(num_clients)]
        for label in classes:
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            for labels, parts in ((train_labels, train_parts), (test_labels, test_parts)):
                indices = np.flatnonzero(labels == label)
                rng.shuffle(indices)
                counts = rng.multinomial(len(indices), proportions)
                cursor = 0
                for client_id, count in enumerate(counts):
                    parts[client_id].extend(indices[cursor : cursor + count].tolist())
                    cursor += count
        if min(map(len, train_parts)) >= min_samples and min(map(len, test_parts)) >= 1:
            for parts in (train_parts, test_parts):
                for indices in parts:
                    rng.shuffle(indices)
            return train_parts, test_parts
    raise RuntimeError(
        f"Could not create non-empty Dirichlet split after {max_attempts} attempts; "
        "increase samples/min_client_samples or alpha"
    )


def _balanced_transport(
    labels: np.ndarray,
    client_profiles: np.ndarray,
    rng: np.random.Generator,
):
    """Allocate every example while keeping client sizes within one sample.

    Iterative proportional fitting turns the sampled client label profiles into
    a matrix whose row sums are equal client quotas and whose column sums match
    the dataset class counts. Largest-remainder rounding then produces integer
    allocations with the same margins.
    """
    labels = np.asarray(labels, dtype=np.int64)
    num_clients, num_classes = client_profiles.shape
    class_counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    client_sizes = np.full(num_clients, len(labels) // num_clients, dtype=np.int64)
    client_sizes[: len(labels) % num_clients] += 1
    matrix = np.maximum(client_profiles, 1e-12)
    for _ in range(1000):
        matrix *= (client_sizes / matrix.sum(axis=1))[:, None]
        matrix *= (class_counts / matrix.sum(axis=0))[None, :]
    allocation = np.floor(matrix).astype(np.int64)
    row_left = client_sizes - allocation.sum(axis=1)
    col_left = class_counts.astype(np.int64) - allocation.sum(axis=0)
    fractional = matrix - allocation
    while row_left.sum() > 0:
        candidates = np.argwhere(
            (row_left[:, None] > 0) & (col_left[None, :] > 0)
        )
        scores = fractional[candidates[:, 0], candidates[:, 1]]
        best = candidates[int(np.argmax(scores))]
        client_id, label = int(best[0]), int(best[1])
        allocation[client_id, label] += 1
        row_left[client_id] -= 1
        col_left[label] -= 1
        fractional[client_id, label] = -1.0
    parts = [[] for _ in range(num_clients)]
    for label in range(num_classes):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        cursor = 0
        for client_id in range(num_clients):
            count = int(allocation[client_id, label])
            parts[client_id].extend(indices[cursor : cursor + count].tolist())
            cursor += count
    for part in parts:
        rng.shuffle(part)
    return parts


def balanced_label_skew_partitions(
    train_labels,
    test_labels,
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
):
    num_classes = int(max(max(train_labels), max(test_labels))) + 1
    profiles = rng.dirichlet(np.full(num_classes, alpha), size=num_clients)
    return (
        _balanced_transport(np.asarray(train_labels), profiles, rng),
        _balanced_transport(np.asarray(test_labels), profiles, rng),
    )


def _make_dataset(texts, labels, indices, tokenizer, max_length, id_offset=0):
    return TextDataset(
        [texts[i] for i in indices],
        [labels[i] for i in indices],
        tokenizer,
        max_length,
        [id_offset + int(i) for i in indices],
    )


def load_federated_data(cfg: DataConfig, tokenizer, seed: int):
    spec = dataset_spec(cfg.name)
    repository_root = Path(__file__).resolve().parents[1]
    local_train = repository_root / "dataset" / spec["local_name"] / "train.parquet"
    local_test = repository_root / "dataset" / spec["local_name"] / "test.parquet"
    if local_train.exists() and local_test.exists():
        raw = load_dataset(
            "parquet",
            data_files={"train": str(local_train), "test": str(local_test)},
            cache_dir=cfg.cache_dir,
        )
    else:
        raw = load_dataset(spec["hub_name"], cache_dir=cfg.cache_dir)
    rng = np.random.default_rng(seed)
    train_texts, train_labels = _balanced_subset(
        raw["train"][spec["text_field"]],
        raw["train"]["label"],
        cfg.max_train_samples,
        rng,
    )
    test_texts, test_labels = _balanced_subset(
        raw["test"][spec["text_field"]],
        raw["test"]["label"],
        cfg.max_test_samples,
        rng,
    )
    if cfg.partition_mode == "label_skew_equal":
        train_parts, test_parts = balanced_label_skew_partitions(
            train_labels,
            test_labels,
            cfg.num_clients,
            cfg.dirichlet_alpha,
            rng,
        )
    elif cfg.partition_mode == "dirichlet_quantity_skew":
        train_parts, test_parts = shared_dirichlet_partitions(
            train_labels,
            test_labels,
            cfg.num_clients,
            cfg.dirichlet_alpha,
            cfg.min_client_samples,
            rng,
        )
    else:
        raise ValueError(
            "partition_mode must be label_skew_equal or dirichlet_quantity_skew"
        )
    if min(map(len, train_parts)) < cfg.min_client_samples:
        raise ValueError("At least one client has too few training examples")
    if min(map(len, test_parts)) < cfg.min_client_test_samples:
        raise ValueError(
            "At least one client has too few test examples; use equal-size partitioning, "
            "fewer clients, or the full test set"
        )

    clients = []
    for client_id, train_indices in enumerate(train_parts):
        val_size = max(1, int(round(len(train_indices) * cfg.val_fraction)))
        if len(train_indices) - val_size < 1:
            val_size = 1
        validation_indices = train_indices[:val_size]
        local_train_indices = train_indices[val_size:]
        clients.append(
            ClientData(
                client_id=client_id,
                train=_make_dataset(
                    train_texts, train_labels, local_train_indices, tokenizer, cfg.max_length
                ),
                validation=_make_dataset(
                    train_texts, train_labels, validation_indices, tokenizer, cfg.max_length
                ),
                test=_make_dataset(
                    test_texts,
                    test_labels,
                    test_parts[client_id],
                    tokenizer,
                    cfg.max_length,
                    id_offset=10_000_000,
                ),
            )
        )
    metadata = {
        "dataset": cfg.name,
        "partition_mode": cfg.partition_mode,
        "num_labels": spec["num_labels"],
        "client_sizes": [
            {
                "client_id": client.client_id,
                "train": len(client.train),
                "validation": len(client.validation),
                "test": len(client.test),
                "train_label_histogram": np.bincount(
                    client.train.labels, minlength=spec["num_labels"]
                ).tolist(),
                "test_label_histogram": np.bincount(
                    client.test.labels, minlength=spec["num_labels"]
                ).tolist(),
            }
            for client in clients
        ],
    }
    return clients, metadata
