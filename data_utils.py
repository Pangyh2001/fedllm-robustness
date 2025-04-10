import random
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def load_and_split_dataset(dataset_name, tokenizer, max_samples, max_length, num_clients):
    """Load dataset, split it for federated learning, and create client datasets"""
    if dataset_name == 'imdb':
        texts, labels = load_imdb_dataset(max_samples)
        num_labels = 2
    elif dataset_name == 'agnews':
        texts, labels = load_agnews_dataset(max_samples)
        num_labels = 4
    elif dataset_name == 'bbcnews':
        texts, labels = load_bbcnews_dataset(max_samples)
        num_labels = 5
    elif dataset_name == 'reuters':
        texts, labels = load_reuters_dataset(max_samples)
        num_labels = 8
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.4, stratify=labels, random_state=42
    )
    
    # Create test dataset
    test_dataset = TextDataset(X_test, y_test, tokenizer, max_length)
    
    # Split training data for clients (non-iid split with Dirichlet distribution)
    client_data = split_data_for_clients(X_train, y_train, num_clients, num_labels)
    
    # Create client datasets
    client_datasets = []
    for i in range(num_clients):
        client_X, client_y = client_data[i]
        client_dataset = TextDataset(client_X, client_y, tokenizer, max_length)
        client_datasets.append(client_dataset)
    
    return client_datasets, test_dataset

def load_imdb_dataset(max_samples=None):
    """Load IMDB dataset with optional sample limit"""
    dataset = load_dataset("imdb")
    texts = []
    labels = []
    
    # Get data from training set
    train_texts = [example["text"] for example in dataset["train"]]
    train_labels = [example["label"] for example in dataset["train"]]
    
    # Limit samples if specified
    if max_samples and max_samples < len(train_texts):
        # Ensure class balance
        pos_indices = [i for i, label in enumerate(train_labels) if label == 1]
        neg_indices = [i for i, label in enumerate(train_labels) if label == 0]
        
        samples_per_class = max_samples // 2
        selected_pos = pos_indices[:samples_per_class]
        selected_neg = neg_indices[:samples_per_class]
        
        selected_indices = selected_pos + selected_neg
        texts = [train_texts[i] for i in selected_indices]
        labels = [train_labels[i] for i in selected_indices]
    else:
        texts = train_texts
        labels = train_labels
    
    return texts, labels

def load_agnews_dataset(max_samples=None):
    """Load AG News dataset with optional sample limit"""
    dataset = load_dataset("ag_news")
    texts = []
    labels = []
    
    # Get data from training set
    train_texts = [example["text"] for example in dataset["train"]]
    train_labels = [example["label"] for example in dataset["train"]]
    
    # Limit samples if specified
    if max_samples and max_samples < len(train_texts):
        # Try to balance classes
        class_indices = [[] for _ in range(4)]  # AG News has 4 classes
        for i, label in enumerate(train_labels):
            class_indices[label].append(i)
        
        samples_per_class = max_samples // 4
        selected_indices = []
        for class_idx in class_indices:
            selected_indices.extend(class_idx[:samples_per_class])
        
        texts = [train_texts[i] for i in selected_indices]
        labels = [train_labels[i] for i in selected_indices]
    else:
        texts = train_texts
        labels = train_labels
    
    return texts, labels

def load_bbcnews_dataset(max_samples=None):
    """Load BBC News dataset with optional sample limit"""
    dataset = load_dataset("SetFit/bbc-news")
    
    # 首先正确定义基础数据
    train_texts = [example["text"] for example in dataset["train"]]
    train_labels = [example["label"] for example in dataset["train"]]
    texts, labels = [], []

    # 验证原始数据标签
    assert all(0 <= l < 5 for l in train_labels), "原始数据包含无效标签"
    
    # 创建类别索引（这才是class_indices的正确定义）
    class_indices = [[] for _ in range(5)]  # 5个类别
    for idx, label in enumerate(train_labels):
        class_indices[label].append(idx)  # 收集每个类别的索引

    # 限制样本逻辑
    if max_samples and max_samples < len(train_texts):
        # 保证每个类别至少有1个样本
        samples_per_class = max(max_samples // 5, 1)
        
        selected_indices = []
        for class_list in class_indices:
            if len(class_list) > 0:
                selected = class_list[:samples_per_class]
                selected_indices.extend(selected)
        
        # 如果样本不足，随机补充
        if len(selected_indices) < max_samples:
            remaining = max_samples - len(selected_indices)
            extra_indices = random.sample(
                [i for i in range(len(train_texts)) if i not in selected_indices],
                remaining
            )
            selected_indices.extend(extra_indices)
        
        # 最终获取数据
        texts = [train_texts[i] for i in selected_indices]
        labels = [train_labels[i] for i in selected_indices]
    else:
        texts = train_texts
        labels = train_labels

    # 最终验证
    assert len(texts) == len(labels), "数据与标签长度不一致"
    assert all(0 <= l < 5 for l in labels), f"发现无效标签: {set(l for l in labels if not 0<=l<5)}"
    print("Unique labels:", set(labels))

    return texts, labels

def load_reuters_dataset(max_samples=None):
    """Load Reuters dataset with optional sample limit"""
    dataset = load_dataset("yangwang825/reuters-21578")
    texts = []
    labels = []
    
    # Get data from training set
    train_texts = [example["text"] for example in dataset["train"]]
    train_labels = [example["label"] for example in dataset["train"]]
    
    # Limit samples if specified
    if max_samples and max_samples < len(train_texts):
        # Try to balance classes
        class_indices = [[] for _ in range(8)]  # Reuters has 8 classes
        for i, label in enumerate(train_labels):
            class_indices[label].append(i)
        
        samples_per_class = max_samples // 8
        selected_indices = []
        for class_idx in class_indices:
            selected_indices.extend(class_idx[:samples_per_class])
        
        texts = [train_texts[i] for i in selected_indices]
        labels = [train_labels[i] for i in selected_indices]
    else:
        texts = train_texts
        labels = train_labels
    
    return texts, labels

def split_data_for_clients(X, y, num_clients, num_classes, alpha=0.5):
    """
    Split data for clients using Dirichlet distribution for non-IID setting
    alpha controls the degree of non-IID (lower = more skewed)
    """
    client_data = [[] for _ in range(num_clients)]
    
    # Group data by class
    class_idxs = [[] for _ in range(num_classes)]
    for idx, label in enumerate(y):
        class_idxs[label].append(idx)
    
    # For each class, distribute data to clients according to Dirichlet distribution
    for class_idx, idxs in enumerate(class_idxs):
        # Generate distribution for this class
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Calculate number of samples per client for this class
        client_sample_sizes = (np.array(proportions) * len(idxs)).astype(int)
        client_sample_sizes[-1] = len(idxs) - np.sum(client_sample_sizes[:-1])  # Ensure we use all samples
        
        # Distribute indices
        start_idx = 0
        for client_idx in range(num_clients):
            end_idx = start_idx + client_sample_sizes[client_idx]
            client_data[client_idx].extend(idxs[start_idx:end_idx])
            start_idx = end_idx
    
    # Create final client datasets
    final_client_data = []
    for client_idx in range(num_clients):
        client_indices = client_data[client_idx]
        client_X = [X[i] for i in client_indices]
        client_y = [y[i] for i in client_indices]
        final_client_data.append((client_X, client_y))
    
    return final_client_data