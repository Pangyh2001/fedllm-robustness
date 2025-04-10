import os
import argparse
import torch
import numpy as np
from datetime import datetime
from server import FederatedServer
from client import FederatedClient
from data_utils import load_and_split_dataset
from models import load_model
import random
import json


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser(description='Federated Learning with LLMs')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='imdb', choices=['imdb', 'agnews','bbcnews','reuters'], 
                        help='Dataset to use')
    parser.add_argument('--max_samples', type=int, default=2000, 
                        help='Maximum number of samples to use per dataset')
    parser.add_argument('--max_length', type=int, default=256, 
                        help='Maximum sequence length')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='google/gemma-1.1-2b-it', 
                        help='Model name or path')
    
    # Federated learning parameters
    parser.add_argument('--num_clients', type=int, default=5, 
                        help='Number of clients')
    parser.add_argument('--client_fraction', type=float, default=1.0, 
                        help='Fraction of clients to use in each round')
    parser.add_argument('--num_rounds', type=int, default=10, 
                        help='Number of federated rounds')
    parser.add_argument('--local_epochs', type=int, default=1, 
                        help='Number of local epochs per round')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=2e-4, 
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    
    # Algorithm parameters
    parser.add_argument('--algorithm', type=str, default='fedavg', 
                        choices=['fedavg', 'cat', 'cat2'], 
                        help='Federated learning algorithm')
    parser.add_argument('--epsilon', type=float, default=0.05, 
                        help='Epsilon for adversarial training')
    
    # CAT2 parameters
    parser.add_argument('--adv_weight', type=float, default=0.3,
                        help='Weight for adversarial loss in CAT2')
    parser.add_argument('--confidence_threshold', type=float, default=0.9, 
                        help='Confidence threshold for CAT2 algorithm')
    parser.add_argument('--batch_threshold', type=float, default=0.9,  # 设置这个的时候，注意batch_size，根据batch_size来设置这个参数
                        help='Batch threshold for CAT2 algorithm')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output', 
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Set random seed
    setup_seed(args.seed)
    
    # TODO 这里设置LLM的类别
    dataset_num_labels = {
        'imdb': 2,
        'agnews': 4,
        'bbcnews': 5,
        'reuters': 8,
        # 可以继续扩展
    }
    
    num_labels = dataset_num_labels.get(args.dataset)


    # Load model and tokenizer
    model, tokenizer, device = load_model(args.model_name, num_labels)
    
    # Prepare dataset and split for clients
    train_data, test_data = load_and_split_dataset(
        args.dataset, 
        tokenizer, 
        args.max_samples,
        args.max_length,
        args.num_clients
    )
    
    # Initialize clients
    clients = []
    for i in range(args.num_clients):
        clients.append(
            FederatedClient(
                client_id=i,
                train_data=train_data[i],
                model=model,
                tokenizer=tokenizer,
                device=device,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                algorithm=args.algorithm,
                epsilon=args.epsilon,
                confidence_threshold=args.confidence_threshold,
                batch_threshold=args.batch_threshold,
                adv_weight=args.adv_weight
            )
        )
    
    # Initialize server
    server = FederatedServer(
        global_model=model,
        clients=clients,
        test_data=test_data,
        device=device,
        client_fraction=args.client_fraction,
        algorithm=args.algorithm,
        epsilon=args.epsilon
    )
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        args.output_dir, 
        f"{args.model_name}_{args.dataset}_{args.num_clients}clients_{args.algorithm}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Save hyperparameters
    args_dict = vars(args)
    # 定义将要保存的 JSON 文件路径
    json_path = os.path.join(output_dir, "args.json")

    # 写入 JSON 文件，indent 参数用来格式化输出，使其更容易阅读
    with open(json_path, "w") as f:
        json.dump(args_dict, f, indent=4)


    # Start federated learning
    server.train(args.num_rounds, output_dir)
    
    print(f"Federated learning completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()