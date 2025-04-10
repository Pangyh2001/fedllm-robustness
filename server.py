import time
import torch
import numpy as np
import os
import copy
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from adversarial import AdversarialAttack

class FederatedServer:
    def __init__(
        self, 
        global_model, 
        clients, 
        test_data, 
        device,
        client_fraction=1.0,
        algorithm='fedavg',
        epsilon=0.05
    ):
        self.global_model = global_model
        self.clients = clients
        self.test_data = test_data
        self.device = device
        self.client_fraction = client_fraction
        self.algorithm = algorithm
        self.epsilon = epsilon
        
    def select_clients(self):
        """Randomly select a fraction of clients"""
        num_clients = max(1, int(self.client_fraction * len(self.clients)))
        return random.sample(self.clients, num_clients)
    
    def aggregate_parameters(self, client_parameters):
        """Aggregate client parameters using FedAvg"""
        # Simple average of client parameters
        aggregated_params = [
            np.mean(
                [client_params[i] for client_params in client_parameters], 
                axis=0
            )
            for i in range(len(client_parameters[0]))
        ]
        
        # Update global model with aggregated parameters
        global_params = self.global_model.state_dict()
        for i, (key, _) in enumerate(global_params.items()):
            global_params[key] = torch.tensor(aggregated_params[i]).to(self.device)
            
        self.global_model.load_state_dict(global_params)
    
    def evaluate(self, adversarial=False):
        """Evaluate the global model on the test data"""
        self.global_model.eval()
        
        test_loader = DataLoader(
            self.test_data, 
            batch_size=8, 
            shuffle=False
        )
        
        correct = 0
        total = 0
        attack = AdversarialAttack(self.global_model, self.epsilon) if adversarial else None
        
        progress_bar = tqdm(test_loader, desc=f"Evaluating {'adversarial' if adversarial else 'normal'}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            if adversarial:
                with torch.enable_grad():
                    adv_embeds = attack.generate(input_ids, attention_mask, labels)
                
                with torch.no_grad():
                    outputs = self.global_model(
                        inputs_embeds=adv_embeds,
                        attention_mask=attention_mask
                    )
            else:
                with torch.no_grad():
                    outputs = self.global_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
            
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({"Accuracy": f"{100 * correct / total:.2f}%"})
        
        accuracy = correct / total
        print(f"Test Accuracy ({'Adversarial' if adversarial else 'Normal'}): {accuracy:.4f}")
        return accuracy
    
    def train(self, num_rounds, output_dir):
        """Train the global model using federated learning"""
        # Initialize arrays to store metrics
        all_losses = []  # [client_id][round][epoch]
        accuracies = []
        adv_accuracies = []
        round_times = []  # 记录每一轮的运行时间
        
        for round_num in range(num_rounds):
            print(f"\n--- Round {round_num+1}/{num_rounds} ---")
            
            # 记录当前轮次开始时间
            round_start_time = time.time()
            
            # Select clients for this round
            selected_clients = self.select_clients()
            print(f"Selected {len(selected_clients)} clients for training")
            
            # Train on selected clients
            client_parameters = []
            client_losses = [[] for _ in range(len(self.clients))]
            
            for client in selected_clients:
                # Update client's local model with global model
                client.update_local_model(self.global_model)
                
                # Perform local training
                losses = client.train()
                client_losses[client.client_id] = losses
                
                # Collect updated parameters
                client_parameters.append(client.get_parameters())
            
            all_losses.append(client_losses)
            
            # Aggregate parameters
            self.aggregate_parameters(client_parameters)
            
            # Evaluate global model
            accuracy = self.evaluate(adversarial=False)
            accuracies.append(accuracy)
            
            # Evaluate with adversarial examples
            adv_accuracy = self.evaluate(adversarial=True)
            adv_accuracies.append(adv_accuracy)

            # 计算本轮的运行时间
            round_end_time = time.time()
            duration = round_end_time - round_start_time
            round_times.append(duration)
            print(f"Round {round_num+1} duration: {duration:.2f} seconds")
        
        # Save results
        self._save_results(output_dir, all_losses, accuracies, adv_accuracies)

            # 将每轮运行时间保存到 time.npy 文件中
        time_file = os.path.join(output_dir, "time.npy")
        np.save(time_file, np.array(round_times))
        print(f"Run times for each round have been saved to {time_file}")
        
    def _save_results(self, output_dir, all_losses, accuracies, adv_accuracies):
        """Save the training results and model"""
        # Save metrics
        np.save(os.path.join(output_dir, "loss.npy"), np.array(all_losses))
        np.save(os.path.join(output_dir, "accuracy.npy"), np.array(accuracies))
        np.save(os.path.join(output_dir, "adv_accuracy.npy"), np.array(adv_accuracies))
        
        # Save model and tokenizer
        self.global_model.save_pretrained(os.path.join(output_dir, "final_model"))
        
        # Save a text summary
        with open(os.path.join(output_dir, "results.txt"), "w") as f:
            f.write(f"Algorithm: {self.algorithm}\n")
            f.write(f"Final accuracy: {accuracies[-1]:.4f}\n")
            f.write(f"Final adversarial accuracy: {adv_accuracies[-1]:.4f}\n")
            
            # Log accuracy progression
            f.write("\nAccuracy progression:\n")
            for i, acc in enumerate(accuracies):
                f.write(f"Round {i+1}: {acc:.4f}\n")
                
            f.write("\nAdversarial accuracy progression:\n")
            for i, acc in enumerate(adv_accuracies):
                f.write(f"Round {i+1}: {acc:.4f}\n")