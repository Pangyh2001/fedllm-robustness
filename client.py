import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import copy
from adversarial import AdversarialAttack
from tqdm import tqdm

class FederatedClient:
    def __init__(
        self, 
        client_id, 
        train_data, 
        model, 
        tokenizer, 
        device, 
        local_epochs=1, 
        batch_size=4, 
        lr=2e-4,
        algorithm='fedavg',
        epsilon=0.05,
        confidence_threshold=0.9,
        batch_threshold=0.7,
        adv_weight=0.3
    ):
        self.client_id = client_id
        self.train_data = train_data
        self.device = device
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.confidence_threshold = confidence_threshold
        self.batch_threshold = batch_threshold
        self.adv_weight = adv_weight
        
        # Create a copy of the model for local training
        self.model = None
        self.base_model = model
        self.tokenizer = tokenizer
        
    def update_local_model(self, global_model):
        """Update local model with global model parameters"""
        self.model = copy.deepcopy(global_model)
        
    def train(self):
        """Train the local model"""
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        
        train_loader = DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        epoch_losses = []
        
        if self.algorithm == 'fedavg':
            # Standard training
            for epoch in range(self.local_epochs):
                total_loss = 0
                progress_bar = tqdm(train_loader, desc=f"Client {self.client_id} - Epoch {epoch+1}/{self.local_epochs}")
                for batch in progress_bar:
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                    labels = batch["labels"].to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(**inputs, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
                epoch_loss = total_loss / len(train_loader)
                epoch_losses.append(epoch_loss)
                print(f"Client {self.client_id} - Epoch {epoch+1}/{self.local_epochs} | Avg Loss: {epoch_loss:.4f}")
        
        elif self.algorithm in ['cat', 'cat2']:
            # Adversarial training (CAT or CAT2)
            attack = AdversarialAttack(self.model, self.epsilon)
            
            for epoch in range(self.local_epochs):
                total_loss = 0
                progress_bar = tqdm(train_loader, desc=f"Client {self.client_id} - Epoch {epoch+1}/{self.local_epochs}")
                for batch in progress_bar:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # For CAT2, decide whether to use adversarial training based on confidence
                    use_adversarial = True
                    if self.algorithm == 'cat2':
                        # Get model predictions and confidence
                        with torch.no_grad():
                            outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask
                            )
                            probs = torch.softmax(outputs.logits, dim=1)
                            confidences, _ = torch.max(probs, dim=1)
                            
                            # Check if confidence meets threshold
                            high_conf_ratio = (confidences >= self.confidence_threshold).float().mean().item()
                            use_adversarial = high_conf_ratio < self.batch_threshold
                    
                    if use_adversarial:
                        # Generate adversarial embeddings
                        adv_embeddings = attack.generate(input_ids, attention_mask, labels)
                        
                        # Compute normal loss
                        normal_outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        normal_loss = normal_outputs.loss
                        
                        # Compute adversarial loss
                        adv_outputs = self.model(
                            inputs_embeds=adv_embeddings,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        adv_loss = adv_outputs.loss
                        
                        # Combined loss (weighted)
                        loss = (1-self.adv_weight) * normal_loss + self.adv_weight * adv_loss
                    else:
                        # Use normal training if confidence is high enough
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
                epoch_loss = total_loss / len(train_loader)
                epoch_losses.append(epoch_loss)
                print(f"Client {self.client_id} - Epoch {epoch+1}/{self.local_epochs} | Avg Loss: {epoch_loss:.4f}")
        
        return epoch_losses
    
    def get_parameters(self):
        """Return model parameters as a list of NumPy arrays"""
        return [val.cpu().detach().to(torch.float).numpy() for _, val in self.model.state_dict().items()]