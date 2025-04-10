import torch

class AdversarialAttack:
    """Implements FGSM-based adversarial attacks in embedding space"""
    def __init__(self, model, epsilon=0.1):
        self.model = model
        self.epsilon = epsilon
        self.embed = model.get_input_embeddings()
        
    def generate(self, input_ids, attention_mask, labels):
        """Generate adversarial embeddings using FGSM in embedding space"""
        embeddings = self.embed(input_ids).detach().requires_grad_(True)
        
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, device=input_ids.device)
        
        outputs = self.model(
            inputs_embeds=embeddings, 
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        
        loss.backward()
        grad = embeddings.grad.data
        
        # Apply FGSM perturbation
        perturbations = self.epsilon * torch.sign(grad)
        adv_embeddings = embeddings + perturbations
        
        return adv_embeddings.detach()