import matplotlib.pyplot as plt
import numpy as np
import os

def plot_metrics(output_dir, loss, accuracy, adv_accuracy):
    """Plot and save metrics visualizations"""
    # Create figures directory
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # Process client losses
    avg_losses = []
    for round_losses in loss:
        # Average across clients and epochs
        round_avg = np.mean([np.mean(client_loss) if client_loss else 0 for client_loss in round_losses])
        avg_losses.append(round_avg)
    
    # Plot average loss per round
    plt.figure(figsize=(10, 6))
    plt.plot(avg_losses, 'b-', marker='o')
    plt.title('Average Loss per Round')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'loss.png'))
    plt.close()
    
    # Plot accuracy metrics
    plt.figure(figsize=(10, 6))
    plt.plot(accuracy, 'g-', marker='o', label='Normal Accuracy')
    plt.plot(adv_accuracy, 'r-', marker='x', label='Adversarial Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'accuracy.png'))
    plt.close()

def save_hyperparameters(output_dir, args):
    """Save hyperparameters to a file"""
    with open(os.path.join(output_dir, "hyperparameters.txt"), "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")