import torch
import matplotlib.pyplot as plt
import seaborn as sns


def plot_latent_space_simplex(model, loader, device, epoch):
    """
    Plots the latent space projected onto 2D. 
    Since K=3, the simplex is a triangle in 3D space. 
    We can plot just the first 2 components (z0, z1) which creates a right triangle.
    
    Args:
        model: The trained ccVAE model
        loader: DataLoader for the dataset
        device: torch device (cpu or cuda)
        epoch: Current epoch number
        digits: List of digit indices used (e.g., [1, 4, 5])
    """
    model.eval()
    z_points = []
    labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # Get posterior parameters lambda, then sample points
            lam = model.encoder(x)
            z = model.sample_batched(lam)
            z_points.append(z.cpu())
            labels.append(y.cpu())
            if len(z_points) * x.size(0) > 500: 
                break  # Only plot 500 points
            
    z_all = torch.cat(z_points, dim=0).numpy()
    labels_all = torch.cat(labels, dim=0).numpy()
    
    plt.figure(figsize=(6, 6))
    
    # Get unique digits in sorted order
    unique_digits = sorted(set(labels_all))
    
    # Scatter plot with per-digit coloring
    for digit in unique_digits:
        mask = [label == digit for label in labels_all]
        z_digit = z_all[mask]
        plt.scatter(z_digit[:, 0], z_digit[:, 1], label=f"Digit {digit}", alpha=0.7, s=30)
    
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title(f"Latent Space (K=3) - Epoch {epoch}")
    plt.legend(title="Digits", loc='best')
    plt.grid(True)
    
    # Draw the boundaries of the simplex (z0 + z1 <= 1, z0>=0, z1>=0)
    plt.plot([0, 1], [1, 0], 'k--', alpha=0.5, label='Simplex boundary')
    plt.plot([0, 0], [0, 1], 'k--', alpha=0.5)
    plt.plot([0, 1], [0, 0], 'k--', alpha=0.5)
    
    plt.savefig(f"results/latent_epoch_{epoch}.png")
    plt.close()


    