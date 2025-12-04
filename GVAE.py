"""
Implementation of a Gaussian Variational AutoEncoder for a MNIST dataset
Encoder:
    Linear(784 - 500)
    BatchNorm()
    ReLU()
    Linear(500 - 10)

The Latent Variable Z will have d=10 dimensions

Decoder:
    Linear(10 - 500)
    BatchNorm()
    ReLU()
    Linear(500 - 784)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import math
import matplotlib
matplotlib.use("Agg") 
from latent_visualization import latent_visualization

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(44)

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dims, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims, latent_dim)

    def forward(self, x):
        h = self.net(x)

        # Bottleneck
        z_mu = self.fc_mu(h)
        z_logvar = self.fc_logvar(h)
        
        return z_mu, z_logvar
    
class BernoulliDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims), 
            nn.BatchNorm1d(hidden_dims), 
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dim)
        )

    def forward(self, z):
        logits = self.net(z)

        # return logits (use BCEWithLogitsLoss)
        return logits
    
class GVAE(nn.Module):
    def __init__(self, input_dim, enc_hidden_dims, dec_hidden_dims, latent_dim):
        """
        input_dim: flattened input size (e.g. 28*28)
        enc_hidden_dims:  encoder hidden sizes
        dec_hidden_dims:  decoder hidden sizes
        latent_dim: K
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = MLPEncoder(input_dim, enc_hidden_dims, latent_dim)
        self.decoder = BernoulliDecoder(latent_dim, dec_hidden_dims, input_dim)

    def sampling(self, z_mu, z_logvar):
        """
        reparameterization trick to sample
        """
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        z = z_mu + eps * std
        
        return z
    
    def forward(self, x):
        """
        x: flattened input (batch, input_dim) with values in [0,1] for Bernoulli decoding
        returns: reconstruction logits, z_mu, z_logvar, z
        """

        # Encoder step to get the posterior parameters
        z_mu, z_logvar = self.encoder(x)  # (batch, K)

        # Sample 
        z = self.sampling(z_mu, z_logvar)  # z in simplex

        # Decoder step
        logits = self.decoder(z)  # (batch, input_dim)
        
        return logits, z_mu, z_logvar, z


def plot_latent_space_simplex(model, loader, device, epoch):
    """
    Plots the latent space projected onto 2D. 
    Since K=3, the simplex is a triangle in 3D space. 
    We can plot just the first 2 components (z0, z1) which creates a right triangle.
    
    Args:
        model: The trained GVAE model
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
            z_mu, z_logvar = model.encoder(x)
            z = model.sampling(z_mu, z_logvar)  # z in simplex
            z = F.softmax(z, dim=1) # Map to simplex
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
    
    plt.savefig(f"GVAE/latent_epoch_{epoch}.png")
    plt.close()

def gvae_eblo_loss(model, x):
    """
    Compute negative ELBO (loss to minimize) for Bernoulli decoder.
    x: (batch, input_dim) values in {0,1} or [0,1]
    returns loss (scalar), recon_loss (scalar), kl (scalar), z (tensor)
    """
    logits, z_mu, z_logvar, z = model(x)

    # Reconstruction: bernoulli likelihood -> BCEWithLogits
    bce = F.binary_cross_entropy_with_logits(logits, x, reduction='none')
    recon_per_sample = bce.sum(dim=1)  # per example reconstruction negative log-likelihood
    
    # Compute the mean for the batch
    recon_loss = recon_per_sample.mean()

    # KL
    z_sigma = torch.exp(0.5 * z_logvar)
    kld_loss = -0.5 * torch.sum(1 + torch.log(z_sigma**2) - z_mu**2 - z_sigma**2, dim=1)
    kl = kld_loss.mean()  # mean reduction over batch

    # ELBO
    loss = recon_loss + kl
    return loss, recon_loss, kl, z


def train_loop(model, optimizer, train_loader, device):
        
        tot_loss = 0.0
        tot_recon = 0.0
        tot_kl = 0.0
        n_samples = 0
        model.train()

        for x, _ in train_loader:
            x = x.to(device)

            #Reset Gradients
            optimizer.zero_grad()

            #Foward Pass
            loss, recon, kl, _ = gvae_eblo_loss(model, x)

            #BackProp
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            tot_loss += loss.item() * batch_size
            tot_recon += recon.item() * batch_size
            tot_kl += kl.item() * batch_size
            n_samples += batch_size

        mean_loss = tot_loss / n_samples
        mean_recon = tot_recon / n_samples
        mean_kl = tot_kl / n_samples

        print(f"Negative ELBO: {mean_loss:.4f} | Recon Loss: {mean_recon:.4f} | KL: {mean_kl:.4f}")

        return mean_loss, mean_recon, mean_kl          

def test_loop(model, test_loader, device):
    model.eval()
    tot_loss, tot_recon, tot_kl = 0.0, 0.0, 0.0
    n_samples = 0

    with torch.no_grad():
        labels = [] # For visualiation
        zs = [] # For visualization

        for x, label in test_loader:
            x = x.to(device)

            # Foward Pass
            loss, recon, kl, z = gvae_eblo_loss(model, x)

            batch_size = x.size(0)
            tot_loss += loss.item() * batch_size
            tot_recon += recon.item() * batch_size
            tot_kl += kl.item() * batch_size
            n_samples += batch_size

            labels.append(label)
            zs.append(z)

    mean_loss = tot_loss / n_samples
    mean_recon = tot_recon / n_samples
    mean_kl = tot_kl / n_samples
    print(f"Negative ELBO: {mean_loss:.4f} | Recon Loss: {mean_recon:.4f} | KL: {mean_kl:.4f}")

    labels = torch.cat(labels, dim=0)
    zs = torch.cat(zs, dim=0)

    plot_latent_space_simplex(model, test_loader, device, 50)

    latent_visualization(zs, labels, "GVAE")

    return mean_loss, mean_recon, mean_kl

def gaussian_log_prob(z, mu, logvar, eps=1e-8):
    """
    log N(z | mu, sigma^2) = -0.5 * [log(2*pi) + logvar + (z - mu)^2 / sigma^2]
    z: (K, B, latent_dim) or (batch, latent_dim)
    mu: (batch, latent_dim) or (1, B, latent_dim)
    logvar: (batch, latent_dim) or (1, B, latent_dim)
    Returns log probability per sample
    """
    sigma_sq = torch.exp(logvar).clamp(min=eps)
    log_p = -0.5 * (math.log(2 * math.pi) + logvar + (z - mu) ** 2 / sigma_sq)
    return log_p.sum(dim=-1)

@torch.no_grad()
def montecarlo_nll(model, data_loader, device, K=500):
    """
    Monte Carlo (importance-weighted) estimate of the marginal log-likelihood:
        p(x) ≈ (1/K) Σ_i p(x|z_i)p(z_i)/q(z_i)
    Returns the mean NLL (nats per image) for Gaussian VAE.
    
    For Gaussian VAE:
    - q(z|x) = N(mu(x), sigma^2(x))  (posterior)
    - p(z) = N(0, I)                  (prior)
    - p(x|z) = Bernoulli (via logits from decoder)
    """
    model.eval()
    total_nll = 0.0
    n_samples = 0

    for x, _ in data_loader:
        x = x.to(device)
        B = x.size(0)

        # ---- 1) Encode: q(z|x) = N(z_mu, z_logvar)
        z_mu, z_logvar = model.encoder(x)  # (B, latent_dim)

        # ---- 2) Sample z_i ~ q(z|x) K times
        # We need K samples per batch element
        z_mu_rep = z_mu.unsqueeze(0).expand(K, -1, -1)  # (K, B, latent_dim)
        z_logvar_rep = z_logvar.unsqueeze(0).expand(K, -1, -1)  # (K, B, latent_dim)
        
        # Reparameterization sampling
        std_rep = torch.exp(0.5 * z_logvar_rep)
        eps = torch.randn_like(std_rep)
        z = z_mu_rep + eps * std_rep  # (K, B, latent_dim)

        # ---- 3) Compute p(x|z_i)
        logits = model.decoder(z.view(-1, model.latent_dim))  # (K*B, input_dim)
        log_p_x_given_z = -F.binary_cross_entropy_with_logits(
            logits, x.repeat(K, 1), reduction="none"
        ).sum(dim=1)
        log_p_x_given_z = log_p_x_given_z.view(K, B)

        # ---- 4) Compute p(z_i) = N(0, I) and q(z_i|x) = N(z_mu, z_logvar)
        # Prior: standard normal N(0, I)
        log_p_z = gaussian_log_prob(z, torch.zeros_like(z), torch.zeros_like(z))  # (K, B)
        
        # Posterior: N(z_mu, z_logvar)
        log_q_z = gaussian_log_prob(z, z_mu_rep, z_logvar_rep)  # (K, B)

        # ---- 5) Importance weights w_i = p(x|z_i)p(z_i)/q(z_i)
        log_w = log_p_x_given_z + log_p_z - log_q_z   # (K, B)

        # ---- 6) Monte Carlo marginal log-likelihood (log-sum-exp trick)
        log_w_max, _ = torch.max(log_w, dim=0, keepdim=True)
        log_p_x = log_w_max.squeeze(0) + torch.log(
            torch.exp(log_w - log_w_max).mean(dim=0)
        ) - math.log(K)

        # ---- 7) Average over batch
        batch_nll = (-log_p_x).sum().item()
        total_nll += batch_nll
        n_samples += B

    mean_nll = total_nll / n_samples
    print(f"Estimated NLL (Monte Carlo, K={K}): {mean_nll:.4f}")
    return mean_nll

def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)

def train_from_scratch(model, train_loader, test_loader, device):
    # Init model weights
    model.apply(initialize_weights)    

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(1, 51):
        print(f"Epoch {epoch}")
        print("--------")
        mean_train_loss, mean_train_recon, mean_train_kl = train_loop(model, optimizer, train_loader, device)
        print("")
 
    #### Small Visualization
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():

                batch, _ = next(iter(test_loader))
                batch = batch[:16].to(device)
                x = batch  

                # Forward pass (posterior sample)
                logits, z_mu, z_logvar, z = model(x)
                recon_imgs = torch.sigmoid(logits).view(-1, 1, 28, 28)

                # Stack originals (top) and reconstructions (bottom)
                comparison = torch.cat([x.view(-1, 1, 28, 28), recon_imgs])
                torchvision.utils.save_image(
                    comparison,
                    f"GVAE/recon_images/recon_test_epoch_{epoch:02d}.png",
                    nrow=16
                )

                plot_latent_space_simplex(model, test_loader, device, epoch)
    return model

def main():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    #### IMPORTANT, the torch.gamma and torch.digamma explode when using mps (floating point errors)
   #  elif torch.backends.mps.is_available():
   #  device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Current device:", device)

    # Flatten
    transform = T.Compose([
    T.ToTensor(),         
    lambda t: t.view(-1)   
        ])
    trainset_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset_full = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Filter out 1, 8, 4
    target_digits = [1, 8, 4]

    def get_indices(dataset, digits):
        indices = []
        for i in range(len(dataset)):
            if dataset.targets[i].item() in digits:
                indices.append(i)
        return indices

    train_idx = get_indices(trainset_full, target_digits)
    test_idx = get_indices(testset_full, target_digits)

    train_subset = Subset(trainset_full, train_idx)
    test_subset = Subset(testset_full, test_idx)

    train_loader = DataLoader(train_subset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=100, shuffle=True)

    input_dim = 28 * 28
    latent_dim = 3

    model = GVAE(input_dim=input_dim,
                   enc_hidden_dims=500,
                   dec_hidden_dims=500,
                   latent_dim=latent_dim
                   ).to(device)

    # Check if trained model exists
    if os.path.exists("GVAE/GVAE_checkpoint.pth"):
        model.load_state_dict(torch.load("GVAE/GVAE_checkpoint.pth", weights_only=True, map_location=device))
    else:
        model = train_from_scratch(model, train_loader, test_loader, device)
        torch.save(model.state_dict(), "GVAE/GVAE_checkpoint.pth")

    print("-------------")
    print("Test Results:")
    montecarlo_nll(model, test_loader, device, K=500)
    mean_test_loss, mean_test_recon, mean_test_kl = test_loop(model, test_loader, device)


def test_checkpoint():
    """
    Read checkpoint files and conduct some tests
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    #### IMPORTANT, the torch.gamma and torch.digamma explode when using mps (floating point errors)
   #  elif torch.backends.mps.is_available():
   #  device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Current device:", device)

    # Flatten
    transform = T.Compose([
    T.ToTensor(),         
    lambda t: t.view(-1)   
        ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=100, shuffle=True, num_workers = 0)

    input_dim = 28 * 28

    latent_dims = [5, 10, 50, 100]

    for d in latent_dims:
        model = GVAE(input_dim=input_dim,
                   enc_hidden_dims=500,
                   dec_hidden_dims=500,
                   latent_dim=d
                   ).to(device)
        
        if os.path.exists(f"GVAE/latent_dim_{d}/GVAE_checkpoint.pth"):
            model.load_state_dict(torch.load(f"GVAE/latent_dim_{d}/GVAE_checkpoint.pth", weights_only=True, map_location=device))
        else:
            print(f"latent_dim_{d} checkpoint file not found!")
            return None
        
        ### Do Your Tests Here ###
        
        # Calc MonteCarlo_NLL
        # nll = montecarlo_nll(model, test_loader, device, K=500)
        # print(f"MonteCarlo_NLL for latent_dim_{d}: {nll:.4f}")

        # Get dimension-wise reconstruction
        dimension_one_hot = torch.nn.functional.one_hot(torch.arange(0, d)).to(device)
        logits = model.decoder(dimension_one_hot.to(torch.float32))
        recon_img = torch.sigmoid(logits).view(-1, 1, 28, 28)

        torchvision.utils.save_image(
            recon_img,
            f"GVAE/dimension_wise_one_hot_recon_with_latent_dim_{d}.png",
            nrow=10
        )

if __name__ == "__main__":
    main()
    # test_checkpoint()
