import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import time
from cc_functions import cc_log_prob_torch, sample_cc_ordered_reparam, lambda_to_eta 
from plots import plot_latent_space_simplex
from latent_visualization import latent_visualization

# torch.set_default_dtype(torch.float64)
torch.manual_seed(48)
os.makedirs("results", exist_ok=True)
EPS = 1e-6




class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),

            nn.Linear(hidden_dims, latent_dim)

        )

    def forward(self, x):
        logits = self.net(x)
        lambdas = F.softmax(logits, dim=1)
        return lambdas


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


class CCVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, K):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, hidden_dims, latent_dim)
        self.decoder = BernoulliDecoder(latent_dim, hidden_dims, input_dim)
        self.latent_dim = latent_dim
        self.K = K

    def sample_batched(self, lam):
        """
        Uses the optimized vectorized sampler from cc_sampler.py
        """

        # 1. Sample using the vectorized rejection sampler
        z_full = sample_cc_ordered_reparam(lam)

        # 2. Safety Clamp & Normalize
        # Ensure numerical stability (sum to 1, no negative values)
        z_full = torch.clamp(z_full, min=0.0, max=1.0)
        z_full = z_full / (z_full.sum(dim=-1, keepdim=True) + 1e-6)

        return z_full

    def forward(self, x):
        lambds = self.encoder(x)  # shape: (batch_size, K)
        etas = lambda_to_eta(lambds)
        z = self.sample_batched(lambds)
        # shape: (batch_size, K)
        logits = self.decoder(z.float())  # shape: (batch_size, input_dim)

        return logits, etas, z


def elbo_loss(model, x, eta_p=None, beta=1.0):
    logits, eta_q, z = model(x)

    # --- Reconstruction Loss (per-sample mean scale) ---
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
    recon = recon.view(x.size(0), -1).sum(dim=1)  # sum over pixels → shape [B]
    recon_loss = recon.mean()                     # mean over batch → scalar

    # --- Fixed uniform prior η_p = 0 (no random jitter!) ---
    if eta_p is None:
        eta_p = torch.zeros_like(eta_q)

    
    # --- Differentiable log-density KL estimation ---
    log_q = cc_log_prob_torch(z, eta_q)            # [B]
    log_p = cc_log_prob_torch(z, eta_p)            # [B]
    kl_per_sample = log_q - log_p                  # [B]
    kl = kl_per_sample.mean()                      # scalar

    loss = recon_loss + beta * kl
    return loss, recon_loss, kl, z  # KL is now the real per-sample mean (nats)


def get_beta(epoch, beta_start=0.0, beta_end=1.0, warmup_epochs=10):
    t = min(epoch / warmup_epochs, 1.0)
    return beta_start + t * (beta_end - beta_start)

def test_loop(model, test_loader, device):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    labels = []
    zs = []

    for x, label in test_loader:
        x = x.to(device)

        loss, recon_loss, kl, z = elbo_loss(model, x)

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl.item()

        labels.append(label)
        zs.append(z)

    avg_loss = total_loss / len(test_loader)
    avg_recon = total_recon / len(test_loader)
    avg_kl = total_kl / len(test_loader)

    print(f"Negative ELBO: {avg_loss:.4f}, Recon (nats): {avg_recon:.4f}, KL (nats): {avg_kl:.4f}")

    labels = torch.cat(labels, dim=0)
    zs = torch.cat(zs, dim=0)

    latent_visualization(zs, labels, "CCVAE")

    return avg_loss, avg_recon, avg_kl

def train_loop(model, trainloader, optimizer, device, beta):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    for x, _ in trainloader:
        x = x.to(device)

        optimizer.zero_grad()
        loss, recon_loss, kl, _ = elbo_loss(model, x, beta=beta)


        loss.backward()

        # Clip grads to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl.item()

    avg_loss = total_loss / len(trainloader)
    avg_recon = total_recon / len(trainloader)
    avg_kl = total_kl / len(trainloader)

    print(f"Train Loss: {avg_loss:.4f}, Recon (nats): {avg_recon:.4f}, KL (nats): {avg_kl:.4f}")
    return avg_loss, avg_recon, avg_kl

def train_from_scratch(model, train_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)   

    for epoch in range(1, 51):
        print(f"Epoch {epoch} ---------------------")
        beta = get_beta(epoch, beta_start=0.05, beta_end=1.0, warmup_epochs=20)
        avg_loss = train_loop(model, train_loader, optimizer, device, beta=beta)

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                x_real, _ = next(iter(test_loader))
                x_real = x_real.to(device)
                x_real = x_real[:16]

                # Forward pass
                logits, _, _ = model(x_real)
                x_recon = torch.sigmoid(logits)

                x_real_img = x_real.view(-1, 1, 28, 28)
                x_recon_img = x_recon.view(-1, 1, 28, 28)

                comparison = torch.cat([x_real_img, x_recon_img])

                save_path = f"results/reconstruction_epoch_{epoch}.png"
                torchvision.utils.save_image(comparison, save_path, nrow=16)

                if latent_dim == 3:
                    plot_latent_space_simplex(model, test_loader, device, epoch)
    
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = T.Compose([T.ToTensor(), lambda t: t.view(-1)])

    # Load FULL MNIST
    trainset_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset_full = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # --- FILTER FOR DIGITS 0, 1, 2 ---
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

    model = CCVAE(input_dim=input_dim, hidden_dims=512, latent_dim=latent_dim, K=latent_dim).to(device)

    # Check if trained model exists
    if os.path.exists("CCVAE_checkpoint.pth"):
        model.load_state_dict(torch.load("CCVAE_checkpoint.pth", weights_only=True, map_location=device))
    else:
        print(f"Training CC-VAE for digits {target_digits}, using latent dim {latent_dim}")
        model = train_from_scratch(model, train_loader, device)
        torch.save(model.state_dict(), "CCVAE_checkpoint.pth")

    print("-------------")
    print("Test Results:")
    mean_test_loss, mean_test_recon, mean_test_kl = test_loop(model, test_loader, device)