
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch import Tensor
from torch.distributions import Distribution
EPS = 1e-6
def inv_cdf_torch(u, l):
    """
    Inverse CDF of the continuous Bernoulli distribution.
    [cite_start]Used for reparameterization in CC sampling[cite: 600].
    """
    near_half = (l > 0.499) & (l < 0.501)
    safe_l = l.clamp(EPS, 1 - EPS)
    u = u.clamp(EPS, 1 - EPS)
    
    num = torch.log(u * (2 * safe_l - 1) + 1 - safe_l) - torch.log(1 - safe_l)
    den = torch.log(safe_l) - torch.log(1 - safe_l)
    x = num / den
    return torch.where(near_half, u, x)

def sample_cc_ordered_reparam(lam):
    """
    [cite_start]Ordered Rejection Sampler (Algorithm 1 in the CC paper [cite: 164]).
    Sorts lambda (lam) to maximize acceptance rate.
    lam: [B, K]
    Returns: [B, K] sample on the simplex.
    """
    B, K = lam.shape
    
    # 1. Sort lambda descending (largest to smallest)
    lam_sorted, indices = torch.sort(lam, dim=1, descending=True)
    
    # 2. Prepare Parameters (CB params = lam_i / (lam_i + lam_1))
    lam_1 = lam_sorted[:, 0].unsqueeze(1) 
    lam_rest = lam_sorted[:, 1:] 
    
    cb_params = lam_rest / (lam_rest + lam_1 + EPS)

    final_x_rest = torch.zeros_like(lam_rest)
    active_mask = torch.ones(B, dtype=torch.bool, device=lam.device)
    max_attempts = 20000
    
    for _ in range(max_attempts):
        if not active_mask.any():
            break
        
        n_active = active_mask.sum()
        u = torch.rand(n_active, K-1, device=lam.device, dtype=lam.dtype)
        
        active_params = cb_params[active_mask]
        # 4. Inverse CDF (Differentiable)
        x_cand = inv_cdf_torch(u, active_params) 
        
        # 5. Check constraint (Sum <= 1)
        sums = x_cand.sum(dim=1)
        accepted_now = (sums <= 1.0)
        
        # 6. Update tensors
        active_indices = torch.nonzero(active_mask).squeeze(-1)
        accepted_indices = active_indices[accepted_now]
        
        if accepted_indices.numel() > 0:
            final_x_rest[accepted_indices] = x_cand[accepted_now]
            # Clone mask before modification to satisfy autograd
            active_mask = active_mask.clone()
            active_mask[accepted_indices] = False
            
    # 7. Calculate the Slack Variable: x_1 = 1 - sum(x_rest)
    x_1 = (1.0 - final_x_rest.sum(dim=1, keepdim=True)).clamp(min=EPS)
    x_sorted = torch.cat([x_1, final_x_rest], dim=1)
    
    # 8. Unsort back to original order
    x_final = torch.zeros_like(lam)
    x_final.scatter_(1, indices, x_sorted)
    
    return x_final

def lambda_to_eta(lam: Tensor) -> Tensor:
    #[cite_start]"""Converts mean parameter lambda [B, K] to natural parameter eta [B, K-1][cite: 91]."""
    lam = lam.clamp(min=EPS, max=1.0) 
    last = lam[:, -1].unsqueeze(1)
    eta = torch.log(lam / (last + EPS))
    eta = torch.clamp(eta, -5, 5)
    return eta[:, :-1]

def numerical_stable_eta_diffs(eta_diffs, eps=1e-6):
     ### Numerical stability: replace small differences with eps
    small_diff_mask = torch.abs(eta_diffs) < eps
    sign_matrix = torch.sign(eta_diffs)
    sign_matrix[sign_matrix == 0] = 1.0 
    safe_diffs = sign_matrix * eps

    eta_diffs = torch.where(small_diff_mask, safe_diffs, eta_diffs)

    return eta_diffs

def compute_inv_C_eta(eta, dtype=torch.float32):

    """
    Computes the inverse of the normalizing constant Z(eta) = 1 / C(eta) for the CC distribution.
    
    Args:
        eta (torch.Tensor): Natural parameters of the CC distribution of dimension K and N observations.
                            Shape: (N, K-1)
    Returns:
        Z_eta (torch.Tensor): Z(eta)
                              Shape: (N)
    
    To get log C(eta) just compute -log Z(eta)
    """

    eps = 1e-6
    n, K_minus_1 = eta.shape
    K = K_minus_1 + 1
    
    # add eta_K = 0
    zeros = torch.zeros(n, 1, dtype=dtype, device=eta.device)
    eta_pad = torch.cat([eta, zeros], dim=1) # Shape: (n, K)
    # 3. Reshape for broadcasting
    rows = eta_pad.unsqueeze(2) # shape: (n, K, 1)   
    cols = eta_pad.unsqueeze(1) # shape: (n, 1, K)
    eta_diffs = cols - rows # shape: (n, K , K)
    
    # Check if eta_diffs is numerically stable
    eta_diffs = numerical_stable_eta_diffs(eta_diffs, eps)
    
    # Set eta_i - eta_i = 1 to avoid divisions by zero
    eta_diffs = eta_diffs + torch.eye(K, dtype=dtype, device=eta.device)


    numer = torch.exp(eta_pad) # shape: (n, K)

    denom = torch.prod(eta_diffs, dim=1) # shape: (n, K) where the last column corresponds to exp(eta_K) = 1
    Z_eta = torch.sum(numer / denom, dim=1) # Colapse columns, shape: (n)
    #print(Z_eta)
    Z_eta = torch.clamp(Z_eta, min=eps)
    return -torch.log(Z_eta)



def cc_log_norm_const_torch(eta: Tensor) -> Tensor:
    """
    [cite_start]Calculates log C(eta) using the Exact Formula (Eq 7 in the paper [cite: 124]).
    [cite_start]Note: Calculation is done in double precision for stability[cite: 134].
    """
    original_dtype = eta.dtype
    eta = eta.double()
    eta = torch.clamp(eta, -5,5)
    B, K_minus_1 = eta.shape
    K = K_minus_1 + 1
    device = eta.device
    
    # [cite_start]1. Construct full eta (append 0 for the Kth component) [cite: 92]
    eta_full = torch.cat([eta, torch.zeros(B, 1, device=device, dtype=eta.dtype)], dim=1)
    
    # 2. Add Jitter
    jitter = torch.arange(K, device=device) * 1e-5
    eta_full = eta_full + jitter.unsqueeze(0)

    # 3. Compute the denominator product: prod_{i!=k} (eta_i - eta_k) 
    eta_i = eta_full.unsqueeze(1) 
    eta_k = eta_full.unsqueeze(2)
    diffs = eta_i - eta_k 
    diffs = diffs + torch.eye(K)
    
    eye_mask = torch.eye(K, device=device).bool().unsqueeze(0).expand(B, -1, -1)
    diffs[eye_mask] = 1.0 
    
    log_diffs_abs = diffs.abs().log()
    diffs_sign = diffs.sign()
    
    log_denom = log_diffs_abs.sum(dim=1) 
    denom_sign = diffs_sign.prod(dim=1) 
    
    log_terms_mag = eta_full - log_denom
    terms_sign = denom_sign
    
    # 4. Sum the terms: S = sum_k (T_k)
    max_log_mag, _ = log_terms_mag.max(dim=1, keepdim=True)
    sum_scaled = torch.sum(terms_sign * torch.exp(log_terms_mag - max_log_mag), dim=1)
    
    # 5. Multiply by (-1)^(K+1) 
    global_sign = (-1)**(K + 1)
    total_sum_signed = global_sign * sum_scaled
    
    log_inv_C = max_log_mag.squeeze() + torch.log(total_sum_signed.clamp(min=EPS))
    
    # Return log C = - log(C^-1)
    return -log_inv_C.to(dtype=original_dtype)

def cc_log_prob_torch(sample: Tensor, eta: Tensor) -> Tensor:
    """
    [cite_start]Calculates the log-density p(z | eta) = eta^T * z + log C(eta)[cite: 94].
    sample: [B, K], eta: [B, K-1]
    Returns: [B]
    """
    n, K_minus_1 = eta.shape
    aug_eta = torch.cat([eta, torch.zeros(n, 1, device=eta.device, dtype=eta.dtype)], dim=-1)
    
    # Exponent term: eta^T * z
    exponent = torch.sum(sample * aug_eta, dim=1) 
    
    # Log Normalizer term
    log_norm_const = compute_inv_C_eta(eta)
    
    return exponent + log_norm_const 