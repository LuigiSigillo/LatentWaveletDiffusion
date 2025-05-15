import torch
import torch.nn.functional as F

def compute_wavelet_attention(latent, dwt, return_visualization=False):
    """
    Compute wavelet-based attention map from a VAE latent.

    Args:
        latent (Tensor): Latent tensor of shape (B, C, H, W)
        dwt: Wavelet transform module
        return_visualization (bool): If True, return visualization components

    Returns:
        attn_map (Tensor): Attention map of shape (B, H, W), values in [0, 1]
        vis_dict (dict, optional): Dictionary with visualization components if return_visualization=True
    """
    B, C, H, W = latent.shape
    
    # Store original dtype and cast to float32 for wavelets processing
    original_dtype = latent.dtype
    latent_float32 = latent.to(dtype=torch.float32)

    # Apply 2D DWT to each channel
    LL, high = dwt(latent_float32)  # LL: low-frequency, high[0]: (B, 3, C, H//2, W//2)
    LH, HL, HH = high[0][:, 0], high[0][:, 1], high[0][:, 2]  # each: (B, C, H//2, W//2)

    # Compute average energy across channels
    LH_energy = LH.pow(2).mean(dim=1)  # (B, H//2, W//2)
    HL_energy = HL.pow(2).mean(dim=1)
    HH_energy = HH.pow(2).mean(dim=1)

    # Sum high-frequency energies
    HF_energy = LH_energy + HL_energy + HH_energy  # (B, H//2, W//2)

    # Upsample to match original spatial resolution
    attn_map_raw = F.interpolate(HF_energy.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)

    # Normalize to [0, 1] per sample
    attn_map = (attn_map_raw - attn_map_raw.amin(dim=(1, 2), keepdim=True)) / \
               (attn_map_raw.amax(dim=(1, 2), keepdim=True) - attn_map_raw.amin(dim=(1, 2), keepdim=True) + 1e-8)

    # Convert back to original dtype 
    attn_map = attn_map.to(dtype=original_dtype)  # shape: (B, H, W)
    
    if return_visualization:
        vis_dict = {
            'LL': LL.detach(),
            'LH': LH.detach(),
            'HL': HL.detach(),
            'HH': HH.detach(),
            'LH_energy': LH_energy.detach(),
            'HL_energy': HL_energy.detach(),
            'HH_energy': HH_energy.detach(),
            'HF_energy': HF_energy.detach(),
            'attn_map_raw': attn_map_raw.detach()
        }
        return attn_map, vis_dict
    
    return attn_map,None  # shape: (B, H, W)

def get_mask_batch(A, l, T, timesteps, return_visualization=False):
    """
    Vectorized version of get_mask for a batch of timesteps.
    
    Args:
        A (Tensor): Wavelet attention map, shape (B, H, W), values in [0, 1]
        l (float): Lower bound (e.g., 0.1)
        T (int): Total number of timesteps
        timesteps (Tensor): Tensor of shape (B,) with values in [0, T]
        return_visualization (bool): If True, return visualization components
    
    Returns:
        M (Tensor): Binary mask, shape (B, 1, H, W)
        vis_dict (dict, optional): Dictionary with visualization components if return_visualization=True
    """
    B, H, W = A.shape
    device = A.device
    # Ensure consistent dtype and device for calculations
    original_dtype = A.dtype
    A_float = A.to(dtype=torch.float32, device=device)
    
    # Make sure timesteps is on the same device as A
    timesteps = timesteps.to(device=device)
    t_matrix = timesteps.view(B, 1, 1).expand(B, H, W).to(dtype=torch.float32, device=device)
    
    # Calculate A_float + l for visualization
    A_plus_l = A_float + l
    
    # Calculate thresholds
    thresholds = T * A_plus_l  # shape (B, H, W)
    
    # Generate mask
    M = (thresholds >= t_matrix).float()  # shape (B, H, W)
    
    # Return mask with the original dtype
    mask = M.unsqueeze(1).to(dtype=original_dtype, device=device)  # shape (B, 1, H, W)
    
    if return_visualization:
        vis_dict = {
            'A_float': A_float.detach(),
            'A_plus_l': A_plus_l.detach(),
            't_matrix': t_matrix.detach(),
            'thresholds': thresholds.detach()
        }
        return mask, vis_dict
    
    return mask,None  # shape (B, 1, H, W)

def get_mask(A, l=0.1, T=1000, t=0):
    """
    Compute the binary mask M(t) for time-dependent adaptive refinement.

    Args:
        A (Tensor): Attention map of shape (B, H, W), values in [0, 1]
        l (float): Lower bound ratio (e.g., 0.1) ensuring all regions are refined at least l*T steps
        T (int): Total number of diffusion steps
        t (int): Current timestep (0 <= t <= T)

    Returns:
        M (Tensor): Binary mask of shape (B, H, W), with values {0, 1}
    """
    # Compute threshold matrix: T * (A + l)
    threshold = T * (A + l)  # shape: (B, H, W)

    # Mask is 1 where threshold >= t
    M = (threshold >= t).float()  # shape: (B, H, W)

    return M


import matplotlib.pyplot as plt

def show_mask(A, M, sample_idx=0):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(A[sample_idx].detach().cpu(), cmap='viridis')
    axs[0].set_title('Wavelet Attention Map')
    
    # Check if M has a channel dimension
    if M.dim() == 4:  # (B, 1, H, W)
        mask_to_show = M[sample_idx, 0].detach().cpu()
    else:  # (B, H, W)
        mask_to_show = M[sample_idx].detach().cpu()
        
    axs[1].imshow(mask_to_show, cmap='gray')
    axs[1].set_title('Binary Mask at timestep')
    plt.savefig("mask.png", dpi=300)

if __name__ == "__main__":
    # Example usage
    B, C, H, W = 2, 3, 64, 64
    T = 1000
    l = 0.1
    t = 500
    latent = torch.randn(B, C, H, W)  # Example latent tensor
    #get latent z from vae
    # z = vae.encode(input)  # Example encoding step

    # Example setup
    from pytorch_wavelets import DWTForward
    A = compute_wavelet_attention(latent, DWTForward(J=1,wave="haar"))  # (B, H, W)
    T = 1000
    l = 0.1
    M = get_mask(A, l, T, 500)
    show_mask(A, M, sample_idx=0)

    # for t in reversed(range(T)):
    #     M = get_mask(A, l, T, t)
    #     # Use M for masked diffusion step here
    #     show_mask(A, M, sample_idx=0)