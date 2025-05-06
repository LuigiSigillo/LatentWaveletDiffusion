import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import pytorch_wavelets as ptwt
import torchvision.transforms as transforms
from matplotlib.colors import LogNorm

def download_image(url):
    """Download an image from URL"""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return img

def plot_wavelet_decomposition(image, wavelet_transform, level=1, figsize=(15, 10)):
    """Perform wavelet decomposition and plot the results"""
    # Convert image to PyTorch tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Perform wavelet decomposition
    yl, yh = wavelet_transform(img_tensor)
    
    # Convert to numpy for visualization
    coeffs_lowpass = yl.squeeze().cpu().numpy()
    coeffs_highpass = [yh_level.squeeze().cpu().numpy() for yh_level in yh]
    
    # Create figure for visualization
    fig, axes = plt.subplots(2, 5, figsize=figsize)  # Changed from 2, 4 to 2, 5
    fig.suptitle(f'Wavelet Decomposition (Level {level})', fontsize=16)
    
    # Plot original image
    axes[0, 0].imshow(np.array(image))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Plot low-pass (LL) component
    ll_img = np.transpose(coeffs_lowpass, (1, 2, 0))
    ll_img = (ll_img - ll_img.min()) / (ll_img.max() - ll_img.min())
    axes[0, 1].imshow(ll_img)
    axes[0, 1].set_title('LL (Approximation)')
    axes[0, 1].axis('off')
    
    # Plot high-pass components (LH, HL, HH) for the first level
    highpass_names = ['LH (Horizontal)', 'HL (Vertical)', 'HH (Diagonal)']
    
    for i, name in enumerate(highpass_names):
        # Get component (for RGB, average across channels)
        component = coeffs_highpass[0][:, i, :, :]
        component_viz = np.abs(np.transpose(component, (1, 2, 0)))
        
        # Normalize for visualization
        if component_viz.max() > component_viz.min():
            component_viz = (component_viz - component_viz.min()) / (component_viz.max() - component_viz.min())
            
        axes[0, i+1].imshow(component_viz)  # Changed from i+2 to i+1
        axes[0, i+1].set_title(name)
        axes[0, i+1].axis('off')
    
    # Create energy maps
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Energy Maps', fontsize=14)
    
    # Calculate and plot energy maps for each subband
    for i, name in enumerate(['LL (Energy)', 'LH (Energy)', 'HL (Energy)', 'HH (Energy)']):
        if i == 0:
            # Energy map for LL
            energy = np.mean(coeffs_lowpass**2, axis=0)
        else:
            # Energy maps for LH, HL, HH
            energy = np.mean(coeffs_highpass[0][:, i-1, :, :]**2, axis=0)
        
        im = axes[1, i+1].imshow(energy, cmap='viridis', norm=LogNorm(vmin=max(energy.min(), 1e-8)))
        axes[1, i+1].set_title(name)
        axes[1, i+1].axis('off')
        plt.colorbar(im, ax=axes[1, i+1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig, (coeffs_lowpass, coeffs_highpass)

def create_combined_energy_map(highpass_coeffs, weights=[0.2, 0.3, 0.5]):
    """Create a combined energy map with different weights for LH, HL, HH"""
    # Get energy for each high-pass component
    lh_energy = np.mean(highpass_coeffs[0][:, 0, :, :]**2, axis=0)
    hl_energy = np.mean(highpass_coeffs[0][:, 1, :, :]**2, axis=0)
    hh_energy = np.mean(highpass_coeffs[0][:, 2, :, :]**2, axis=0)
    
    # Normalize each energy map
    lh_norm = lh_energy / lh_energy.max()
    hl_norm = hl_energy / hl_energy.max()
    hh_norm = hh_energy / hh_energy.max()
    
    # Combine with weights
    combined = weights[0] * lh_norm + weights[1] * hl_norm + weights[2] * hh_norm
    
    # Normalize the combined map
    combined = combined / combined.max()
    
    return combined

def visualize_attention_mask_application(image, energy_map, threshold=0.3):
    """Visualize how the energy map would be applied as an attention mask"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(np.array(image))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Energy map as attention
    im = axes[1].imshow(energy_map, cmap='inferno')
    axes[1].set_title('Combined Energy Map (Attention)')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Thresholded mask for time-dependent diffusion
    mask = np.zeros_like(energy_map)
    mask[energy_map > threshold] = 1.0
    
    axes[2].imshow(mask, cmap='viridis')
    axes[2].set_title(f'Thresholded Mask (t={threshold})')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    # Set up wavelet transform
    wave = ptwt.DWTForward(J=1, mode='zero', wave='db2')
    
    # Download sample image
    # url = "https://pytorch.org/assets/images/sample/sample5.jpg"
    # print(f"Downloading sample image from {url}")
    # image = download_image(url)
    image = Image.open("/mnt/share/Luigi/Documents/URAE/dataset/test_images/image_00000001.jpg").convert('RGB')  # Replace with your image path
    # Resize for faster processing
    image = image.resize((512, 512))
    
    # Plot wavelet decomposition
    print("Performing wavelet decomposition...")
    fig1, (lowpass, highpass) = plot_wavelet_decomposition(image, wave)
    
    # Create combined energy map with custom weights
    print("Creating combined energy map...")
    combined_energy = create_combined_energy_map(highpass)
    
    # Visualize how the energy map would be applied
    print("Visualizing attention mask application...")
    fig2 = visualize_attention_mask_application(image, combined_energy)
    
    # Show plots
    plt.savefig("./testing.png", dpi=300)
    
    print("Done!")

if __name__ == "__main__":
    main()