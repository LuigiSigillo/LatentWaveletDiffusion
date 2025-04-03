import os
import torch
import argparse
import shutil
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
from vae_finetune_diffusability import SpectralAnalyzer

def rename_model_files(checkpoint_path, model_subfolder=""):
    """
    Rename model files to the format expected by AutoencoderKL.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        model_subfolder: Subfolder within checkpoint where model is stored
    
    Returns:
        bool: True if files were renamed, False otherwise
    """
    # Check if model is in a subfolder
    dir_path = checkpoint_path
    if model_subfolder:
        potential_path = os.path.join(checkpoint_path, model_subfolder)
        if os.path.exists(potential_path):
            dir_path = potential_path
    
    # Check for model.safetensors and rename it
    source_file = os.path.join(dir_path, "model.safetensors")
    target_file = os.path.join(dir_path, "diffusion_pytorch_model.safetensors")
    
    if os.path.exists(source_file) and not os.path.exists(target_file):
        print(f"Renaming {source_file} to {target_file}")
        shutil.copy2(source_file, target_file)
        return True
    
    # Check for pytorch_model.bin and rename it
    source_file = os.path.join(dir_path, "pytorch_model.bin")
    target_file = os.path.join(dir_path, "diffusion_pytorch_model.bin")
    
    if os.path.exists(source_file) and not os.path.exists(target_file):
        print(f"Renaming {source_file} to {target_file}")
        shutil.copy2(source_file, target_file)
        return True
    
    return False

def evaluate_checkpoints(
    checkpoint_dir,
    data_dir,
    output_dir="spectral_analysis_results",
    image_size=256,
    batch_size=8,
    device="cuda",
    with_tracking=False,
    checkpoint_prefix="checkpoint-",
    model_subfolder="vae",
    rename_files=True
):
    """
    Evaluate VAE checkpoints using SpectralAnalyzer.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoint folders
        data_dir (str): Directory containing images for evaluation
        output_dir (str): Directory to save analysis results
        image_size (int): Image size for evaluation
        batch_size (int): Batch size for evaluation
        device (str): Device to run evaluation on ('cuda' or 'cpu')
        with_tracking (bool): Whether to log results to wandb
        checkpoint_prefix (str): Prefix of checkpoint folders
        model_subfolder (str): Subfolder where model files are stored within checkpoint
        rename_files (bool): Whether to rename model files to the expected format
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize spectral analyzer
    spectral_analyzer = SpectralAnalyzer(device)
    
    # Prepare dataset for evaluation
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
    ])
    
    # Load test images
    image_paths = [
        os.path.join(data_dir, fname)
        for fname in os.listdir(data_dir)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]
    
    # Take a batch of images for evaluation
    eval_images = []
    for i in range(min(batch_size, len(image_paths))):
        img = Image.open(image_paths[i]).convert("RGB")
        eval_images.append(transform(img))
    
    # Stack images to create a batch
    eval_batch = torch.stack(eval_images).to(device)
    
    # Find all checkpoint folders
    checkpoint_folders = []
    if os.path.isdir(checkpoint_dir):
        for item in os.listdir(checkpoint_dir):
            if item.startswith(checkpoint_prefix) and os.path.isdir(os.path.join(checkpoint_dir, item)):
                try:
                    step = int(item.replace(checkpoint_prefix, ""))
                    checkpoint_folders.append((step, os.path.join(checkpoint_dir, item)))
                except ValueError:
                    print(f"Skipping folder {item} - couldn't extract step number")
    
    # Sort checkpoints by step
    checkpoint_folders.sort(key=lambda x: x[0])
    
    # Add final model if it exists (not in a checkpoint folder)
    if os.path.exists(os.path.join(checkpoint_dir, "config.json")):
        final_step = max([step for step, _ in checkpoint_folders]) + 1000 if checkpoint_folders else 10000
        checkpoint_folders.append((final_step, checkpoint_dir))
    
    # Create dataframe to store metrics
    metrics_data = {
        "step": [],
        "high_freq_suppression": [],
        "orig_high_freq_ratio": [],
        "recon_high_freq_ratio": [],
        "orig_low_freq_ratio": [],
        "recon_low_freq_ratio": []
    }
    
    # Process each checkpoint
    for step, checkpoint_path in tqdm(checkpoint_folders, desc="Processing checkpoints"):
        print(f"Processing checkpoint at step {step}: {checkpoint_path}")
        
        try:
            # Check if model is in a subfolder
            model_path = checkpoint_path
            if model_subfolder:
                potential_model_path = os.path.join(checkpoint_path, model_subfolder)
                if os.path.exists(potential_model_path):
                    model_path = potential_model_path
                    print(f"Found model in subfolder: {model_path}")
            
            # Rename model files if needed
            if rename_files:
                renamed = rename_model_files(checkpoint_path, model_subfolder)
                if renamed:
                    print(f"Model files renamed in {checkpoint_path}")
            
            # # List files in the model path to help with debugging
            # print(f"Files in model path {model_path}:")
            # for file in os.listdir(model_path):
            #     print(f"  - {file}")
            
            # Load model
            vae = AutoencoderKL.from_pretrained(model_path).to(device)
            vae.eval()
            
            # Generate reconstructions
            with torch.no_grad():
                reconstructions = vae(eval_batch).sample
            
            # Analyze spectral properties
            # metrics = spectral_analyzer.analyze(
            #     eval_batch, 
            #     reconstructions, 
            #     step, 
            #     log_to_wandb=with_tracking
            # )
            
            # Compute DCT coefficients for the image batch
            dct_coeffs = compute_dct_batch(eval_batch)
            dct_coeffs_recon = compute_dct_batch(reconstructions)
            
            # Analyze frequency bands
            freq_metrics = frequency_bands_analysis(dct_coeffs, device=device)
            freq_metrics_recon = frequency_bands_analysis(dct_coeffs_recon, device=device)
            print("Frequency Metrics:")
            for key, value in freq_metrics.items():
                print(f"{key}: {value:.4f}")
                # total_energy: 32818843648.0000
                # low_freq_energy: 32750628864.0000
                # mid_freq_energy: 68190384.0000
                # high_freq_energy: 299134.7188
                # low_freq_ratio: 0.9979
                # mid_freq_ratio: 0.0021
                # high_freq_ratio: 0.0000
            
            # Compute PSNR between original and reconstructed images
            image_psnr = psnr(eval_batch, reconstructions)
            print(f"\nPSNR between original and reconstruction: {image_psnr:.2f} dB")
            
            # Save metrics
            metrics_data["step"].append(step)
            # metrics_data["high_freq_suppression"].append(metrics["high_freq_suppression"])
            # metrics_data["orig_high_freq_ratio"].append(metrics["orig_high_freq_ratio"])
            # metrics_data["recon_high_freq_ratio"].append(metrics["recon_high_freq_ratio"])
            # metrics_data["orig_low_freq_ratio"].append(metrics["orig_low_freq_ratio"])
            # metrics_data["recon_low_freq_ratio"].append(metrics["recon_low_freq_ratio"])
            metrics_data["orig_high_freq_ratio"].append(freq_metrics["high_freq_ratio"])
            metrics_data["recon_high_freq_ratio"].append(freq_metrics_recon["high_freq_ratio"])
            metrics_data["orig_low_freq_ratio"].append(freq_metrics["low_freq_ratio"])
            metrics_data["recon_low_freq_ratio"].append(freq_metrics_recon["low_freq_ratio"])
            metrics_data["high_freq_suppression"].append(1.0 - (freq_metrics_recon["high_freq_ratio"] / freq_metrics["high_freq_ratio"]))

            # Save a visualization of originals vs reconstructions
            save_reconstruction_comparison(eval_batch, reconstructions, step, output_dir)
            
        except Exception as e:
            print(f"Error processing checkpoint {checkpoint_path}: {e}")
            # print("Trying to list directory contents to debug:")
            # try:
            #     files = os.listdir(checkpoint_path)
            #     print(f"Files in {checkpoint_path}:")
            #     for file in files:
            #         print(f"  - {file}")
            #         if os.path.isdir(os.path.join(checkpoint_path, file)):
            #             subfolder_files = os.listdir(os.path.join(checkpoint_path, file))
            #             print(f"  Files in {file}/:")
            #             for subfile in subfolder_files:
            #                 print(f"    - {subfile}")
            # except Exception as list_error:
            #     print(f"Error listing directory: {list_error}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(os.path.join(output_dir, "spectral_metrics.csv"), index=False)
    
    # Plot metrics
    plot_metrics(metrics_data, output_dir)
    
    return metrics_data

def save_reconstruction_comparison(originals, reconstructions, step, output_dir):
    """Save a comparison of original and reconstructed images."""
    os.makedirs(os.path.join(output_dir, "reconstructions"), exist_ok=True)
    
    # Create a figure with subplots
    n_images = min(4, originals.shape[0])
    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 4, 8))
    
    # Normalize images from [-1, 1] to [0, 1] for visualization
    orig_images = (originals[:n_images] + 1) / 2
    recon_images = (reconstructions[:n_images] + 1) / 2
    
    # Plot each image
    for i in range(n_images):
        # Original image
        orig_img = orig_images[i].permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")
        
        # Reconstructed image
        recon_img = recon_images[i].permute(1, 2, 0).cpu().numpy()
        axes[1, i].imshow(recon_img)
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reconstructions", f"comparison_step_{step}.png"))
    plt.close(fig)

def plot_metrics(metrics_data, output_dir):
    """Plot metrics over training steps."""
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    # Plot high frequency suppression
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_data["step"], metrics_data["high_freq_suppression"], 'o-')
    plt.xlabel("Training Step")
    plt.ylabel("High Frequency Suppression")
    plt.title("High Frequency Suppression Across Checkpoints")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "plots", "high_freq_suppression.png"))
    plt.close()
    
    # Plot high frequency ratios
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_data["step"], metrics_data["orig_high_freq_ratio"], 'o-', label="Original")
    plt.plot(metrics_data["step"], metrics_data["recon_high_freq_ratio"], 'o-', label="Reconstructed")
    plt.xlabel("Training Step")
    plt.ylabel("High Frequency Ratio")
    plt.title("High Frequency Content Across Checkpoints")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "plots", "high_freq_ratios.png"))
    plt.close()
    
    # Plot low frequency ratios
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_data["step"], metrics_data["orig_low_freq_ratio"], 'o-', label="Original")
    plt.plot(metrics_data["step"], metrics_data["recon_low_freq_ratio"], 'o-', label="Reconstructed")
    plt.xlabel("Training Step")
    plt.ylabel("Low Frequency Ratio")
    plt.title("Low Frequency Content Across Checkpoints")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "plots", "low_freq_ratios.png"))
    plt.close()




import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Optional: For SSIM you could use skimage.metrics, but here we implement PSNR.
def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # assuming images normalized between 0 and 1
    return 20 * torch.log10(max_pixel / torch.sqrt(mse)).item()

# --------------------------
# 2D DCT using FFT-based approximation
# --------------------------
def dct2d(x):
    """
    Compute a type-II DCT of a 2D signal x using FFT.
    x: torch tensor of shape (H, W)
    Returns: DCT coefficients, same shape as x.
    """
    # Apply the DCT-II algorithm via FFT as described in many online implementations.
    # Note: This is a simplified implementation.
    x = x.float()
    H, W = x.shape
    # Create a symmetric extension
    x_ext = torch.cat([x, torch.flip(x, dims=[0])], dim=0)
    x_ext = torch.cat([x_ext, torch.flip(x_ext, dims=[1])], dim=1)
    # Compute the FFT and take the real part
    X = torch.fft.fft2(x_ext)
    # Extract the top-left quadrant, scaled appropriately
    X = X[:H, :W].real
    return X

# --------------------------
# Frequency Bands Analysis Function
# --------------------------
def frequency_bands_analysis(dct_coeffs, device='cpu'):
    """
    Analyze energy in different frequency bands from DCT coefficients.
    dct_coeffs: Tensor of shape (B, C, H, W)
    Returns a dictionary with total energy and energy ratios for low, mid, high frequencies.
    """
    batch_size, channels, height, width = dct_coeffs.shape
    
    # Create frequency band indices
    low_idx = int(height * 0.25)  # low frequencies: top-left quarter
    mid_idx = int(height * 0.75)  # mid frequencies: between 25%-75%
    
    # Compute energy (squared magnitude) of coefficients
    energy = dct_coeffs.pow(2)
    total_energy = energy.sum(dim=[1,2,3]).mean().item()
    
    # Low frequency energy: top-left corner of the DCT map
    low_energy = energy[:, :, :low_idx, :low_idx].sum(dim=[1,2,3]).mean().item()
    
    # Create mid-frequency mask: exclude top-left and bottom-right parts
    mid_mask = torch.ones((height, width), device=device)
    mid_mask[:low_idx, :low_idx] = 0
    mid_mask[mid_idx:, mid_idx:] = 0
    mid_energy = (energy * mid_mask.view(1,1,height,width)).sum(dim=[1,2,3]).mean().item()
    
    # Create high-frequency mask: select bottom-right quadrant (approx.) or everything beyond mid_idx rows/cols
    high_mask = torch.zeros((height, width), device=device)
    high_mask[mid_idx:, :] = 1
    high_mask[:, mid_idx:] = 1
    high_energy = (energy * high_mask.view(1,1,height,width)).sum(dim=[1,2,3]).mean().item()
    
    return {
        "total_energy": total_energy,
        "low_freq_energy": low_energy,
        "mid_freq_energy": mid_energy,
        "high_freq_energy": high_energy,
        "low_freq_ratio": low_energy / total_energy if total_energy > 0 else 0,
        "mid_freq_ratio": mid_energy / total_energy if total_energy > 0 else 0,
        "high_freq_ratio": high_energy / total_energy if total_energy > 0 else 0,
    }

# --------------------------
# Helper function to compute DCT on a batch of images
# --------------------------
def compute_dct_batch(x):
    """
    Compute a 2D DCT for each channel of each image in the batch.
    x: Tensor of shape (B, C, H, W)
    Returns: Tensor of same shape containing DCT coefficients.
    """
    B, C, H, W = x.shape
    dct_out = torch.zeros_like(x)
    for b in range(B):
        for c in range(C):
            dct_out[b, c] = dct2d(x[b, c])
    return dct_out

# # --------------------------
# # Main function: Load image, compute metrics, and plot results
# # --------------------------
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Load an example image (adjust path as needed) and normalize to [0, 1]
#     img_path = "example.jpg"  # replace with your image path
#     img = Image.open(img_path).convert("RGB")
#     img = img.resize((256, 256))  # resize for simplicity; adjust as needed
#     img = np.array(img).astype(np.float32) / 255.0
#     img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)  # shape: (1, 3, H, W)

    # # Compute DCT coefficients for the image batch
    # dct_coeffs = compute_dct_batch(img)

    # # Analyze frequency bands
    # freq_metrics = frequency_bands_analysis(dct_coeffs, device=device)
    # print("Frequency Metrics:")
    # for key, value in freq_metrics.items():
    #     print(f"{key}: {value:.4f}")

    # # For demonstration, assume we have a reconstruction from the VAE
    # # Here we simulate it by adding a small perturbation to the original image.
    # reconstructed = img + 0.05 * torch.randn_like(img)
    # reconstructed = torch.clamp(reconstructed, 0, 1)

    # # Compute PSNR between original and reconstructed images
    # image_psnr = psnr(img, reconstructed)
    # print(f"\nPSNR between original and reconstruction: {image_psnr:.2f} dB")

    # # Plot original and reconstructed images
    # img_np = img.squeeze(0).permute(1,2,0).cpu().numpy()
    # rec_np = reconstructed.squeeze(0).permute(1,2,0).cpu().numpy()

    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(img_np)
    # ax[0].set_title("Original")
    # ax[0].axis("off")
    # ax[1].imshow(rec_np)
    # ax[1].set_title("Reconstructed")
    # ax[1].axis("off")
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VAE checkpoints using SpectralAnalyzer")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory containing checkpoint folders")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing test images")
    parser.add_argument("--output_dir", type=str, default="spectral_analysis", help="Output directory")
    parser.add_argument("--image_size", type=int, default=256, help="Image size for evaluation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--with_tracking", action="store_true", help="Whether to track with wandb")
    parser.add_argument("--model_subfolder", type=str, default="", help="Subfolder within checkpoint where model is stored")
    parser.add_argument("--no_rename", action="store_true", help="Do not rename model files")
    args = parser.parse_args()
    
    evaluate_checkpoints(
        args.checkpoint_dir,
        args.data_dir,
        args.output_dir,
        args.image_size,
        args.batch_size,
        args.device,
        args.with_tracking,
        model_subfolder=args.model_subfolder,
        rename_files=not args.no_rename
    )
