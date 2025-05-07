import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchvision.models import inception_v3
from scipy import linalg
import numpy as np
from PIL import Image
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class SingleFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and 
                            f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

class PairedImagesDataset(Dataset):
    def __init__(self, real_folder, generated_folder, transform=None):
        self.real_folder = real_folder
        self.generated_folder = generated_folder
        self.transform = transform
        
        self.image_files = [f for f in os.listdir(real_folder) if os.path.isfile(os.path.join(real_folder, f)) and 
                            f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        real_img_path = os.path.join(self.real_folder, self.image_files[idx])
        gen_img_path = os.path.join(self.generated_folder, self.image_files[idx])
        
        real_image = Image.open(real_img_path).convert('RGB')
        
        # Check if corresponding generated image exists
        if os.path.exists(gen_img_path):
            gen_image = Image.open(gen_img_path).convert('RGB')
        else:
            # Try common variations in filenames
            base_name, ext = os.path.splitext(self.image_files[idx])
            alt_filenames = [
                f"{base_name}_generated{ext}",
                f"{base_name}_gen{ext}",
                f"gen_{base_name}{ext}"
            ]
            
            found = False
            for alt_name in alt_filenames:
                alt_path = os.path.join(self.generated_folder, alt_name)
                if os.path.exists(alt_path):
                    gen_image = Image.open(alt_path).convert('RGB')
                    found = True
                    break
            
            if not found:
                print(f"Warning: No matching generated image found for {self.image_files[idx]}")
                # Return a blank image or placeholder
                gen_image = Image.new('RGB', real_image.size, (0, 0, 0))
        
        if self.transform:
            real_image = self.transform(real_image)
            gen_image = self.transform(gen_image)
            
        return real_image, gen_image

def get_activations(images, model, batch_size=64, dims=2048, device='cuda'):
    model.eval()
    
    if len(images) % batch_size != 0:
        n_batches = len(images) // batch_size + 1
    else:
        n_batches = len(images) // batch_size
        
    n_used_imgs = n_batches * batch_size
    
    pred_arr = np.empty((n_used_imgs, dims))
    
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        
        batch = images[start:end]
        batch = batch.to(device)
        
        with torch.no_grad():
            pred = model(batch)[0]
        
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you don't use pretrained=True or if the model is resnet.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
            
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        
        pred_arr[start:end] = pred
    
    return pred_arr

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Frechet Distance between two multivariate Gaussians."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def calculate_fid(real_activations, generated_activations):
    # Calculate mean and covariance statistics
    mu_real = np.mean(real_activations, axis=0)
    sigma_real = np.cov(real_activations, rowvar=False)
    
    mu_gen = np.mean(generated_activations, axis=0)
    sigma_gen = np.cov(generated_activations, rowvar=False)
    
    # Calculate Frechet Distance
    fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    
    return fid_value

def calculate_nmse(real_images, generated_images):
    """Calculate Normalized Mean Squared Error."""
    mse = torch.mean((real_images - generated_images) ** 2, dim=[1, 2, 3])
    norm = torch.mean(real_images ** 2, dim=[1, 2, 3])
    nmse = torch.mean(mse / (norm + 1e-8))
    return nmse.item()

def calculate_psnr_batch(real_images, generated_images):
    """Calculate PSNR for a batch of images."""
    # Convert to numpy arrays in range [0, 255]
    real_np = real_images.cpu().numpy().transpose(0, 2, 3, 1) * 255
    gen_np = generated_images.cpu().numpy().transpose(0, 2, 3, 1) * 255
    
    psnr_values = []
    for i in range(real_np.shape[0]):
        psnr_value = psnr(real_np[i], gen_np[i], data_range=255)
        psnr_values.append(psnr_value)
    
    return np.mean(psnr_values)

def calculate_ssim_batch(real_images, generated_images):
    """Calculate SSIM for a batch of images."""
    # Convert to numpy arrays in range [0, 255]
    real_np = real_images.cpu().numpy().transpose(0, 2, 3, 1) * 255
    gen_np = generated_images.cpu().numpy().transpose(0, 2, 3, 1) * 255
    
    ssim_values = []
    for i in range(real_np.shape[0]):
        ssim_value = ssim(real_np[i], gen_np[i], multichannel=True, data_range=255)
        ssim_values.append(ssim_value)
    
    return np.mean(ssim_values)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate image quality metrics')
    parser.add_argument('--real', type=str, required=True, help='Path to real images folder')
    parser.add_argument('--generated', type=str, required=True, help='Path to generated images folder')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--metrics', type=str, default='all', 
                        help='Metrics to evaluate (comma-separated list of: fid,nmse,psnr,ssim,lpips)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if CUDA is available when device is set to 'cuda'
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    
    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # For metrics that don't require inception preprocessing
    basic_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Determine which metrics to compute
    metrics_to_compute = args.metrics.lower().split(',') if args.metrics != 'all' else ['fid', 'nmse', 'psnr', 'ssim', 'lpips']
    
    # Results dictionary
    results = {}
    
    # Calculate FID if requested
    if 'fid' in metrics_to_compute:
        print("Calculating FID...")
        
        # Load inception model
        inception_model = inception_v3(pretrained=True, transform_input=False)
        inception_model.fc = nn.Identity()  # Remove final fully connected layer
        inception_model = inception_model.to(device)
        
        # Create datasets and loaders
        real_dataset = SingleFolderDataset(args.real, transform=transform)
        real_loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        generated_dataset = SingleFolderDataset(args.generated, transform=transform)
        generated_loader = DataLoader(generated_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # Get activations
        real_activations = []
        for batch in real_loader:
            batch = batch.to(device)
            with torch.no_grad():
                activations = inception_model(batch)[0]
                activations = activations.squeeze(-1).squeeze(-1).cpu().numpy()
                real_activations.append(activations)
        real_activations = np.vstack(real_activations)
        
        generated_activations = []
        for batch in generated_loader:
            batch = batch.to(device)
            with torch.no_grad():
                activations = inception_model(batch)[0]
                activations = activations.squeeze(-1).squeeze(-1).cpu().numpy()
                generated_activations.append(activations)
        generated_activations = np.vstack(generated_activations)
        
        # Calculate FID
        fid_value = calculate_fid(real_activations, generated_activations)
        results['FID'] = fid_value
        print(f"FID: {fid_value:.4f}")
    
    # For NMSE, PSNR, SSIM, and LPIPS, we need paired images
    if any(m in metrics_to_compute for m in ['nmse', 'psnr', 'ssim', 'lpips']):
        paired_dataset = PairedImagesDataset(args.real, args.generated, transform=basic_transform)
        paired_loader = DataLoader(paired_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # Initialize counters for averaged metrics
        total_nmse = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        total_lpips = 0.0
        num_batches = 0
        
        # Initialize LPIPS model if needed
        if 'lpips' in metrics_to_compute:
            lpips_model = lpips.LPIPS(net='alex').to(device)
        
        # Process each batch
        for real_batch, gen_batch in paired_loader:
            real_batch = real_batch.to(device)
            gen_batch = gen_batch.to(device)
            
            if 'nmse' in metrics_to_compute:
                nmse_value = calculate_nmse(real_batch, gen_batch)
                total_nmse += nmse_value
            
            if 'psnr' in metrics_to_compute:
                psnr_value = calculate_psnr_batch(real_batch, gen_batch)
                total_psnr += psnr_value
            
            if 'ssim' in metrics_to_compute:
                ssim_value = calculate_ssim_batch(real_batch, gen_batch)
                total_ssim += ssim_value
            
            if 'lpips' in metrics_to_compute:
                with torch.no_grad():
                    lpips_value = lpips_model(real_batch, gen_batch).mean().item()
                total_lpips += lpips_value
            
            num_batches += 1
        
        # Calculate averages
        if 'nmse' in metrics_to_compute:
            avg_nmse = total_nmse / num_batches
            results['NMSE'] = avg_nmse
            print(f"NMSE: {avg_nmse:.4f}")
        
        if 'psnr' in metrics_to_compute:
            avg_psnr = total_psnr / num_batches
            results['PSNR'] = avg_psnr
            print(f"PSNR: {avg_psnr:.4f} dB")
        
        if 'ssim' in metrics_to_compute:
            avg_ssim = total_ssim / num_batches
            results['SSIM'] = avg_ssim
            print(f"SSIM: {avg_ssim:.4f}")
        
        if 'lpips' in metrics_to_compute:
            avg_lpips = total_lpips / num_batches
            results['LPIPS'] = avg_lpips
            print(f"LPIPS: {avg_lpips:.4f}")
    
    # Print summary of all results
    print("\n--- Summary of Results ---")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()