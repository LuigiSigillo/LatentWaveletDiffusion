import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
import os
from glob import glob
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mean_squared_error
from scipy.stats import wasserstein_distance
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_wavelets import DWTForward, DWTInverse
from tqdm import tqdm
import logging # Import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom collate function (if you decide to use it later)
# def collate_skip_none(batch):
#     batch = [item for item in batch if item is not None]
#     if not batch:
#         return None 
#     return torch.utils.data.dataloader.default_collate(batch)

class ImageDataset(Dataset):
    """Dataset for loading and preprocessing images for frequency analysis"""
    
    def __init__(self, directory, img_size=(256, 256)):
        """
        Initialize dataset with images from a directory.
        
        Args:
            directory (str): Directory containing images
            img_size (tuple): Target size for resizing images
        """
        self.image_paths = []
        for root, _, files in os.walk(directory):
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                self.image_paths.extend(glob(os.path.join(root, ext)))
                          
        if not self.image_paths:
            raise ValueError(f"No images found in {directory}")  
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
        
        self.img_size = img_size
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        max_attempts = 5  # Limit recursive attempts to avoid infinite loops
        return self._get_item_with_retry(idx, attempts=0, max_attempts=max_attempts)
    
    def _get_item_with_retry(self, idx, attempts=0, max_attempts=5):
        """Helper method to get an item with retry logic to avoid infinite recursion"""
        if attempts >= max_attempts:
            # If too many attempts, return a simple valid tensor
            print(f"Too many failed attempts to load images, returning dummy tensor")
            return torch.zeros((1, *self.img_size), dtype=torch.float32)
            
        img_path = self.image_paths[idx]
        
        # Load image
        try:
            img = imread(img_path)
            
            # Check if image is valid
            if img is None or img.size == 0:
                raise ValueError(f"Empty image data from {img_path}")
                
        except Exception as e:
            # print(f"Error loading image {img_path}: {e}")
            # Try a different image instead of recursive call
            new_idx = np.random.randint(0, len(self.image_paths))
            return self._get_item_with_retry(new_idx, attempts + 1, max_attempts)
            
        # Handle images with different channel counts
        try:
            if len(img.shape) > 2:
                if img.shape[2] == 4:  # RGBA image
                    # Drop alpha channel
                    img = img[:, :, :3]
                # Convert RGB to grayscale
                if img.shape[2] >= 3:
                    img = rgb2gray(img)
                    
            # Ensure image is 2D (grayscale)
            if len(img.shape) != 2:
                print(f"Unexpected image shape after conversion: {img.shape} for {img_path}")
                new_idx = np.random.randint(0, len(self.image_paths))
                return self._get_item_with_retry(new_idx, attempts + 1, max_attempts)
                
            # Resize and convert to tensor (now guaranteed to be grayscale)
            img_tensor = self.transform(img.astype(np.float32))
            
            # Ensure tensor has 1 channel
            if img_tensor.shape[0] > 1:
                img_tensor = img_tensor[0].unsqueeze(0)
                
            return img_tensor
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            new_idx = np.random.randint(0, len(self.image_paths))
            return self._get_item_with_retry(new_idx, attempts + 1, max_attempts)
            
class FrequencyAnalyzer:
    """
    A class to analyze frequency characteristics of images, including:
    - Power Spectral Density (PSD) via 2D Fourier Transform
    - Wavelet-based energy preservation using PyTorch
    """
    
    def __init__(self, img_size=(256, 256), batch_size=16):
        """
        Initialize the analyzer with a standard image size.
        
        Args:
            img_size (tuple): Target image size for analysis
            batch_size (int): Batch size for processing images
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def load_and_preprocess(self, img_path):
        """
        Load an image and preprocess it for analysis.
        
        Args:
            img_path (str): Path to the image
            
        Returns:
            np.ndarray: Preprocessed grayscale image
        """
        # Load image
        img = imread(img_path)
        
        # Handle grayscale vs RGB
        if len(img.shape) > 2 and img.shape[2] == 3:
            img = rgb2gray(img)
        
        # Resize image
        img = resize(img, self.img_size, anti_aliasing=True, mode='reflect')
        
        return img
    
    def compute_psd_2d(self, img):
        """
        Compute 2D Power Spectral Density.
        
        Args:
            img (np.ndarray): Input grayscale image
            
        Returns:
            np.ndarray: 2D PSD
        """
        # Apply window function to reduce edge effects
        window = np.outer(np.hanning(img.shape[0]), np.hanning(img.shape[1]))
        img_windowed = img * window
        
        # Compute 2D FFT
        f_transform = np.fft.fft2(img_windowed)
        # Shift zero frequency to center
        f_transform_shifted = np.fft.fftshift(f_transform)
        # Compute power spectrum
        psd_2d = np.abs(f_transform_shifted)**2
        
        return psd_2d
    
    def radial_average(self, psd_2d):
        """
        Perform radial averaging of 2D PSD to get 1D frequency profile.
        
        Args:
            psd_2d (np.ndarray): 2D PSD array
            
        Returns:
            tuple: (frequency bins, radially averaged PSD)
        """
        # Get image dimensions
        h, w = psd_2d.shape
        
        # Create coordinate grid
        y, x = np.indices(psd_2d.shape)
        center_y, center_x = h // 2, w // 2
        
        # Calculate distance from center for each pixel
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        r = r.astype(int)
        
        # Maximum radius is half the minimum dimension
        max_radius = min(center_x, center_y)
        
        # Initialize arrays to store values
        radial_prof = np.zeros(max_radius)
        bin_counts = np.zeros(max_radius)
        
        # Sum values in each radial bin
        for i in range(h):
            for j in range(w):
                rad = r[i, j]
                if rad < max_radius:
                    radial_prof[rad] += psd_2d[i, j]
                    bin_counts[rad] += 1
        
        # Avoid division by zero
        bin_counts[bin_counts == 0] = 1
        
        # Average values in each bin
        radial_prof /= bin_counts
        
        # Create frequency bins (cycles/pixel)
        freq_bins = np.linspace(0, 0.5, max_radius)  # Nyquist frequency is 0.5 cycles/pixel
        
        return freq_bins, radial_prof
    
    def analyze_psd(self, img):
        """
        Analyze the Power Spectral Density of an image.
        
        Args:
            img (np.ndarray): Input grayscale image
            
        Returns:
            tuple: (frequency bins, radially averaged PSD)
        """
        psd_2d = self.compute_psd_2d(img)
        return self.radial_average(psd_2d)
    
    def compute_wavelet_energy(self, img):
        """
        Compute wavelet-based energy ratios using PyTorch Wavelets.
        
        Args:
            img (np.ndarray): Input grayscale image
            
        Returns:
            dict: Dictionary containing wavelet energy metrics
        """
        try:
            # Check the input image for proper size and content
            if img.shape[0] < 4 or img.shape[1] < 4:
                print(f"Image too small for wavelet transform: {img.shape}")
                raise ValueError("Image too small for wavelet")
                
            # Ensure image has even dimensions (required by some wavelet implementations)
            h, w = img.shape
            if h % 2 != 0 or w % 2 != 0:
                # Pad to even dimensions
                h_pad = 0 if h % 2 == 0 else 1
                w_pad = 0 if w % 2 == 0 else 1
                img = np.pad(img, ((0, h_pad), (0, w_pad)), mode='reflect')
            
            # Check for NaN or Inf values
            if np.isnan(img).any() or np.isinf(img).any():
                print("Image contains NaN or Inf values")
                # Replace problematic values
                img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
                
            # Check value range
            if img.min() < -10 or img.max() > 10:
                print(f"Unusual image value range: min={img.min()}, max={img.max()}")
                # Normalize to 0-1 range
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                
            # Convert numpy array to PyTorch tensor
            img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
            
            # Make sure tensor values are valid for computation
            if torch.isnan(img_tensor).any() or torch.isinf(img_tensor).any():
                img_tensor = torch.nan_to_num(img_tensor, nan=0.0, posinf=1.0, neginf=0.0)
                
            # Initialize wavelet transform with more robust parameters
            # Using db1 (same as Haar) but can be more stable in some implementations
            dwt = DWTForward(J=1, wave='haar')
            
            # Move to GPU if available
            img_tensor = img_tensor.to(self.device)
            dwt = dwt.to(self.device)
            
            # Apply wavelet transform
            # Returns a tuple: (LL, [LH, HL, HH])
            coeffs = dwt(img_tensor)
            
            # Check if coeffs tuple is valid and contains at least one element
            if not coeffs or len(coeffs) == 0:
                print("Warning: Wavelet transform returned empty or invalid coefficients.")
                raise ValueError("Invalid wavelet coefficients")

            # Extract coefficients - safely handle potential dimension issues
            LL = coeffs[0]  # Low-low component

            # Check if high frequency components exist in the tuple
            if len(coeffs) > 1 and coeffs[1] is not None:
                # Check if the high frequency list itself is not empty
                if len(coeffs[1]) > 0:
                    # Extract high frequency components with proper handling of the shape
                    # The pytorch_wavelets package returns [1, 1, 3, H, W] for high frequency components
                    # where the 3 channels are LH, HL, HH
                    high_freq = coeffs[1][0]
                    
                    # Handle the specific shape we're seeing: [1, 1, 3, H, W]
                    if high_freq.dim() == 5 and high_freq.shape[2] == 3:
                        # Extract the three components (LH, HL, HH) from the 3rd dimension
                        LH = high_freq[:, :, 0, :, :]
                        HL = high_freq[:, :, 1, :, :]
                        HH = high_freq[:, :, 2, :, :]
                        
                        # # Remove extra dimensions to match LL shape
                        # LH = LH.reshape(LH.shape[0], LH.shape[3], LH.shape[4])
                        # HL = HL.reshape(HL.shape[0], HL.shape[3], HL.shape[4])
                        # HH = HH.reshape(HH.shape[0], HH.shape[3], HH.shape[4])
                        
                        # # Add channel dimension to match LL shape
                        # LH = LH.unsqueeze(1)
                        # HL = HL.unsqueeze(1)
                        # HH = HH.unsqueeze(1)
                    # Handle the standard expected shape [B, C, H, W]
                    elif high_freq.dim() == 4 and high_freq.shape[1] >= 3:
                        LH, HL, HH = high_freq[:, 0:1, :, :], high_freq[:, 1:2, :, :], high_freq[:, 2:3, :, :]
                    else:
                        # For any other unexpected shape, log it once and create placeholder tensors
                        if not hasattr(self, '_logged_shape_warning'):
                            print(f"Unexpected high frequency components shape: {high_freq.shape}")
                            print("This message will only be shown once.")
                            self._logged_shape_warning = True
                        
                        # Create placeholder components
                        LH = torch.zeros_like(LL)
                        HL = torch.zeros_like(LL)
                        HH = torch.zeros_like(LL)
                    
                    # Compute energy in each component
                    ll_energy = torch.sum(LL**2).item()
                    lh_energy = torch.sum(LH**2).item()
                    hl_energy = torch.sum(HL**2).item()
                    hh_energy = torch.sum(HH**2).item()
                    
                    # Total high-frequency energy
                    high_freq_energy = lh_energy + hl_energy + hh_energy
                    
                    # Ratio of high to low frequency energy with safeguards
                    energy_ratio = high_freq_energy / ll_energy if ll_energy > 1e-8 else 0
                    
                    return {
                        'LL_energy': ll_energy,
                        'LH_energy': lh_energy, 
                        'HL_energy': hl_energy,
                        'HH_energy': hh_energy,
                        'high_freq_energy': high_freq_energy,
                        'energy_ratio': energy_ratio
                    }
                else:
                    # Handle case where coeffs[1] is an empty list
                    print("Warning: Wavelet transform returned empty high frequency component list.")
                    if LL is not None:
                        ll_energy = torch.sum(LL**2).item()
                    else:
                        ll_energy = 0.0
                    # Fallback to FFT approximation or return zeros
            else:
                # Handle case where coeffs tuple only has LL component (len(coeffs) <= 1)
                print("Warning: Wavelet transform did not return high frequency components tuple element.")
                if LL is not None:
                    ll_energy = torch.sum(LL**2).item()
                else:
                    ll_energy = 0.0
                # Fallback to FFT approximation or return zeros

            # Fallback logic (moved from inner else block to handle all missing HF cases)
            try:
                # Compute a simple approximation of high frequency energy using FFT
                fft = torch.fft.fft2(img_tensor.squeeze())
                fft_shift = torch.fft.fftshift(fft)
                h, w = fft_shift.shape
                
                # Create a simple approximation of wavelet decomposition using frequency regions
                # Low freq is center region, high freq is other regions
                center_h, center_w = h // 2, w // 2
                mask_lh = torch.zeros((h, w), device=self.device)
                mask_lh[:center_h, center_w:] = 1.0  # top-right quadrant
                
                mask_hl = torch.zeros((h, w), device=self.device)
                mask_hl[center_h:, :center_w] = 1.0  # bottom-left quadrant
                
                mask_hh = torch.zeros((h, w), device=self.device)
                mask_hh[center_h:, center_w:] = 1.0  # bottom-right quadrant
                
                # Apply masks
                lh_approx = torch.sum((torch.abs(fft_shift) * mask_lh)**2).item()
                hl_approx = torch.sum((torch.abs(fft_shift) * mask_hl)**2).item()
                hh_approx = torch.sum((torch.abs(fft_shift) * mask_hh)**2).item()
                
                high_freq_energy = lh_approx + hl_approx + hh_approx
                energy_ratio = high_freq_energy / ll_energy if ll_energy > 1e-8 else 0
                
                return {
                    'LL_energy': ll_energy,
                    'LH_energy': lh_approx,
                    'HL_energy': hl_approx,
                    'HH_energy': hh_approx,
                    'high_freq_energy': high_freq_energy,
                    'energy_ratio': energy_ratio
                }
            except Exception as e_fft:
                print(f"Failed to approximate wavelet components via FFT: {e_fft}")
                return {
                    'LL_energy': ll_energy,
                    'LH_energy': 0.0,
                    'HL_energy': 0.0,
                    'HH_energy': 0.0,
                    'high_freq_energy': 0.0,
                    'energy_ratio': 0.0
                }
        except Exception as e:
            print(f"Error in wavelet computation: {e}")
            # Return a default set of values if computation fails
            return {
                'LL_energy': 0.0,
                'LH_energy': 0.0, 
                'HL_energy': 0.0,
                'HH_energy': 0.0,
                'high_freq_energy': 0.0,
                'energy_ratio': 0.0
            }

    def analyze_directory(self, dir_path, method_name=None):
        """
        Analyze all images in a directory and compute average metrics using PyTorch and batched processing
        with running averages to save memory.
        
        Args:
            dir_path (str): Path to directory containing images
            method_name (str, optional): Name of the method for labeling
            
        Returns:
            dict: Dictionary containing average metrics for the directory
        """
        # Create dataset and dataloader
        try:
            dataset = ImageDataset(dir_path, self.img_size)
            # Consider using collate_fn=collate_skip_none if image loading errors are frequent
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4) 
        except ValueError as e:
            logging.error(f"Error creating dataset for {dir_path}: {e}")
            raise
        
        # Initialize accumulators for running averages
        psd_sum = None
        wavelet_sum = None
        processed_count = 0
        freq_bins = None
        
        # Default wavelet metrics structure (initialized when first valid result is found)
        default_wavelet_keys = ['LL_energy', 'LH_energy', 'HL_energy', 'HH_energy', 'high_freq_energy', 'energy_ratio']

        # Process images in batches
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Analyzing {method_name or os.path.basename(dir_path)}"):
            try:
                # Skip empty batches (can happen with collate_skip_none or if dataset is empty)
                if batch is None or batch.size(0) == 0:
                    logging.warning(f"Skipping empty or invalid batch {batch_idx}")
                    continue
                    
                # Move batch to device
                batch = batch.to(self.device)
                
                # Process each image in the batch
                for i in range(batch.size(0)):
                    img_tensor = None # Ensure img_tensor is defined in this scope
                    try:
                        img_tensor = batch[i]
                        
                        # Check tensor dimensions
                        if img_tensor.dim() != 3 or img_tensor.size(0) != 1:
                            logging.warning(f"Skipping malformed tensor with shape: {img_tensor.shape}")
                            continue
                        
                        # Convert to numpy for PSD analysis
                        img_np = img_tensor.squeeze().cpu().numpy()
                        
                        # Check for NaN or Inf values
                        if np.isnan(img_np).any() or np.isinf(img_np).any():
                            logging.warning("Skipping image with NaN or Inf values")
                            continue
                        
                        # --- Analyze PSD ---
                        current_freq_bins = None
                        radial_psd = None
                        try:
                            current_freq_bins, radial_psd = self.analyze_psd(img_np)
                            # Store freq_bins only once
                            if freq_bins is None:
                                freq_bins = current_freq_bins
                            # Initialize or add to psd_sum
                            if psd_sum is None:
                                psd_sum = radial_psd.copy() # Important: copy the array
                            else:
                                # Ensure shapes match before adding
                                if psd_sum.shape == radial_psd.shape:
                                    psd_sum += radial_psd
                                else:
                                     logging.warning(f"PSD shape mismatch: expected {psd_sum.shape}, got {radial_psd.shape}. Skipping PSD update for this image.")
                                     # Don't update PSD sum, but might continue with wavelet if needed
                                     # Or decide to skip the image entirely: continue 
                        except Exception as e:
                            logging.error(f"Error in PSD analysis for an image in batch {batch_idx}: {e}")
                            continue # Skip this image entirely if PSD fails

                        # --- Compute wavelet energy ---
                        wavelet_metrics = None
                        try:
                            wavelet_metrics = self.compute_wavelet_energy(img_np)
                            # Initialize wavelet_sum if first valid result
                            if wavelet_sum is None:
                                wavelet_sum = {key: 0.0 for key in default_wavelet_keys}
                            # Add metrics to sum
                            for key in default_wavelet_keys:
                                if key in wavelet_metrics:
                                     wavelet_sum[key] += wavelet_metrics[key]
                                else:
                                     logging.warning(f"Key '{key}' not found in wavelet_metrics for an image.")

                        except Exception as e:
                            logging.error(f"Error in wavelet analysis for an image in batch {batch_idx}: {e}")
                            # If wavelet fails, we already added PSD, but don't increment count
                            continue # Skip incrementing count for this image

                        # --- Increment count only if both PSD and Wavelet succeeded ---
                        processed_count += 1
                    
                    except Exception as e:
                        # Catch errors processing a single image within the batch
                        logging.error(f"Error processing image {i} in batch {batch_idx}: {e}")
                        # Continue to the next image in the batch
            
            except Exception as e:
                # Catch errors related to the whole batch (e.g., moving to device)
                logging.error(f"Critical error processing batch {batch_idx}: {e}")
                # Continue to the next batch

        # --- Calculate final averages ---
        avg_psd = None
        avg_wavelet = None

        if processed_count > 0:
            if psd_sum is not None:
                 avg_psd = psd_sum / processed_count
            else:
                 logging.warning(f"Processed {processed_count} images but PSD sum is None for {dir_path}.")
            
            if wavelet_sum is not None:
                 avg_wavelet = {key: value / processed_count for key, value in wavelet_sum.items()}
            else:
                 logging.warning(f"Processed {processed_count} images but Wavelet sum is None for {dir_path}.")

        # Handle cases where no images were processed successfully
        if processed_count == 0:
            logging.warning(f"WARNING: No valid images processed successfully from directory: {dir_path}")
            # Use default values
            freq_bins = np.linspace(0, 0.5, self.img_size[0] // 2) # Default based on img_size
            avg_psd = np.zeros_like(freq_bins)
            avg_wavelet = {key: 0.0 for key in default_wavelet_keys}
        elif freq_bins is None: # Handle case where count > 0 but freq_bins wasn't set (shouldn't happen ideally)
             logging.warning("Processed images but failed to capture frequency bins. Using default.")
             freq_bins = np.linspace(0, 0.5, self.img_size[0] // 2) 
             if avg_psd is None: # If PSD sum was also None
                 avg_psd = np.zeros_like(freq_bins)

        # Ensure avg_wavelet is initialized if it's still None
        if avg_wavelet is None:
             avg_wavelet = {key: 0.0 for key in default_wavelet_keys}
             
        logging.info(f"Successfully analyzed {processed_count} images from {dir_path}")
        
        return {
            'method_name': method_name if method_name else os.path.basename(dir_path),
            'freq_bins': freq_bins,
            'avg_psd': avg_psd,
            'wavelet_metrics': avg_wavelet
        }
    
    def compare_methods(self, real_data_dir, method_dirs):
        """
        Compare frequency characteristics of different methods to real data.
        
        Args:
            real_data_dir (str): Directory with real images
            method_dirs (dict): Dictionary mapping method names to directories
            
        Returns:
            dict: Comparison results
        """
        # Analyze real data
        real_results = self.analyze_directory(real_data_dir, 'Real')
        
        # Analyze each method
        method_results = {}
        for method_name, dir_path in method_dirs.items():
            method_results[method_name] = self.analyze_directory(dir_path, method_name)
        
        # Compare methods to real data
        comparison = {}
        for method_name, results in method_results.items():
            # PSD similarity metrics
            psd_mse = mean_squared_error(real_results['avg_psd'], results['avg_psd'])
            psd_emd = wasserstein_distance(real_results['avg_psd'], results['avg_psd'])
            
            # Wavelet energy comparison
            real_energy_ratio = real_results['wavelet_metrics']['energy_ratio']
            method_energy_ratio = results['wavelet_metrics']['energy_ratio']
            energy_ratio_diff = method_energy_ratio / real_energy_ratio if real_energy_ratio > 0 else 0
            
            comparison[method_name] = {
                'psd_mse': psd_mse,
                'psd_emd': psd_emd,  # Earth Mover's Distance (Wasserstein)
                'energy_ratio': method_energy_ratio,
                'energy_ratio_vs_real': energy_ratio_diff
            }
        
        return {
            'real_results': real_results,
            'method_results': method_results,
            'comparison': comparison
        }
    
    def plot_spectral_analysis(self, comparison_results, save_path=None, log_scale=True):
        """
        Plot the spectral analysis results.
        
        Args:
            comparison_results (dict): Results from compare_methods
            save_path (str, optional): Path to save the figure
            log_scale (bool): Whether to use log scale for PSD
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        plt.figure(figsize=(10, 6))
        
        real_results = comparison_results['real_results']
        method_results = comparison_results['method_results']
        
        # Plot real data PSD
        freq_bins = real_results['freq_bins']
        plt.plot(freq_bins, real_results['avg_psd'], 'k-', linewidth=2, label='Real')
        
        # Plot each method's PSD
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        for i, (method_name, results) in enumerate(method_results.items()):
            color = colors[i % len(colors)]
            plt.plot(freq_bins, results['avg_psd'], color=color, linewidth=1.5, label=method_name)
        
        plt.xlabel('Spatial Frequency (cycles/pixel)')
        plt.ylabel('Power Spectral Density')
        
        if log_scale:
            plt.yscale('log')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title('Average Power Spectral Density Comparison')
        
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
        
        return plt.gcf()
    
    def create_wavelet_energy_table(self, comparison_results):
        """
        Create a table of wavelet energy metrics.
        
        Args:
            comparison_results (dict): Results from compare_methods
            
        Returns:
            pandas.DataFrame: Table of wavelet energy metrics
        """
        data = []
        
        # Add real data first
        real_metrics = comparison_results['real_results']['wavelet_metrics']
        real_row = {
            'Method': 'Real',
            'Low Freq Energy (LL²)': real_metrics['LL_energy'],
            'High Freq Energy (LH²+HL²+HH²)': real_metrics['high_freq_energy'],
            'Energy Ratio': real_metrics['energy_ratio'],
            'vs Real': 1.0  # Reference point
        }
        data.append(real_row)
        
        # Add each method
        for method_name, comparison in comparison_results['comparison'].items():
            method_metrics = comparison_results['method_results'][method_name]['wavelet_metrics']
            method_row = {
                'Method': method_name,
                'Low Freq Energy (LL²)': method_metrics['LL_energy'],
                'High Freq Energy (LH²+HL²+HH²)': method_metrics['high_freq_energy'],
                'Energy Ratio': method_metrics['energy_ratio'],
                'vs Real': comparison['energy_ratio_vs_real']
            }
            data.append(method_row)
        
        df = pd.DataFrame(data)
        return df


# Example usage
def main(paths_dict_img_generated,real_dir=None):
    # Create output directories if they don't exist
    figures_path = os.path.join(os.path.dirname(paths_dict_img_generated['WALD']),'figures')
    results_path = os.path.join(os.path.dirname(paths_dict_img_generated['WALD']),'results')
    os.makedirs(figures_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    
    # Initialize analyzer with GPU acceleration
    analyzer = FrequencyAnalyzer(img_size=(256, 256), batch_size=16)
    
    # Define directories for each method
    # real_dir = '/leonardo_scratch/large/userexternal/lsigillo/laion_high_res_images_2K'

    # Add frequency band analysis
    def analyze_frequency_bands(comparison_results):
        """Analyze the PSD in specific frequency bands"""
        print("\nFrequency Band Analysis:")
        
        # Define frequency bands (in cycles/pixel)
        bands = {
            'Low': (0.0, 0.1),
            'Mid': (0.1, 0.3),
            'High': (0.3, 0.5)
        }
        
        real_psd = comparison_results['real_results']['avg_psd']
        freq_bins = comparison_results['real_results']['freq_bins']
        
        band_results = {}
        
        # For each method
        for method_name, results in comparison_results['method_results'].items():
            method_psd = results['avg_psd']
            band_results[method_name] = {}
            
            # For each frequency band
            for band_name, (low, high) in bands.items():
                # Find indices for this band
                indices = np.where((freq_bins >= low) & (freq_bins <= high))[0]
                
                if len(indices) == 0:
                    continue
                
                # Extract PSD values for this band
                real_band_psd = real_psd[indices]
                method_band_psd = method_psd[indices]
                
                # Calculate metrics
                band_mse = mean_squared_error(real_band_psd, method_band_psd)
                band_emd = wasserstein_distance(real_band_psd, method_band_psd)
                
                # Store results
                band_results[method_name][band_name] = {
                    'MSE': band_mse,
                    'EMD': band_emd
                }
        
        # Print results
        for band_name in bands.keys():
            print(f"\n{band_name} Frequency Band (MSE, lower is better):")
            for method_name in band_results.keys():
                if band_name in band_results[method_name]:
                    print(f"  {method_name}: {band_results[method_name][band_name]['MSE']:.6f}")
        
        return band_results
    
    # Compare methods with GPU-accelerated processing
    print("Starting analysis of image directories...")
    comparison_results = analyzer.compare_methods(real_dir, paths_dict_img_generated)
    
    # Plot spectral analysis
    print("Generating spectral analysis plot...")
    analyzer.plot_spectral_analysis(comparison_results, save_path=os.path.join(figures_path,'spectral_analysis.pdf'))
    
    # Create wavelet energy table
    wavelet_table = analyzer.create_wavelet_energy_table(comparison_results)
    print("\nWavelet Energy Analysis:")
    print(wavelet_table)
    
    # Save the table
    wavelet_table.to_csv(os.path.join(results_path,'wavelet_energy.csv'), index=False)
    
    # Print percentage comparisons for paper - only if methods are available
    real_ratio = comparison_results['real_results']['wavelet_metrics']['energy_ratio']
    wald_ratio = comparison_results['method_results']['WALD']['wavelet_metrics']['energy_ratio']
    
    # # Check if LDM and YODA methods are in the results
    # print(f"\nPercentage Comparisons:")
    # if 'LDM' in comparison_results['method_results']:
    #     ldm_ratio = comparison_results['method_results']['LDM']['wavelet_metrics']['energy_ratio']
    #     wald_vs_ldm = (wald_ratio / ldm_ratio - 1) * 100 if ldm_ratio > 0 else 0
    #     print(f"WALD preserves {wald_vs_ldm:.1f}% more high-frequency energy compared to LDM")
    # else:
    #     print("LDM method not available for comparison")
        
    # if 'YODA' in comparison_results['method_results']:
    #     yoda_ratio = comparison_results['method_results']['YODA']['wavelet_metrics']['energy_ratio']
    #     wald_vs_yoda = (wald_ratio / yoda_ratio - 1) * 100 if yoda_ratio > 0 else 0
    #     print(f"WALD preserves {wald_vs_yoda:.1f}% more high-frequency energy compared to YODA")
    # else:
    #     print("YODA method not available for comparison")
    
    # Print PSD similarity metrics
    print("\nPSD Similarity Metrics (lower is better):")
    for method, metrics in comparison_results['comparison'].items():
        print(f"{method}: MSE = {metrics['psd_mse']:.6f}, EMD = {metrics['psd_emd']:.6f}")
        
    # Perform frequency band analysis
    band_results = analyze_frequency_bands(comparison_results)
    
    # Add perceptual correlation analysis
    print("\nPerceptual Correlation Analysis:")
    print("Note: For a complete perceptual analysis, consider implementing LPIPS or MS-SSIM")
    print("These metrics would provide better correlation with human perception")
    
    print("\nAnalysis complete! Results saved to figures/ and results/ directories.")


if __name__ == "__main__":
        
    method_dirs = {
        'WALD': "/mnt/share/Luigi/Documents/URAE/src/output/URAE_VAE_SE_WAV_ATT_LAION",
        # 'LDM': './data/ldm_images',
        # 'YODA': './data/yoda_images',
        'URAE': '/mnt/share/Luigi/Documents/URAE/src/output/URAE_original_trained_by_me'
    }

    real_dir = "/mnt/share/Luigi/Documents/URAE/dataset/laion_high_resolution_images"    
    main(
        method_dirs,
        real_dir
    )