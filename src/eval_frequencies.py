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
import logging
import random # Add this import

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# class ImageDataset(Dataset):
#     """Dataset for loading and preprocessing images for frequency analysis"""
#     def __init__(self, directory, img_size=(256, 256), max_images=None):
#         """
#         Initialize dataset with images from a directory.
#         Args:
#             directory (str): Directory containing images
#             img_size (tuple): Target size for resizing images
#             max_images (int, optional): Maximum number of images to load
#         """
#         self.image_paths = []
#         for root, _, files in os.walk(directory):
#             for ext in ['*.jpg', '*.png', '*.jpeg']:
#                 self.image_paths.extend(glob(os.path.join(root, ext)))
#         if not self.image_paths:
#             raise ValueError(f"No images found in {directory}")
            
#         # Limit the number of images if max_images is specified
#         if max_images is not None and max_images > 0:
#             self.image_paths = self.image_paths[:max_images]

#         self.transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize(img_size),
#             transforms.ToTensor(),
#         ])
#         self.img_size = img_size

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         max_attempts = 5  # Limit recursive attempts to avoid infinite loops
#         return self._get_item_with_retry(idx, attempts=0, max_attempts=max_attempts)

#     def _get_item_with_retry(self, idx, attempts=0, max_attempts=5):
#         """Helper method to get an item with retry logic to avoid infinite loops"""
#         if attempts >= max_attempts:
#             print(f"Too many failed attempts to load images, returning dummy tensor")
#             return torch.zeros((1, *self.img_size), dtype=torch.float32)
#         img_path = self.image_paths[idx]
#         try:
#             img = imread(img_path)
#             if img is None or img.size == 0:
#                 raise ValueError(f"Empty image data from {img_path}")
#         except Exception as e:
#             return self._get_item_with_retry(np.random.randint(0, len(self.image_paths)), attempts + 1, max_attempts)

#         try:
#             if len(img.shape) > 2:
#                 if img.shape[2] == 4:
#                     img = img[:, :, :3]
#                 if img.shape[2] >= 3:
#                     img = rgb2gray(img)
#             if len(img.shape) != 2:
#                 return self._get_item_with_retry(np.random.randint(0, len(self.image_paths)), attempts + 1, max_attempts)

#             img_tensor = self.transform(img.astype(np.float32))
#             if img_tensor.shape[0] > 1:
#                 img_tensor = img_tensor[0].unsqueeze(0)
#             return img_tensor
#         except Exception as e:
#             return self._get_item_with_retry(np.random.randint(0, len(self.image_paths)), attempts + 1, max_attempts)
           
# class FrequencyAnalyzer:
#     """
#     A class to analyze frequency characteristics of images, including:
#     - Power Spectral Density (PSD) via 2D Fourier Transform
#     - Wavelet-based energy preservation using PyTorch
#     """
    
#     def __init__(self, img_size=(256, 256), batch_size=16):
#         """
#         Initialize the analyzer with a standard image size.
        
#         Args:
#             img_size (tuple): Target image size for analysis
#             batch_size (int): Batch size for processing images
#         """
#         self.img_size = img_size
#         self.batch_size = batch_size
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Using device: {self.device}")
    
#     def load_and_preprocess(self, img_path):
#         """
#         Load an image and preprocess it for analysis.
        
#         Args:
#             img_path (str): Path to the image
            
#         Returns:
#             np.ndarray: Preprocessed grayscale image
#         """
#         # Load image
#         img = imread(img_path)
        
#         # Handle grayscale vs RGB
#         if len(img.shape) > 2 and img.shape[2] == 3:
#             img = rgb2gray(img)
        
#         # Resize image
#         img = resize(img, self.img_size, anti_aliasing=True, mode='reflect')
        
#         return img
    
#     def compute_psd_2d(self, img):
#         """
#         Compute 2D Power Spectral Density.
        
#         Args:
#             img (np.ndarray): Input grayscale image
            
#         Returns:
#             np.ndarray: 2D PSD
#         """
#         # Apply window function to reduce edge effects
#         window = np.outer(np.hanning(img.shape[0]), np.hanning(img.shape[1]))
#         img_windowed = img * window
        
#         # Compute 2D FFT
#         f_transform = np.fft.fft2(img_windowed)
#         # Shift zero frequency to center
#         f_transform_shifted = np.fft.fftshift(f_transform)
#         # Compute power spectrum
#         psd_2d = np.abs(f_transform_shifted)**2
        
#         return psd_2d
    
#     def radial_average(self, psd_2d):
#         """
#         Perform radial averaging of 2D PSD to get 1D frequency profile.
        
#         Args:
#             psd_2d (np.ndarray): 2D PSD array
            
#         Returns:
#             tuple: (frequency bins, radially averaged PSD)
#         """
#         # Get image dimensions
#         h, w = psd_2d.shape
        
#         # Create coordinate grid
#         y, x = np.indices(psd_2d.shape)
#         center_y, center_x = h // 2, w // 2
        
#         # Calculate distance from center for each pixel
#         r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
#         r = r.astype(int)
        
#         # Maximum radius is half the minimum dimension
#         max_radius = min(center_x, center_y)
        
#         # Initialize arrays to store values
#         radial_prof = np.zeros(max_radius)
#         bin_counts = np.zeros(max_radius)
        
#         # Sum values in each radial bin
#         for i in range(h):
#             for j in range(w):
#                 rad = r[i, j]
#                 if rad < max_radius:
#                     radial_prof[rad] += psd_2d[i, j]
#                     bin_counts[rad] += 1
        
#         # Avoid division by zero
#         bin_counts[bin_counts == 0] = 1
        
#         # Average values in each bin
#         radial_prof /= bin_counts
        
#         # Create frequency bins (cycles/pixel)
#         freq_bins = np.linspace(0, 0.5, max_radius)  # Nyquist frequency is 0.5 cycles/pixel
        
#         return freq_bins, radial_prof
    
#     def analyze_psd(self, img):
#         """
#         Analyze the Power Spectral Density of an image.
        
#         Args:
#             img (np.ndarray): Input grayscale image
            
#         Returns:
#             tuple: (frequency bins, radially averaged PSD)
#         """
#         psd_2d = self.compute_psd_2d(img)
#         return self.radial_average(psd_2d)
    
#     def compute_wavelet_energy(self, img):
#         """
#         Compute wavelet-based energy ratios using PyTorch Wavelets.
        
#         Args:
#             img (np.ndarray): Input grayscale image
            
#         Returns:
#             dict: Dictionary containing wavelet energy metrics
#         """
#         try:
#             # Check the input image for proper size and content
#             if img.shape[0] < 4 or img.shape[1] < 4:
#                 print(f"Image too small for wavelet transform: {img.shape}")
#                 raise ValueError("Image too small for wavelet")
                
#             # Ensure image has even dimensions (required by some wavelet implementations)
#             h, w = img.shape
#             if h % 2 != 0 or w % 2 != 0:
#                 # Pad to even dimensions
#                 h_pad = 0 if h % 2 == 0 else 1
#                 w_pad = 0 if w % 2 == 0 else 1
#                 img = np.pad(img, ((0, h_pad), (0, w_pad)), mode='reflect')
            
#             # Check for NaN or Inf values
#             if np.isnan(img).any() or np.isinf(img).any():
#                 print("Image contains NaN or Inf values")
#                 # Replace problematic values
#                 img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
                
#             # Check value range
#             if img.min() < -10 or img.max() > 10:
#                 print(f"Unusual image value range: min={img.min()}, max={img.max()}")
#                 # Normalize to 0-1 range
#                 img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                
#             # Convert numpy array to PyTorch tensor
#             img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
            
#             # Make sure tensor values are valid for computation
#             if torch.isnan(img_tensor).any() or torch.isinf(img_tensor).any():
#                 img_tensor = torch.nan_to_num(img_tensor, nan=0.0, posinf=1.0, neginf=0.0)
                
#             # Initialize wavelet transform with more robust parameters
#             # Using db1 (same as Haar) but can be more stable in some implementations
#             dwt = DWTForward(J=1, wave='haar')
            
#             # Move to GPU if available
#             img_tensor = img_tensor.to(self.device)
#             dwt = dwt.to(self.device)
            
#             # Apply wavelet transform
#             # Returns a tuple: (LL, [LH, HL, HH])
#             coeffs = dwt(img_tensor)
            
#             # Check if coeffs tuple is valid and contains at least one element
#             if not coeffs or len(coeffs) == 0:
#                 print("Warning: Wavelet transform returned empty or invalid coefficients.")
#                 raise ValueError("Invalid wavelet coefficients")

#             # Extract coefficients - safely handle potential dimension issues
#             LL = coeffs[0]  # Low-low component

#             # Check if high frequency components exist in the tuple
#             if len(coeffs) > 1 and coeffs[1] is not None:
#                 # Check if the high frequency list itself is not empty
#                 if len(coeffs[1]) > 0:
#                     # Extract high frequency components with proper handling of the shape
#                     # The pytorch_wavelets package returns [1, 1, 3, H, W] for high frequency components
#                     # where the 3 channels are LH, HL, HH
#                     high_freq = coeffs[1][0]
                    
#                     # Handle the specific shape we're seeing: [1, 1, 3, H, W]
#                     if high_freq.dim() == 5 and high_freq.shape[2] == 3:
#                         # Extract the three components (LH, HL, HH) from the 3rd dimension
#                         LH = high_freq[:, :, 0, :, :]
#                         HL = high_freq[:, :, 1, :, :]
#                         HH = high_freq[:, :, 2, :, :]
                        
#                         # # Remove extra dimensions to match LL shape
#                         # LH = LH.reshape(LH.shape[0], LH.shape[3], LH.shape[4])
#                         # HL = HL.reshape(HL.shape[0], HL.shape[3], HL.shape[4])
#                         # HH = HH.reshape(HH.shape[0], HH.shape[3], HH.shape[4])
                        
#                         # # Add channel dimension to match LL shape
#                         # LH = LH.unsqueeze(1)
#                         # HL = HL.unsqueeze(1)
#                         # HH = HH.unsqueeze(1)
#                     # Handle the standard expected shape [B, C, H, W]
#                     elif high_freq.dim() == 4 and high_freq.shape[1] >= 3:
#                         LH, HL, HH = high_freq[:, 0:1, :, :], high_freq[:, 1:2, :, :], high_freq[:, 2:3, :, :]
#                     else:
#                         # For any other unexpected shape, log it once and create placeholder tensors
#                         if not hasattr(self, '_logged_shape_warning'):
#                             print(f"Unexpected high frequency components shape: {high_freq.shape}")
#                             print("This message will only be shown once.")
#                             self._logged_shape_warning = True
                        
#                         # Create placeholder components
#                         LH = torch.zeros_like(LL)
#                         HL = torch.zeros_like(LL)
#                         HH = torch.zeros_like(LL)
                    
#                     # Compute energy in each component
#                     ll_energy = torch.sum(LL**2).item()
#                     lh_energy = torch.sum(LH**2).item()
#                     hl_energy = torch.sum(HL**2).item()
#                     hh_energy = torch.sum(HH**2).item()
                    
#                     # Total high-frequency energy
#                     high_freq_energy = lh_energy + hl_energy + hh_energy
                    
#                     # Ratio of high to low frequency energy with safeguards
#                     energy_ratio = high_freq_energy / ll_energy if ll_energy > 1e-8 else 0
                    
#                     return {
#                         'LL_energy': ll_energy,
#                         'LH_energy': lh_energy, 
#                         'HL_energy': hl_energy,
#                         'HH_energy': hh_energy,
#                         'high_freq_energy': high_freq_energy,
#                         'energy_ratio': energy_ratio
#                     }
#                 else:
#                     # Handle case where coeffs[1] is an empty list
#                     print("Warning: Wavelet transform returned empty high frequency component list.")
#                     if LL is not None:
#                         ll_energy = torch.sum(LL**2).item()
#                     else:
#                         ll_energy = 0.0
#                     # Fallback to FFT approximation or return zeros
#             else:
#                 # Handle case where coeffs tuple only has LL component (len(coeffs) <= 1)
#                 print("Warning: Wavelet transform did not return high frequency components tuple element.")
#                 if LL is not None:
#                     ll_energy = torch.sum(LL**2).item()
#                 else:
#                     ll_energy = 0.0
#                 # Fallback to FFT approximation or return zeros

#             # Fallback logic (moved from inner else block to handle all missing HF cases)
#             try:
#                 # Compute a simple approximation of high frequency energy using FFT
#                 fft = torch.fft.fft2(img_tensor.squeeze())
#                 fft_shift = torch.fft.fftshift(fft)
#                 h, w = fft_shift.shape
                
#                 # Create a simple approximation of wavelet decomposition using frequency regions
#                 # Low freq is center region, high freq is other regions
#                 center_h, center_w = h // 2, w // 2
#                 mask_lh = torch.zeros((h, w), device=self.device)
#                 mask_lh[:center_h, center_w:] = 1.0  # top-right quadrant
                
#                 mask_hl = torch.zeros((h, w), device=self.device)
#                 mask_hl[center_h:, :center_w] = 1.0  # bottom-left quadrant
                
#                 mask_hh = torch.zeros((h, w), device=self.device)
#                 mask_hh[center_h:, center_w:] = 1.0  # bottom-right quadrant
                
#                 # Apply masks
#                 lh_approx = torch.sum((torch.abs(fft_shift) * mask_lh)**2).item()
#                 hl_approx = torch.sum((torch.abs(fft_shift) * mask_hl)**2).item()
#                 hh_approx = torch.sum((torch.abs(fft_shift) * mask_hh)**2).item()
                
#                 high_freq_energy = lh_approx + hl_approx + hh_approx
#                 energy_ratio = high_freq_energy / ll_energy if ll_energy > 1e-8 else 0
                
#                 return {
#                     'LL_energy': ll_energy,
#                     'LH_energy': lh_approx,
#                     'HL_energy': hl_approx,
#                     'HH_energy': hh_approx,
#                     'high_freq_energy': high_freq_energy,
#                     'energy_ratio': energy_ratio
#                 }
#             except Exception as e_fft:
#                 print(f"Failed to approximate wavelet components via FFT: {e_fft}")
#                 return {
#                     'LL_energy': ll_energy,
#                     'LH_energy': 0.0,
#                     'HL_energy': 0.0,
#                     'HH_energy': 0.0,
#                     'high_freq_energy': 0.0,
#                     'energy_ratio': 0.0
#                 }
#         except Exception as e:
#             print(f"Error in wavelet computation: {e}")
#             # Return a default set of values if computation fails
#             return {
#                 'LL_energy': 0.0,
#                 'LH_energy': 0.0, 
#                 'HL_energy': 0.0,
#                 'HH_energy': 0.0,
#                 'high_freq_energy': 0.0,
#                 'energy_ratio': 0.0
#             }

#     def analyze_directory(self, dir_path, method_name=None, max_images=None):
#         """
#         Analyze all images in a directory and compute average metrics using PyTorch and batched processing
#         with running averages to save memory.
        
#         Args:
#             dir_path (str): Path to directory containing images
#             method_name (str, optional): Name of the method for labeling
#             max_images (int, optional): Maximum number of images to analyze
            
#         Returns:
#             dict: Dictionary containing average metrics for the directory
#         """
#         # Create dataset and dataloader
#         try:
#             dataset = ImageDataset(dir_path, self.img_size, max_images=max_images)
#             # Consider using collate_fn=collate_skip_none if image loading errors are frequent
#             dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4) 
#         except ValueError as e:
#             logging.error(f"Error creating dataset for {dir_path}: {e}")
#             raise
            
#         # Initialize accumulators for running averages
#         psd_sum = None
#         wavelet_sum = None
#         processed_count = 0
#         freq_bins = None
        
#         # Default wavelet metrics structure (initialized when first valid result is found)
#         default_wavelet_keys = ['LL_energy', 'LH_energy', 'HL_energy', 'HH_energy', 'high_freq_energy', 'energy_ratio']

#         # Process images in batches
#         for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Analyzing {method_name or os.path.basename(dir_path)}"):
#             try:
#                 # Skip empty batches (can happen with collate_skip_none or if dataset is empty)
#                 if batch is None or batch.size(0) == 0:
#                     logging.warning(f"Skipping empty or invalid batch {batch_idx}")
#                     continue
                    
#                 # Move batch to device
#                 batch = batch.to(self.device)
                
#                 # Process each image in the batch
#                 for i in range(batch.size(0)):
#                     img_tensor = None # Ensure img_tensor is defined in this scope
#                     try:
#                         img_tensor = batch[i]
                        
#                         # Check tensor dimensions
#                         if img_tensor.dim() != 3 or img_tensor.size(0) != 1:
#                             logging.warning(f"Skipping malformed tensor with shape: {img_tensor.shape}")
#                             continue
                        
#                         # Convert to numpy for PSD analysis
#                         img_np = img_tensor.squeeze().cpu().numpy()
                        
#                         # Check for NaN or Inf values
#                         if np.isnan(img_np).any() or np.isinf(img_np).any():
#                             logging.warning("Skipping image with NaN or Inf values")
#                             continue
                        
#                         # --- Analyze PSD ---
#                         current_freq_bins = None
#                         radial_psd = None
#                         try:
#                             current_freq_bins, radial_psd = self.analyze_psd(img_np)
#                             # Store freq_bins only once
#                             if freq_bins is None:
#                                 freq_bins = current_freq_bins
#                             # Initialize or add to psd_sum
#                             if psd_sum is None:
#                                 psd_sum = radial_psd.copy() # Important: copy the array
#                             else:
#                                 # Ensure shapes match before adding
#                                 if psd_sum.shape == radial_psd.shape:
#                                     psd_sum += radial_psd
#                                 else:
#                                      logging.warning(f"PSD shape mismatch: expected {psd_sum.shape}, got {radial_psd.shape}. Skipping PSD update for this image.")
#                                      # Don't update PSD sum, but might continue with wavelet if needed
#                                      # Or decide to skip the image entirely: continue 
#                         except Exception as e:
#                             logging.error(f"Error in PSD analysis for an image in batch {batch_idx}: {e}")
#                             continue # Skip this image entirely if PSD fails

#                         # --- Compute wavelet energy ---
#                         wavelet_metrics = None
#                         try:
#                             wavelet_metrics = self.compute_wavelet_energy(img_np)
#                             # Initialize wavelet_sum if first valid result
#                             if wavelet_sum is None:
#                                 wavelet_sum = {key: 0.0 for key in default_wavelet_keys}
#                             # Add metrics to sum
#                             for key in default_wavelet_keys:
#                                 if key in wavelet_metrics:
#                                      wavelet_sum[key] += wavelet_metrics[key]
#                                 else:
#                                      logging.warning(f"Key '{key}' not found in wavelet_metrics for an image.")

#                         except Exception as e:
#                             logging.error(f"Error in wavelet analysis for an image in batch {batch_idx}: {e}")
#                             # If wavelet fails, we already added PSD, but don't increment count
#                             continue # Skip incrementing count for this image

#                         # --- Increment count only if both PSD and Wavelet succeeded ---
#                         processed_count += 1
                    
#                     except Exception as e:
#                         # Catch errors processing a single image within the batch
#                         logging.error(f"Error processing image {i} in batch {batch_idx}: {e}")
#                         # Continue to the next image in the batch
            
#             except Exception as e:
#                 # Catch errors related to the whole batch (e.g., moving to device)
#                 logging.error(f"Critical error processing batch {batch_idx}: {e}")
#                 # Continue to the next batch

#         # --- Calculate final averages ---
#         avg_psd = None
#         avg_wavelet = None

#         if processed_count > 0:
#             if psd_sum is not None:
#                  avg_psd = psd_sum / processed_count
#             else:
#                  logging.warning(f"Processed {processed_count} images but PSD sum is None for {dir_path}.")
            
#             if wavelet_sum is not None:
#                  avg_wavelet = {key: value / processed_count for key, value in wavelet_sum.items()}
#             else:
#                  logging.warning(f"Processed {processed_count} images but Wavelet sum is None for {dir_path}.")

#         # Handle cases where no images were processed successfully
#         if processed_count == 0:
#             logging.warning(f"WARNING: No valid images processed successfully from directory: {dir_path}")
#             # Use default values
#             freq_bins = np.linspace(0, 0.5, self.img_size[0] // 2) # Default based on img_size
#             avg_psd = np.zeros_like(freq_bins)
#             avg_wavelet = {key: 0.0 for key in default_wavelet_keys}
#         elif freq_bins is None: # Handle case where count > 0 but freq_bins wasn't set (shouldn't happen ideally)
#              logging.warning("Processed images but failed to capture frequency bins. Using default.")
#              freq_bins = np.linspace(0, 0.5, self.img_size[0] // 2) 
#              if avg_psd is None: # If PSD sum was also None
#                  avg_psd = np.zeros_like(freq_bins)

#         # Ensure avg_wavelet is initialized if it's still None
#         if avg_wavelet is None:
#              avg_wavelet = {key: 0.0 for key in default_wavelet_keys}
             
#         logging.info(f"Successfully analyzed {processed_count} images from {dir_path}")
        
#         return {
#             'method_name': method_name if method_name else os.path.basename(dir_path),
#             'freq_bins': freq_bins,
#             'avg_psd': avg_psd,
#             'wavelet_metrics': avg_wavelet
#         }
#     def compare_methods(self, real_data_dir, method_dirs, max_images=None):
#         """
#         Compare frequency characteristics of different methods to real data.
        
#         Args:
#             real_data_dir (str): Directory with real images
#             method_dirs (dict): Dictionary mapping method names to directories
#             max_images (int, optional): Maximum number of images to analyze per directory
            
#         Returns:
#             dict: Comparison results
#         """
#         # Analyze real data
#         real_results = self.analyze_directory(real_data_dir, 'Real', max_images=max_images)
        
#         # Analyze each method
#         method_results = {}
#         for method_name, dir_path in method_dirs.items():
#             method_results[method_name] = self.analyze_directory(dir_path, method_name, max_images=max_images)
#         # Compare methods to real data
#         comparison = {}
#         for method_name, results in method_results.items():
#             # PSD similarity metrics
#             psd_mse = mean_squared_error(real_results['avg_psd'], results['avg_psd'])
#             psd_emd = wasserstein_distance(real_results['avg_psd'], results['avg_psd'])
            
#             # Wavelet energy comparison
#             real_energy_ratio = real_results['wavelet_metrics']['energy_ratio']
#             method_energy_ratio = results['wavelet_metrics']['energy_ratio']
#             energy_ratio_diff = method_energy_ratio / real_energy_ratio if real_energy_ratio > 0 else 0
            
#             comparison[method_name] = {
#                 'psd_mse': psd_mse,
#                 'psd_emd': psd_emd,  # Earth Mover's Distance (Wasserstein)
#                 'energy_ratio': method_energy_ratio,
#                 'energy_ratio_vs_real': energy_ratio_diff
#             }
        
#         return {
#             'real_results': real_results,
#             'method_results': method_results,
#             'comparison': comparison
#         }
    
#     def plot_spectral_analysis(self, comparison_results, save_path=None, log_scale=True):
#         """
#         Plot the spectral analysis results.
        
#         Args:
#             comparison_results (dict): Results from compare_methods
#             save_path (str, optional): Path to save the figure
#             log_scale (bool): Whether to use log scale for PSD
            
#         Returns:
#             matplotlib.figure.Figure: The generated figure
#         """
#         plt.figure(figsize=(10, 6))
        
#         real_results = comparison_results['real_results']
#         method_results = comparison_results['method_results']
        
#         # Plot real data PSD
#         freq_bins = real_results['freq_bins']
#         plt.plot(freq_bins, real_results['avg_psd'], 'k-', linewidth=2, label='Real')
        
#         # Plot each method's PSD
#         colors = ['r', 'g', 'b', 'c', 'm', 'y']
#         for i, (method_name, results) in enumerate(method_results.items()):
#             color = colors[i % len(colors)]
#             plt.plot(freq_bins, results['avg_psd'], color=color, linewidth=1.5, label=method_name)
        
#         plt.xlabel('Spatial Frequency (cycles/pixel)')
#         plt.ylabel('Power Spectral Density')
        
#         if log_scale:
#             plt.yscale('log')
        
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.title('Average Power Spectral Density Comparison')
        
#         if save_path:
#             plt.tight_layout()
#             plt.savefig(save_path, dpi=300)
        
#         return plt.gcf()
    
#     def create_wavelet_energy_table(self, comparison_results):
#         """
#         Create a table of wavelet energy metrics.
        
#         Args:
#             comparison_results (dict): Results from compare_methods
            
#         Returns:
#             pandas.DataFrame: Table of wavelet energy metrics
#         """
#         data = []
        
#         # Add real data first
#         real_metrics = comparison_results['real_results']['wavelet_metrics']
#         real_row = {
#             'Method': 'Real',
#             'Low Freq Energy (LL²)': real_metrics['LL_energy'],
#             'High Freq Energy (LH²+HL²+HH²)': real_metrics['high_freq_energy'],
#             'Energy Ratio': real_metrics['energy_ratio'],
#             'vs Real': 1.0  # Reference point
#         }
#         data.append(real_row)
        
#         # Add each method
#         for method_name, comparison in comparison_results['comparison'].items():
#             method_metrics = comparison_results['method_results'][method_name]['wavelet_metrics']
#             method_row = {
#                 'Method': method_name,
#                 'Low Freq Energy (LL²)': method_metrics['LL_energy'],
#                 'High Freq Energy (LH²+HL²+HH²)': method_metrics['high_freq_energy'],
#                 'Energy Ratio': method_metrics['energy_ratio'],
#                 'vs Real': comparison['energy_ratio_vs_real']
#             }
#             data.append(method_row)
        
#         df = pd.DataFrame(data)
#         return df
    
#     def compute_wdp(self, real_img_tensor, gen_img_tensor):
#         """
#         Compute Wavelet Detail Preservation (WDP) score.
#         Args:
#             real_img_tensor (torch.Tensor): Ground truth image (1xHxW)
#             gen_img_tensor (torch.Tensor): Generated image (1xHxW)
#         Returns:
#             float: WDP score ∈ [0, 1]
#         """
#         dwt = DWTForward(J=1, wave='haar').to(self.device)
#         _, [real_H] = dwt(real_img_tensor.unsqueeze(0))
#         _, [gen_H] = dwt(gen_img_tensor.unsqueeze(0))

#         real_H = torch.cat(real_H.unbind(0), dim=0)
#         gen_H = torch.cat(gen_H.unbind(0), dim=0)

#         diff = torch.norm(real_H - gen_H, p=2)
#         norm = torch.norm(real_H, p=2)
#         return float(torch.exp(-diff / (norm + 1e-8)))
    
#     def create_wdp_table(self, real_dir, method_dirs, results_path, max_images=None):
#         """
#         Create a table comparing WDP scores between methods and real images.
        
#         Args:
#             real_dir (str): Directory with real images
#             method_dirs (dict): Dictionary mapping method names to directories
#             results_path (str): Path to save results
#             max_images (int, optional): Maximum number of images to analyze per directory
#         """
#         results = []

#         real_dataset = ImageDataset(real_dir, self.img_size, max_images=max_images)
#         real_loader = DataLoader(real_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
#         real_images = []
#         for batch in real_loader:
#             real_images.extend([b.to(self.device) for b in batch])

#         for method_name, dir_path in method_dirs.items():
#             gen_dataset = ImageDataset(dir_path, self.img_size)
#             gen_loader = DataLoader(gen_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

#             wdp_scores = []
#             for i, gen_batch in tqdm(enumerate(gen_loader)):
#                 if i >= len(real_images):
#                     break
#                 real_img = real_images[i].to(self.device)
#                 for gen_img in gen_batch:
#                     if gen_img.dim() == 2:
#                         gen_img = gen_img.unsqueeze(0)
#                     wdp = self.compute_wdp(real_img, gen_img.to(self.device))
#                     wdp_scores.append(wdp)

#             results.append({
#                 'Method': method_name,
#                 'WDP': np.mean(wdp_scores),
#                 'WDP_std': np.std(wdp_scores)
#             })

#         df = pd.DataFrame(results)
#         df.to_csv(os.path.join(results_path,'detail_preservation.csv'), index=False)
#         return df

# def main(paths_dict_img_generated, real_dir=None, max_images=None):
#     figures_path = os.path.join(os.path.dirname(paths_dict_img_generated['WALD']), 'figures')
#     results_path = os.path.join(os.path.dirname(paths_dict_img_generated['WALD']), 'results')
#     os.makedirs(figures_path, exist_ok=True)
#     os.makedirs(results_path, exist_ok=True)

#     analyzer = FrequencyAnalyzer(img_size=(2048, 2048), batch_size=4)

#     print("Starting analysis of image directories...")
#     comparison_results = analyzer.compare_methods(real_dir, paths_dict_img_generated, max_images=max_images)

#     print("Generating spectral analysis plot...")
#     analyzer.plot_spectral_analysis(comparison_results, save_path=os.path.join(figures_path, 'spectral_analysis.pdf'))

#     print("Creating wavelet energy table...")
#     wavelet_table = analyzer.create_wavelet_energy_table(comparison_results)
#     print("\nWavelet Energy Analysis:")
#     print(wavelet_table)

#     print("Creating WDP metric table...")
#     wdp_table = analyzer.create_wdp_table(real_dir, paths_dict_img_generated, results_path, max_images=max_images)
#     print("\nWavelet Detail Preservation (WDP):")
#     print(wdp_table)

#     print("\nAnalysis complete! Results saved to figures/ and results/ directories.")

# if __name__ == "__main__":
    # import argparse
    
    # parser = argparse.ArgumentParser(description='Analyze image frequencies')
    # parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to analyze per directory')
    # args = parser.parse_args()
    
    # method_dirs = {
    #     'WALD': "/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/output/URAE_VAE_SE_WAV_ATT_LAION/HPDv2_test_set",
    #     'URAE': '/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/output/URAE_original_trained_by_me/HPDv2_test_set'
    # }
    # real_dir = "/leonardo_scratch/large/userexternal/lsigillo/HPDv2/test"
    # main(method_dirs, real_dir, args.max_images)


import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pytorch_wavelets import DWTForward, DWTInverse
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import make_grid
from typing import List, Tuple, Dict, Optional
import random # Add this import

class WaveletMetrics:
    """
    Classe per calcolare metriche basate su wavelet per valutare la qualità delle immagini
    in termini di dettagli ad alta/bassa frequenza.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', wave='db2', J=4, resize=1024):
        """
        Inizializza la classe WaveletMetrics.
        
        Args:
            device: dispositivo su cui eseguire i calcoli ('cuda' o 'cpu')
            wave: tipo di wavelet (default: 'db2', Daubechies 2)
            J: livelli di decomposizione wavelet (default: 4)
            resize: dimensione a cui ridimensionare le immagini (default: 1024)
                   Se None, non viene applicato ridimensionamento
        """
        self.device = device
        self.wave = wave
        self.J = J
        self.resize = resize
        self.dwt = DWTForward(J=self.J, wave=self.wave, mode='zero').to(self.device)
        self.idwt = DWTInverse(wave=self.wave, mode='zero').to(self.device)
        
    def preprocess_image(self, image_path: str, resize=1024) -> torch.Tensor:
        """
        Carica e preprocessa un'immagine per l'analisi wavelet.
        
        Args:
            image_path: percorso dell'immagine
            resize: dimensione a cui ridimensionare l'immagine (default: 1024)
                   Se None, non viene applicato ridimensionamento
            
        Returns:
            Tensore dell'immagine normalizzata [1, 3, H, W]
        """
        # Carica l'immagine
        img = Image.open(image_path).convert('RGB')
        
        # Ridimensiona mantenendo le proporzioni se richiesto
        if resize is not None:
            # Imposta entrambe dimensioni allo stesso valore per evitare problemi di broadcasting
            img = img.resize((resize, resize), Image.LANCZOS)
            
        # Trasforma in tensore e normalizza
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        return img_tensor
    
    def compute_wavelet_decomposition(self, img_tensor: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Calcola la decomposizione wavelet dell'immagine.
        
        Args:
            img_tensor: tensore dell'immagine [B, C, H, W]
            
        Returns:
            Tuple contenente coefficienti di approssimazione e dettagli
        """
        # Per ogni canale
        coeffs_list = []
        for c in range(img_tensor.shape[1]):
            channel = img_tensor[:, c:c+1, :, :]
            coeffs = self.dwt(channel)
            coeffs_list.append(coeffs)
            
        # Combina i coefficienti di tutti i canali
        yl = torch.cat([c[0] for c in coeffs_list], dim=1)
        yh = [torch.cat([c[1][j] for c in coeffs_list], dim=1) for j in range(self.J)]
        
        return yl, yh
    
    def compare_images(self, 
                    gen_img_path: str, 
                    real_img_path: str,
                    visualize: bool = True) -> Dict[str, float]:
        """
        Confronta un'immagine generata con una reale utilizzando tutte le metriche di frequenza.
        
        Args:
            gen_img_path: percorso dell'immagine generata
            real_img_path: percorso dell'immagine reale
            visualize: se True, visualizza la decomposizione wavelet e lo spettro
            
        Returns:
            Dizionario con tutte le metriche calcolate
        """
        # Caricamento e preprocessing
        gen_img = self.preprocess_image(gen_img_path, resize=self.resize)
        real_img = self.preprocess_image(real_img_path, resize=self.resize)
        
        # Calcolo metriche esistenti
        gen_ratio = self.ratio_high_low_frequency_energy(gen_img)
        real_ratio = self.ratio_high_low_frequency_energy(real_img)
        
        quality_metrics = self.wavelet_quality_assessment(gen_img, real_img)
        
        gen_spectrum = self.power_spectrum_analysis(gen_img)
        real_spectrum = self.power_spectrum_analysis(real_img)
        
        # Nuove metriche
        # 1. Metriche energia wavelet avanzate
        gen_wavelet_energy = self.wavelet_subband_energy_analysis(gen_img)
        real_wavelet_energy = self.wavelet_subband_energy_analysis(real_img)
        
        # 2. Analisi spettrale avanzata con log-log slope
        gen_enhanced_spectrum = self.enhanced_power_spectrum_analysis(gen_img)
        real_enhanced_spectrum = self.enhanced_power_spectrum_analysis(real_img)
        
        # 3. Indice enfasi alta frequenza
        hf_emphasis = self.high_frequency_emphasis_index(gen_img, real_img)
        
        # 4. FSIM
        fsim_value = self.feature_similarity_index(gen_img, real_img)
        
        # 5. MS-SSIM
        ms_ssim_value = self.multi_scale_ssim(gen_img, real_img)
        
        # Risultati
        results = {
            'gen_high_low_ratio': gen_ratio,
            'real_high_low_ratio': real_ratio,
            'ratio_difference': abs(gen_ratio - real_ratio),
            'wavelet_quality': quality_metrics,
            'gen_spectrum': gen_spectrum,
            'real_spectrum': real_spectrum,
            
            # Nuove metriche
            'gen_wavelet_energy': gen_wavelet_energy,
            'real_wavelet_energy': real_wavelet_energy,
            'gen_enhanced_spectrum': gen_enhanced_spectrum,
            'real_enhanced_spectrum': real_enhanced_spectrum,
            'hf_emphasis': hf_emphasis,
            'fsim': fsim_value,
            'ms_ssim': ms_ssim_value
        }
            
        # Visualizzazione opzionale
        if visualize:
            self.visualize_wavelet_decomposition(gen_img, title='Decomposizione Wavelet - Immagine Generata')
            self.visualize_wavelet_decomposition(real_img, title='Decomposizione Wavelet - Immagine Reale')
            
            # Confronto diretto
            plt.figure(figsize=(15, 7))
            
            plt.subplot(1, 2, 1)
            gen_img_np = (gen_img[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
            plt.imshow(gen_img_np)
            plt.title(f'Generata (Ratio HF/LF: {gen_ratio:.4f})')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            real_img_np = (real_img[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
            plt.imshow(real_img_np)
            plt.title(f'Reale (Ratio HF/LF: {real_ratio:.4f})')
            plt.axis('off')
            
            plt.suptitle('Confronto immagini - Rapporto alte/basse frequenze', fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.savefig("img_comparison_HI_LO.png")
            
            # Confronto spettro di potenza
            plt.figure(figsize=(15, 5))
            
            # Distribuzione energia per bande di frequenza
            labels = ['Basse freq.', 'Medie freq.', 'Alte freq.']
            gen_values = [gen_spectrum['avg_low_freq_energy'], 
                         gen_spectrum['avg_mid_freq_energy'], 
                         gen_spectrum['avg_high_freq_energy']]
            real_values = [real_spectrum['avg_low_freq_energy'], 
                          real_spectrum['avg_mid_freq_energy'], 
                          real_spectrum['avg_high_freq_energy']]
            
            x = np.arange(len(labels))
            width = 0.35
            
            plt.bar(x - width/2, gen_values, width, label='Generata')
            plt.bar(x + width/2, real_values, width, label='Reale')
            
            plt.ylabel('Energia normalizzata')
            plt.title('Distribuzione energia nelle bande di frequenza')
            plt.xticks(x, labels)
            plt.legend()
            plt.tight_layout()
            plt.savefig("img_distrib_HI_LO.png")
        
        # Add new visualizations for the advanced metrics
        plt.figure(figsize=(15, 10))
        
        # Plot per energy subbands
        plt.subplot(2, 2, 1)
        bands = ['LL', 'Level1', 'Level2']
        if self.J > 2:
            bands.extend([f'Level{j+1}' for j in range(2, self.J)])
        
        gen_values = [gen_wavelet_energy['ll_energy_ratio']]
        gen_values.extend([gen_wavelet_energy[f'level{j+1}_energy_ratio'] for j in range(self.J)])
        
        real_values = [real_wavelet_energy['ll_energy_ratio']]
        real_values.extend([real_wavelet_energy[f'level{j+1}_energy_ratio'] for j in range(self.J)])
        
        x = np.arange(len(bands))
        width = 0.35
        
        plt.bar(x - width/2, gen_values, width, label='Generata')
        plt.bar(x + width/2, real_values, width, label='Reale')
        plt.ylabel('Energia normalizzata')
        plt.title('Distribuzione energia nei sottobandi wavelet')
        plt.xticks(x, bands)
        plt.legend()
        
        # Plot frequenza vs potenza logaritmici
        if 'avg_log_log_slope' in gen_enhanced_spectrum and 'avg_log_log_slope' in real_enhanced_spectrum:
            plt.subplot(2, 2, 2)
            plt.text(0.5, 0.5, f"Log-log slope (Gen): {gen_enhanced_spectrum['avg_log_log_slope']:.3f}\n"
                   f"Log-log slope (Real): {real_enhanced_spectrum['avg_log_log_slope']:.3f}", 
                   ha='center', va='center', fontsize=12)
            plt.axis('off')
        
        # Plot metriche percettive
        plt.subplot(2, 2, 3)
        metrics = ['FSIM', 'MS-SSIM', 'Wavelet Score']
        values = [fsim_value, ms_ssim_value, quality_metrics['overall_wavelet_score']]
        plt.bar(metrics, values, color=['green', 'blue', 'purple'])
        plt.ylim(0, 1)
        plt.title('Metriche percettive')
        
        # Plot filtri alta frequenza
        plt.subplot(2, 2, 4)
        hf_metrics = ['hf_emphasis_index', 'laplacian_variance', 'sobel_variance']
        gen_hf_values = [hf_emphasis[f'gen_{m}'] for m in hf_metrics]
        real_hf_values = [hf_emphasis[f'real_{m}'] for m in hf_metrics]
        
        plt.bar(np.arange(len(hf_metrics)) - width/2, gen_hf_values, width, label='Generata')
        plt.bar(np.arange(len(hf_metrics)) + width/2, real_hf_values, width, label='Reale')
        plt.title('Metriche filtri alta frequenza')
        plt.xticks(np.arange(len(hf_metrics)), ['HF Index', 'Laplacian', 'Sobel'])
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("advanced_metrics.png")
    
        return results
    
    def power_spectrum_analysis(self, img_tensor: torch.Tensor) -> Dict[str, float]:
        """
        Analizza lo spettro di potenza dell'immagine per valutare la distribuzione
        dell'energia nelle diverse frequenze.
        
        Args:
            img_tensor: tensore dell'immagine [B, C, H, W]
            
        Returns:
            Dizionario con metriche dello spettro di potenza
        """
        results = {}
        
        # Media su tutti i canali
        for c in range(img_tensor.shape[1]): # Corrected loop
            channel = img_tensor[0, c].cpu().numpy()
            
            # FFT 2D
            fft = np.fft.fft2(channel)
            fft_shifted = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shifted)
            power_spectrum = magnitude ** 2
            
            # Normalizzazione
            power_spectrum_norm = power_spectrum / np.sum(power_spectrum)
            
            # Creazione di anelli concentrici per analizzare frequenze diverse
            h, w = power_spectrum.shape
            center_y, center_x = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            
            # Definizione delle bande di frequenza (in percentuale della frequenza massima)
            max_dist = np.sqrt(center_y**2 + center_x**2)
            low_freq_mask = distances <= 0.15 * max_dist
            mid_freq_mask = (distances > 0.15 * max_dist) & (distances <= 0.4 * max_dist)
            high_freq_mask = distances > 0.4 * max_dist
            
            # Energia in ciascuna banda
            low_freq_energy = np.sum(power_spectrum_norm[low_freq_mask])
            mid_freq_energy = np.sum(power_spectrum_norm[mid_freq_mask])
            high_freq_energy = np.sum(power_spectrum_norm[high_freq_mask])
            
            # Salvataggio risultati per canale
            results[f'channel_{c}_low_freq_energy'] = float(low_freq_energy)
            results[f'channel_{c}_mid_freq_energy'] = float(mid_freq_energy)
            results[f'channel_{c}_high_freq_energy'] = float(high_freq_energy)
            results[f'channel_{c}_high_to_low_ratio'] = float(high_freq_energy / (low_freq_energy + 1e-8))
        
        # Media tra i canali
        results['avg_low_freq_energy'] = np.mean([results[f'channel_{c}_low_freq_energy'] for c in range(img_tensor.shape[1])])
        results['avg_mid_freq_energy'] = np.mean([results[f'channel_{c}_mid_freq_energy'] for c in range(img_tensor.shape[1])])
        results['avg_high_freq_energy'] = np.mean([results[f'channel_{c}_high_freq_energy'] for c in range(img_tensor.shape[1])])
        results['avg_high_to_low_ratio'] = np.mean([results[f'channel_{c}_high_to_low_ratio'] for c in range(img_tensor.shape[1])])
        
        return results
    
    def visualize_wavelet_decomposition(self, img_tensor: torch.Tensor, figsize=(15, 10), title='Decomposizione Wavelet'):
        """
        Visualizza la decomposizione wavelet dell'immagine.
        
        Args:
            img_tensor: tensore dell'immagine [B, C, H, W]
            figsize: dimensioni della figura
            title: titolo della figura
        """
        # Decomposizione wavelet
        yl, yh = self.compute_wavelet_decomposition(img_tensor)
        
        # Preparazione della visualizzazione
        plt.figure(figsize=figsize)
        
        # Immagine originale
        plt.subplot(2, self.J + 1, 1)
        img_np = (img_tensor[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        plt.imshow(img_np)
        plt.title('Immagine originale')
        plt.axis('off')
        
        # Approssimazione (basse frequenze)
        plt.subplot(2, self.J + 1, 2)
        approx = yl[0].mean(dim=0).cpu().numpy()
        plt.imshow(approx, cmap='gray')
        plt.title('Approssimazione (basse freq.)')
        plt.axis('off')
        
        # Dettagli per ogni livello (alte frequenze)
        for j in range(self.J):
            plt.subplot(2, self.J + 1, j + 3)
            # Visualizziamo la magnitudine dei coefficienti di dettaglio
            detail = torch.sqrt(torch.sum(yh[j][0]**2, dim=0)).cpu().numpy()
            plt.imshow(detail, cmap='viridis')
            plt.title(f'Dettagli livello {j+1}')
            plt.axis('off')
        
        # Visualizzazione dello spettro di potenza
        for c in range(min(3, img_tensor.shape[1])):
            channel = img_tensor[0, c].cpu().numpy()
            
            # FFT 2D
            fft = np.fft.fft2(channel)
            fft_shifted = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shifted)
            power_spectrum = np.log(magnitude + 1e-8)  # Log per visualizzazione migliore
            
            plt.subplot(2, self.J + 1, self.J + 2 + c)
            plt.imshow(power_spectrum, cmap='magma')
            plt.title(f'Spettro potenza (canale {c})')
            plt.axis('off')
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.9)
        plt.savefig("spectr_power_channel.png")
    
    def ratio_high_low_frequency_energy(self, img_tensor: torch.Tensor) -> float:
        """
        Calcola il rapporto tra l'energia delle alte frequenze e quella delle basse frequenze.
        
        Args:
            img_tensor: tensore dell'immagine [B, C, H, W]
            
        Returns:
            Rapporto energia alte/basse frequenze
        """
        # Decomposizione wavelet
        yl, yh = self.compute_wavelet_decomposition(img_tensor)
        
        # Energia basse frequenze (coefficienti di approssimazione)
        low_freq_energy = torch.sum(yl ** 2).item()
        
        # Energia alte frequenze (coefficienti di dettaglio)
        high_freq_energy = 0.0
        for j in range(self.J):
            high_freq_energy += torch.sum(yh[j] ** 2).item()
        
        # Rapporto energie
        ratio = high_freq_energy / (low_freq_energy + 1e-8)
        
        return ratio
    
    def wavelet_quality_assessment(self, 
                                  gen_img_tensor: torch.Tensor, 
                                  real_img_tensor: torch.Tensor) -> Dict[str, float]:
        """
        Valuta la qualità dell'immagine generata rispetto a quella reale utilizzando
        i coefficienti wavelet.
        
        Args:
            gen_img_tensor: tensore dell'immagine generata [B, C, H, W]
            real_img_tensor: tensore dell'immagine reale [B, C, H, W]
            
        Returns:
            Dizionario con diverse metriche di qualità
        """
        # Verifica che le immagini abbiano le stesse dimensioni
        if gen_img_tensor.shape != real_img_tensor.shape:
            print(f"Warning: Images have different shapes: {gen_img_tensor.shape} vs {real_img_tensor.shape}")
            # Ridimensiona l'immagine generata per farla corrispondere a quella reale
            real_img_tensor = F.interpolate(real_img_tensor, size=(gen_img_tensor.shape[2], gen_img_tensor.shape[3]), 
                                         mode='bilinear', align_corners=False)
        
        try:
            # Decomposizione wavelet per entrambe le immagini
            yl_gen, yh_gen = self.compute_wavelet_decomposition(gen_img_tensor)
            yl_real, yh_real = self.compute_wavelet_decomposition(real_img_tensor)
            
            # Assicuriamoci che i tensori abbiano le stesse dimensioni per il calcolo MSE
            if yl_gen.shape != yl_real.shape:
                # Ridimensiona il tensore più piccolo alla dimensione del più grande
                if yl_gen.numel() <= yl_real.numel():
                    yl_gen = F.interpolate(yl_gen, size=(yl_real.shape[2], yl_real.shape[3]), 
                                        mode='bilinear', align_corners=False)
                else:
                    yl_real = F.interpolate(yl_real, size=(yl_gen.shape[2], yl_gen.shape[3]), 
                                         mode='bilinear', align_corners=False)
            
            # Differenza nei coefficienti di approssimazione (basse frequenze)
            approx_diff = F.mse_loss(yl_gen, yl_real).item()
            
            # Differenze nei coefficienti di dettaglio per ogni livello (alte frequenze)
            detail_diff = []
            detail_ssim = []
            
            for j in range(min(len(yh_gen), len(yh_real))):
                # Gestisci il caso in cui i coefficienti abbiano dimensioni diverse
                if yh_gen[j].shape != yh_real[j].shape:
                    if yh_gen[j].dim() == 5 and yh_real[j].dim() == 5:
                        # Per tensori 5D [B, C, O, H, W], ridimensiona solo H e W
                        # Estrai le dimensioni
                        _, _, _, h_gen, w_gen = yh_gen[j].shape
                        _, _, _, h_real, w_real = yh_real[j].shape
                        
                        if h_gen * w_gen <= h_real * w_real:
                            # Ridimensiona yh_gen per farlo corrispondere a yh_real
                            yh_gen_resized = []
                            for o in range(yh_gen[j].shape[2]):
                                yh_gen_o = yh_gen[j][:, :, o]
                                yh_gen_o = F.interpolate(yh_gen_o, size=(h_real, w_real), 
                                                     mode='bilinear', align_corners=False)
                                yh_gen_resized.append(yh_gen_o.unsqueeze(2))
                            yh_gen[j] = torch.cat(yh_gen_resized, dim=2)
                        else:
                            # Ridimensiona yh_real per farlo corrispondere a yh_gen
                            yh_real_resized = []
                            for o in range(yh_real[j].shape[2]):
                                yh_real_o = yh_real[j][:, :, o]
                                yh_real_o = F.interpolate(yh_real_o, size=(h_gen, w_gen), 
                                                      mode='bilinear', align_corners=False)
                                yh_real_resized.append(yh_real_o.unsqueeze(2))
                            yh_real[j] = torch.cat(yh_real_resized, dim=2)
                
                try:
                    diff_j = F.mse_loss(yh_gen[j], yh_real[j]).item()
                    detail_diff.append(diff_j)
                    
                    # Calcolo SSIM sui coefficienti wavelet
                    ssim_j = self._ssim(yh_gen[j], yh_real[j])
                    detail_ssim.append(ssim_j)
                except Exception as e:
                    print(f"Error calculating metrics for level {j}: {e}")
                    detail_diff.append(1.0)  # Valore di default alto per MSE (peggiore)
                    detail_ssim.append(0.0)  # Valore di default basso per SSIM (peggiore)
            
            # Riempi i livelli mancanti con valori predefiniti
            while len(detail_diff) < self.J:
                detail_diff.append(1.0)
                detail_ssim.append(0.0)
            
            # SSIM sui coefficienti di approssimazione
            approx_ssim = self._ssim(yl_gen, yl_real)
            
            # Risultati
            results = {
                'approx_mse': approx_diff,
                'approx_ssim': approx_ssim,
            }
            
            for j in range(self.J):
                if j < len(detail_diff):
                    results[f'detail_level{j+1}_mse'] = detail_diff[j]
                    results[f'detail_level{j+1}_ssim'] = detail_ssim[j]
                else:
                    results[f'detail_level{j+1}_mse'] = 1.0
                    results[f'detail_level{j+1}_ssim'] = 0.0
                
            # Score complessivo (media pesata)
            weight_approx = 0.4
            weight_details = [0.3, 0.2, 0.1, 0.0][:self.J]  # Pesi decrescenti per livelli di dettaglio
            if self.J > 3:
                # Riscala i pesi per livelli aggiuntivi
                sum_weights = weight_approx + sum(weight_details[:self.J])
                weight_approx = weight_approx / sum_weights
                weight_details = [w / sum_weights for w in weight_details[:self.J]]
            
            # overall_score = weight_approx * (1 - approx_diff) + \
            #                sum(w * (1 - d) for w, d in zip(weight_details, detail_diff[:self.J]))
            # Fix 2: Use exponential decay function (better approach)
            overall_score = weight_approx * np.exp(-approx_diff) + \
               sum(w * np.exp(-d) for w, d in zip(weight_details, detail_diff[:self.J]))
            results['overall_wavelet_score'] = overall_score
            
            return results
            
        except Exception as e:
            print(f"Error in wavelet quality assessment: {e}")
            # Ritorna valori predefiniti in caso di errore
            default_results = {
                'approx_mse': 1.0,
                'approx_ssim': 0.0,
                'overall_wavelet_score': 0.0
            }
            for j in range(self.J):
                default_results[f'detail_level{j+1}_mse'] = 1.0
                default_results[f'detail_level{j+1}_ssim'] = 0.0
            
            return default_results
    
    def _ssim(self, x, y):
        """
        Calcola SSIM tra due tensori.
        Per i coefficienti di dettaglio (che sono 5D), calcola SSIM per ogni orientazione
        e poi fa la media.
        """
        # Valori costanti per SSIM
        C1 = (0.01 * 2) ** 2
        C2 = (0.03 * 2) ** 2
        
        try:
            # Gestisce sia tensori 4D (coefficienti di approssimazione)
            # che tensori 5D (coefficienti di dettaglio)
            if x.dim() == 5:  # [B, C, O, H, W] dove O è l'orientazione
                # Media su tutte le orientazioni
                ssim_values = []
                for o in range(x.shape[2]):
                    x_o = x[:, :, o]  # [B, C, H, W]
                    y_o = y[:, :, o]  # [B, C, H, W]
                    
                    # Verifica dimensioni compatibili
                    if x_o.shape != y_o.shape:
                        if x_o.numel() <= y_o.numel():
                            x_o = F.interpolate(x_o, size=(y_o.shape[2], y_o.shape[3]), 
                                             mode='bilinear', align_corners=False)
                        else:
                            y_o = F.interpolate(y_o, size=(x_o.shape[2], x_o.shape[3]), 
                                             mode='bilinear', align_corners=False)
                    
                    # Media
                    mu_x = F.avg_pool2d(x_o, kernel_size=11, stride=1, padding=5)
                    mu_y = F.avg_pool2d(y_o, kernel_size=11, stride=1, padding=5)
                    mu_x_sq = mu_x ** 2
                    mu_y_sq = mu_y ** 2
                    mu_xy = mu_x * mu_y
                    
                    # Varianza e covarianza
                    sigma_x_sq = F.avg_pool2d(x_o ** 2, kernel_size=11, stride=1, padding=5) - mu_x_sq
                    sigma_y_sq = F.avg_pool2d(y_o ** 2, kernel_size=11, stride=1, padding=5) - mu_y_sq
                    sigma_xy = F.avg_pool2d(x_o * y_o, kernel_size=11, stride=1, padding=5) - mu_xy
                    
                    # SSIM
                    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                              ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
                    
                    ssim_values.append(torch.mean(ssim_map).item())
                
                return sum(ssim_values) / len(ssim_values) if ssim_values else 0.0
            else:  # Tensore 4D [B, C, H, W]
                # Verifica dimensioni compatibili
                if x.shape != y.shape:
                    if x.numel() <= y.numel():
                        x = F.interpolate(x, size=(y.shape[2], y.shape[3]), 
                                       mode='bilinear', align_corners=False)
                    else:
                        y = F.interpolate(y, size=(x.shape[2], x.shape[3]), 
                                       mode='bilinear', align_corners=False)
                
                # Media
                mu_x = F.avg_pool2d(x, kernel_size=11, stride=1, padding=5)
                mu_y = F.avg_pool2d(y, kernel_size=11, stride=1, padding=5)
                mu_x_sq = mu_x ** 2
                mu_y_sq = mu_y ** 2
                mu_xy = mu_x * mu_y
                
                # Varianza e covarianza
                sigma_x_sq = F.avg_pool2d(x ** 2, kernel_size=11, stride=1, padding=5) - mu_x_sq
                sigma_y_sq = F.avg_pool2d(y ** 2, kernel_size=11, stride=1, padding=5) - mu_y_sq
                sigma_xy = F.avg_pool2d(x * y, kernel_size=11, stride=1, padding=5) - mu_xy
                
                # SSIM
                ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                          ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
                
                return torch.mean(ssim_map).item()
        except Exception as e:
            print(f"Error calculating SSIM: {e}")
            return 0.0  # Valore di default in caso di errore
    
    def batch_evaluate(self, 
                      gen_img_paths: List[str], 
                      real_img_paths: List[str],
                      summary: bool = True) -> Dict[str, float]:
        """
        Valuta un batch di immagini generate rispetto alle reali.
        
        Args:
            gen_img_paths: lista dei percorsi delle immagini generate
            real_img_paths: lista dei percorsi delle immagini reali
            summary: se True, stampa un riepilogo dei risultati
            
        Returns:
            Dizionario con le metriche medie
        """
        assert len(gen_img_paths) == len(real_img_paths), "Il numero di immagini generate e reali deve essere uguale"
        
        all_results = []
        for i, (gen_path, real_path) in tqdm(enumerate(zip(gen_img_paths, real_img_paths))):
            try:
                # print(f"Processing image pair {i+1}/{len(gen_img_paths)}")
                results = self.compare_images(gen_path, real_path, visualize=False)
                all_results.append(results)
            except Exception as e:
                print(f"Error processing image pair {gen_path} and {real_path}: {e}")
                # Continua con la prossima coppia di immagini
        
        if not all_results:
            print("No valid results to summarize!")
            return {}
            
        # Calcolo medie
        avg_results = {
            'avg_gen_high_low_ratio': np.mean([r['gen_high_low_ratio'] for r in all_results]),
            'avg_real_high_low_ratio': np.mean([r['real_high_low_ratio'] for r in all_results]),
            'avg_ratio_difference': np.mean([r['ratio_difference'] for r in all_results]),
            'avg_wavelet_quality_score': np.mean([r['wavelet_quality']['overall_wavelet_score'] for r in all_results]),
            'avg_gen_high_freq_energy': np.mean([r['gen_spectrum']['avg_high_freq_energy'] for r in all_results]),
            'avg_real_high_freq_energy': np.mean([r['real_spectrum']['avg_high_freq_energy'] for r in all_results]),
        }
        
        if summary:
            print("===== RIEPILOGO VALUTAZIONE =====")
            print(f"Numero immagini valutate: {len(gen_img_paths)}")
            print(f"Rapporto alte/basse frequenze - Generato: {avg_results['avg_gen_high_low_ratio']:.4f}")
            print(f"Rapporto alte/basse frequenze - Reale: {avg_results['avg_real_high_low_ratio']:.4f}")
            print(f"Differenza media rapporto: {avg_results['avg_ratio_difference']:.4f}")
            print(f"Punteggio qualità wavelet: {avg_results['avg_wavelet_quality_score']:.4f}")
            print(f"Energia alte frequenze - Generato: {avg_results['avg_gen_high_freq_energy']:.4f}")
            print(f"Energia alte frequenze - Reale: {avg_results['avg_real_high_freq_energy']:.4f}")
            
            # Visualizzazione statistica
            plt.figure(figsize=(15, 5))
            
            # Distribuzione rapporto alte/basse frequenze
            plt.subplot(1, 2, 1)
            gen_ratios = [r['gen_high_low_ratio'] for r in all_results]
            real_ratios = [r['real_high_low_ratio'] for r in all_results]
            plt.hist(gen_ratios, alpha=0.5, label='Generato')
            plt.hist(real_ratios, alpha=0.5, label='Reale')
            plt.xlabel('Rapporto alte/basse frequenze')
            plt.ylabel('Numero immagini')
            plt.legend()
            plt.title('Distribuzione del rapporto alte/basse frequenze')
            
            # Punteggi qualità
            plt.subplot(1, 2, 2)
            quality_scores = [r['wavelet_quality']['overall_wavelet_score'] for r in all_results]
            plt.hist(quality_scores, bins=10)
            plt.xlabel('Punteggio qualità')
            plt.ylabel('Numero immagini')
            plt.title('Distribuzione dei punteggi di qualità')
            
            plt.tight_layout()
            plt.savefig("distr_benchmark.png")
        
        return avg_results
    #####
    def wavelet_subband_energy_analysis(self, img_tensor: torch.Tensor) -> Dict[str, float]:
        """
        Calcola metriche dettagliate sull'energia nei diversi sottobandi wavelet.
        
        Args:
            img_tensor: tensore dell'immagine [B, C, H, W]
            
        Returns:
            Dizionario con metriche sull'energia nei sottobandi
        """
        # Decomposizione wavelet
        yl, yh = self.compute_wavelet_decomposition(img_tensor)
        
        # Inizializza dizionario risultati
        results = {}
        
        # Energia basse frequenze (coefficienti di approssimazione - LL)
        ll_energy = torch.sum(yl ** 2).item()
        results['ll_energy'] = ll_energy
        
        # Energia alte frequenze per ogni livello e orientazione
        total_hf_energy = 0.0
        for j in range(self.J):
            level_energy = torch.sum(yh[j] ** 2).item()
            results[f'level{j+1}_energy'] = level_energy
            total_hf_energy += level_energy
            
            # Se i coefficienti sono organizzati per orientazione (come in PyTorch Wavelets)
            if yh[j].dim() == 5:  # [B, C, O, H, W] dove O è l'orientazione
                for o in range(yh[j].shape[2]):
                    # Estrai energia per orientazione (LH, HL, HH)
                    orientation_names = ['LH', 'HL', 'HH']
                    if o < len(orientation_names):
                        o_name = orientation_names[o]
                        o_energy = torch.sum(yh[j][:, :, o] ** 2).item()
                        results[f'level{j+1}_{o_name}_energy'] = o_energy
        
        # Energia totale
        total_energy = ll_energy + total_hf_energy
        results['total_energy'] = total_energy
        
        # Normalizzazione rispetto all'energia totale
        results['ll_energy_ratio'] = ll_energy / total_energy if total_energy > 0 else 0
        results['hf_energy_ratio'] = total_hf_energy / total_energy if total_energy > 0 else 0
        
        for j in range(self.J):
            if f'level{j+1}_energy' in results:
                results[f'level{j+1}_energy_ratio'] = results[f'level{j+1}_energy'] / total_energy if total_energy > 0 else 0
        
        # Rapporto energia alte/basse frequenze (già implementato in ratio_high_low_frequency_energy)
        results['hf_lf_ratio'] = total_hf_energy / ll_energy if ll_energy > 0 else 0
        
        return results

    def enhanced_power_spectrum_analysis(self, img_tensor: torch.Tensor) -> Dict[str, float]:
        """
        Analisi avanzata dello spettro di potenza, includendo il calcolo della pendenza log-log.
        
        Args:
            img_tensor: tensore dell'immagine [B, C, H, W]
            
        Returns:
            Dizionario con metriche avanzate dello spettro di potenza
        """
        results = {}
        
        # Calcola lo spettro di potenza per ogni canale
        for c in range(img_tensor.shape[1]): # Corrected loop
            channel = img_tensor[0, c].cpu().numpy()
            
            # FFT 2D
            fft = np.fft.fft2(channel)
            fft_shifted = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shifted)
            power_spectrum = magnitude ** 2
            
            # Normalizzazione
            power_spectrum_norm = power_spectrum / np.sum(power_spectrum)
            
            # Calcolo della distanza radiale dal centro
            h, w = power_spectrum.shape
            center_y, center_x = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            
            # Raggruppa per distanza per calcolo media radiale
            max_dist = int(np.sqrt(center_y**2 + center_x**2))
            radial_prof = np.zeros(max_dist)
            bin_counts = np.zeros(max_dist)
            
            # Calcola media radiale
            for i in range(h):
                for j in range(w):
                    rad = int(distances[i, j])
                    if rad < max_dist:
                        radial_prof[rad] += power_spectrum_norm[i, j]
                        bin_counts[rad] += 1
            
            # Evita divisione per zero
            bin_counts[bin_counts == 0] = 1
            radial_prof /= bin_counts
            
            # Calcolo della pendenza log-log (escludendo DC e frequenze molto basse)
            # Usa solo la parte centrale dello spettro per evitare artefatti
            start_idx = max(1, int(max_dist * 0.05))  # Escludi frequenze molto basse
            end_idx = int(max_dist * 0.8)  # Escludi frequenze molto alte (spesso rumorose)
            
            # Assicurati che abbiamo abbastanza punti
            if end_idx - start_idx > 10:
                # Crea array di frequenze (asse x)
                freqs = np.arange(start_idx, end_idx)
                
                # Log-log plot
                log_freqs = np.log(freqs)
                log_power = np.log(radial_prof[start_idx:end_idx])
                
                # Elimina eventuali -inf (log di zero)
                valid_indices = np.isfinite(log_power)
                if np.sum(valid_indices) > 10:  # Assicurati di avere abbastanza punti validi
                    # Calcola pendenza usando regressione lineare
                    from scipy.stats import linregress
                    slope, intercept, r_value, p_value, std_err = linregress(
                        log_freqs[valid_indices], 
                        log_power[valid_indices]
                    )
                    
                    results[f'channel_{c}_log_log_slope'] = slope
                    results[f'channel_{c}_log_log_r_squared'] = r_value**2
            
            # Bande di frequenza (come nell'implementazione esistente)
            max_dist = np.sqrt(center_y**2 + center_x**2)
            low_freq_mask = distances <= 0.15 * max_dist
            mid_freq_mask = (distances > 0.15 * max_dist) & (distances <= 0.4 * max_dist)
            high_freq_mask = distances > 0.4 * max_dist
            
            # Energia in ciascuna banda
            low_freq_energy = np.sum(power_spectrum_norm[low_freq_mask])
            mid_freq_energy = np.sum(power_spectrum_norm[mid_freq_mask])
            high_freq_energy = np.sum(power_spectrum_norm[high_freq_mask])
            
            # Salvataggio risultati per canale
            results[f'channel_{c}_low_freq_energy'] = float(low_freq_energy)
            results[f'channel_{c}_mid_freq_energy'] = float(mid_freq_energy)
            results[f'channel_{c}_high_freq_energy'] = float(high_freq_energy)
            results[f'channel_{c}_high_to_low_ratio'] = float(high_freq_energy / (low_freq_energy + 1e-8))
        
        # Media tra i canali
        slopes = [results[f'channel_{c}_log_log_slope'] for c in range(img_tensor.shape[1]) 
                  if f'channel_{c}_log_log_slope' in results]
        if slopes:
            results['avg_log_log_slope'] = np.mean(slopes)
        
        # Media dei valori per banda (come nell'implementazione esistente)
        results['avg_low_freq_energy'] = np.mean([results[f'channel_{c}_low_freq_energy'] 
                                                 for c in range(img_tensor.shape[1])])
        results['avg_mid_freq_energy'] = np.mean([results[f'channel_{c}_mid_freq_energy'] 
                                                 for c in range(img_tensor.shape[1])])
        results['avg_high_freq_energy'] = np.mean([results[f'channel_{c}_high_freq_energy'] 
                                                  for c in range(img_tensor.shape[1])])
        results['avg_high_to_low_ratio'] = np.mean([results[f'channel_{c}_high_to_low_ratio'] 
                                                   for c in range(img_tensor.shape[1])])
        
        return results

    def high_frequency_emphasis_index(self, gen_img_tensor: torch.Tensor, 
                                    real_img_tensor: torch.Tensor = None) -> Dict[str, float]:
        """
        Calcola l'indice di enfasi alta frequenza usando filtri passa-alto.
        
        Args:
            gen_img_tensor: tensore dell'immagine generata [B, C, H, W]
            real_img_tensor: tensore dell'immagine reale [B, C, H, W] (opzionale)
            
        Returns:
            Dizionario con l'indice di enfasi alta frequenza e distanze relative
        """
        results = {}
        
        # Definisci filtri passa-alto
        # 1. Laplaciano (rileva bordi in tutte le direzioni)
        laplacian_filter = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).to(self.device).view(1, 1, 3, 3)
        
        # 2. Sobel (rileva bordi orizzontali e verticali)
        sobel_h_filter = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).to(self.device).view(1, 1, 3, 3)
        
        sobel_v_filter = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).to(self.device).view(1, 1, 3, 3)
        
        # Calcola il contenuto ad alta frequenza per l'immagine generata
        gen_hf_metrics = self._apply_high_pass_filters(gen_img_tensor, laplacian_filter, 
                                                    sobel_h_filter, sobel_v_filter)
        for key, value in gen_hf_metrics.items():
            results[f'gen_{key}'] = value
        
        # Se è fornita un'immagine reale, calcola anche per essa e confronta
        if real_img_tensor is not None:
            # Assicurati che le dimensioni corrispondano
            if gen_img_tensor.shape != real_img_tensor.shape:
                real_img_tensor = F.interpolate(real_img_tensor, 
                                            size=(gen_img_tensor.shape[2], gen_img_tensor.shape[3]),
                                            mode='bilinear', 
                                            align_corners=False)
            
            real_hf_metrics = self._apply_high_pass_filters(real_img_tensor, laplacian_filter, 
                                                        sobel_h_filter, sobel_v_filter)
            for key, value in real_hf_metrics.items():
                results[f'real_{key}'] = value
            
            # Calcola distanza L1 e L2 tra i residui ad alta frequenza
            for key in ['laplacian_residual', 'sobel_residual']:
                if f'gen_{key}' in results and f'real_{key}' in results:
                    gen_tensor = results[f'gen_{key}_tensor']
                    real_tensor = results[f'real_{key}_tensor']
                    
                    # Distanza L1 (MAE)
                    l1_dist = torch.mean(torch.abs(gen_tensor - real_tensor)).item()
                    results[f'{key}_l1_distance'] = l1_dist
                    
                    # Distanza L2 (MSE)
                    l2_dist = torch.mean((gen_tensor - real_tensor) ** 2).item()
                    results[f'{key}_l2_distance'] = l2_dist
        
        # Rimuovi i tensori dai risultati (mantieni solo metriche scalari)
        keys_to_remove = [key for key in results if key.endswith('_tensor')]
        for key in keys_to_remove:
            del results[key]
        
        return results

    def _apply_high_pass_filters(self, img_tensor: torch.Tensor, 
                                laplacian_filter: torch.Tensor,
                                sobel_h_filter: torch.Tensor, 
                                sobel_v_filter: torch.Tensor) -> Dict[str, float]:
        """
        Applica filtri passa-alto all'immagine e calcola metriche.
        Helper method per high_frequency_emphasis_index.
        
        Args:
            img_tensor: tensore dell'immagine [B, C, H, W]
            laplacian_filter: filtro laplaciano
            sobel_h_filter: filtro sobel orizzontale
            sobel_v_filter: filtro sobel verticale
            
        Returns:
            Dizionario con metriche
        """
        results = {}
        
        # Applica filtri a ogni canale
        laplacian_outputs = []
        sobel_outputs = []
        
        for c in range(img_tensor.shape[1]):
            # Estrai singolo canale
            channel = img_tensor[:, c:c+1]
            
            # Applica Laplaciano
            laplacian_output = F.conv2d(channel, laplacian_filter, padding=1)
            laplacian_outputs.append(laplacian_output)
            
            # Applica Sobel (magnitudine del gradiente)
            sobel_h_output = F.conv2d(channel, sobel_h_filter, padding=1)
            sobel_v_output = F.conv2d(channel, sobel_v_filter, padding=1)
            sobel_magnitude = torch.sqrt(sobel_h_output**2 + sobel_v_output**2)
            sobel_outputs.append(sobel_magnitude)
        
        # Combina i risultati di tutti i canali
        laplacian_residual = torch.cat(laplacian_outputs, dim=1)
        sobel_residual = torch.cat(sobel_outputs, dim=1)
        
        # Calcola varianza (energia) dei residui
        laplacian_variance = torch.var(laplacian_residual).item()
        sobel_variance = torch.var(sobel_residual).item()
        
        # Calcola altre statistiche
        laplacian_mean = torch.mean(torch.abs(laplacian_residual)).item()
        sobel_mean = torch.mean(sobel_residual).item()
        
        # Salva risultati
        results['laplacian_variance'] = laplacian_variance
        results['sobel_variance'] = sobel_variance
        results['laplacian_mean_abs'] = laplacian_mean
        results['sobel_mean'] = sobel_mean
        
        # Indice di enfasi alta frequenza (normalizzato)
        hf_emphasis_index = (laplacian_variance + sobel_variance) / 2.0
        results['hf_emphasis_index'] = hf_emphasis_index
        
        # Salva tensori per confronto (verranno rimossi nel metodo chiamante)
        results['laplacian_residual_tensor'] = laplacian_residual
        results['sobel_residual_tensor'] = sobel_residual
        
        return results
    
    def feature_similarity_index(self, gen_img_tensor: torch.Tensor, 
                                real_img_tensor: torch.Tensor) -> float:
        """
        Calcola l'indice di similarità di feature (FSIM).
        Implementazione semplificata di Feature Similarity Index che considera 
        mappa di fase e gradiente dell'immagine.
        
        Args:
            gen_img_tensor: tensore dell'immagine generata [B, C, H, W]
            real_img_tensor: tensore dell'immagine reale [B, C, H, W]
            
        Returns:
            Valore FSIM (0-1, dove 1 indica perfetta similarità)
        """
        # Assicurati che le dimensioni corrispondano
        if gen_img_tensor.shape != real_img_tensor.shape:
            real_img_tensor = F.interpolate(real_img_tensor, 
                                          size=(gen_img_tensor.shape[2], gen_img_tensor.shape[3]),
                                          mode='bilinear', 
                                          align_corners=False)
        
        # Converti in scala di grigi se necessario
        if gen_img_tensor.shape[1] > 1:
            gen_gray = 0.299 * gen_img_tensor[:, 0] + 0.587 * gen_img_tensor[:, 1] + 0.114 * gen_img_tensor[:, 2]
            real_gray = 0.299 * real_img_tensor[:, 0] + 0.587 * real_img_tensor[:, 1] + 0.114 * real_img_tensor[:, 2]
            gen_gray = gen_gray.unsqueeze(1)
            real_gray = real_gray.unsqueeze(1)
        else:
            gen_gray = gen_img_tensor
            real_gray = real_img_tensor
        
        # 1. Calcola mappe di gradiente (Sobel)
        sobel_h_filter = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).to(self.device).view(1, 1, 3, 3)
        
        sobel_v_filter = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).to(self.device).view(1, 1, 3, 3)
        
        # Gradiente Gen
        gen_grad_h = F.conv2d(gen_gray, sobel_h_filter, padding=1)
        gen_grad_v = F.conv2d(gen_gray, sobel_v_filter, padding=1)
        gen_grad_mag = torch.sqrt(gen_grad_h**2 + gen_grad_v**2)
        
        # Gradiente Real
        real_grad_h = F.conv2d(real_gray, sobel_h_filter, padding=1)
        real_grad_v = F.conv2d(real_gray, sobel_v_filter, padding=1)
        real_grad_mag = torch.sqrt(real_grad_h**2 + real_grad_v**2)
        
        # 2. Calcola Phase Congruency (semplificato usando l'orientazione del gradiente)
        # L'orientazione del gradiente è un'approssimazione della fase locale
        gen_grad_orient = torch.atan2(gen_grad_v, gen_grad_h)
        real_grad_orient = torch.atan2(real_grad_v, real_grad_h)
        
        # 3. Calcola similarità
        # Costanti per evitare instabilità numerica
        T1 = 0.85  # Soglia per magnitudine gradiente
        T2 = 0.6   # Soglia per congruenza di fase
        
        # Similarità di gradiente (analogamente a SSIM)
        grad_similarity = (2.0 * gen_grad_mag * real_grad_mag + T1) / (gen_grad_mag**2 + real_grad_mag**2 + T1)
        
        # Similarità di fase (usando differenza di orientamento)
        phase_diff = torch.abs(gen_grad_orient - real_grad_orient)
        # Normalizza phase_diff per considerare la circolarità (-π e π sono equivalenti)
        phase_diff = torch.min(phase_diff, 2*np.pi - phase_diff)
        phase_similarity = torch.cos(phase_diff)  # -1 a 1
        phase_similarity = (phase_similarity + 1) / 2  # Scala a 0-1
        phase_similarity = (phase_similarity + T2) / (1.0 + T2)  # Applica soglia
        
        # Pesa similarità con la magnitudine del gradiente (aree con gradiente più alto hanno più peso)
        max_grad = torch.max(gen_grad_mag, real_grad_mag)
        weight = torch.exp(-(max_grad / (max_grad.mean() + 1e-8)))
        
        # FSIM finale
        fsim_map = grad_similarity * phase_similarity
        fsim = torch.sum(fsim_map * weight) / (torch.sum(weight) + 1e-8)
        
        return fsim.item()


    def multi_scale_ssim(self, gen_img_tensor: torch.Tensor, 
                     real_img_tensor: torch.Tensor, 
                     max_levels: int = 5) -> float:
        """
        Calcola l'indice di similarità strutturale multi-scala (MS-SSIM).
        
        Args:
            gen_img_tensor: tensore dell'immagine generata [B, C, H, W]
            real_img_tensor: tensore dell'immagine reale [B, C, H, W]
            max_levels: numero massimo di livelli di scala
            
        Returns:
            Valore MS-SSIM (0-1, dove 1 indica perfetta similarità)
        """
        # Assicurati che le dimensioni corrispondano
        if gen_img_tensor.shape != real_img_tensor.shape:
            real_img_tensor = F.interpolate(real_img_tensor, 
                                        size=(gen_img_tensor.shape[2], gen_img_tensor.shape[3]),
                                        mode='bilinear', 
                                        align_corners=False)
        
        # Determina il numero effettivo di livelli basato sulla dimensione dell'immagine
        min_dim = min(gen_img_tensor.shape[2], gen_img_tensor.shape[3])
        levels = min(max_levels, int(np.log2(min_dim)) - 2)
        levels = max(1, levels)  # Almeno un livello
        
        # Pesi per ogni livello (dando più importanza ai livelli più bassi)
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333][:levels]).to(self.device)
        weights = weights / weights.sum()  # Normalizza
        
        # Costanti per SSIM
        C1 = (0.01 * 2) ** 2
        C2 = (0.03 * 2) ** 2
        
        # Kernel Gaussiano per calcolo media/varianza
        window_size = 11
        sigma = 1.5
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / (2*sigma**2)) 
                            for x in range(window_size)]).to(self.device)
        _1D_window = gauss / gauss.sum()
        _2D_window = _1D_window.unsqueeze(1) @ _1D_window.unsqueeze(0)
        window = _2D_window.unsqueeze(0).unsqueeze(0)
        window = window.expand(gen_img_tensor.shape[1], 1, window_size, window_size)
        
        # Calcola SSIM per ogni livello
        mssim = []
        mcs = []
        
        for _ in range(levels):
            # Calcola SSIM a questo livello
            ssim_val, cs = self._ssim_single_scale(gen_img_tensor, real_img_tensor, window, C1, C2)
            mssim.append(ssim_val)
            mcs.append(cs)
            
            # Ridimensiona per il livello successivo
            if _ < levels - 1:
                gen_img_tensor = F.avg_pool2d(gen_img_tensor, kernel_size=2, padding=0)
                real_img_tensor = F.avg_pool2d(real_img_tensor, kernel_size=2, padding=0)
        
        # Combinazione dei livelli
        mcs = torch.stack(mcs[:-1])  # Tutti i contrast structure tranne l'ultimo
        mssim_val = torch.stack(mssim)
        
        # Calcola MS-SSIM
        ms_ssim = torch.prod(mcs ** weights[:-1]) * (mssim[-1] ** weights[-1])
        
        return ms_ssim.item()

    def _ssim_single_scale(self, img1: torch.Tensor, img2: torch.Tensor, 
                        window: torch.Tensor, C1: float, C2: float) -> Tuple[float, float]:
        """
        Calcola SSIM e Contrast Structure per un singolo livello.
        Helper method per multi_scale_ssim.
        
        Args:
            img1, img2: Tensori immagine [B, C, H, W]
            window: Kernel gaussiano [C, 1, window_size, window_size]
            C1, C2: Costanti per stabilità numerica
            
        Returns:
            Tuple (SSIM, Contrast Structure)
        """
        # Calcola medie
        mu1 = F.conv2d(img1, window, padding=0, groups=img1.shape[1])
        mu2 = F.conv2d(img2, window, padding=0, groups=img2.shape[1])
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Calcola varianze e covarianza
        sigma1_sq = F.conv2d(img1 ** 2, window, padding=0, groups=img1.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, window, padding=0, groups=img2.shape[1]) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=0, groups=img1.shape[1]) - mu1_mu2
        
        # Calcola SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        
        # Media
        ssim_val = torch.mean(ssim_map)
        cs = torch.mean(cs_map)
        
        return ssim_val, cs


def generate_plots(methods_results, real_reference, comparison_data, detailed_metrics, results_dir):
    """
    Generate comparison plots for all methods
    
    Args:
        methods_results: Dictionary with results for each method
        real_reference: Dictionary with reference values from real images
        comparison_data: Data formatted for comparison
        detailed_metrics: Dictionary mapping metric names to their keys
        results_dir: Directory to save the plots
    """
    # 1. Basic metrics comparison (bar chart)
    metrics_to_plot = ['High/Low Frequency Ratio', 'Wavelet Quality Score', 
                      'High Frequency Energy', 'FSIM', 'MS-SSIM']
    
    plt.figure(figsize=(18, 8))
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 3, i+1)
        
        # Get index of this metric in the DataFrame
        metric_idx = list(detailed_metrics.keys()).index(metric)
        
        # Get values for all methods and real reference
        methods = list(methods_results.keys()) + ['Real Reference']
        values = [comparison_data[method][metric_idx] for method in methods]
        
        # Create color map with real reference in a different color
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink'][:len(methods)-1] + ['red']
        
        # Create bar chart
        bars = plt.bar(methods, values, color=colors)
        plt.title(metric)
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for j, (bar, v) in enumerate(zip(bars, values)):
            plt.text(bar.get_x() + bar.get_width()/2, v + 0.01, 
                    f"{v:.3f}", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'basic_metrics_comparison.png'), dpi=300)
    
    # 2. Frequency distribution visualization
    plt.figure(figsize=(15, 8))
    
    # High frequency metrics
    hf_metrics = ['HF Emphasis Index', 'Log-log Slope']
    metric_indices = [list(detailed_metrics.keys()).index(m) for m in hf_metrics]
    
    for i, (metric, idx) in enumerate(zip(hf_metrics, metric_indices)):
        plt.subplot(1, 2, i+1)
        
        # Get values for all methods and real reference
        methods = list(methods_results.keys()) + ['Real Reference']
        values = [comparison_data[method][idx] for method in methods]
        
        # Create color map
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink'][:len(methods)-1] + ['red']
        
        # Create bar chart
        bars = plt.bar(methods, values, color=colors)
        plt.title(f"Frequency Analysis: {metric}")
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for j, (bar, v) in enumerate(zip(bars, values)):
            plt.text(bar.get_x() + bar.get_width()/2, v + 0.01, 
                    f"{v:.3f}", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'frequency_distribution_comparison.png'), dpi=300)
    
    # 3. Save a combined chart for paper/presentation use
    plt.figure(figsize=(12, 10))
    
    # Select most important metrics
    key_metrics = ['Wavelet Quality Score', 'FSIM', 'MS-SSIM', 'HF Emphasis Index']
    metric_indices = [list(detailed_metrics.keys()).index(m) for m in key_metrics]
    
    methods = list(methods_results.keys()) + ['Real Reference']
    
    # Set width of bars
    barWidth = 0.25
    
    # Set positions of bar on X axis
    r = np.arange(len(key_metrics))
    positions = [r]
    
    for i in range(1, len(methods)):
        positions.append([x + barWidth for x in positions[i-1]])
    
    # Create bars
    for i, method in enumerate(methods):
        values = [comparison_data[method][idx] for idx in metric_indices]
        plt.bar(positions[i], values, width=barWidth, edgecolor='grey', 
               label=method, alpha=0.8)
    
    # Add labels and legend
    plt.xlabel('Metrics', fontweight='bold')
    plt.ylabel('Scores', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(key_metrics))], key_metrics, rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.title('Comprehensive Quality Evaluation Across Methods')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'comprehensive_comparison.png'), dpi=300)



def main(args):
    """
    Main function to orchestrate the frequency analysis process.
    
    Args:
        args: Command line arguments from argparse
    """
    # Check if method names and dirs have the same length
    if len(args.methods) != len(args.method_dirs):
        raise ValueError("Number of method names and directories must match!")
    
    # Create a dictionary mapping method names to their directories
    method_dirs = {name: dir for name, dir in zip(args.methods, args.method_dirs)}
    
    # Initialize WaveletMetrics with the specified parameters
    metrics = WaveletMetrics(wave=args.wavelet, J=args.levels, resize=args.resize)
    
    # Create output directory
    if args.output_dir:
        results_dir = args.output_dir
    else:
        results_dir = os.path.join(os.path.dirname(args.method_dirs[0]), 'frequency_analysis_results')
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Starting evaluation with {args.max_images} images from {len(method_dirs)} methods...")
    print(f"Results will be saved to: {results_dir}")
    
    # Load real image paths
    real_img_paths_all = [os.path.join(args.real_dir, f) for f in os.listdir(args.real_dir)
                         if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(real_img_paths_all) == 0:
        raise ValueError(f"No valid images found in real directory: {args.real_dir}")
    
    # Evaluate each method and save results
    methods_results = {}
    
    try:
        # Create a detailed metrics list for comprehensive comparison
        detailed_metrics = {
            'High/Low Frequency Ratio': 'avg_gen_high_low_ratio',
            'Ratio Difference from Real': 'avg_ratio_difference',
            'Wavelet Quality Score': 'avg_wavelet_quality_score',
            'High Frequency Energy': 'avg_gen_high_freq_energy',
            'FSIM': 'avg_fsim',  # New metric
            'MS-SSIM': 'avg_ms_ssim',  # New metric
            'HF Emphasis Index': 'avg_hf_emphasis',  # New metric
            'Log-log Slope': 'avg_log_log_slope'  # New metric
        }
        
        # Create result storage for real references
        real_reference = {}
        
        # Process each method
        for method_name, method_dir in method_dirs.items():
            print(f"\n===== Evaluating {method_name} Method =====")
            
            # Get image paths for this method
            gen_img_paths_all = [os.path.join(method_dir, f) for f in os.listdir(method_dir)
                              if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            if len(gen_img_paths_all) == 0:
                print(f"Warning: No valid images found in method directory: {method_dir}")
                continue

            # Determine the number of pairs to sample
            max_pairs = min(len(gen_img_paths_all), len(real_img_paths_all))
            
            num_to_select = max_pairs
            if args.max_images is not None and args.max_images > 0:
                num_to_select = min(args.max_images, max_pairs)

            if num_to_select == 0:
                print(f"Warning: No image pairs to select for method {method_name}")
                continue

            # Generate random indices for sampling
            available_indices = list(range(max_pairs))
            chosen_indices = random.sample(available_indices, num_to_select)
            
            # Select the images based on the chosen_indices
            gen_img_paths = [gen_img_paths_all[i] for i in chosen_indices]
            real_img_paths = [real_img_paths_all[i] for i in chosen_indices]
            
            print(f"Selected {len(gen_img_paths)} image pairs for evaluation for method {method_name}.")

            # Run batch evaluation
            results = metrics.batch_evaluate(
                gen_img_paths, # Use selected paths
                real_img_paths, # Use selected paths
                summary=True
            )
            
            # Store all individual results for calculating aggregated metrics
            all_results = []
            for i, (gen_path, real_path) in enumerate(zip(gen_img_paths, real_img_paths)): # Use selected paths
                try:
                    img_result = metrics.compare_images(gen_path, real_path, visualize=False)
                    all_results.append(img_result)
                except Exception as e:
                    print(f"Error processing pair {i}: {e}")
            
            # Calculate additional metrics from individual results
            if all_results:
                # Add NaN checking for FSIM
                fsim_values = [r.get('fsim', 0.0) for r in all_results]
                fsim_values = [v for v in fsim_values if not np.isnan(v)]  # Filter out NaN values
                results['avg_fsim'] = np.mean(fsim_values) if fsim_values else 0.0
                
                # Add NaN checking for MS-SSIM
                ms_ssim_values = [r.get('ms_ssim', 0.0) for r in all_results]
                ms_ssim_values = [v for v in ms_ssim_values if not np.isnan(v)]  # Filter out NaN values
                results['avg_ms_ssim'] = np.mean(ms_ssim_values) if ms_ssim_values else 0.0
                
                # Add NaN checking for HF emphasis
                hf_emphasis_values = []
                for r in all_results:
                    if 'hf_emphasis' in r and 'gen_hf_emphasis_index' in r['hf_emphasis']:
                        value = r['hf_emphasis']['gen_hf_emphasis_index']
                        if not np.isnan(value):
                            hf_emphasis_values.append(value)
                results['avg_hf_emphasis'] = np.mean(hf_emphasis_values) if hf_emphasis_values else 0.0
                
                # Calculate log-log slope metrics (negative values are expected and correct)
                # Natural images have power spectrum that falls off with frequency (~1/f^2 relationship)
                log_slopes = []
                for r in all_results:
                    if 'gen_enhanced_spectrum' in r and 'avg_log_log_slope' in r['gen_enhanced_spectrum']:
                        slope = r['gen_enhanced_spectrum']['avg_log_log_slope']
                        if not np.isnan(slope) and -5.0 < slope < 0:
                            log_slopes.append(slope)
                results['avg_log_log_slope'] = np.mean(log_slopes) if log_slopes else -2.0  # Default to typical value
                
                # Print explanation for log-log slope
                if log_slopes:
                    print(f"  Log-log slope avg: {results['avg_log_log_slope']:.3f} (negative is expected)")
                    print("  Note: Log-log slope is typically negative for natural images (-1 to -3)")
                    print("  More negative values indicate smoother images with less high-frequency content")
                    print("  Less negative values indicate more textured images with more details")
                
                # Store real metrics for comparison with similar NaN handling
                if not real_reference:
                    real_reference['avg_high_low_ratio'] = results.get('avg_real_high_low_ratio', 0.0)
                    real_reference['avg_high_freq_energy'] = results.get('avg_real_high_freq_energy', 0.0)
                    
                    # Calculate real log_log_slope based on the *currently selected* real images for this method
                    # This ensures consistency if different methods end up with different numbers of pairs
                    # due to their own gen_img_paths limitations.
                    # However, for a global real_reference, it might be better to calculate it once
                    # on a fixed random subset of real_img_paths_all if args.max_images is used,
                    # or on all real_img_paths_all if args.max_images is not restrictive.
                    # For now, this uses the 'real' part of the current method's evaluation.
                    current_method_real_log_slopes = []
                    for r_idx, r_data in enumerate(all_results): # Iterate through current method's all_results
                        if 'real_enhanced_spectrum' in r_data and 'avg_log_log_slope' in r_data['real_enhanced_spectrum']:
                            slope = r_data['real_enhanced_spectrum']['avg_log_log_slope']
                            if not np.isnan(slope) and -5.0 < slope < 0:
                                current_method_real_log_slopes.append(slope)
                    
                    # Update real_reference only if it's the first method or if we want to average across methods
                    # For simplicity, let's assume the first method's 'real' values are representative enough
                    # or that batch_evaluate's real_... metrics are sufficient.
                    if not real_reference.get('avg_log_log_slope_calculated', False):
                        real_reference['avg_log_log_slope'] = np.mean(current_method_real_log_slopes) if current_method_real_log_slopes else -2.0
                        real_reference['avg_log_log_slope_calculated'] = True


            # Store results
            methods_results[method_name] = results
            
        # Create comparison DataFrame
        print("\n===== Comparison between Methods =====")
        
        # Initialize data for DataFrame
        comparison_data = {'Metric': list(detailed_metrics.keys())}
        
        # Add data for each method
        for method_name, results in methods_results.items():
            method_values = []
            for metric_name, metric_key in detailed_metrics.items():
                # Add the metric value if it exists, otherwise use 0.0
                method_values.append(results.get(metric_key, 0.0))
            comparison_data[method_name] = method_values
        
        # Add real reference data
        real_values = [
            real_reference.get('avg_high_low_ratio', 0.0),
            0.0,  # No difference from itself
            1.0,  # Perfect quality score
            real_reference.get('avg_high_freq_energy', 0.0),
            1.0,  # Perfect FSIM with itself
            1.0,  # Perfect MS-SSIM with itself
            0.0,  # No reference for HF emphasis (or could be calculated on real_images if desired)
            real_reference.get('avg_log_log_slope', -2.0) # Use the calculated or default
        ]
        comparison_data['Real Reference'] = real_values
        
        # Create DataFrame and print
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df)
        
        # Save comparison to CSV
        comparison_df.to_csv(os.path.join(results_dir, 'methods_comparison.csv'), index=False)
        
        # Generate plots if not disabled
        if not args.no_plots:
            generate_plots(methods_results, real_reference, comparison_data, detailed_metrics, results_dir)
            print(f"Plots saved to: {results_dir}")
        else:
            print("Plot generation disabled by --no_plots flag")
        
        print(f"\nEvaluation complete! Results saved to: {results_dir}")
        
    except Exception as e:
        import traceback
        print(f"Error during evaluation: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Wavelet-based frequency analysis for image comparison')
    parser.add_argument('--real_dir', type=str, default="/leonardo_scratch/large/userexternal/lsigillo/HPDv2/test",
                      help='Directory containing real images')
    parser.add_argument('--methods', type=str, nargs='+', 
                       default=['WALD', 'URAE'],
                       help='Method names to evaluate')
    parser.add_argument('--method_dirs', type=str, nargs='+',
                       default=["/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/output/URAE_VAE_SE_WAV_ATT_LAION/HPDv2_test_set",
                                "/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/output/URAE_original_trained_by_me/HPDv2_test_set"],
                       help='Directories containing generated images for each method')
    parser.add_argument('--max_images', type=int, default=100, # Keep default or user can change
                       help='Maximum number of images to evaluate (randomly selected)')
    parser.add_argument('--resize', type=int, default=512,
                       help='Resize images to this size for analysis')
    parser.add_argument('--wavelet', type=str, default='haar',
                       help='Wavelet type (haar, db2, etc.)')
    parser.add_argument('--levels', type=int, default=2,
                       help='Number of wavelet decomposition levels')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save results (default: creates frequency_analysis_results in parent dir)')
    parser.add_argument('--no_plots', action='store_true',
                       help='Disable plot generation and saving')
    
    args = parser.parse_args()
    
    main(args)

