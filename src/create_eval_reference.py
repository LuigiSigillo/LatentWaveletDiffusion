#!/usr/bin/env python3
import os
import random
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

def is_image(filename):
    """Check if a file is an image based on its extension."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'}
    return os.path.splitext(filename.lower())[1] in image_extensions

def is_valid_image(filepath):
    """Try to open the image to verify it's valid."""
    try:
        with Image.open(filepath) as img:
            # Force loading the image data
            # img.verify()
            return True
    except Exception as e:
        return False

def random_image_selection(source_dir, destination_dir, num_images=3200, seed=None):
    """
    Randomly select and copy images from source directory to destination directory.
    
    Args:
        source_dir (str): Path to source directory
        destination_dir (str): Path to destination directory
        num_images (int): Number of images to select and copy
        seed (int, optional): Random seed for reproducibility
    """
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Collect all files with image extensions
    print("Finding image files...")
    image_candidates = []
    for f in tqdm(os.listdir(source_dir)):
        if is_image(f) and os.path.isfile(os.path.join(source_dir, f)):
            image_candidates.append(f)
    
    print(f"Found {len(image_candidates)} files with image extensions in {source_dir}")
    
    if len(image_candidates) < num_images:
        print(f"Warning: Only {len(image_candidates)} potential images available, which is less than requested {num_images}")
    
    # Randomly select and validate images until we have enough
    valid_images = []
    invalid_count = 0
    attempts = 0
    max_attempts = min(len(image_candidates) * 2, 100000)  # Avoid infinite loops
    
    # Shuffle once instead of repeatedly sampling
    random.shuffle(image_candidates)
    candidate_index = 0
    
    print(f"Selecting {num_images} valid images...")
    with tqdm(total=num_images) as pbar:
        while len(valid_images) < num_images and attempts < max_attempts and candidate_index < len(image_candidates):
            # Get next candidate
            image = image_candidates[candidate_index]
            candidate_index += 1
            attempts += 1
            
            # Validate the image
            filepath = os.path.join(source_dir, image)
            if is_valid_image(filepath):
                valid_images.append(image)
                pbar.update(1)
            else:
                invalid_count += 1
    
    if len(valid_images) < num_images:
        print(f"Warning: Could only find {len(valid_images)} valid images after checking {attempts} files")
    
    print(f"Found {len(valid_images)} valid images, rejected {invalid_count} invalid images")
    
    # Copy selected images to destination
    print(f"Copying images to {destination_dir}")
    copy_success = 0
    for image in tqdm(valid_images):
        source_path = os.path.join(source_dir, image)
        dest_path = os.path.join(destination_dir, image)
        try:
            shutil.copy2(source_path, dest_path)
            copy_success += 1
        except (shutil.Error, OSError) as e:
            print(f"Error copying {image}: {e}")
    
    print(f"Successfully copied {copy_success} images to {destination_dir}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly select and copy images from a source folder to a destination folder")
    parser.add_argument("--source_dir", help="Source directory containing images")
    parser.add_argument("--destination_dir", help="Destination directory to copy selected images")
    parser.add_argument("--num", type=int, default=3200, help="Number of images to select (default: 3200)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    random_image_selection(args.source_dir, args.destination_dir, args.num, args.seed)