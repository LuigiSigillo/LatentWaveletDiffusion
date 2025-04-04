import duckdb
import requests
import json
import os
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import logging
import argparse
import hashlib
import glob
import re
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_parquet_files(folder_path):
    """
    Find all parquet files in the given directory
    
    Args:
        folder_path (str): Path to the folder containing parquet files
    
    Returns:
        list: List of paths to parquet files
    """
    if not os.path.isdir(folder_path):
        logger.error(f"The path {folder_path} is not a directory")
        return []
    
    parquet_files = glob.glob(os.path.join(folder_path, "*.parquet"))
    logger.info(f"Found {len(parquet_files)} parquet files in {folder_path}")
    return parquet_files

def filter_parquet_files(parquet_files, min_width=1920, min_height=1080):
    """
    Filter parquet files to get rows with WIDTH >= min_width and HEIGHT >= min_height
    
    Args:
        parquet_files (list): List of paths to parquet files
        min_width (int): Minimum width for filtering images
        min_height (int): Minimum height for filtering images
    
    Returns:
        DataFrame: Filtered data with columns URL, TEXT, WIDTH, HEIGHT
    """
    conn = duckdb.connect(database=':memory:')
    
    if not parquet_files:
        logger.error("No parquet files found to process")
        return conn.execute("SELECT URL, TEXT, WIDTH, HEIGHT FROM (VALUES) LIMIT 0").fetchdf()
    
    # Get total row count before filtering
    count_queries = []
    for file_path in parquet_files:
        count_queries.append(f"SELECT COUNT(*) FROM read_parquet('{file_path}')")
    
    total_count_query = "+".join([f"({q})" for q in count_queries])
    total_count = conn.execute(f"SELECT {total_count_query}").fetchone()[0]
    logger.info(f"Total rows before filtering: {total_count}")
    
    # Filter the data
    queries = []
    for file_path in parquet_files:
        queries.append(
            f"SELECT * FROM read_parquet('{file_path}') "
            f"WHERE WIDTH >= {min_width} AND HEIGHT >= {min_height}"
        )
    
    query = " UNION ALL ".join(queries) if len(queries) > 1 else queries[0]
    
    logger.info(f"Filtering parquet files for images with width >= {min_width} and height >= {min_height}...")
    filtered_data = conn.execute(query).fetchdf()
    filtered_count = len(filtered_data)
    logger.info(f"Filtered data contains {filtered_count} rows ({filtered_count/total_count:.2%} of original data)")
    
    return filtered_data

def download_image(url, save_path):
    """
    Download an image from a URL and save it
    
    Args:
        url (str): URL of the image to download
        save_path (str): Path where to save the image
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        # Remove partially downloaded file if it exists
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
                logger.debug(f"Removed partial download: {save_path}")
            except Exception as remove_error:
                logger.warning(f"Could not remove partial download {save_path}: {remove_error}")
        return False

def save_json_metadata(text, save_path, original_filename, url):
    """
    Save metadata (prompt) to a JSON file
    
    Args:
        text (str): The prompt text
        save_path (str): Path to save the JSON file
        original_filename (str): Original filename from URL
        url (str): Original URL of the image
        
    Returns:
        bool: True if successful, False otherwise
    """
    metadata = {
        "prompt": text,
        "original_filename": original_filename,
        "source_url": url
        # "generated_prompt" field would be added later when using VLMs
    }
    
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {save_path}: {e}")
        return False

def get_filename_from_url(url):
    """
    Extract the original filename from URL
    
    Args:
        url (str): The URL of the image
        
    Returns:
        tuple: (base_name, ext) where base_name is the original filename and ext is the file extension
    """
    url_path = url.split('/')[-1]
    base_name = os.path.splitext(url_path)[0]
    ext = os.path.splitext(url_path)[1] or '.jpg'  # Default to .jpg if no extension
    
    # If base_name is empty, use a placeholder
    if not base_name:
        base_name = "unknown"
    
    return base_name, ext

def process_row(row, output_dir, file_id):
    """
    Process a single row: download image and save JSON metadata
    
    Args:
        row (Series): Row from DataFrame with URL and TEXT
        output_dir (str): Directory to save files
        file_id (int): Progressive ID for the filename
        
    Returns:
        dict: Result information
    """
    url = row['URL']
    text = row['TEXT']
    
    # Get original filename from URL for metadata
    original_filename, ext = get_filename_from_url(url)
    
    # Use progressive ID for actual filenames
    image_filename = f"image_{file_id:08d}{ext}"
    json_filename = f"image_{file_id:08d}.json"
    
    image_path = os.path.join(output_dir, image_filename)
    json_path = os.path.join(output_dir, json_filename)
    
    # Download image first
    image_success = download_image(url, image_path)
    
    # Only save JSON metadata if image download was successful
    json_success = False
    if image_success:
        json_success = save_json_metadata(text, json_path, original_filename, url)
        
        # If JSON creation fails, remove the image to keep pairs consistent
        if not json_success and os.path.exists(image_path):
            try:
                os.remove(image_path)
                logger.debug(f"Removed image due to JSON creation failure: {image_path}")
            except Exception as remove_error:
                logger.warning(f"Could not remove image {image_path} after JSON creation failed: {remove_error}")
    
    return {
        "url": url,
        "id": file_id,
        "image_path": image_path if image_success else None,
        "json_path": json_path if json_success else None,
        "original_filename": original_filename,
        "success": image_success and json_success
    }

def download_images_with_metadata(filtered_data, output_dir, max_workers=10, limit=None):
    """
    Download images and save metadata from filtered data
    
    Args:
        filtered_data (DataFrame): DataFrame containing URLs and texts
        output_dir (str): Directory to save images and JSON files
        max_workers (int): Maximum number of concurrent downloads
        limit (int, optional): Maximum number of files to download
        
    Returns:
        list: Results of the download operations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply limit if provided
    if limit and limit > 0 and limit < len(filtered_data):
        logger.info(f"Limiting downloads to {limit} images (out of {len(filtered_data)} available)")
        filtered_data = filtered_data.head(limit)
    
    results = []
    total = len(filtered_data)
    
    logger.info(f"Starting download of {total} images...")
    
    # Create a dictionary to map futures to (url, file_id) pairs
    future_to_data = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks with progressive file IDs
        for idx, (_, row) in enumerate(filtered_data.iterrows(), 1):
            future = executor.submit(process_row, row, output_dir, idx)
            future_to_data[future] = (row['URL'], idx)
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_data), total=total):
            url, file_id = future_to_data[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {url} (ID: {file_id}): {e}")
                results.append({
                    "url": url, 
                    "id": file_id,
                    "success": False, 
                    "error": str(e)
                })
    
    # Summarize results
    successes = sum(1 for r in results if r.get("success", False))
    logger.info(f"Downloaded {successes} of {total} images successfully")
    
    return results

def save_filtered_data(filtered_data, output_path):
    """Save filtered data to a parquet file"""
    # Ensure directory exists but treat output_path as a file
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if there's a directory component
        os.makedirs(output_dir, exist_ok=True)
    
    filtered_data.to_parquet(output_path)
    logger.info(f"Saved filtered data to {output_path}")


def cleanup_folder(folder_path):
    """
    Remove non-JSON/JPG files and orphaned JSON files from a folder.

    Args:
        folder_path (str): The path to the folder to clean up.
    """
    allowed_extensions = {'.json', '.jpg', '.jpeg'}
    json_files = set()
    jpg_files = set()

    # First, collect all JSON and JPG filenames (without extension) and remove invalid files
    for filename in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            name, ext = os.path.splitext(filename)
            ext_lower = ext.lower()

            if ext_lower in allowed_extensions:
                if ext_lower == '.json':
                    json_files.add(name)
                else:  # .jpg or .jpeg
                    jpg_files.add(name)
            else:
                try:
                    os.remove(file_path)
                    logger.info(f"Removed non-image/JSON file: {file_path}")
                except Exception as e:
                    logger.error(f"Error removing {file_path}: {e}")

    # Identify JSON files without a corresponding JPG file
    orphaned_json_files = json_files - jpg_files

    # Remove orphaned JSON files
    for name in orphaned_json_files:
        json_path = os.path.join(folder_path, name + '.json')
        try:
            os.remove(json_path)
            logger.info(f"Removed orphaned JSON file: {json_path}")
        except FileNotFoundError:
            logger.warning(f"JSON file not found: {json_path}")
        except Exception as e:
            logger.error(f"Error removing {json_path}: {e}")


def remove_corrupted_images(folder_path):
    """
    Check for corrupted images in a folder and remove them along with their associated JSON and .safetensors files.

    Args:
        folder_path (str): Path to the folder containing images and associated files.
    """
    for filename in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                # Attempt to open the image to check if it's valid
                with Image.open(file_path) as img:
                    img.verify()  # Verify the image integrity
            except Exception as e:
                logger.warning(f"Corrupted image detected and removed: {file_path} ({e})")
                try:
                    os.remove(file_path)  # Remove the corrupted image
                except Exception as remove_error:
                    logger.error(f"Error removing corrupted image {file_path}: {remove_error}")
                
                # Remove the associated JSON file
                json_file = os.path.splitext(file_path)[0] + '.json'
                if os.path.exists(json_file):
                    try:
                        os.remove(json_file)
                        logger.info(f"Removed associated JSON file: {json_file}")
                    except Exception as json_remove_error:
                        logger.error(f"Error removing JSON file {json_file}: {json_remove_error}")
                


def main():
    parser = argparse.ArgumentParser(description="Filter parquet files and download images with metadata")
    parser.add_argument("--parquet-folder", required=True, help="Path to folder containing parquet files")
    parser.add_argument("--output-dir", required=True, help="Directory to save images and JSON files")
    parser.add_argument("--filtered-output", help="File path to save filtered parquet data")
    parser.add_argument("--max-workers", type=int, default=10, help="Maximum number of concurrent downloads")
    parser.add_argument("--limit", type=int, default=20000, help="Maximum number of files to download (default: 20000)")
    parser.add_argument("--min-width", type=int, default=1920, help="Minimum image width for filtering (default: 1920)")
    parser.add_argument("--min-height", type=int, default=1080, help="Minimum image height for filtering (default: 1080)")
    
    args = parser.parse_args()
    
    # Find parquet files in the specified folder
    parquet_files = find_parquet_files(args.parquet_folder)
    
    # Filter parquet files
    filtered_data = filter_parquet_files(parquet_files, args.min_width, args.min_height)
    
    # Save filtered data if output path is provided
    if args.filtered_output:
        save_filtered_data(filtered_data, args.filtered_output)
    
    # Download images and create JSON files
    results = download_images_with_metadata(filtered_data, args.output_dir, args.max_workers, args.limit)
    
    # Save results summary
    results_path = os.path.join(args.output_dir, "download_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    logger.info(f"Download results saved to {results_path}")

    remove_corrupted_images(args.output_dir)
    cleanup_folder(args.output_dir)
    # i want to summarize the results of the download counting the number of images downloaded after the cleaning
    downloaded_images = len([f for f in os.listdir(args.output_dir) if f.lower().endswith(('.jpg', '.jpeg'))])
    logger.info(f"Total images downloaded after cleaning: {downloaded_images}")


if __name__ == "__main__":
    main()
    
