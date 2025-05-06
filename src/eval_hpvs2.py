import os
from glob import glob
from PIL import Image  # Removed ImageResampling import
import numpy as np
import pyiqa  # Added import for pyiqa
import lpips
import torch
import random  # Added import for random
from eval import calculate_lpips
# import
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
from tqdm import tqdm
import hpsv2
from pipeline_flux import FluxPipeline
from transformer_flux import FluxTransformer2DModel
from multiprocessing import Process
import argparse


def calculate_fid_for_all(generated_folder, reference_folder):
    """
    Compute FID for all generated subfolders against all reference images.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fid_metric = pyiqa.create_metric("fid", device=device)

    # Compute FID for the entire generated folder against the reference folder
    fid_score = fid_metric(generated_folder, reference_folder)
    print(f"FID score for all generated subfolders against all reference images: {fid_score:.4f}")
    return fid_score

def calculate_fid_and_lpips(generated_folder, reference_folder, compute_fid=True, compute_lpips=True):
    """
    Compute FID and/or LPIPS for generated subfolders against reference images.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fid_metric = pyiqa.create_metric("fid", device=device) if compute_fid else None

    # Get subfolders in the generated folder
    generated_subfolders = [os.path.join(generated_folder, subfolder) for subfolder in os.listdir(generated_folder) if os.path.isdir(os.path.join(generated_folder, subfolder))]

    fid_values = []
    lpips_values = []

    for gen_subfolder in generated_subfolders:
        # Get generated and reference images
        generated_images = sorted(glob(os.path.join(gen_subfolder, "*.jpg")))
        reference_images = sorted(glob(os.path.join(reference_folder, "*.jpg")))

        # Randomly select a subset of reference images if there are more than generated images
        if len(reference_images) > len(generated_images):
            reference_images = random.sample(reference_images, len(generated_images))
            # Create a temporary folder for the sampled reference images
            temp_ref_folder = "./tmp/sampled_reference"
            os.makedirs(temp_ref_folder, exist_ok=True)
            for img_path in reference_images:
                os.symlink(img_path, os.path.join(temp_ref_folder, os.path.basename(img_path)))

            # Compute FID using the temporary folder
            if compute_fid:
                fid_score = fid_metric(gen_subfolder, temp_ref_folder)
                print(f"FID score between {gen_subfolder} and sampled reference images: {fid_score:.4f}")
                fid_values.append(fid_score)

            # Clean up the temporary folder
            for file in os.listdir(temp_ref_folder):
                os.unlink(os.path.join(temp_ref_folder, file))
            os.rmdir(temp_ref_folder)
        else:
            # Compute FID for the current subfolder using fid_metric
            if compute_fid:
                fid_score = fid_metric(gen_subfolder, reference_folder)
                print(f"FID score between {gen_subfolder} and {reference_folder}: {fid_score:.4f}")
                fid_values.append(fid_score)

        # Compute LPIPS for the current subfolder using the helper function
        if compute_lpips:
            lpips_score = calculate_lpips(generated_images, reference_images)
            print(f"LPIPS score between {gen_subfolder} and {reference_folder}: {lpips_score:.4f}")
            lpips_values.append(lpips_score)

    # Print results
    if compute_fid:
        avg_fid = np.mean(fid_values)
        print(f"FID for each subfolder: {fid_values}")
        print(f"Average FID: {avg_fid}")

    if compute_lpips:
        avg_lpips = np.mean(lpips_values)
        print(f"LPIPS for each subfolder: {lpips_values}")
        print(f"Average LPIPS: {avg_lpips}")

def calculate_lpips_for_all(generated_folder, reference_folder):
    """
    Compute LPIPS for all generated subfolders against all reference images.
    """

    # Get all generated and reference images
    generated_images = sorted(glob(os.path.join(generated_folder, "**", "*.jpg"), recursive=True))
    reference_images = sorted(glob(os.path.join(reference_folder, "*.jpg")))

    # Match the number of reference images to the generated images if needed
    if len(reference_images) > len(generated_images):
        reference_images = random.sample(reference_images, len(generated_images))

    # Compute LPIPS for all images
    lpips_scores = []
    calculate_lpips(generated_images, reference_images)
    avg_lpips = print(f"LPIPS score for all generated subfolders against all reference images: {avg_lpips:.4f}")
    return avg_lpips

def set_hf_cache_dir(cache_dir):
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir
        #also the torch cache when downloading the model
        os.environ["TORCH_HOME"] = cache_dir
        print(f"Set Huggingface and torch cache to: {cache_dir}")


def pickScore_calc_probs(prompt, images, device, processor,model):
    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)


    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        
        if len(images) == 1:
            # get probabilities if you have only one image to choose from
            probs = scores
        else:
            # get probabilities if you have multiple images to choose from
            probs = torch.softmax(scores, dim=-1)
    
    return probs.cpu().tolist()

def calculate_average_pickscore_from_prompts(all_prompts, output_dir):
    """
    Iterates over styles and their prompts, calculates PickScore for each image-prompt pair,
    and computes the average score for each style.

    Args:
        all_prompts (dict): Dictionary containing styles as keys and lists of prompts as values.
        output_dir (str): Path to the directory containing generated images for each style.

    Returns:
        dict: A dictionary with style names as keys and their average scores as values.
    """
    results = {}
        # load model
    device = "cuda"
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

    processor = AutoProcessor.from_pretrained(processor_name_or_path, cache_dir="/leonardo_scratch/large/userexternal/lsigillo/")
    model = AutoModel.from_pretrained(model_pretrained_name_or_path, cache_dir="/leonardo_scratch/large/userexternal/lsigillo/").eval().to(device)

    for style, prompts in tqdm(all_prompts.items(), total=len(all_prompts), desc="Calculating PickScores"):
        style_dir = os.path.join(output_dir, style)
        if not os.path.exists(style_dir):
            print(f"Directory for style '{style}' not found, skipping...")
            continue

        # Load images for the current style
        image_files = sorted([os.path.join(style_dir, f) for f in os.listdir(style_dir) if f.endswith(".jpg")])
        if len(image_files) != len(prompts):
            print(f"Mismatch between number of images and prompts for style '{style}', skipping...")
            continue

        # Calculate scores for each image-prompt pair
        scores = []
        for image_file, prompt in zip(image_files, prompts):
            pil_image = Image.open(image_file)
            score = pickScore_calc_probs(prompt, [pil_image], device, processor, model)
            scores.append(score[0])  # Assuming single image, take the first score

        # Compute the average score for the style
        avg_score = np.mean(scores)
        results[style] = avg_score
        print(f"Average PickScore for style '{style}': {avg_score:.4f}")

    return results


def DPG_generate_images_for_gpu(prompt_files, prompt_folder, output_folder, 
                            checkpoint_path, device_str, cache_dir, height, width, 
                            gen_seed,img_per_prompt=4, save_grid=True):
    """
    Generate images for a subset of prompts on a specific GPU.

    Args:
        prompt_files (list): List of prompt filenames to process.
        prompt_folder (str): Path to the folder containing text files with prompts.
        output_folder (str): Path to the folder where generated images will be saved.
        checkpoint_path (str): Path to the model checkpoint.
        device_str (str): GPU device string (e.g., "cuda:0").
        cache_dir (str): Cache directory for model loading.
        height (int): Height of the generated images.
        width (int): Width of the generated images.
        gen_seed (torch.Generator): Random seed generator.
    """
    os.makedirs(output_folder, exist_ok=True)
    import gc

    # Load the model on the specified GPU
    bfl_repo = "black-forest-labs/FLUX.1-dev"
    device = torch.device(device_str)
    dtype = torch.bfloat16
    transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, cache_dir=cache_dir)
    pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=transformer, torch_dtype=dtype, cache_dir=cache_dir)
    pipe.scheduler.config.use_dynamic_shifting = False
    pipe.scheduler.config.time_shift = 10
    pipe = pipe.to(device)

    # Load LoRA weights
    pipe.load_lora_weights(checkpoint_path)
    pipe = pipe.to(device)
    with torch.no_grad():
        for prompt_file in tqdm(prompt_files, total=len(prompt_files), desc=f"Processing on {device_str}"):
            prompt_path = os.path.join(prompt_folder, prompt_file)
            with open(prompt_path, "r") as file:
                prompt = file.read().strip()
            if os.path.join(output_folder, os.path.splitext(prompt_file)[0] + ".jpg") in os.listdir(output_folder):
                print(f"Image for prompt '{prompt_file}' already exists, skipping...")
                continue
            # Generate n images for the prompt
            images = pipe(
                prompt,
                height=height,
                width=width,
                guidance_scale=3.5,
                num_inference_steps=28,
                max_sequence_length=512,
                generator=gen_seed,
                ntk_factor=10,
                proportional_attention=True,
                num_images_per_prompt=img_per_prompt,
            ).images
            if save_grid:
                # Arrange the images in a 2x2 grid
                grid_width, grid_height = images[0].size
                grid = Image.new("RGB", (grid_width * 2, grid_height * 2))

                grid.paste(images[0], (0, 0))
                grid.paste(images[1], (grid_width, 0))
                grid.paste(images[2], (0, grid_height))
                grid.paste(images[3], (grid_width, grid_height))

                # Save the grid image with the same name as the prompt file (but with .jpg extension)
                output_path = os.path.join(output_folder, os.path.splitext(prompt_file)[0] + ".jpg")
                grid.save(output_path)
                # print(f"Generated and saved image for prompt: {prompt_file} on {device_str}")
                # Clear GPU memory
                del images, grid
            else:
                # Save each image with the same name as the prompt file (but with .jpg extension)
                for i, image in enumerate(images):
                    output_path = os.path.join(output_folder, os.path.splitext(prompt_file)[0] + f"_{i}.jpg")
                    image.save(output_path)
                    # print(f"Generated and saved image for prompt: {prompt_file} on {device_str}")
                    # Clear GPU memory
                    del image
            torch.cuda.empty_cache()
            gc.collect()

def generate_images_from_prompts_parallel(prompt_folder, output_folder, checkpoint_path, 
                                            cache_dir, height=2048, width=2048, gen_seed=None,
                                            dpg=True):
    """
    Generate images based on prompts in text files using multiple GPUs.

    Args:
        prompt_folder (str): Path to the folder containing text files with prompts.
        output_folder (str): Path to the folder where generated images will be saved.
        checkpoint_path (str): Path to the model checkpoint.
        cache_dir (str): Cache directory for model loading.
        height (int): Height of the generated images.
        width (int): Width of the generated images.
        gen_seed (torch.Generator): Random seed generator.
    """
    import json
    if dpg:
        prompt_files = [f for f in os.listdir(prompt_folder) if f.endswith(".txt")]
    else:
        # Load the JSON file
        with open(prompt_folder, 'r') as f:
            prompt_files = json.load(f)
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("No GPUs available for parallel processing.")

    # Split the prompt files into chunks for each GPU
    chunks = [prompt_files[i::num_gpus] for i in range(num_gpus)]

    # Create and start a process for each GPU
    processes = []
    for i, chunk in enumerate(chunks):
        device_str = f"cuda:{i}"
        if dpg:
            process = Process(
                target=DPG_generate_images_for_gpu,
                args=(chunk, prompt_folder, output_folder, checkpoint_path, device_str, cache_dir, height, width, gen_seed)
            )
        else:
            process = Process(target=HPDv2_generate_images,
                              args=(chunk, output_folder, checkpoint_path, device_str, cache_dir, height, width, gen_seed))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()


def HPDv2_generate_images(data, output_dir, checkpoint_path, device_str,
                           cache_dir, height, width, gen_seed,):
    """
    Generate images based on prompts from a JSON file and save them with 
    the specified filenames to the output directory.
    
    Args:
        json_file (str): Path to the JSON file containing prompts and image paths
        output_dir (str): Directory where to save the generated images
        model_id (str): Hugging Face model ID for the image generation model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    import gc

    # Load the model on the specified GPU
    bfl_repo = "black-forest-labs/FLUX.1-dev"
    device = torch.device(device_str)
    dtype = torch.bfloat16
    transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, cache_dir=cache_dir)
    pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=transformer, torch_dtype=dtype, cache_dir=cache_dir)
    pipe.scheduler.config.use_dynamic_shifting = False
    pipe.scheduler.config.time_shift = 10
    pipe = pipe.to(device)

    # Load LoRA weights
    pipe.load_lora_weights(checkpoint_path)
    pipe = pipe.to(device)

    
    # Process each entry in the JSON file
    for idx, entry in tqdm(enumerate(data)):
        if "prompt" not in entry or "image_path" not in entry:
            print(f"Skipping entry {idx}: Missing prompt or image_path")
            continue
            
        prompt = entry["prompt"]
        image_paths = entry["image_path"]
        
        # print(f"Processing entry {idx+1}/{len(data)}: {prompt[:50]}...")
        with torch.no_grad():
            # Generate the required number of images
            num_images = len(image_paths)
            
            # Generate images in batches if there are more than 3 images
            if num_images > 3:
                # Process in batches of 3
                batch_size = 3
                
                for batch_start in range(0, num_images, batch_size):
                    batch_end = min(batch_start + batch_size, num_images)
                    batch_count = batch_end - batch_start
                    batch_paths = image_paths[batch_start:batch_end]
                    
                    print(f"Generating batch {batch_start//batch_size + 1}: images {batch_start+1}-{batch_end} of {num_images}")
                    batch_images = pipe(
                        prompt,
                        height=height,
                        width=width,
                        guidance_scale=3.5,
                        num_inference_steps=28,
                        max_sequence_length=512,
                        generator=gen_seed,
                        ntk_factor=10,
                        proportional_attention=True,
                        num_images_per_prompt=batch_count,
                    ).images
                    
                    # Save each image immediately with its specified filename
                    for i, (image, filename) in enumerate(zip(batch_images, batch_paths)):
                        save_path = os.path.join(output_dir, filename)
                        print(f"Saving image {batch_start+i+1}/{num_images}: {save_path}")
                        image.save(save_path)
                    
                    # Clean up after each batch
                    del batch_images
                    torch.cuda.empty_cache()
                    gc.collect()
            else:
                # Process all at once if 3 or fewer images
                images = pipe(
                    prompt,
                    height=height,
                    width=width,
                    guidance_scale=3.5,
                    num_inference_steps=28,
                    max_sequence_length=512,
                    generator=gen_seed,
                    ntk_factor=10,
                    proportional_attention=True,
                    num_images_per_prompt=num_images,
                ).images
                
                # Save each image with its specified filename
                for i, (image, filename) in enumerate(zip(images, image_paths)):
                    save_path = os.path.join(output_dir, filename)
                    print(f"Saving image {i+1}/{num_images}: {save_path}")
                    image.save(save_path)
                
                del images
        
        torch.cuda.empty_cache()
        gc.collect()
        # print(f"Finished entry {idx+1}/{len(data)}")


if __name__ == "__main__":
    set_hf_cache_dir("/leonardo_scratch/large/userexternal/lsigillo/")
    
    parser = argparse.ArgumentParser(description="Generate images from JSON prompts")
    parser.add_argument("--json_file", type=str, default="/leonardo_scratch/large/userexternal/lsigillo/HPDv2/test.json", help="Path to the JSON file")
    parser.add_argument("--output_dir", type=str, help="Directory to save the generated images")
    parser.add_argument("--checkpoint_path", type=str, default="/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/URAE_VAE_SE_WAV_ATT_LAION/checkpoint-2000", help="Model ID from Hugging Face")
    parser.add_argument("--cache_dir", type=str, default="/leonardo_scratch/large/userexternal/lsigillo/", help="Cache directory for model loading")
    parser.add_argument("--height", type=int, default=2048, help="Height of the generated images")
    parser.add_argument("--width", type=int, default=2048, help="Width of the generated images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--generated_folder_hpdv2", type=str, default="/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/output/URAE_VAE_SE_WAV_ATT_LAION", help="Path to the generated images folder")
    parser.add_argument("--reference_folder_hpdv2", type=str, default="/leonardo_scratch/large/userexternal/lsigillo/HPDv2/test", help="Path to the reference images folder")
    parser.add_argument("--prompt_folder_dpg", type=str, default="/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ELLA/dpg_bench/prompts", help="Path to the prompt folder")
    args = parser.parse_args()
    gen_seed = torch.manual_seed(seed=args.seed)
    
    # # GENERATE IMAGES HPDv2
    # generate_images_from_prompts_parallel(args.json_file, os.path.join(args.output_dir,"HPDv2"), args.checkpoint_path, 
    #                                     cache_dir="/leonardo_scratch/large/userexternal/lsigillo/",
    #                                        height=2048, width=2048, gen_seed=gen_seed, dpg=False)   
     
    # # GENERATE IMAGES DPG
    # output_dir_dpg = os.path.join(args.output_dir,"DPG")
    # generate_images_from_prompts_parallel(args.prompt_folder_dpg, output_dir_dpg, 
    #                                       args.checkpoint_path, args.cache_dir, gen_seed=gen_seed, dpg=True)   
    
    # os.system(f"bash /leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ELLA/dpg_bench/dist_eval.sh {output_dir_dpg} {2048}")
    
    
    # generated_folder = args.checkpoint_path.replace("ckpt", "output")
    # #replace the last part of the path with HPDv2
    # generated_folder = os.path.join(os.path.dirname(generated_folder), "HPDv2")    # Check if the generated folder exists
    # # Compute both FID and LPIPS
    # calculate_fid_and_lpips(generated_folder, args.reference_folder_hpdv2, compute_fid=False, compute_lpips=True)

    # # Compute FID for all generated subfolders against all reference images
    # calculate_fid_for_all(generated_folder, args.reference_folder_hpdv2)

    # # Compute LPIPS for all generated subfolders against all reference images
    # calculate_lpips_for_all(generated_folder, args.reference_folder_hpdv2)
    # all_prompts = hpsv2.benchmark_prompts('all')
    # average_scores = calculate_average_pickscore_from_prompts(all_prompts, generated_folder)
    # print("Average scores for all subfolders:", average_scores)
    # # Calculate the average
    # average_score = np.mean(list(average_scores.values()))

    # # Print the average
    # print(f"Overall average score: {average_score:.4f}")

    import torch
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", 
                                            cache_dir=args.cache_dir,
                                            torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")

    image = pipe(
        "A capybara holding a sign that reads Hello World",
        num_inference_steps=28,
        guidance_scale=3.5,
        width=2048,
        height=2048,
    ).images[0]
    image.save("capybara.png")