from diffusers import Lumina2Pipeline
import os
import torch
import hpsv2
from tqdm import tqdm
import argparse
import multiprocessing
from diffusers import FluxPipeline
from diffusers import FluxTransformer2DModel

def generate_images_for_style(style, prompts, device_id, generated_path, height, width, seed, cache_dir):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)  # Assign specific GPU
    device = torch.device("cuda:"+str(device_id))
    dtype = torch.bfloat16
    if "LUMINA" in generated_path:
        # Load the model and pipeline
        pipe = Lumina2Pipeline.from_pretrained("Alpha-VLLM/Lumina-Image-2.0", torch_dtype=dtype, 
                                            cache_dir=cache_dir)
        # pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        pipe.to(device)
    else:
        bfl_repo = "black-forest-labs/FLUX.1-dev"
        transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, 
                                                             cache_dir=cache_dir)
        pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=transformer, torch_dtype=dtype, 
                                            cache_dir=cache_dir)
        pipe.scheduler.config.use_dynamic_shifting = False
        pipe.scheduler.config.time_shift = 10
        pipe = pipe.to(device)
    # Set up output directory
    # name_exp = checkpoint_path.split("/")[-2]
    output_dir = os.path.join(generated_path, style)
    print("generating images in  ",output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Generate images for the given style
    gen_seed = torch.Generator("cpu").manual_seed(seed)
    for idx, prompt in tqdm(enumerate(prompts), total=len(prompts), desc=f"Generating {style} images"):
        #check if image exists
        if os.path.exists(os.path.join(output_dir, f"{idx:05d}.jpg")):
            print(f"Image {idx:05d}.jpg already exists. Skipping.")
            continue
        
        if "LUMINA" in generated_path:
            image = pipe(
                prompt,
                height=height,
                width=width,
                guidance_scale=4.0,
                num_inference_steps=28,
                cfg_trunc_ratio=0.25,
                cfg_normalization=True,
                generator=gen_seed
            ).images[0]
        else:
            image = pipe(
                prompt,
                height=height,
                width=width,
                guidance_scale=3.5,
                num_inference_steps=28,
                max_sequence_length=512,
                generator=gen_seed,
                # ntk_factor=10,
                # proportional_attention=True
        ).images[0]
        image.save(os.path.join(output_dir, f"{idx:05d}.jpg"))

def parallel_generate_images(generated_path, checkpoint_path, device_ids, height=2048, width=2048, seed=888, cache_dir=None):
    # Get benchmark prompts
    all_prompts = hpsv2.benchmark_prompts('all')
    styles = list(all_prompts.keys())
    # Check if only one GPU is available
    if len(device_ids) == 1:
        print("Only one GPU detected. Running processes sequentially.")
        for style in styles:
            prompts = all_prompts[style]
            device_id = device_ids[0]  # Use the single available GPU
            generate_images_for_style(
                style, prompts, device_id, generated_path, height, width, seed, cache_dir
            )
    else:
        # Create a process for each style, assigning GPUs in a round-robin fashion
        processes = []
        for i, style in enumerate(styles):
            prompts = all_prompts[style]
            device_id = device_ids[i] 
            print("Launching images for style:", style, "with genereted path", generated_path)
            p = multiprocessing.Process(
                target=generate_images_for_style,
                args=(style, prompts, device_id, generated_path, height, width, seed, cache_dir)
            )
            processes.append(p)
            p.start()

        # Wait for all processes to complete
        for p in processes:
            p.join()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Image generation and evaluation script")
    
    # General arguments
    parser.add_argument('--checkpoint', type=str,
                       help='model name or path to the checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    # Generation options
    parser.add_argument('--generate', action='store_true',
                       help='Generate images using the checkpoint')
    parser.add_argument('--height', type=int, default=2048,
                       help='Height of generated images')
    parser.add_argument('--width', type=int, default=2048,
                       help='Width of generated images')
    parser.add_argument('--seed', type=int, default=8888,
                       help='Random seed for generation')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='/leonardo_scratch/large/userexternal/lsigillo/')
    return parser.parse_args()

# def set_hf_cache_dir(cache_dir):
#     if cache_dir:
#         #os.environ["HF_HOME"] = cache_dir
#         os.system("export HF_HOME=" + cache_dir)
#         #also the torch cache when downloading the model
#         # os.environ["TORCH_HOME"] = cache_dir
#         os.system("export TORCH_HOME=" + cache_dir)
#         print(f"Set Huggingface and torch cache to: {cache_dir}")



if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    args = parse_arguments()
    # set_hf_cache_dir(args.cache_dir)
    root_path_proj = "/leonardo_work/IscrC_UniMod/luigi/urae/src/output"
    generated_path = os.path.join(root_path_proj,args.checkpoint,"HPDv2")
    
    print(f"Generated images will be saved to: {generated_path}")
    
    # Get available GPU IDs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs available. Exiting.")
        exit(1)
    device_ids = list(range(num_gpus))  # Generate GPU IDs [0, 1, ..., num_gpus-1]
    print(f"Available GPUs: {device_ids}")

    if args.generate:
        print(f"Generating images using checkpoint: {args.checkpoint}")
        hps2_score = parallel_generate_images(
            generated_path,
            args.checkpoint,
            device_ids,  # Pass the dynamically detected GPU IDs
            height=args.height,
            width=args.width,
            seed=args.seed,
            cache_dir=args.cache_dir
        )
        print(f"HPS2 evaluation score: {hps2_score}")
