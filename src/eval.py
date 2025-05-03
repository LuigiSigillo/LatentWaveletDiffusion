import pyiqa
import torch
import os
import torch
from huggingface_hub import hf_hub_download
# # Add the parent directory of 'pipeline_flux' to the Python path
# print(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
from pipeline_flux import FluxPipeline
from transformer_flux import FluxTransformer2DModel
import hpsv2
from tqdm import tqdm
import argparse
import multiprocessing

# Set HF_HOME and TRANSFORMERS_CACHE environment variables
def set_hf_cache_dir(cache_dir):
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir
        #also the torch cache when downloading the model
        os.environ["TORCH_HOME"] = cache_dir
        print(f"Set Huggingface and torch cache to: {cache_dir}")

def evaluate_image_quality(metric_name, dist_path, ref_path=None, dataset_name=None, dataset_res=None, dataset_split=None):
    """
    Evaluate image quality using specified metric.
    
    Args:
        metric_name (str): Name of the metric to use (e.g., 'lpips', 'fid', 'maniqa', 'qualiclip')
        dist_path (str): Path to distorted image or directory
        ref_path (str, optional): Path to reference image or directory. Required for FR metrics.
        dataset_name (str, optional): Dataset name for FID. Only used with FID metric.
        dataset_res (int, optional): Dataset resolution for FID. Only used with FID metric.
        dataset_split (str, optional): Dataset split for FID. Only used with FID metric.
    
    Returns:
        float: Computed quality score
    """
    # Check if CUDA is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    
    # Create metric
    iqa_metric = pyiqa.create_metric(metric_name, device=device)
    
    # Display if lower or higher score is better
    print(f"Metric: {metric_name}, Lower is better: {iqa_metric.lower_better}")
    
    # Identify if the metric is no-reference (NR)
    nr_metrics = ['maniqa', 'qualiclip', 'niqe', 'brisque']
    is_nr_metric = metric_name.lower() in nr_metrics
    
    # Compute score
    if metric_name.lower() == 'fid':
        if ref_path is not None:
            score = iqa_metric(dist_path, ref_path)
            print(f"FID score between {dist_path} and {ref_path}: {score:.4f}")
        elif dataset_name is not None:
            score = iqa_metric(dist_path, dataset_name=dataset_name, 
                             dataset_res=dataset_res, dataset_split=dataset_split)
            print(f"FID score between {dist_path} and {dataset_name} ({dataset_res}px, {dataset_split}): {score:.4f}")
        else:
            raise ValueError("For FID metric, either ref_path or dataset_name must be provided")
    elif is_nr_metric:
        # Handle no-reference metrics (MANIQA, QualiCLIP, etc.)
        if os.path.isfile(dist_path):
            score = iqa_metric(dist_path)
            print(f"{metric_name} score for {dist_path}: {score:.4f}")
        elif os.path.isdir(dist_path):
            # Process all images in directory
            total_score = 0
            count = 0
            for root, _, files in os.walk(dist_path):
                for filename in tqdm(files):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                        img_path = os.path.join(root, filename)
                        img_score = iqa_metric(img_path)
                        # print(f"{metric_name} score for {img_path}: {img_score:.4f}")
                        total_score += img_score
                        count += 1
            
            if count == 0:
                raise ValueError(f"No valid images found in directory {dist_path}")
            
            score = total_score / count
            print(f"Average {metric_name} score for all images in {dist_path}: {score}")
        else:
            raise ValueError(f"Invalid path for {metric_name}: {dist_path}")
    else:
        if ref_path is None:
            raise ValueError(f"Reference path required for {metric_name} metric")
        
        # Check if paths are files or directories
        if os.path.isfile(dist_path) and os.path.isfile(ref_path):
            score = iqa_metric(dist_path, ref_path)
            print(f"{metric_name} score between {dist_path} and {ref_path}: {score:.4f}")
        else:
            score = calculate_lpips(dist_path, ref_path)
    return score


def generate_images(root_path_proj, checkpoint_path,device_str, height=2048, width=2048,seed=888, cache_dir=None):
    #load original model
    bfl_repo="black-forest-labs/FLUX.1-dev"
    device = torch.device(device_str)
    dtype = torch.bfloat16
    transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, cache_dir=cache_dir)
    pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=transformer, torch_dtype=dtype, cache_dir=cache_dir)
    pipe.scheduler.config.use_dynamic_shifting = False
    pipe.scheduler.config.time_shift = 10
    pipe = pipe.to(device)
    
    #our trained URAE
    pipe.load_lora_weights(checkpoint_path)
    pipe = pipe.to(device)

    # Get benchmark prompts (<style> = all, anime, concept-art, paintings, photo)
    all_prompts = hpsv2.benchmark_prompts('all') 
    gen_seed = torch.manual_seed(seed=seed)
    #take name by the checkpoint name
    name_exp = checkpoint_path.split("/")[-2]
    print(name_exp)
    # Iterate over the benchmark prompts to generate images
    for style, prompts in tqdm(all_prompts.items(), total=len(all_prompts), desc="Generating images"):
        # Create the directory if it doesn't exist
        output_dir = os.path.join(root_path_proj,"src","output",name_exp, style)
        print(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        for idx, prompt in tqdm(enumerate(prompts), total=len(prompts), desc=f"Generating {style} images"):
            # image = TextToImageModel(prompt)
            image = pipe(
                prompt,
                height=height,
                width=width,
                guidance_scale=3.5,
                num_inference_steps=28,
                max_sequence_length=512,
                generator=gen_seed,
                ntk_factor=10,
                proportional_attention=True
            ).images[0]
            # TextToImageModel is the model you want to evaluate
            image.save(os.path.join(root_path_proj,"src","output",name_exp, style, f"{idx:05d}.jpg")) 
            # <image_path> is the folder path to store generated images, as the input of hpsv2.evaluate().

    hps2 = hpsv2.evaluate("output/"+checkpoint_path.split("/")[-2],
            device=device) 
    return hps2


def generate_images_for_style(style, prompts, device_id, root_path_proj, checkpoint_path, height, width, seed, cache_dir):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)  # Assign specific GPU
    device = torch.device("cuda:"+str(device_id))
    dtype = torch.bfloat16

    # Load the model and pipeline
    bfl_repo = "black-forest-labs/FLUX.1-dev"
    transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, cache_dir=cache_dir)
    pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=transformer, torch_dtype=dtype, cache_dir=cache_dir)
    pipe.scheduler.config.use_dynamic_shifting = False
    pipe.scheduler.config.time_shift = 10
    pipe = pipe.to(device)

    # Load LoRA weights
    pipe.load_lora_weights(checkpoint_path)
    pipe = pipe.to(device)

    # Set up output directory
    name_exp = checkpoint_path.split("/")[-2]
    output_dir = os.path.join(root_path_proj, "src", "output", name_exp, style)
    os.makedirs(output_dir, exist_ok=True)

    # Generate images for the given style
    gen_seed = torch.manual_seed(seed=seed)
    for idx, prompt in tqdm(enumerate(prompts), total=len(prompts), desc=f"Generating {style} images"):
        image = pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=3.5,
            num_inference_steps=28,
            max_sequence_length=512,
            generator=gen_seed,
            ntk_factor=10,
            proportional_attention=True
        ).images[0]
        image.save(os.path.join(output_dir, f"{idx:05d}.jpg"))

def parallel_generate_images(root_path_proj, checkpoint_path, device_ids, height=2048, width=2048, seed=888, cache_dir=None):
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
                style, prompts, device_id, root_path_proj, checkpoint_path, height, width, seed, cache_dir
            )
    else:
        # Create a process for each style, assigning GPUs in a round-robin fashion
        processes = []
        for i, style in enumerate(styles):
            prompts = all_prompts[style]
            device_id = device_ids[i] 
            p = multiprocessing.Process(
                target=generate_images_for_style,
                args=(style, prompts, device_id, root_path_proj, checkpoint_path, height, width, seed, cache_dir)
            )
            processes.append(p)
            p.start()

        # Wait for all processes to complete
        for p in processes:
            p.join()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Image generation and evaluation script")
    
    # General arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to the checkpoint')
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
                       help='')
    # Evaluation options
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate image quality')
    parser.add_argument('--metrics', nargs='+', default=['fid', 'lpips', 'maniqa', 'qualiclip'],
                       help='List of metrics for evaluation (default: fid lpips maniqa qualiclip)')
    parser.add_argument('--use_custom_paths', action='store_true',
                       help='Use custom paths instead of deriving from checkpoint')
    parser.add_argument('--dist_path', type=str, default=None,
                       help='Custom path to distorted images (only if use_custom_paths is True)')
    parser.add_argument('--ref_path', type=str, default=None,
                       help='Custom path to reference images (only if use_custom_paths is True)')
    
    # FID specific options
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='Dataset name for FID')
    parser.add_argument('--dataset_res', type=int, default=None,
                       help='Dataset resolution for FID')
    parser.add_argument('--dataset_split', type=str, default=None,
                       help='Dataset split for FID')
    
    # List metrics option
    parser.add_argument('--list_metrics', action='store_true',
                       help='List all available metrics')
    
    return parser.parse_args()

def calculate_lpips(dist_path, ref_path):
    from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity
    from PIL import Image
    from torchvision import transforms

    # Get all image files from both directories
    dist_images = []
    ref_images = []
    
    # Collect distorted images
    for root, _, files in os.walk(dist_path):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                dist_images.append(os.path.join(root, filename))
    
    # Collect reference images
    for root, _, files in os.walk(ref_path):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                ref_images.append(os.path.join(root, filename))
    
    # Sort images to ensure consistent pairing
    dist_images.sort()
    ref_images.sort()
    
    # Make sure we have equal number of images
    min_len = min(len(dist_images), len(ref_images))
    dist_images = dist_images[:min_len]
    ref_images = ref_images[:min_len]
    
    print(f"Processing {min_len} image pairs for LPIPS calculation")

    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, dist_images, ref_images):
            self.dist_images = dist_images
            self.ref_images = ref_images
            self.transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
            ])
        
        def __len__(self):
            return len(self.dist_images)
        
        def __getitem__(self, idx):
            # Open images
            dist_img = Image.open(self.dist_images[idx])
            ref_img = Image.open(self.ref_images[idx])
            
            # Convert RGBA to RGB if needed
            if dist_img.mode == 'RGBA':
                dist_img = dist_img.convert('RGB')
            if ref_img.mode == 'RGBA':
                ref_img = ref_img.convert('RGB')
            
            # Transform images
            dist_tensor = self.transform(dist_img)
            ref_tensor = self.transform(ref_img)
            
            # Ensure both have 3 channels
            if dist_tensor.shape[0] != 3 or ref_tensor.shape[0] != 3:
                print(f"Warning: Unusual channel count in image pair {idx}: {dist_tensor.shape}, {ref_tensor.shape}")
                # Force 3 channels if needed (take first 3 or add zeros)
                if dist_tensor.shape[0] > 3:
                    dist_tensor = dist_tensor[:3, :, :]
                elif dist_tensor.shape[0] < 3:
                    zeros = torch.zeros(3-dist_tensor.shape[0], 512, 512)
                    dist_tensor = torch.cat([dist_tensor, zeros], dim=0)
                
                if ref_tensor.shape[0] > 3:
                    ref_tensor = ref_tensor[:3, :, :]
                elif ref_tensor.shape[0] < 3:
                    zeros = torch.zeros(3-ref_tensor.shape[0], 512, 512)
                    ref_tensor = torch.cat([ref_tensor, zeros], dim=0)
            
            return dist_tensor, ref_tensor
    
    dataset = ImageDataset(dist_images, ref_images)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    
    total_score = 0
    batch_count = 0
    
    try:
        for (img1, img2) in tqdm(dataloader):
            # Validate shapes before computing metric
            if img1.shape != img2.shape:
                print(f"Warning: Shape mismatch detected: {img1.shape} vs {img2.shape}")
                continue
                
            batch_score = learned_perceptual_image_patch_similarity(img1, img2, net_type='squeeze', normalize=True)
            total_score += batch_score
            batch_count += 1
    except Exception as e:
        print(f"Error during LPIPS calculation: {e}")
        # Print the shapes of tensors for debugging
        for i, (img1, img2) in enumerate(dataset):
            print(f"Pair {i}: {img1.shape}, {img2.shape}")
            if i > 5:  # Just sample a few for debugging
                break
        raise
    
    if batch_count == 0:
        raise ValueError("No valid image pairs were processed for LPIPS calculation")
        
    return total_score / batch_count


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    args = parse_arguments()
    set_hf_cache_dir(args.cache_dir)
    # List available metrics if requested
    if args.list_metrics:
        print("Available metrics:")
        print(pyiqa.list_models())
        exit(0)
    
    # Derive paths from checkpoint
    checkpoint_path = args.checkpoint
    name_exp = checkpoint_path.split("/")[-2]
    root_path_proj = os.path.abspath(os.getcwd())
    generated_path = os.path.join(root_path_proj,"src",f"output/{name_exp}")
    print(f"Generated images will be saved to: {generated_path}")
    
    # Get available GPU IDs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs available. Exiting.")
        exit(1)
    device_ids = list(range(num_gpus))  # Generate GPU IDs [0, 1, ..., num_gpus-1]
    print(f"Available GPUs: {device_ids}")

    # # Generate images if requested
    # if args.generate:
    #     print(f"Generating images using checkpoint: {args.checkpoint}")
    #     hps2_score = generate_images(
    #         root_path_proj,
    #         args.checkpoint,
    #         args.device,
    #         height=args.height,
    #         width=args.width,
    #         seed=args.seed,
    #         cache_dir=args.cache_dir
    #     )
    #     print(f"HPS2 evaluation score: {hps2_score}")
    if args.generate:
        print(f"Generating images using checkpoint: {args.checkpoint}")
        hps2_score = parallel_generate_images(
            root_path_proj,
            args.checkpoint,
            device_ids,  # Pass the dynamically detected GPU IDs
            height=args.height,
            width=args.width,
            seed=args.seed,
            cache_dir=args.cache_dir
        )
        print(f"HPS2 evaluation score: {hps2_score}")
    # Evaluate with custom metrics if requested
    if args.evaluate:
        if not args.metrics:
            print("Error: No metrics specified. Using default metrics: fid, lpips, maniqa, qualiclip")
            args.metrics = ['lpips', 'maniqa', 'qualiclip','fid']
        
        # Determine paths based on user choice
        dist_path = args.dist_path if args.use_custom_paths else generated_path
        ref_path = args.ref_path
        print(f"Distorted images path: {dist_path}")
        print(f"Reference images path: {ref_path}")
        print(f"Using custom paths: {args.use_custom_paths}")
        print(f"Generated images path: {generated_path}")
        # If no paths were specified or derived
        if not dist_path:
            print("Error: No path for distorted images. Either generate images, specify --dist_path, or use a valid checkpoint.")
            exit(1)
        
        # Store results for all metrics
        results = {}

        # Run evaluation for each metric
        for metric in args.metrics:
            print(f"Evaluating with metric: {metric}")
            
            try:
                # Skip metrics that require reference images if no reference is provided
                if metric.lower() not in ['maniqa', 'qualiclip', 'niqe', 'brisque', 'fid'] and ref_path is None:
                    print(f"Skipping {metric} - reference path required but not provided")
                    continue
                # Run evaluation
                score = evaluate_image_quality(
                    metric,
                    dist_path,
                    ref_path=ref_path,
                    dataset_name=args.dataset_name,
                    dataset_res=args.dataset_res,
                    dataset_split=args.dataset_split
                )

                results[metric] = score

                print(f"Final {metric} score ", score)
                
            except Exception as e:
                print(f"Error evaluating with {metric}: {str(e)}")
        
        hps2 = hpsv2.evaluate(dist_path) 
        results["hps2"] = hps2
        # Print summary of all results
        if results:
            print("\n=== Evaluation Summary ===")
            for metric, score in results.items():
                print(f"{metric}: ", score)
        #write on a file txt with the name of the checkpoint that genreate those mtrics
        with open(os.path.join(root_path_proj,"src",f"output/{name_exp}", "evaluation_results.txt"), "w") as f:
            for metric, score in results.items():
                f.write(f"{metric}: " + score +"\n")
            print(f"Evaluation results saved to: {os.path.join(root_path_proj,'src',f'output/{name_exp}', 'evaluation_results.txt')}")
    
    # If neither generation nor evaluation was requested
    if not args.generate and not args.evaluate and not args.list_metrics:
        print("Error: You must specify at least one operation: --generate, --evaluate, or --list_metrics")
        print("Use --help for more information")