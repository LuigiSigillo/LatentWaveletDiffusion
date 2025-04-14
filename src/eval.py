import os
import torch
import argparse
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from typing import Optional, List, Dict, Any
from huggingface_hub import hf_hub_download

# For image generation (from notebook)
try:
    from pipeline_flux import FluxPipeline
    from transformer_flux import FluxTransformer2DModel
except ImportError:
    print("Warning: FLUX pipeline modules not found. Image generation will not work.")

def load_images_from_folder(folder_path: str, 
                           transform: Optional[transforms.Compose] = None,
                           max_images: Optional[int] = None) -> List[torch.Tensor]:
    """Load images from a folder and apply transformations."""
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder {folder_path} does not exist")
    
    # Get all image files recursively
    image_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_files.append(os.path.join(root, file))
    
    if max_images is not None:
        image_files = image_files[:max_images]
    
    images = []
    for img_path in tqdm(image_files, desc="Loading images"):
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                if transform:
                    img_tensor = transform(img)
                else:
                    default_transform = transforms.Compose([
                        transforms.ToTensor(),
                    ])
                    img_tensor = default_transform(img)
                images.append(img_tensor)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    return images

def calculate_fid(generated_images: List[torch.Tensor], 
                 real_images: List[torch.Tensor] = None,
                 feature_dims: int = 2048) -> float:
    """Calculate FID score between generated images and real images."""
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fid = FrechetInceptionDistance(feature=feature_dims, normalize=True)
        fid = fid.to(device)
        
        # If no real images provided, use generated as both sets (for self-consistency)
        if real_images is None:
            print("No reference images provided. FID will measure self-consistency.")
            halfway = len(generated_images) // 2
            real_images = generated_images[halfway:]
            generated_images = generated_images[:halfway]
            if len(generated_images) < 10:
                print("Warning: Few images for meaningful FID calculation")
        
        # Process generated images
        for img in tqdm(generated_images, desc="Processing generated images for FID"):
            img = img.to(device)
            # FID expects images in [0, 255] range
            img = (img * 255).byte()
            if img.shape[0] == 3:  # Ensure NCHW format
                img = img.unsqueeze(0)
            fid.update(img, real=False)
        
        # Process real images
        for img in tqdm(real_images, desc="Processing real images for FID"):
            img = img.to(device)
            img = (img * 255).byte()
            if img.shape[0] == 3:
                img = img.unsqueeze(0)
            fid.update(img, real=True)
        
        # Calculate FID
        return fid.compute().item()
    except ImportError:
        print("FID calculation requires torchmetrics. Install with: pip install torchmetrics")
        return float('nan')
    except Exception as e:
        print(f"Error calculating FID: {e}")
        return float('nan')

def calculate_lpips(generated_images: List[torch.Tensor], 
                   reference_images: List[torch.Tensor] = None) -> float:
    """Calculate LPIPS score between generated images and reference images."""
    try:
        import lpips
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lpips_model = lpips.LPIPS(net='alex').to(device)
        
        # If no reference images, compare consecutive images
        if reference_images is None:
            print("No reference images provided. Using consecutive image comparisons for LPIPS.")
            lpips_scores = []
            for i in tqdm(range(len(generated_images)-1), desc="Calculating LPIPS"):
                img1 = generated_images[i].unsqueeze(0).to(device)
                img2 = generated_images[i+1].unsqueeze(0).to(device)
                
                with torch.no_grad():
                    lpips_score = lpips_model(img1, img2)
                lpips_scores.append(lpips_score.item())
        else:
            # Make sure we have the same number of generated and reference images
            n = min(len(generated_images), len(reference_images))
            
            lpips_scores = []
            for i in tqdm(range(n), desc="Calculating LPIPS"):
                img1 = generated_images[i].unsqueeze(0).to(device)
                img2 = reference_images[i].unsqueeze(0).to(device)
                
                with torch.no_grad():
                    lpips_score = lpips_model(img1, img2)
                lpips_scores.append(lpips_score.item())
        
        return np.mean(lpips_scores)
    except ImportError:
        print("LPIPS not installed. Install with: pip install lpips")
        return float('nan')
    except Exception as e:
        print(f"Error calculating LPIPS: {e}")
        return float('nan')

def calculate_man_iqa(images: List[torch.Tensor]) -> float:
    """Calculate MAN-IQA scores for the images."""
    try:
        from pyiqa import create_metric
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create MAN-IQA metric
        metric = create_metric('maniqa', device=device)
        
        scores = []
        for img in tqdm(images, desc="Calculating MAN-IQA"):
            # MAN-IQA expects images in range [0, 1]
            img_tensor = img.unsqueeze(0).to(device)
            with torch.no_grad():
                score = metric(img_tensor).item()
            scores.append(score)
        
        return np.mean(scores)
    except ImportError:
        print("MAN-IQA not available. Install with: pip install pyiqa")
        return float('nan')
    except Exception as e:
        print(f"Error calculating MAN-IQA: {e}")
        return float('nan')

def calculate_qualiclip(images: List[torch.Tensor], prompts: List[str] = None) -> float:
    """Calculate a CLIP-based image quality score."""
    try:
        import clip
        import torch.nn.functional as F
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        # Process images
        image_features_list = []
        for img in tqdm(images, desc="Processing images for CLIP score"):
            # Convert tensor to PIL for CLIP preprocessing
            pil_img = transforms.ToPILImage()(img)
            processed_img = preprocess(pil_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = model.encode_image(processed_img)
                image_features = F.normalize(image_features, dim=-1)
                image_features_list.append(image_features)
        
        if prompts and len(prompts) == len(images):
            # Calculate text-image alignment scores
            scores = []
            for i, prompt in enumerate(tqdm(prompts, desc="Calculating prompt-image alignment")):
                text = clip.tokenize([prompt]).to(device)
                with torch.no_grad():
                    text_features = model.encode_text(text)
                    text_features = F.normalize(text_features, dim=-1)
                
                # Calculate cosine similarity
                similarity = (100 * (image_features_list[i] @ text_features.T)).item()
                scores.append(similarity)
            
            return np.mean(scores)
        else:
            print("No prompts provided for QualiCLIP. Using image-image consistency.")
            # Calculate image-image consistency as alternative
            similarities = []
            for i in range(len(image_features_list)):
                for j in range(i+1, len(image_features_list)):
                    sim = (100 * (image_features_list[i] @ image_features_list[j].T)).item()
                    similarities.append(sim)
            
            return np.mean(similarities) if similarities else float('nan')
    
    except ImportError:
        print("CLIP not installed. Install with: pip install ftfy regex tqdm git+https://github.com/openai/CLIP.git")
        return float('nan')
    except Exception as e:
        print(f"Error calculating CLIP-based score: {e}")
        return float('nan')

def calculate_pickscore(images: List[torch.Tensor], prompts: List[str]) -> float:
    """Calculate Pickscore for images based on text prompts."""
    try:
        from pickscore import PickScore
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pick_score = PickScore(device=device)
        
        if len(prompts) != len(images):
            print("Warning: Number of prompts and images must match for Pickscore")
            return float('nan')
        
        # Convert tensor images to PIL images for Pickscore
        pil_images = [transforms.ToPILImage()(img) for img in images]
        
        # Calculate scores
        scores = pick_score.compute_image_text_similarity(pil_images, prompts)
        return scores.mean().item()
    
    except ImportError:
        print("Pickscore not installed. Install with: pip install pickscore")
        return float('nan')
    except Exception as e:
        print(f"Error calculating Pickscore: {e}")
        return float('nan')

def setup_flux_model(device: str = None, ckpt_path: str = None):
    """Set up the FLUX model for image generation."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    dtype = torch.bfloat16
    bfl_repo = "black-forest-labs/FLUX.1-dev"
    
    try:
        # Load transformer model
        transformer = FluxTransformer2DModel.from_pretrained(
            bfl_repo, subfolder="transformer", torch_dtype=dtype
        )
        
        # Load pipeline
        pipe = FluxPipeline.from_pretrained(
            bfl_repo, transformer=transformer, torch_dtype=dtype
        )
        
        # Configure scheduler
        pipe.scheduler.config.use_dynamic_shifting = False
        pipe.scheduler.config.time_shift = 10
        
        # Download and load URAE adapter weights if needed
        if not os.path.exists('ckpt/urae_2k_adapter.safetensors'):
            os.makedirs('ckpt', exist_ok=True)
            hf_hub_download(
                repo_id="Huage001/URAE", 
                filename='urae_2k_adapter.safetensors', 
                local_dir='ckpt', 
                local_dir_use_symlinks=False
            )
        
        pipe.load_lora_weights("ckpt/urae_2k_adapter.safetensors")
        
        # Load custom checkpoint if provided
        if ckpt_path and os.path.exists(ckpt_path):
            pipe.load_lora_weights(ckpt_path)
            print(f"Loaded custom checkpoint: {ckpt_path}")
            
        pipe = pipe.to(device)
        return pipe
        
    except Exception as e:
        print(f"Error setting up FLUX model: {e}")
        return None

def generate_images(pipe, output_dir: str, prompts: List[str], 
                   height: int = 2048, width: int = 2048, 
                   seed: int = 8888):
    """Generate images using the FLUX pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    
    images = []
    for idx, prompt in enumerate(tqdm(prompts, desc="Generating images")):
        image = pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=3.5,
            num_inference_steps=28,
            max_sequence_length=512,
            generator=torch.manual_seed(seed),
            ntk_factor=10,
            proportional_attention=True
        ).images[0]
        
        # Save the image
        image_path = os.path.join(output_dir, f"{idx:05d}.jpg")
        image.save(image_path)
        images.append(image)
    
    return images

def evaluate_images(
    generated_folder: str, 
    reference_folder: Optional[str] = None,
    prompts_file: Optional[str] = None,
    prompts_list: Optional[List[str]] = None,
    max_images: Optional[int] = None,
    eval_hpsv2: bool = True
) -> Dict[str, float]:
    """Evaluate images using multiple metrics."""
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Standard size for inception-based metrics
        transforms.ToTensor(),
    ])
    
    # Load images
    generated_images = load_images_from_folder(generated_folder, transform, max_images)
    print(f"Loaded {len(generated_images)} generated images")
    
    reference_images = None
    if reference_folder and os.path.exists(reference_folder):
        reference_images = load_images_from_folder(reference_folder, transform, max_images)
        print(f"Loaded {len(reference_images)} reference images")
    
    # Load prompts
    prompts = []
    if prompts_list:
        prompts = prompts_list
    elif prompts_file and os.path.exists(prompts_file):
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f]
    
    # Match number of prompts to images
    if prompts:
        if len(prompts) < len(generated_images):
            print(f"Warning: Fewer prompts ({len(prompts)}) than images ({len(generated_images)})")
            prompts = prompts + [""] * (len(generated_images) - len(prompts))
        if len(prompts) > len(generated_images):
            prompts = prompts[:len(generated_images)]
        
        print(f"Using {len(prompts)} prompts for evaluation")
    
    # Calculate metrics
    results = {}
    
    print("Calculating FID score...")
    results["FID"] = calculate_fid(generated_images, reference_images)
    
    print("Calculating LPIPS score...")
    results["LPIPS"] = calculate_lpips(generated_images, reference_images)
    
    print("Calculating MAN-IQA score...")
    results["MAN-IQA"] = calculate_man_iqa(generated_images)
    
    print("Calculating QualiCLIP score...")
    results["QualiCLIP"] = calculate_qualiclip(generated_images, prompts)
    
    if prompts:
        print("Calculating Pickscore...")
        results["Pickscore"] = calculate_pickscore(generated_images, prompts)
    
    # Run hpsv2 evaluation if requested
    if eval_hpsv2:
        try:
            import hpsv2
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("Running HPS-V2 evaluation...")
            hpsv2_results = hpsv2.evaluate(generated_folder, device=device)
            
            # Merge results
            for k, v in hpsv2_results.items():
                results[f"HPS-{k}"] = v
        except ImportError:
            print("HPS-V2 not installed. Skipping hpsv2 evaluation.")
        except Exception as e:
            print(f"Error in HPS-V2 evaluation: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='FLUX Image Generation and Evaluation')
    
    # General options
    parser.add_argument('--mode', type=str, choices=['generate', 'evaluate', 'both'], default='evaluate',
                      help='Operation mode: generate images, evaluate existing ones, or both')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (e.g., "cuda:0", "cuda:3")')
    
    # Generation options
    parser.add_argument('--output_dir', type=str, default='output/generated',
                      help='Directory to save generated images')
    parser.add_argument('--height', type=int, default=2048,
                      help='Image height for generation')
    parser.add_argument('--width', type=int, default=2048,
                      help='Image width for generation')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to custom checkpoint for the model')
    parser.add_argument('--seed', type=int, default=8888,
                      help='Random seed for generation')
    
    # Evaluation options
    parser.add_argument('--generated', type=str, default=None,
                      help='Path to folder containing generated images')
    parser.add_argument('--reference', type=str, default=None,
                      help='Path to folder containing reference/real images')
    parser.add_argument('--prompts', type=str, default=None,
                      help='Path to text file containing prompts (one per line)')
    parser.add_argument('--max_images', type=int, default=None,
                      help='Maximum number of images to evaluate')
    parser.add_argument('--no_hpsv2', action='store_true',
                      help='Skip HPS-V2 evaluation')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ['generate', 'both']:
        if not args.prompts:
            parser.error("--prompts is required when generating images")
    
    if args.mode in ['evaluate', 'both']:
        if args.mode == 'evaluate' and not args.generated:
            parser.error("--generated is required when evaluating without generation")
    
    # Set device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load prompts
    prompts = []
    if args.prompts and os.path.exists(args.prompts):
        with open(args.prompts, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f]
        print(f"Loaded {len(prompts)} prompts from {args.prompts}")
    
    # Generate images if requested
    if args.mode in ['generate', 'both']:
        pipe = setup_flux_model(device, args.checkpoint)
        if pipe is None:
            print("Failed to set up FLUX model. Exiting.")
            return
        
        print(f"Generating {len(prompts)} images...")
        generate_images(
            pipe, 
            args.output_dir, 
            prompts,
            height=args.height,
            width=args.width,
            seed=args.seed
        )
        print(f"Images generated and saved to {args.output_dir}")
    
    # Evaluate images
    if args.mode in ['evaluate', 'both']:
        generated_dir = args.generated if args.generated else args.output_dir
        
        print(f"Evaluating images in {generated_dir}...")
        results = evaluate_images(
            generated_dir,
            args.reference,
            args.prompts,
            prompts if args.mode == 'both' else None,
            args.max_images,
            not args.no_hpsv2
        )
        
        # Print results
        print("\nEvaluation Results:")
        for metric, score in results.items():
            if np.isnan(score):
                print(f"{metric}: Not calculated")
            else:
                print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    main()