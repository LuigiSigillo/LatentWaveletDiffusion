import os
import argparse
import logging
import math
import random
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
import wandb
from PIL import Image
import PIL
import bitsandbytes as bnb
import torch_dct as tdct  # pip install torch-dct
from pytorch_wavelets import DWTForward  # pip install pytorch_wavelets
import einops
import lpips  # Add LPIPS import - pip install lpips

# Setup logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageDataset(Dataset):
    def __init__(self, data_dir, image_size=256):
        self.data_dir = data_dir
        self.image_paths = [
            os.path.join(data_dir, fname)
            for fname in os.listdir(data_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]
        if not self.image_paths:
            raise FileNotFoundError(f"No image files found in {data_dir}.")
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            # Try to open the image - first attempt
            with open(image_path, 'rb') as f:
                image = Image.open(f)
                # The load() call below will catch truncated files
                image.load()
                image = image.convert("RGB")
                image = self.transform(image)
                return {"pixel_values": image}
        except (IOError, PIL.UnidentifiedImageError, OSError) as e:
            # Check specifically for truncated file error
            if "truncated" in str(e).lower():
                logger.warning(f"Skipping truncated image {image_path}: {str(e)}")
            else:
                logger.warning(f"Error loading image {image_path}: {str(e)}")
            
            # Try the next image instead
            next_idx = (idx + 1) % len(self)
            logger.info(f"Trying next image at index {next_idx}")
            return self.__getitem__(next_idx)
        except Exception as e:
            # Catch any other unexpected errors
            logger.error(f"Unexpected error loading image {image_path}: {str(e)}")
            
            # Create a blank/noise image as fallback
            image_size = self.transform.transforms[0].size
            dummy_image = torch.randn(3, image_size, image_size)
            dummy_image = torch.clamp(dummy_image, -1, 1)  # Ensure in [-1, 1] range like normalized images
            logger.info(f"Returning random noise image for {image_path}")
            return {"pixel_values": dummy_image}

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained VAE using AutoencoderKL")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, 
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the training data")
    parser.add_argument("--output_dir", type=str, default="vae-finetuned", help="Directory to save model and results")
    parser.add_argument("--image_size", type=int, default=256, help="Image size for training (square)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.99, help="Adam beta2")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--max_train_steps", type=int, default=10000, help="Max training steps")
    parser.add_argument("--num_train_epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="Learning rate scheduler")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Learning rate warmup steps")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision training")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8-bit Adam")
    parser.add_argument("--allow_tf32", action="store_true", help="Allow TF32 precision on Ampere GPUs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpointing_steps", type=int, default=2000, help="Save checkpoint every X steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--with_tracking", action="store_true", help="Enable experiment tracking with WandB")
    parser.add_argument("--report_to", type=str, default="wandb", help="Tracking system to use")
    parser.add_argument("--tracker_project_name", type=str, default="vae-finetuning", help="WandB project name")
    parser.add_argument("--kl_weight", type=float, default=0, help="Weight for KL divergence loss (disabled for SE-regularized models)")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder parameters")
    parser.add_argument("--freeze_decoder", action="store_true", help="Freeze decoder parameters")
    parser.add_argument("--revision", type=str, default=None, help="Revision of pretrained model identifier")
    parser.add_argument("--subfolder", type=str, default=None, help="Subfolder of the pretrained model if part of a larger model")
    parser.add_argument("--validation_prompts", type=str, nargs="+", default=None, help="Validation prompts")
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--regularization_alpha", type=float, default=0.1, help="Weight for regularization loss")
    parser.add_argument("--lpips_weight", type=float, default=0.05, help="Weight for LPIPS perceptual loss")
    parser.add_argument("--latent_channels", type=int, default=16, help="Number of latent channels")
    parser.add_argument("--ema_decay", type=float, default=0.9998, help="EMA decay rate (half-life of 5000 steps)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set up accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to if args.with_tracking else None,
        project_dir=args.output_dir
    )
    
    # Make sure the output directory exists
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(f"{args.output_dir}/samples", exist_ok=True)
        
    # Initialize random seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Load pretrained VAE model
    logger.info(f"Loading pretrained VAE from {args.pretrained_model_name_or_path}")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder=args.subfolder,
        revision=args.revision,
    )
    
    # Create a copy for EMA
    ema_vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder=args.subfolder,
        revision=args.revision,
    )
    # Freeze EMA model parameters for now (we'll update them in the training loop)
    for param in ema_vae.parameters():
        param.requires_grad = False
    
    # Freeze only the last normalization and output convolution layers of the decoder
    # as per the paper's recommendation
    freeze_count = 0
    total_params = 0
    
    # Count total parameters for percentage calculation
    for param in vae.parameters():
        total_params += param.numel()
    
    # Freeze only decoder's last layers
    for name, param in vae.decoder.named_parameters():
        total_params += param.numel()
        # Look for the last normalization layer and output convolution
        if "norm_out" in name or "conv_out" in name or "out.weight" in name or "out.bias" in name:
            param.requires_grad = False
            freeze_count += param.numel()
            logger.info(f"Freezing decoder parameter: {name}")
    
    logger.info(f"Frozen {freeze_count} parameters out of {total_params} total parameters ({(freeze_count/total_params)*100:.2f}%)")
    
    # Additional freezing if specified
    if args.freeze_encoder:
        logger.info("Freezing all encoder parameters")
        for param in vae.encoder.parameters():
            param.requires_grad = False
    
    if args.freeze_decoder:
        logger.info("Freezing all decoder parameters")
        for param in vae.decoder.parameters():
            param.requires_grad = False
    
    # Create dataset and dataloader
    dataset = ImageDataset(args.data_dir, image_size=args.image_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    # Initialize LPIPS VGG model for perceptual loss
    lpips_model = lpips.LPIPS(net='vgg').to(accelerator.device)
    # Freeze LPIPS model parameters
    for param in lpips_model.parameters():
        param.requires_grad = False
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {trainable_params}")
    
    # Create optimizer with AdamW as specified
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                filter(lambda p: p.requires_grad, vae.parameters()),
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )
        except ImportError:
            logger.warning("bitsandbytes not installed, falling back to regular AdamW")
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, vae.parameters()),
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, vae.parameters()),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    
    # Scheduler and steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    
    if args.max_train_steps is None:
        args.max_train_steps = 10000  # Set default to 10,000 steps
        overrode_max_train_steps = True
    
    if args.num_train_epochs is None:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        logger.info(f"Calculated number of epochs: {args.num_train_epochs}")
        
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    
    # Prepare model, optimizer and dataloader
    vae, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        vae, optimizer, dataloader, lr_scheduler
    )
    # Move EMA model to device
    ema_vae = ema_vae.to(accelerator.device)
    
    # Calculate total batch size
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info(f"Training for {args.max_train_steps} steps with batch size {args.batch_size} (total seen images: ~{args.max_train_steps * args.batch_size})")
    
    # Initialize tracking
    if accelerator.is_main_process and args.with_tracking:
        experiment_config = vars(args)
        # Add model info to config
        experiment_config["model_parameters"] = {
            "latent_channels": args.latent_channels,
            "scaling_factor": vae.config.scaling_factor,
            "block_out_channels": vae.config.block_out_channels,
            "frozen_parameters": freeze_count,
            "frozen_percentage": (freeze_count/total_params)*100,
        }
        experiment_config["trainable_parameters"] = trainable_params
        
        wandb.init(
            project=args.tracker_project_name,
            config=experiment_config,
        )
        wandb.watch(vae)
    
    # Resume checkpoint if specified
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        
        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            
            initial_global_step = global_step
            
            if overrode_max_train_steps:
                args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
                
    else:
        initial_global_step = 0
    
    # Progress bar
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    # Function to update EMA model
    def update_ema_model(step):
        # Calculate EMA decay rate for half-life of 5000 steps
        with torch.no_grad():
            # Unwrap model if needed
            unwrapped_model = accelerator.unwrap_model(vae)
            for ema_param, model_param in zip(ema_vae.parameters(), unwrapped_model.parameters()):
                ema_param.data = args.ema_decay * ema_param.data + (1 - args.ema_decay) * model_param.data
    
    # Function to evaluate and generate samples
    def evaluate_and_sample(step, fixed_visualization_batch=None):
        if not accelerator.is_main_process:
            return

        vae.eval()
        ema_vae.eval()
        with torch.no_grad():
            # Fixed batch for consistent visualization
            if fixed_visualization_batch is None:
                eval_batch_fixed = next(iter(dataloader))
                fixed_visualization_batch = eval_batch_fixed["pixel_values"][:8].to(accelerator.device)
            
            fixed_eval_images = fixed_visualization_batch

            # Dynamically changing batch for evaluation
            eval_batch_dynamic = next(iter(dataloader))
            dynamic_eval_images = eval_batch_dynamic["pixel_values"][:8].to(accelerator.device)

            # Generate reconstructions with main model for fixed batch
            fixed_reconstructed = vae(fixed_eval_images).sample
            # Option 2: Using mean - deterministic encoding
            fixed_latents = vae.encode(fixed_eval_images).latent_dist.mean
            fixed_decoded = vae.decode(fixed_latents).sample
            fixed_ema_reconstructed = ema_vae(fixed_eval_images).sample

            # Calculate losses for fixed batch
            fixed_rec_loss = F.mse_loss(fixed_reconstructed, fixed_eval_images).item()
            # Calculate LPIPS loss for fixed batch
            fixed_reconstr_0_1 = (fixed_reconstructed + 1) / 2
            fixed_eval_images_0_1 = (fixed_eval_images + 1) / 2
            fixed_lpips_loss = lpips_model(fixed_reconstr_0_1, fixed_eval_images_0_1).mean().item()
            
            fixed_kl_loss = vae.encode(fixed_eval_images).latent_dist.kl().mean().item()
            #downscale_factor = random.choice([0.5, 0.25])  # 2x or 4x downsampling
            #x_down = F.interpolate(fixed_eval_images, scale_factor=downscale_factor, mode='bilinear', align_corners=False)
            #x_recon_down = F.interpolate(fixed_decoded, scale_factor=downscale_factor, mode='bilinear', align_corners=False)
            # fixed_scale_loss = F.mse_loss(x_recon_down, x_down, reduction='sum').item()
            # New code using einops (consistent with training):
            scale_factor = float(1 / random.choice([0.5, 0.25]))  # Convert 0.5/0.25 to 2/4
            interp_kwargs = dict(pattern='b c (h sh) (w sw) -> b c h w', reduction='mean', 
                                sh=int(scale_factor), sw=int(scale_factor))
                                
            x_down = einops.reduce(fixed_eval_images, **interp_kwargs)
            x_recon_down = einops.reduce(fixed_decoded, **interp_kwargs)
            fixed_scale_loss = F.mse_loss(x_recon_down, x_down, reduction='mean').item()
            # Total loss for fixed batch
            fixed_loss = fixed_rec_loss + args.kl_weight * fixed_kl_loss + args.regularization_alpha * fixed_scale_loss + args.lpips_weight * fixed_lpips_loss

            # Log metrics and images for the fixed batch
            if args.with_tracking:
                wandb.log({
                    "validation_fixed/reconstruction_loss": fixed_rec_loss,
                    "validation_fixed/kl_loss": fixed_kl_loss,
                    "validation_fixed/scale_loss": fixed_scale_loss,
                    "validation_fixed/lpips_loss": fixed_lpips_loss,
                    "validation_fixed/loss": fixed_loss,
                    "validation_fixed/downscale_factor": 1/scale_factor if args.regularization_alpha > 0 else 0,
                    # Add other logs here
                }, step=step, commit=False)

            # Generate reconstructions with main model for dynamic batch
            dynamic_reconstructed = vae(dynamic_eval_images).sample
            dynamic_latents = vae.encode(dynamic_eval_images).latent_dist.sample()
            dynamic_decoded = vae.decode(dynamic_latents).sample
            dynamic_ema_reconstructed = ema_vae(dynamic_eval_images).sample

            # Calculate losses for dynamic batch
            dynamic_rec_loss = F.mse_loss(dynamic_reconstructed, dynamic_eval_images).item()
            # Calculate LPIPS loss for dynamic batch
            dynamic_reconstr_0_1 = (dynamic_reconstructed + 1) / 2
            dynamic_eval_images_0_1 = (dynamic_eval_images + 1) / 2
            dynamic_lpips_loss = lpips_model(dynamic_reconstr_0_1, dynamic_eval_images_0_1).mean().item()
            
            dynamic_kl_loss = vae.encode(dynamic_eval_images).latent_dist.kl().mean().item()
            #x_down_dynamic = F.interpolate(dynamic_eval_images, scale_factor=downscale_factor, mode='bilinear', align_corners=False)
            #x_recon_down_dynamic = F.interpolate(dynamic_decoded, scale_factor=downscale_factor, mode='bilinear', align_corners=False)
            #dynamic_scale_loss = F.mse_loss(x_recon_down_dynamic, x_down_dynamic, reduction='sum').item()
            x_down_dynamic = einops.reduce(dynamic_eval_images, **interp_kwargs)
            x_recon_down_dynamic = einops.reduce(dynamic_decoded, **interp_kwargs)
            dynamic_scale_loss = F.mse_loss(x_recon_down_dynamic, x_down_dynamic, reduction='mean').item()
            # Total loss for dynamic batch
            dynamic_loss = dynamic_rec_loss + args.kl_weight * dynamic_kl_loss + args.regularization_alpha * dynamic_scale_loss + args.lpips_weight * dynamic_lpips_loss

            # Log metrics and images for the dynamic batch
            if args.with_tracking:
                wandb.log({
                    "validation_dynamic/reconstruction_loss": dynamic_rec_loss,
                    "validation_dynamic/kl_loss": dynamic_kl_loss,
                    "validation_dynamic/scale_loss": dynamic_scale_loss,
                    "validation_dynamic/lpips_loss": dynamic_lpips_loss,
                    "validation_dynamic/loss": dynamic_loss,
                    # Add other logs here
                }, step=step, commit=False)
            
            # Add spectral analysis for fixed batch
            spectral_metrics_fixed = spectral_analyzer.analyze(
                fixed_eval_images, 
                fixed_reconstructed, 
                step, 
                log_to_wandb=args.with_tracking
            )

            # Add spectral analysis for dynamic batch
            spectral_metrics_dynamic = spectral_analyzer.analyze(
                dynamic_eval_images, 
                dynamic_reconstructed, 
                step, 
                log_to_wandb=args.with_tracking
            )

            # Log spectral metrics
            if args.with_tracking:
                wandb.log({
                    "validation_fixed/high_freq_suppression": spectral_metrics_fixed["high_freq_suppression"],
                    "validation_dynamic/high_freq_suppression": spectral_metrics_dynamic["high_freq_suppression"],
                }, step=step)
        vae.train()
        return fixed_visualization_batch
    
    # Training loop
    global_step = initial_global_step
    vae.train()
    fixed_visualization_batch = None
    # Initialize spectral analyzer
    spectral_analyzer = SpectralAnalyzer(accelerator.device)
    
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(vae):
                pixel_values = batch["pixel_values"].to(accelerator.device)
                
                # Encode the input images to get the latent distribution
                encoded = vae.encode(pixel_values)
                
                # Option 1: Sample from distribution (or use mean if desired)
                latents = encoded.latent_dist.mean
                
                # Decode the latents to reconstruct the images
                decoded = vae.decode(latents)
                
                # Reconstruction loss
                reconstr_image = decoded.sample
                reconstruction_loss = F.mse_loss(reconstr_image, pixel_values, reduction="mean")
                

                # LPIPS perceptual loss
                if args.lpips_weight > 0:
                    # Scale images from [-1, 1] to [0, 1] for LPIPS
                    reconstr_image_0_1 = (reconstr_image + 1) / 2
                    pixel_values_0_1 = (pixel_values + 1) / 2
                    lpips_loss = lpips_model(reconstr_image_0_1, pixel_values_0_1).mean()
                else:
                    lpips_loss = torch.tensor(0.0).to(accelerator.device)
                
                # KL divergence loss (disabled for SE-regularized models as per paper)
                kl_loss = encoded.latent_dist.kl().mean() if args.kl_weight > 0 else torch.tensor(0.0).to(accelerator.device)
                
                # Apply downsampling regularization similar to the video model but for images
                if args.regularization_alpha > 0:
                    # Randomly choose scale factor from [2, 4] (inverse of [0.5, 0.25])
                    scale_factor = float(1 / random.choice([0.5, 0.25]))
                    
                    # Downsample both original images and latents
                    interp_kwargs = dict(pattern='b c (h sh) (w sw) -> b c h w', reduction='mean', 
                                        sh=int(scale_factor), sw=int(scale_factor))
                    
                    # Downsample original images and latent representations
                    pixels_down = einops.reduce(pixel_values, **interp_kwargs)
                    
                    # For latents, we need to handle the different structure
                    latents_down = einops.reduce(latents, **interp_kwargs)
                    
                    # Generate new predictions from downsampled latents
                    decoded_down = vae.decode(latents_down)
                    reconstr_down = decoded_down.sample
                    
                    # Compute loss between the new predictions and downsampled originals
                    scale_loss = F.mse_loss(reconstr_down, pixels_down, reduction='mean')
                else:
                    scale_loss = torch.tensor(0.0).to(accelerator.device)
                    
                # Total loss with regularization parameters
                loss = reconstruction_loss + args.kl_weight * kl_loss + args.regularization_alpha * scale_loss + args.lpips_weight * lpips_loss

                # Log metrics
                if args.with_tracking and accelerator.is_main_process:
                    wandb.log({
                        "train/reconstruction_loss": reconstruction_loss.detach().item(),
                        "train/kl_loss": kl_loss.detach().item(),
                        "train/scale_loss": scale_loss.detach().item(),
                        "train/lpips_loss": lpips_loss.detach().item(),  # Log LPIPS loss
                        "train/loss": loss.detach().item(),
                        "train/downscale_factor": 1/scale_factor if args.regularization_alpha > 0 else 0,
                    }, step=global_step)
                  
                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Update progress
            if accelerator.sync_gradients:
                # Update EMA model
                update_ema_model(global_step)
                progress_bar.update(1)
                global_step += 1
                
                # Logging
                if accelerator.is_main_process:
                    logs = {
                        "loss": loss.detach().item(),
                        "reconstruction_loss": reconstruction_loss.detach().item(),
                        "kl_loss": kl_loss.detach().item(),
                        "scale_loss": scale_loss.detach().item(),
                        "lpips_loss": lpips_loss.detach().item(),  # Add LPIPS to logs
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                        "epoch": epoch,
                        "images_seen": global_step * args.batch_size * accelerator.num_processes,
                    }
                    progress_bar.set_postfix(**logs)
                    if args.with_tracking:
                        accelerator.log(logs, step=global_step)
                
                # Run validation
                if global_step % args.validation_steps == 0:
                    fixed_visualization_batch = evaluate_and_sample(global_step, fixed_visualization_batch)
                
                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        # Save accelerator state
                        accelerator.save_state(save_path)
                        # Unwrap model for proper saving
                        unwrapped_vae = accelerator.unwrap_model(vae)
                        
                        # Save model with proper naming convention (config.json and diffusion_pytorch_model.safetensors)
                        unwrapped_vae.save_pretrained(
                            save_path,
                            safe_serialization=True,  # Use safetensors format
                        )
                        
                        # Also save EMA model
                        ema_save_path = os.path.join(save_path, "ema")
                        os.makedirs(ema_save_path, exist_ok=True)
                        ema_vae.save_pretrained(
                            ema_save_path,
                            safe_serialization=True,  # Use safetensors format
                        )
                        
                        logger.info(f"Saved checkpoint at step {global_step} to {save_path}")
            
            # Break if max steps reached
            if global_step >= args.max_train_steps:
                break
        
        # Break if max steps reached
        if global_step >= args.max_train_steps:
            break
    
    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Unwrap the model
        vae = accelerator.unwrap_model(vae)
        
        # Save both the regular model and EMA model
        vae.save_pretrained(args.output_dir)
        ema_vae.save_pretrained(os.path.join(args.output_dir, "ema"))
        # Generate final samples
        evaluate_and_sample(global_step, fixed_visualization_batch)
        
        logger.info(f"Training completed after {global_step} steps. Total images seen: {global_step * args.batch_size * accelerator.num_processes}")
    accelerator.end_training()

class SpectralAnalyzer:
    """Monitors spectral properties of images during VAE training."""
    def __init__(self, device, n_bands=3):
        self.device = device
        self.n_bands = n_bands
        
        # Initialize wavelet transform
        self.dwt = DWTForward(J=1, mode='zero', wave='haar').to(device)
    
    def compute_dct(self, images):
        """Compute DCT coefficients for batch of images."""
        # Move to CPU for DCT if using scipy
        batch_size, channels, height, width = images.shape
        dct_coeffs = torch.zeros_like(images)
        for c in range(channels):
            # Using torch-dct which supports GPU computation
            dct_coeffs[:, c] = tdct.dct_2d(images[:, c])
        return dct_coeffs        
    
    def compute_wavelets(self, images):
        """Compute wavelet decomposition of images."""
        # Returns low-frequency approximation and list of high-frequency details
        return self.dwt(images)
    
    def frequency_bands_analysis(self, dct_coeffs):
        """Analyze energy in different frequency bands (low, mid, high)."""
        batch_size, channels, height, width = dct_coeffs.shape
        # Create frequency band masks
        low_idx = int(height * 0.25)  # Low freq: 0-25%
        mid_idx = int(height * 0.75)  # Mid freq: 25-75%
        
        # Calculate energy (squared magnitude) of coefficients
        energy = dct_coeffs.pow(2)
        total_energy = energy.sum(dim=[1, 2, 3]).mean().item()
        # Low frequency energy (top-left corner of DCT)
        low_energy = energy[:, :, :low_idx, :low_idx].sum(dim=[1, 2, 3]).mean().item()
        # Mid frequency energy
        mid_mask = torch.ones((height, width), device=self.device)
        mid_mask[:low_idx, :low_idx] = 0
        mid_mask[mid_idx:, mid_idx:] = 0
        mid_energy = (energy * mid_mask.view(1, 1, height, width)).sum(dim=[1, 2, 3]).mean().item()
        # High frequency energy
        high_mask = torch.zeros((height, width), device=self.device)
        high_mask[mid_idx:, :] = 1
        high_mask[:, mid_idx:] = 1
        high_energy = (energy * high_mask.view(1, 1, height, width)).sum(dim=[1, 2, 3]).mean().item()
        return {
            "total_energy": total_energy,
            "low_freq_energy": low_energy,
            "mid_freq_energy": mid_energy,
            "high_freq_energy": high_energy,
            "low_freq_ratio": low_energy / total_energy if total_energy > 0 else 0,
            "mid_freq_ratio": mid_energy / total_energy if total_energy > 0 else 0,
            "high_freq_ratio": high_energy / total_energy if total_energy > 0 else 0,
        }
    
    def analyze(self, originals, reconstructions, step, log_to_wandb=False):
        """Analyze spectral properties of original and reconstructed images."""
        # Get DCT coefficients
        orig_dct = self.compute_dct(originals)
        recon_dct = self.compute_dct(reconstructions)
        diff_dct = self.compute_dct(originals - reconstructions)
        # Get wavelet coefficients
        orig_wav_ll, orig_wav_bands = self.compute_wavelets(originals)
        recon_wav_ll, recon_wav_bands = self.compute_wavelets(reconstructions)
        # Analyze frequency bands
        orig_metrics = self.frequency_bands_analysis(orig_dct)
        recon_metrics = self.frequency_bands_analysis(recon_dct)
        # Calculate high frequency suppression ratio
        high_freq_suppression = 1.0 - (recon_metrics["high_freq_ratio"] / orig_metrics["high_freq_ratio"]) \
            if orig_metrics["high_freq_ratio"] > 0 else 0.0
        metrics = {
            "high_freq_suppression": high_freq_suppression,
            "orig_high_freq_ratio": orig_metrics["high_freq_ratio"],
            "recon_high_freq_ratio": recon_metrics["high_freq_ratio"],
            "orig_low_freq_ratio": orig_metrics["low_freq_ratio"],
            "recon_low_freq_ratio": recon_metrics["low_freq_ratio"],
        }
        
        # Log visualizations at specified frequency
        if log_to_wandb:
            # Prepare DCT visualization (log scale for better visibility)
            def prepare_dct_viz(dct_coeffs):
                # Take abs and log scale for visualization
                viz = torch.log10(torch.abs(dct_coeffs) + 1e-8)
                # Normalize to [0,1] for visualization
                viz = (viz - viz.min()) / (viz.max() - viz.min() + 1e-8)
                return viz[:, 0:3] if viz.shape[1] >= 3 else viz.repeat(1, 3, 1, 1)
            
            # Create visualization grids
            orig_dct_viz = prepare_dct_viz(orig_dct)
            recon_dct_viz = prepare_dct_viz(recon_dct)
            diff_dct_viz = prepare_dct_viz(diff_dct)
            # Log images and metrics
            wandb.log({
                "spectral/dct_coefficients": wandb.Image(
                    torch.cat([
                        orig_dct_viz[:4].cpu(), 
                        recon_dct_viz[:4].cpu(), 
                        diff_dct_viz[:4].cpu()
                    ], dim=2),
                    caption="Top: Original DCT, Middle: Reconstructed DCT, Bottom: Difference DCT"
                ),
                "spectral/metrics": {
                    f"spectral/{k}": v for k, v in metrics.items()
                }
            }, step=step, commit=False)
            
            # Visualize wavelet decomposition (first level only)
            # This shows how well the model preserves details at different scales
            wavelet_viz = self.visualize_wavelets(orig_wav_ll, orig_wav_bands, recon_wav_ll, recon_wav_bands, step)
            wandb.log({
                "spectral/wavelets": wandb.Image(wavelet_viz),
            }, step=step)
        return metrics        
    
    def visualize_wavelets(self, orig_wav_ll, orig_wav_bands, recon_wav_ll, recon_wav_bands, step):
        """Visualize wavelet decomposition."""
        # Create a grid for visualization
        n_bands = 4
        fig, axes = plt.subplots(n_bands, 2, figsize=(10, 5 * n_bands))
        fig.suptitle(f"Wavelet Decomposition at Step {step}", fontsize=16)
        mapping = {0: "LL", 1: "LH", 2: "HL", 3: "HH"}
        for i in range(n_bands):
            # Get the first image from the batch and first channel
            # PyTorch wavelets output shape is often (batch, channels, height, width)
            if i == 0:
                # Low frequency band
                orig_band = orig_wav_ll.cpu().detach()
                recon_band = recon_wav_ll.cpu().detach()
            else:
                # High frequency bands
                orig_band = orig_wav_bands[0].cpu().detach()
                recon_band = recon_wav_bands[0].cpu().detach()
                # Unpack the high frequency bands
                orig_band = torch.unbind(orig_band, dim=2)[i-1]
                recon_band = torch.unbind(recon_band, dim=2)[i-1]
            
            if len(orig_band.shape) == 4:
                # Take first batch item, all channels
                orig_band = orig_band[0].cpu().detach()
                recon_band = recon_band[0].cpu().detach()
            else:
                # Already extracted batch dimension
                orig_band = orig_band.cpu().detach()
                recon_band = recon_band.cpu().detach()
            
            # Normalize for visualization
            orig_band = (orig_band - orig_band.min()) / (orig_band.max() - orig_band.min() + 1e-8)
            recon_band = (recon_band - recon_band.min()) / (recon_band.max() - recon_band.min() + 1e-8)
            
            # For visualization, use first channel if multichannel
            if len(orig_band.shape) > 2:
                # Convert to numpy and rearrange dimensions: (channels, height, width) -> (height, width, channels)
                orig_band_np = orig_band.permute(1, 2, 0).numpy()
                recon_band_np = recon_band.permute(1, 2, 0).numpy()
                # If channels = 1, remove the channel dimension
                if orig_band.shape[0] == 1:
                    orig_band_np = orig_band_np.squeeze(-1)
                    recon_band_np = recon_band_np.squeeze(-1)
            else:
                # Already in right format (height, width)
                orig_band_np = orig_band.numpy()
                recon_band_np = recon_band.numpy()
            
            axes[i, 0].imshow(orig_band_np, cmap='viridis' )
            axes[i, 0].set_title(f"Original Band {mapping[i]}")
            axes[i, 0].axis("off")
            
            axes[i, 1].imshow(recon_band_np, cmap='viridis' )
            axes[i, 1].set_title(f"Reconstructed Band {mapping[i]}")
            axes[i, 1].axis("off")
        
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs("results/wavelet_viz", exist_ok=True)
        plt.savefig(os.path.join("results/wavelet_viz", f"wavelet_visualization_step_{step}.png"))
        plt.close(fig)
        return fig        

def set_seed(seed):
    """Sets the random seed for reproducibility."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)


if __name__ == "__main__":

    main()