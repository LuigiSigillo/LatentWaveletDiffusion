export CUDA_VISIBLE_DEVICES=0
export VAE_MODEL_NAME="vae_SE_FLUX_512"

python vae_finetune_diffusability.py \
  --pretrained_model_name_or_path "black-forest-labs/FLUX.1-dev" \
  --subfolder "vae" \
  --output_dir ./ckpt/$VAE_MODEL_NAME \
  --image_size 512 \
  --batch_size 3 \
  --num_train_epochs 200 \
  --with_tracking \
  --max_train_steps 60000 \
  --regularization_alpha 0.25 \
  --lpips_weight 0.05 \
  --cache_dir /leonardo_scratch/large/userexternal/lsigillo \
  --data_dir /leonardo_scratch/large/userexternal/lsigillo/Aesthetic-4K/train/ \
  --resume_from_checkpoint "latest"
  # --data_dir /leonardo_scratch/large/userexternal/lsigillo/laion_high_res_images_2K/ \
