export CUDA_VISIBLE_DEVICES=0
export VAE_MODEL_NAME="vae_normal_512"

python vae_finetune_diffusability.py \
  --pretrained_model_name_or_path "black-forest-labs/FLUX.1-dev" \
  --subfolder "vae" \
  --data_dir /mnt/share/Luigi/Documents/URAE/dataset/laion_high_resolution_images \
  --output_dir ./ckpt/$VAE_MODEL_NAME \
  --image_size 512 \
  --batch_size 3 \
  --num_train_epochs 200 \
  --with_tracking \
  --max_train_steps 60000 \
  --regularization_alpha 0 \
  --lpips_weight 0.05
