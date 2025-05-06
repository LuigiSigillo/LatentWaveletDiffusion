export CUDA_VISIBLE_DEVICES=0
export VAE_MODEL_NAME="vae_SE_SANA_512"

python vae_finetune_diffusability.py \
  --pretrained_model_name_or_path "Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers" \
  --subfolder "vae" \
  --data_dir /leonardo_scratch/large/userexternal/lsigillo/laion_high_res_images_2K/ \
  --output_dir ./ckpt/$VAE_MODEL_NAME \
  --image_size 512 \
  --batch_size 3 \
  --num_train_epochs 200 \
  --with_tracking \
  --max_train_steps 60000 \
  --regularization_alpha 0.25 \
  --lpips_weight 0.05 \
  --cache_dir /leonardo_scratch/large/userexternal/lsigillo
