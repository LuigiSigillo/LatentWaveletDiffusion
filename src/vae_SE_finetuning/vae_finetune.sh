CUDA_VISIBILE_DEVICES=0 python vae_finetune_diffusability.py \
  --pretrained_model_name_or_path "black-forest-labs/FLUX.1-dev" \
  --subfolder "vae" \
  --data_dir /mnt/share/Luigi/Documents/URAE/dataset/laion_high_resolution_images \
  --output_dir ./ckpt/vae_new_finetuned_LAION \
  --image_size 256 \
  --batch_size 12 \
  --num_train_epochs 20 \
  --with_tracking \
  --max_train_steps 30000

