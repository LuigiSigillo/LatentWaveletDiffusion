CUDA_VISIBILE_DEVICES=2 python vae_finetune_diffusability.py \
  --pretrained_model_name_or_path "black-forest-labs/FLUX.1-dev" \
  --subfolder "vae" \
  --data_dir /mnt/share/Luigi/Documents/URAE/dataset/laion_high_resolution_images \
  --output_dir ./vae_finetuned_LAION \
  --image_size 256 \
  --batch_size 16 \
  --num_train_epochs 2 \
  --with_tracking \
  --max_train_steps 20000

