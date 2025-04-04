CUDA_VISIBILE_DEVICES=1 python vae_finetune_diffusability.py \
  --pretrained_model_name_or_path "black-forest-labs/FLUX.1-dev" \
  --subfolder "vae" \
  --data_dir /mnt/share/datasets/img_align_celeba \
  --output_dir ./vae-finetuned \
  --image_size 256 \
  --batch_size 16 \
  --num_train_epochs 2 \
  --with_tracking \
  --max_train_steps 20000

