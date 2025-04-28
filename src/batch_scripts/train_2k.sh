export MODEL_NAME="black-forest-labs/FLUX.1-dev"
# export VAE_MODEL_NAME="/mnt/share/Luigi/Documents/URAE/src/vae_SE_finetuning/ckpt/vae_normal_512/checkpoint-60000"
export DATA_DIR="/mnt/share/Luigi/Documents/URAE/dataset/laion_high_resolution_images"
# export LATENT_CODE_DIR=""
export OUTPUT_DIR="/mnt/share/Luigi/Documents/URAE/src/ckpt/URAE_WAV_ATT_LAION"
export PRECISION="bf16"
export CUDA_VISIBLE_DEVICES=6,7
export MASTER_PORT=29501
MASTER_PORT=$MASTER_PORT accelerate launch --main_process_port $MASTER_PORT --num_processes 2 --multi_gpu --mixed_precision $PRECISION src/train_2k.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --dataset_root=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision=$PRECISION \
  --dataloader_num_workers=4 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1. \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --seed="0" \
  --real_prompt_ratio=0.2 \
  --checkpointing_steps=1000 \
  --gradient_checkpointing \
  --wavelet_attention
  # --latent_code_dir=$LATENT_CODE_DIR \
  # --pretrained_vae_path=$VAE_MODEL_NAME \

