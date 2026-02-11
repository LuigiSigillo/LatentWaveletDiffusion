export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export VAE_MODEL_NAME="/path/to/fine-tuned/vae"  # or comment out to use default
export DATA_DIR="/path/to/dataset"
export LATENT_CODE_DIR="/path/to/cached/latents"
export OUTPUT_DIR="/path/to/checkpoints"
export PRECISION="bf16"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=29508

MASTER_PORT=$MASTER_PORT accelerate launch --main_process_port $MASTER_PORT --num_processes 4 --multi_gpu --mixed_precision $PRECISION src/train_4k.py \
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
  --max_train_steps=10000 \
  --seed="0" \
  --real_prompt_ratio=0.2 \
  --checkpointing_steps=1000 \
  --gradient_checkpointing \
  --ntk_factor=10 \
  --proportional_attention \
  --pretrained_lora="src/ckpt/urae_2k_adapter.safetensors" \
  --wavelet_attention \
  --wav_att_l_mask=0.3 \
  --cache_dir=$HF_HOME \
  --latent_code_dir=$LATENT_CODE_DIR \
  --pretrained_vae_path=$VAE_MODEL_NAME \
  --resume_from_checkpoint="latest"
