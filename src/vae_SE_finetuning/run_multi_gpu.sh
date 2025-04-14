#!/bin/bash

# Script to launch multi-GPU training for VAE fine-tuning
# Usage: bash run_multi_gpu.sh

# Configuration
PRETRAINED_MODEL_PATH="black-forest-labs/FLUX.1-dev"  # Replace with your model path
DATA_DIR="/mnt/share/Luigi/Documents/URAE/dataset/laion_high_resolution_image"  # Replace with your data directory
OUTPUT_DIR="/mnt/share/Luigi/Documents/URAE/models/vae_finetuned_LAION"
IMAGE_SIZE=256
BATCH_SIZE=16  # Per GPU batch size
LEARNING_RATE=1e-5
MAX_STEPS=20000
MIXED_PRECISION="bf16"  # Use bf16 for faster training on modern GPUs
export CUDA_VISIBLE_DEVICES=2,3 # Specify the GPUs to use

# Use a custom port to avoid conflicts
# Port 0 means "use the next available port"
MAIN_PORT=0

# Create output directory
mkdir -p $OUTPUT_DIR

# Launch multi-GPU training using accelerate
accelerate launch \
    --multi_gpu \
    --mixed_precision $MIXED_PRECISION \
    --num_processes=2 \
    --main_process_port=$MAIN_PORT \
    vae_finetune_diffusability.py \
    --pretrained_model_name_or_path $PRETRAINED_MODEL_PATH \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --image_size $IMAGE_SIZE \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_train_steps $MAX_STEPS \
    --multi_gpu \
    --sync_batch_stats \
    --with_tracking \
    --allow_tf32

echo "Training completed. Model saved to $OUTPUT_DIR"
