export NUM_WORKERS=4
export DATA_DIR="/mnt/share/Luigi/Documents/URAE/dataset/laion_high_resolution_images"
export OUTPUT_DIR="/mnt/share/Luigi/Documents/URAE/dataset/latents_VAE"
# export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export MODEL_NAME="/mnt/share/Luigi/Documents/URAE/src/vae_SE_finetuning/ckpt/vae_normal_512/checkpoint-60000"
export CUDA_VISIBLE_DEVICES=3,4,6,7
export MASTER_PORT=29601
torchrun --nproc_per_node=$NUM_WORKERS --master_port=$MASTER_PORT src/cache_latent_codes.py \
    --data_root=$DATA_DIR \
    --num_worker=$NUM_WORKERS \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --mixed_precision='bf16' \
    --output_dir=$OUTPUT_DIR \
    --resolution=2560 \