export NUM_WORKERS=4
export DATA_DIR="/path/to/your/dataset"
export OUTPUT_DIR="/path/to/latents"
# export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export MODEL_NAME="/path/to/fine-tuned/vae"  # or use default FLUX VAE
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=29601
torchrun --nproc_per_node=$NUM_WORKERS --master_port=$MASTER_PORT src/cache_latent_codes.py \
    --data_root=$DATA_DIR \
    --num_worker=$NUM_WORKERS \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --mixed_precision='bf16' \
    --output_dir=$OUTPUT_DIR \
    --resolution=2048 \
    --cache_dir=$HF_HOME \