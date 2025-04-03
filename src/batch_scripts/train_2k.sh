export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export DATA_DIR="/mnt/share/datasets/img_align_celeba"
export OUTPUT_DIR="/mnt/share/Luigi/Documents/URAE/src/ckpt"
export PRECISION="bf16"
export CUDA_VISIBLE_DEVICES=0,1
accelerate launch --num_processes 2 --multi_gpu --mixed_precision $PRECISION src/train_2k.py \
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
  --gradient_checkpointing 
