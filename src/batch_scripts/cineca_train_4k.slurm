#!/bin/sh
#SBATCH -A IscrC_NeuroGen
#SBATCH -p boost_usr_prod
#SBATCH --time=23:00:00      
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node=1    
#SBATCH --gres=gpu:4          
#SBATCH --cpus-per-task=16      
#SBATCH --job-name=train_SE_4k
#SBATCH --mem=400G
echo "NODELIST="${SLURM_NODELIST}

cd /leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav
# export WANDB_MODE=offline
# module load anaconda3
source /leonardo_scratch/fast/IscrC_UniMod/luigi/miniconda3/etc/profile.d/conda.sh
conda activate URAE
export http_proxy=http://login07:3168
export https_proxy=http://login07:3168


export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export VAE_MODEL_NAME="/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/vae_SE_finetuning/ckpt/VAE_SE/checkpoint-60000"
export DATA_DIR="/leonardo_scratch/large/userexternal/lsigillo/laion_high_res_images_4K"
export LATENT_CODE_DIR="$FAST/luigi/dataset/laion_2K/se_latents/se_latents_4K"

export OUTPUT_DIR="/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/4K_URAE_VAE_SE_WAV_ATT_LAION"

export PRECISION="bf16"
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=29502

MASTER_PORT=$MASTER_PORT accelerate launch --main_process_port $MASTER_PORT --num_processes 8 --multi_gpu --mixed_precision $PRECISION src/train_4k.py \
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
  --cache_dir=$SCRATCH \
  --latent_code_dir=$LATENT_CODE_DIR \
  --pretrained_vae_path=$VAE_MODEL_NAME \

