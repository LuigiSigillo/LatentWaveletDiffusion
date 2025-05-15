#!/bin/sh
#SBATCH -A IscrC_UniMod
#SBATCH -p boost_usr_prod
#SBATCH --time=23:50:00      
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node=1    
#SBATCH --gres=gpu:4          
#SBATCH --cpus-per-task=16      
#SBATCH --job-name=urae_train_wald_2048_se
#SBATCH --mem=256G
#SBATCH --dependency=afterany:15766615
echo "NODELIST="${SLURM_NODELIST}

cd /leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav
# export WANDB_MODE=offline
# module load anaconda3
source /leonardo_scratch/fast/IscrC_UniMod/luigi/miniconda3/etc/profile.d/conda.sh
conda activate URAE
export http_proxy=http://login07:3168
export https_proxy=http://login07:3168

export MODEL_NAME="black-forest-labs/FLUX.1-dev"
#export VAE_MODEL_NAME="/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/vae_SE_finetuning/ckpt/VAE_SE/checkpoint-60000"
export VAE_MODEL_NAME="/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/vae_SE_finetuning/ckpt/vae_SE_FLUX_512/checkpoint-60000"
export DATA_DIR="/leonardo_scratch/large/userexternal/lsigillo/laion_high_res_images_2K"
export LATENT_CODE_DIR="/leonardo_scratch/large/userexternal/lsigillo/laion_se_latents_2048"
#export LATENT_CODE_DIR="/leonardo_scratch/large/userexternal/lsigillo/laion_se_latents_2560"

export OUTPUT_DIR="/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/l_03_URAE_LAION_2048_SE_AEST"


export PRECISION="bf16"
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=29403
MASTER_PORT=$MASTER_PORT accelerate launch --main_process_port $MASTER_PORT --num_processes 4 --multi_gpu --mixed_precision $PRECISION src/train_2k.py \
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
  --wavelet_attention \
  --wav_att_l_mask=0.3 \
  --cache_dir=$SCRATCH \
  --latent_code_dir=$LATENT_CODE_DIR \
  --pretrained_vae_path=$VAE_MODEL_NAME \

