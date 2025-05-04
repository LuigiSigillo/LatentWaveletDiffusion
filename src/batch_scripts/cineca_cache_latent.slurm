#!/bin/sh
#SBATCH -A IscrC_NeuroGen
#SBATCH -p boost_usr_prod
#SBATCH --time=23:00:00      
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node=1    
#SBATCH --gres=gpu:4          
#SBATCH --cpus-per-task=8      
#SBATCH --job-name=cache_latent

echo "NODELIST="${SLURM_NODELIST}

cd /leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav
# export WANDB_MODE=offline
# module load anaconda3

source /leonardo_scratch/fast/IscrC_UniMod/luigi/miniconda3/etc/profile.d/conda.sh
conda activate URAE
export http_proxy=http://login07:3168
export https_proxy=http://login07:3168


export NUM_WORKERS=4
export DATA_DIR="/leonardo_scratch/large/userexternal/lsigillo/Aesthetic-4K/train"
export OUTPUT_DIR="/leonardo_scratch/large/userexternal/lsigillo/Aesthetic_se_latents_4096"
# export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export MODEL_NAME="/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/vae_SE_finetuning/ckpt/VAE_SE/checkpoint-60000"
# export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "DATA_DIR="${DATA_DIR}
echo "OUTPUT_DIR="${OUTPUT_DIR}
echo "MODEL_NAME="${MODEL_NAME}

torchrun --nproc_per_node=$NUM_WORKERS src/cache_latent_codes.py \
    --data_root=$DATA_DIR \
    --num_worker=$NUM_WORKERS \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --mixed_precision='bf16' \
    --output_dir=$OUTPUT_DIR \
    --resolution=4096 \
    --cache_dir=$SCRATCH \
