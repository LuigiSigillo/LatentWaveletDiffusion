#!/bin/sh
#SBATCH -A IscrC_NeuroGen
#SBATCH -p boost_usr_prod
#SBATCH --time=23:00:00      
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node=1    
#SBATCH --gres=gpu:4          
#SBATCH --cpus-per-task=8      
#SBATCH --job-name=traning_2k

echo "NODELIST="${SLURM_NODELIST}

cd /leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav
export WANDB_MODE=offline
module load anaconda3
source activate URAE

export NUM_WORKERS=4
export DATA_DIR="/leonardo_scratch/large/userexternal/lsigillo/laion_high_res_images"
export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=$NUM_WORKERS src/cache_prompt_embeds.py \
    --data_root=$DATA_DIR \
    --batch_size=128 \
    --num_worker=$NUM_WORKERS \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --mixed_precision='bf16' \
    --output_dir=$DATA_DIR \
    --column="prompt" \
    --max_sequence_length=512 \
    --cache_dir=$SCRATCH
