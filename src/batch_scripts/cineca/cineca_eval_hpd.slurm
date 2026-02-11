#!/bin/sh
#SBATCH -A IscrC_UniMod
#SBATCH -p boost_usr_prod
#SBATCH --time=23:50:00      
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node=1    
#SBATCH --gres=gpu:1         
#SBATCH --cpus-per-task=16      
#SBATCH --job-name=eval_SE_urae_4k
#SBATCH --mem=128G
echo "NODELIST="${SLURM_NODELIST}

cd /leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav
# export WANDB_MODE=offline
# module load anaconda3
source /leonardo_scratch/fast/IscrC_UniMod/luigi/miniconda3/etc/profile.d/conda.sh
conda activate URAE_eval
export http_proxy=http://login07:3168
export https_proxy=http://login07:3168


###############################       2K      ##############
#export CHECKPOINT="/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/l_003_URAE_VAE_NOOO_SE_WAV_ATT_LAION_2048/checkpoint-6000"
#export CHECKPOINT="/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/l_003_URAE_VAE_SE_WAV_ATT_LAION_2048/checkpoint-6000"
#export CHECKPOINT="/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/l_03_URAE_VAE_SE_WAV_ATT_LAION_2560/checkpoint-2000"
#export CHECKPOINT="/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/l_003_URAE_VAE_SE_WAV_ATT_LAION_2560/checkpoint-4000"
#export CHECKPOINT="/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/l_003_URAE_VAE_NOOO_SE_WAV_ATT_LAION_2560/checkpoint-4000"
#export CHECKPOINT="/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/l_03_URAE_VAE_SE_WAV_ATT_LAION_2048/checkpoint-2000"

# export CHECKPOINT="/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/MASK_INPUT_l_03_URAE_LAION_2048_SE/checkpoint-2000"
# export CHECKPOINT="/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/MASK_INPUT_l_03_URAE_LAION_2048/checkpoint-2000"
#python src/eval.py --generate --checkpoint $CHECKPOINT

###############################       4K      ##############
export CHECKPOINT="/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/l_03_4K_URAE_VAE_SE_WAV_ATT_LAION_4096/checkpoint-2000/adapter_weights.safetensors"
python src/eval_4k.py --checkpoint_path $CHECKPOINT --style concept-art