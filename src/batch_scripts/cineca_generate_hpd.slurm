#!/bin/sh
#SBATCH -A IscrC_UniMod
#SBATCH -p boost_usr_prod
#SBATCH --time=23:50:00      
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node=1    
#SBATCH --gres=gpu:4          
#SBATCH --cpus-per-task=16      
#SBATCH --job-name=generate_lumina_hpd
#SBATCH --mem=128G
echo "NODELIST="${SLURM_NODELIST}

cd /leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav
# export WANDB_MODE=offline
# module load anaconda3
source /leonardo_scratch/fast/IscrC_UniMod/luigi/miniconda3/etc/profile.d/conda.sh
conda activate imax
export http_proxy=http://login07:3168
export https_proxy=http://login07:3168
export HF_CACHE=/leonardo_scratch/large/userexternal/lsigillo/

python src/genereate_lumina_gt.py --generate  --checkpoint FLUX