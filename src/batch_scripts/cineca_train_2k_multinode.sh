#!/bin/sh

#SBATCH -A IscrC_NeuroGen
#SBATCH -p boost_usr_prod
#SBATCH --job-name=multinode
#SBATCH -D .
#SBATCH --output=O-%x.%j
#SBATCH --error=E-%x.%j
#SBATCH --nodes=2                 # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:4                # number of GPUs per node
#SBATCH --cpus-per-task=32         # number of cores per tasks
#SBATCH --time=23:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --mem=256G

######################
### Set enviroment ###
######################
cd /leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav

source /leonardo_scratch/fast/IscrC_UniMod/luigi/miniconda3/etc/profile.d/conda.sh
conda activate URAE
export GPUS_PER_NODE=4

######################
export http_proxy=http://login07:3168
export https_proxy=http://login07:3168

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# Get the list of nodes and the first node (master node)
master_node=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
# Get the IP address of the master node
master_ip=$(srun --nodes=1 --ntasks=1 --nodelist=$master_node bash -c "hostname -I | awk '{print \$1}'")
# master_ip=$(srun --nodes=1 --ntasks=1 --nodelist=$master_node bash -c "ip addr show ib0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}'")
######################
# Set environment variables for distributed training
export SLURM_MASTER_ADDR=$master_ip
export SLURM_MASTER_PORT=29301
export SLURM_TOTAL_GPUS=$(($SLURM_NNODES * $SLURM_GPUS_ON_NODE))

#####################
# Optional: Print out the values for debugging
echo "Custom parameter values:"
echo "MASTER ADDRESS: $SLURM_MASTER_ADDR"
echo "MASTER_PORT: $SLURM_MASTER_PORT"
echo "NUMBER OF NODES REQUESTED: $SLURM_NNODES"
echo "NUMBER OF NODES ALLOCATED: $SLURM_JOB_NUM_NODES"
echo "NUMBER OF GPUS PER NODE: $SLURM_GPUS_ON_NODE"
echo "TOTAL GPUS: $SLURM_TOTAL_GPUS" 
echo "MACHINE RANK: $SLURM_NODEID"

#### Set variables ####
export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export VAE_MODEL_NAME="/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/vae_SE_finetuning/ckpt/VAE_SE/checkpoint-60000"
export DATA_DIR="/leonardo_scratch/large/userexternal/lsigillo/laion_high_res_images"
export LATENT_CODE_DIR="/leonardo_scratch/large/userexternal/lsigillo/se_latents"

export OUTPUT_DIR="/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/BIG_URAE_VAE_SE_WAV_ATT_LAION"

export PRECISION="bf16"

export LAUNCHER="accelerate launch \
    --main_process_ip=$SLURM_MASTER_ADDR \
    --main_process_port=$SLURM_MASTER_PORT \
    --num_processes $SLURM_TOTAL_GPUS \
    --num_machines $SLURM_JOB_NUM_NODES \
    --machine_rank=$SLURM_NODEID \
    --mixed_precision $PRECISION \
    --multi_gpu \
    --dynamo_backend=no \
    "
export SCRIPT_ARGS=" \
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
    --cache_dir=$SCRATCH \
    --latent_code_dir=$LATENT_CODE_DIR \
    --pretrained_vae_path=$VAE_MODEL_NAME \
    "

export PYTHON_FILE="src/train_2k.py"
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $PYTHON_FILE $SCRIPT_ARGS" 
srun $CMD