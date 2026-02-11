#!/bin/bash
#SBATCH -A IscrC_NeuroGen
#SBATCH -p boost_usr_prod
#SBATCH --job-name=multinode_training
#SBATCH --output=multinode_%j.out
#SBATCH --error=multinode_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=23:00:00
#SBATCH --mem=256G

# Print SLURM environment variables for debugging
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_NODEID: $SLURM_NODEID"

# Set environment
cd /leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav
source /leonardo_scratch/fast/IscrC_UniMod/luigi/miniconda3/etc/profile.d/conda.sh
conda activate URAE

# Set proxies if needed (consider commenting these out if they cause issues)
export http_proxy=http://login07:3168
export https_proxy=http://login07:3168
export no_proxy=localhost,127.0.0.1

# Set network parameters for distributed training
# Get the master node hostname and IP - using the InfiniBand interface
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" hostname --ip-address | head -n 1)
MASTER_PORT=$(( 10000 + RANDOM % 20000 )) # Choose a random port between 10000-30000

# Use host network for more reliable communication
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_SOCKET_IFNAME=ib0

# Calculate total GPUs
export GPUS_PER_NODE=4
export TOTAL_GPUS=$(( SLURM_NNODES * GPUS_PER_NODE ))

# Print network configuration
echo "Master node: $MASTER_NODE"
echo "Master IP: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Total GPUs: $TOTAL_GPUS"
echo "Node ID: $SLURM_NODEID"

# Define model and data paths
export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export VAE_MODEL_NAME="/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/vae_SE_finetuning/ckpt/VAE_SE/checkpoint-60000"
export DATA_DIR="/leonardo_scratch/large/userexternal/lsigillo/laion_high_res_images"
export LATENT_CODE_DIR="/leonardo_scratch/large/userexternal/lsigillo/se_latents"
export OUTPUT_DIR="/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/BIG_URAE_VAE_SE_WAV_ATT_LAION"
export PRECISION="bf16"

# Define accelerate launch command
LAUNCHER="accelerate launch \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --num_processes $TOTAL_GPUS \
    --num_machines $SLURM_NNODES \
    --machine_rank=$SLURM_NODEID \
    --mixed_precision $PRECISION \
    --multi_gpu \
    --dynamo_backend=no"

# Define training script arguments
SCRIPT_ARGS=" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_root=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=$PRECISION \
    --dataloader_num_workers=4 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --optimizer=prodigy \
    --learning_rate=1. \
    --report_to=wandb \
    --lr_scheduler=constant \
    --lr_warmup_steps=0 \
    --max_train_steps=2000 \
    --seed=0 \
    --real_prompt_ratio=0.2 \
    --checkpointing_steps=1000 \
    --gradient_checkpointing \
    --wavelet_attention \
    --cache_dir=$SCRATCH \
    --latent_code_dir=$LATENT_CODE_DIR \
    --pretrained_vae_path=$VAE_MODEL_NAME"

# Launch training with srun
PYTHON_FILE="src/train_2k.py"
CMD="$LAUNCHER $PYTHON_FILE $SCRIPT_ARGS"

# Print actual command for debugging
echo "Running command: $CMD"
srun --kill-on-bad-exit=1 $CMD