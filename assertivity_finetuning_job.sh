#!/bin/bash
#SBATCH --job-name=wandb_sweep_assertivity          # Job name
#SBATCH --ntasks=4                    # Number of sweep agents (tasks)
#SBATCH --partition=short-unkillable
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=0-03:00:00
#SBATCH --output=sweep_output_%j.log    # Standard output and error log

# Load necessary modules (e.g., for CUDA, Python)
module load python/3.10
source .env

export CUDA_VISIBLE_DEVICES=0

# Run 10 WandB agents in parallel, each in the background
for i in {1..4}
do
    srun wandb agent shahrad_m/assertivity-scibert-finetuning/hijnubt3 &  # Run each agent in the background
done

# Wait for all background processes to finish
wait

echo "All WandB sweep agents have finished."
