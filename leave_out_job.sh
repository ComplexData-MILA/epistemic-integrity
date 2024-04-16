#!/bin/bash
#SBATCH --job-name=assertivity_finetuning          # Job name
#SBATCH --partition=short-unkillable
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --gres=gpu:v100:4
#SBATCH --time=0-03:00:00
#SBATCH --output=finetuning_output_%j.log    # Standard output and error log

# Load necessary modules (e.g., for CUDA, Python)
module load python/3.10
source .env

export CUDA_VISIBLE_DEVICES=0

# List of directories
directories=("Anthropic" "CMV" "GM" "llama3-8b" "Pei")

# Base path
base_path="scibert-finetuning/data"

# Training parameters
epochs=30
batch_size=16
learning_rate=0.0025775875019941174
weight_decay=0.0994490050283763
dropout_rate=0.15954075296691828
warmup_steps=5

output_base_path="$SCRATCH/scibert-finetuned"
# Create output directories if they don't exist
for dir in "${directories[@]}"; do
  output_dir="${output_base_path}/${dir}"
  mkdir -p "$output_dir"
done

# Loop over each directory and train the model
for dir in "${directories[@]}"; do
  train_file="${base_path}/${dir}/train_data.csv"
  test_file="${base_path}/${dir}/test_data.csv"
  output_dir="${output_base_path}/${dir}"
  
  echo "Training model for ${dir}..."
  
  python scibert-finetuning/leave-out-training.py \
    --train_file "$train_file" \
    --test_file "$test_file" \
    --epochs "$epochs" \
    --batch_size "$batch_size" \
    --learning_rate "$learning_rate" \
    --weight_decay "$weight_decay" \
    --dropout_rate "$dropout_rate" \
    --warmup_steps "$warmup_steps" \
    --output_dir "$output_dir"
  
  echo "Finished training model for ${dir}."
done

echo "All models trained."