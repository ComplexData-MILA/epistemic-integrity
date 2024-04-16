import pandas as pd
from certainty_estimator import CertaintyEstimator
import torch
import os
import json

# List of directories
directories = [
    'Anthropic', 'CMV', 'GM', 'llama3-8b', 'Pei'
]

# Base path
base_path = 'scibert-finetuning/data/'

# Initialize CertaintyEstimator
ce = CertaintyEstimator(task="sentence-level", cuda=False)

# Function to compute MSE loss
def compute_mse_loss(test_csv):
    df = pd.read_csv(test_csv)
    texts = df['text'].tolist()
    assertiveness = df['assertiveness'].tolist()

    # Predict certainty
    present_aspect_certainty = ce.predict(texts, get_processed_output=True)

    # Convert to tensors
    present_aspect_certainty = torch.tensor(present_aspect_certainty, dtype=torch.float32)
    assertiveness = torch.tensor(assertiveness, dtype=torch.float32)

    # Standardize the present_aspect_certainty values
    mean_pred = torch.mean(present_aspect_certainty)
    std_pred = torch.std(present_aspect_certainty)
    standardized_pred = (present_aspect_certainty - mean_pred) / std_pred

    # Standardize the assertiveness values
    mean_assert = torch.mean(assertiveness)
    std_assert = torch.std(assertiveness)
    standardized_assert = (assertiveness - mean_assert) / std_assert

    # Compute MSE loss
    mse_loss = torch.nn.MSELoss()
    loss = mse_loss(standardized_pred, standardized_assert)
    return loss.item()

# Compute MSE loss for each directory and save results
results = []
for directory in directories:
    test_csv = os.path.join(base_path, directory, 'test_data.csv')
    mse = compute_mse_loss(test_csv)
    results.append({
        'directory': directory,
        'mse_loss': mse
    })

# Save results to an output file
output_file = 'leave-out-scibert-mse_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f'MSE results saved to {output_file}.')