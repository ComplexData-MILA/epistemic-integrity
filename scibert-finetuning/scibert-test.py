import pandas as pd
from certainty_estimator import CertaintyEstimator
import torch

# load scibert-finetuning/data/test_data.csv
df = pd.read_csv('scibert-finetuning/data/test_data.csv')
texts = df['text'].tolist()
assertiveness = df['assertiveness'].tolist()

# Initialize CertaintyEstimator
ce = CertaintyEstimator(task="sentence-level", cuda=True)

# Predict certainty
present_aspect_certainty = ce.predict(texts, get_processed_output=True)

# Assuming present_aspect_certainty and assertiveness are lists or numpy arrays
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

# Optionally, scale standardized values to the range [0, 1]
min_pred = torch.min(standardized_pred)
max_pred = torch.max(standardized_pred)
scaled_pred = (standardized_pred - min_pred) / (max_pred - min_pred)

min_assert = torch.min(standardized_assert)
max_assert = torch.max(standardized_assert)
scaled_assert = (standardized_assert - min_assert) / (max_assert - min_assert)

# Compute MSE loss between standardized predictions and assertiveness
mse_loss = torch.nn.MSELoss()
loss = mse_loss(scaled_pred, scaled_assert)
print(loss)


# 0.1077