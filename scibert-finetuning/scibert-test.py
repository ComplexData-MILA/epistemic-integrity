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

# Standardize the predictions
mean_pred = torch.mean(present_aspect_certainty)
std_pred = torch.std(present_aspect_certainty)
standardized_pred = (present_aspect_certainty - mean_pred) / std_pred

# Standardize the assertiveness values
mean_assert = torch.mean(assertiveness)
std_assert = torch.std(assertiveness)
standardized_assert = (assertiveness - mean_assert) / std_assert

# Compute MSE loss between standardized predictions and assertiveness
mse_loss = torch.nn.MSELoss()
loss = mse_loss(standardized_pred, standardized_assert)
print(loss)
