# Scibert and GPT 4o finetuning on the assertiveness dataset

- The training dataset can be found data/train_data.csv and test_data.csv.

- Multiple models are fine tuned and trained to be used for assertiveness prediction.

- The models are evaluated on the test dataset and the best model is selected based on the evaluation metrics.

- GPT 4o model and the fine tuned one are closed source as OpenAI does not allow sharing them.

Here are the results of fine tuning/trainin procedure based on the MSE loss on the test set:

Raw Scibert (Pei model): 1.83
Finetuned Scibert with an additional of a projection layer: 1.63
GPT 4o 2024-08-06 version - prompted: 1.35
GPT 4o 2024-08-06 fine tuned: 0.78
Nizhnik model: 1.70

The best model is the GPT 4o 2024-08-06 fine tuned model with a MSE loss of 0.78.