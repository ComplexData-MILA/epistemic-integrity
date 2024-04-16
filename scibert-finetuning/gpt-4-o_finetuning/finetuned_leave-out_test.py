from openai import OpenAI
import os
import pandas as pd
import torch

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("openai_api_key"))

# System message content
CONTENT = """
We will present you with a statement. Your task is to evaluate the linguistic assertivity of it. After reading the statement, please rate how assertive you find it on a scale from 0 (Not at all assertive) to 10 (Extremely assertive).

Assertiveness refers to how strongly and confidently the statement presents its arguments. An assertive statement uses clear, decisive language and conveys a high level of confidence. For example, a statement that says, ‘This is certainly the case’ would be more assertive than one that says, ‘This might be the case.’

Please be consistent in your ratings. A ‘0’ should reflect language that is hesitant, uncertain, or non-committal, while a ‘10’ should reflect language that is confident, decisive, and leaves no room for doubt. ONLY GIVE ME A FLOAT BETWEEN 0 and 10 AS YOUR RESPONSE PLEASE.
"""

# Function to get completion from the model
def get_completion(text, model):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CONTENT},
            {"role": "user", "content": text},
        ]
    )
    return completion

# Function to get responses from the model for a given test file
def get_responses(test_file, model):
    df = pd.read_csv(test_file)
    texts = df['text'].tolist()
    responses = []
    counter = 0
    for text in texts:
        completion = get_completion(text, model)
        value = completion.choices[0].message.content.strip()
        responses.append(float(value))
        counter += 1
        print(f"response {counter}: {value}")
    return responses, df['assertiveness'].tolist()

# Function to calculate the loss
def calculate_loss(predictions, targets):
    predictions = torch.tensor(predictions, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

    mean_pred = torch.mean(predictions)
    std_pred = torch.std(predictions)
    standardized_pred = (predictions - mean_pred) / std_pred

    mean_target = torch.mean(targets)
    std_target = torch.std(targets)
    standardized_target = (targets - mean_target) / std_target

    mse_loss = torch.nn.MSELoss()
    loss = mse_loss(standardized_pred, standardized_target)
    return loss.item()

# Models and corresponding test files
models_and_files = {
    "ft:gpt-4o-2024-08-06:ps-project-research:scibert-finetuning-anthropic-out-round:ADJznRaN": "scibert-finetuning/data/Anthropic/test_data.csv",
    "ft:gpt-4o-2024-08-06:ps-project-research:scibert-finetuning-pei-out-round:ADO6OfwV": "scibert-finetuning/data/Pei/test_data.csv",
    "ft:gpt-4o-2024-08-06:ps-project-research:scibert-finetuning-llama-out-round:ADO8fwTy": "scibert-finetuning/data/llama3-8b/test_data.csv",
    "ft:gpt-4o-2024-08-06:ps-project-research:scibert-finetuning-gm-out-round:ADK1RdW7": "scibert-finetuning/data/GM/test_data.csv",
    "ft:gpt-4o-2024-08-06:ps-project-research:scibert-finetuning-cmv-out-round:ADK5AlGD": "scibert-finetuning/data/CMV/test_data.csv"
}

# Test each model with its corresponding test file
for model, test_file in models_and_files.items():
    print(f"Testing model: {model} with file: {test_file}")
    responses, assertiveness = get_responses(test_file, model)
    loss = calculate_loss(responses, assertiveness)
    print(f"Loss for model {model}: {loss}")