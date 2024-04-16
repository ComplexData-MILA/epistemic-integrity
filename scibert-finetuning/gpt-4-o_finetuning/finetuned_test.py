from openai import OpenAI
import os
import pandas as pd
import torch

client = OpenAI(api_key=os.getenv("openai_api_key"))
CONTENT = \
"""
We will present you with a statement. Your task is to evaluate the linguistic assertivity of it. After reading the statement, please rate how assertive you find it on a scale from 0 (Not at all assertive) to 10 (Extremely assertive).

Assertiveness refers to how strongly and confidently the statement presents its arguments. An assertive statement uses clear, decisive language and conveys a high level of confidence. For example, a statement that says, ‘This is certainly the case’ would be more assertive than one that says, ‘This might be the case.’

Please be consistent in your ratings. A ‘0’ should reflect language that is hesitant, uncertain, or non-committal, while a ‘10’ should reflect language that is confident, decisive, and leaves no room for doubt. ONLY GIVE ME A FLOAT BETWEEN 0 and 10 AS YOUR RESPONSE PLEASE.
"""
# Example completion provided at the beginning
def get_completion(text):
    completion = client.chat.completions.create(
        model="ft:gpt-4o-2024-08-06:ps-project-research:scibert-finetuning-rounded-data:ABojH8kB",
        messages=[
            {"role": "system", "content": CONTENT},
            {"role": "user", "content": text},
        ]
    )
    return completion

df = pd.read_csv('scibert-finetuning/data/test_data.csv')

# Function to create a thread, read the question from scibert-finetuning/data/test_data.csv only the text attribute,
# get the response from the assistant, and add the response to a list after converting it into float
def get_responses():
    texts = df['text'].tolist()
    responses = []
    counter = 0
    for text in texts:
        completion = get_completion(text)
        value = completion.choices[0].message.content.strip()
        responses.append(float(value))
        counter += 1
        print(f"response {counter}: {value}")
    return responses

# Get the responses
responses = get_responses()

assertiveness = torch.tensor(df['assertiveness'].tolist(), dtype=torch.float32)
mean_assert = torch.mean(assertiveness)
std_assert = torch.std(assertiveness)
standardized_assert = (assertiveness - mean_assert) / std_assert

responses = torch.tensor(responses, dtype=torch.float32)
mean_pred = torch.mean(responses)
std_pred = torch.std(responses)
standardized_pred = (responses - mean_pred) / std_pred

mse_loss = torch.nn.MSELoss()
loss = mse_loss(standardized_assert, standardized_pred)
print(loss)

# No leave out scaled: 0.0183

