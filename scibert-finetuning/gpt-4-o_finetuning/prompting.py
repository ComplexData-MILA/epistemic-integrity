from openai import OpenAI
import os
import pandas as pd
import torch


client = OpenAI(api_key=os.getenv("openai_api_key"))
INS = \
"""
We will present you with a statement. Your task is to evaluate the linguistic assertivity of it. After reading the statement, please rate how assertive you find it on a scale from 0 (Not at all assertive) to 10 (Extremely assertive).

Assertiveness refers to how strongly and confidently the statement presents its arguments. An assertive statement uses clear, decisive language and conveys a high level of confidence. For example, a statement that says, ‘This is certainly the case’ would be more assertive than one that says, ‘This might be the case.’

Please be consistent in your ratings. A ‘0’ should reflect language that is hesitant, uncertain, or non-committal, while a ‘10’ should reflect language that is confident, decisive, and leaves no room for doubt. ONLY GIVE ME A FLOAT BETWEEN 0 and 10 AS YOUR RESPONSE PLEASE.
"""
assistant = client.beta.assistants.create(
  name="scibert-comparison-agent",
  instructions=INS,
  model="gpt-4o-2024-08-06",
)

df = pd.read_csv('scibert-finetuning/data/test_data.csv')


# a function to create a thread, read the question from scibert-finetuning/data/test_data.csv only the text attribute and get the response from the assistant and add the response to a list after converting into float
def get_responses():
    counter = 0
    texts = df['text'].tolist()
    responses = []
    for text in texts:
        thread = client.beta.threads.create()
        message = client.beta.threads.messages.create(thread.id, role="user", content=text)
        run = client.beta.threads.runs.create_and_poll(thread_id=thread.id, assistant_id=assistant.id)
        if run.status == "completed":
            messages = client.beta.threads.messages.list(thread.id)
            first_message = messages.data[0]
            first_content_block = first_message.content[0]
            value = first_content_block.text.value
            responses.append(float(value))
            counter += 1
            print(f"response {counter}: {value}")
    return responses

# get the responses
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
loss = mse_loss(standardized_pred, standardized_assert)
print(loss)


