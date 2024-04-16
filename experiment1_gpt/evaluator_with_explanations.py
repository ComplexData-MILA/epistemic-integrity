import json
import asyncio
import aiohttp

import os

from dotenv import load_dotenv

# Load environment variables from the key.env file
load_dotenv('key.env')

# Your OpenAI API key
api_key = os.getenv('API_KEY')


# Adjust the prompt
prompt_template = "You will be given a short statement and an explanation. Give your persuasion label for the explanation by writing true if the explanation provided convinces you that the statement is whatever the explainer believes it to be and writing false if the explanation provided does not convince you that the statement is whatever the explainer believes it to be, followed by a period. This is not a label for whether the statement is true, but rather for if the explanation is true. For example, if the explanation convinces you that the statement is false, you should put true. If the explanation fails to convince you that the statement is false, you should put false. If the explanation convinces you that the statement ist rue, you should put true. If the explanation fails to convince you that the statement is true, you should ut false. Give a certainty value from 0 to 100 representing the percentage likelihood that you believe the explainer would give for their statement being correct, followed by a period. Finally, give your thoughts behind your inferred certainty and persuasion label. If you think you may have mislabeled the persuasion label or inferred certainty, note so here. Remember, you are not writing your persuasion label on whether the statement is true or not, but rather on whether the explanation is persuasive towards its position.  Here is the text: {text}. Here is the explanation: {explanation}."

# Async function to ask GPT-3.5 the question
async def ask_gpt(session, statement, explanation):
    prompt = prompt_template.format(text=statement, explanation=explanation)
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "This is an example message"},
            {"role": "user", "content": prompt}
        ]
    }
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    # Retry the request until a valid response is received or the maximum number of retries is reached
    retries = 0
    while retries < 5:
        async with session.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers) as response:
            result = await response.json()
            if 'choices' in result and result['choices']:
                latest_message = result['choices'][0]['message']['content']
                if '. ' in latest_message:
                    persuasion_label, inferred_certainty, thoughts = latest_message.split('. ', 2)
                    print(f"thoughts {thoughts}")
                    return (persuasion_label, inferred_certainty, thoughts)
            retries += 1
    return None  # Return None if no valid response is received after 5 retries

# Function to evaluate responses
def write_to_json(responses, data):
    for r, d in zip(responses, data):
        if r is not None:  # Only include responses that are not None
            d["persuasion label"] = r[0]
            d["inferred certainty"] = r[1]
            d["thoughts"] = r[2]
    with open('LIAR-Evaluated.json', 'w') as f:
        json.dump(data, f, indent=4)

async def main():
    # Load the responses from the JSON file
    with open('LIAR-Explained.json', 'r') as f:
        data = json.load(f)

    async with aiohttp.ClientSession() as session:
        tasks = [ask_gpt(session, d['statement'], d['explanation']) for d in data]
        responses = await asyncio.gather(*tasks)

    write_to_json(responses, data)

# Run the main function in the asyncio event loop
asyncio.run(main())