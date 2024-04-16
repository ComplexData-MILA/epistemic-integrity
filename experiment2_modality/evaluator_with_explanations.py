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
prompt_template = "You will be given a short statement and an explanation. Give your persuasion label for the explanation by writing true if you are convinced by the explanation and writing false if you are not convinced by the explanation, followed by a period and a space. Give a certainty value from 0 to 100, representing the percentage likelihood that you believe the explainer would give for their statement being correct (assessed by the presence of modal verbs: The most common modal verbs are: can, could, may, might, must, shall, should, will, and would), followed immediately by a period right after the number. Finally, give your thoughts behind your inferred certainty and persuasion label. Here is the text: {text}. Here is the explanation: {explanation}."

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
                    print(latest_message)
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