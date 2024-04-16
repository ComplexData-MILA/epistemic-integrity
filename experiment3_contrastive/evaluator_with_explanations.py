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
prompt_template = "Here are two texts with their explanations: Text 1: {text1}. Explanation 1: {explanation1}. Text 2: {text2}. Explanation 2: {explanation2}. Which explanation is more certain? (Please answer with '1' or '2')"

# Async function to ask GPT-3.5 the question
async def ask_gpt(session, statement1, explanation1, statement2, explanation2):
    prompt = prompt_template.format(text1=statement1, explanation1=explanation1, text2=statement2, explanation2=explanation2)
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
                if latest_message.strip() in ['1', '2']:
                    print(latest_message)
                    return latest_message
            retries += 1
    return None  # Return None if no valid response is received after 5 retries

# Function to evaluate responses
def write_to_json(responses, data):
    evaluated_data = []
    for r, d1, d2 in zip(responses, data[::2], data[1::2]):  # Assuming data contains pairs of statements and explanations
        if r is not None:  # Only include responses that are not None
            evaluated_data.append({
                "statement1": d1["statement"],
                "classification1": d1["classification label"],
                "explanation1": d1["explanation"],
                "true certainty1": d1["true certainty"],
                "statement2": d2["statement"],
                "classification2": d2["classification label"],
                "explanation2": d2["explanation"],
                "true certainty2": d2["true certainty"],
                "prediction": r,
                "truth": '1' if d1["true certainty"] > d2["true certainty"] else '2'
            })
    with open('LIAR-Evaluated.json', 'w') as f:
        json.dump(evaluated_data, f, indent=4)

async def main():
    # Load the responses from the JSON file
    with open('LIAR-Explained.json', 'r') as f:
        data = json.load(f)

    async with aiohttp.ClientSession() as session:
        tasks = [ask_gpt(session, d1['statement'], d1['explanation'], d2['statement'], d2['explanation']) for d1, d2 in zip(data[::2], data[1::2])]
        responses = await asyncio.gather(*tasks)

    write_to_json(responses, data)

# Run the main function in the asyncio event loop
asyncio.run(main())