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

# Async function to ask the model the question
async def ask_model(session, model_name, statement1, explanation1, statement2, explanation2):
    prompt = prompt_template.format(text1=statement1, explanation1=explanation1, text2=statement2, explanation2=explanation2)
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    if model_name == "text-davinci-002":
        data = {
            # "model": model_name,
            "prompt": prompt,
            "max_tokens": 60
        }
        url = "https://api.openai.com/v1/engines/text-davinci-002/completions"
    else:
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "This is an example message"},
                {"role": "user", "content": prompt}
            ]
        }
        url = "https://api.openai.com/v1/chat/completions"
    
    # Async function to ask the model the question
async def ask_model(session, model_name, statement1, explanation1, statement2, explanation2):
    prompt = prompt_template.format(text1=statement1, explanation1=explanation1, text2=statement2, explanation2=explanation2)
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    if model_name == "text-davinci-002":
        data = {
            "prompt": prompt,
            "max_tokens": 60
        }
        url = "https://api.openai.com/v1/engines/text-davinci-002/completions"
    else:
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "This is an example message"},
                {"role": "user", "content": prompt}
            ]
        }
        url = "https://api.openai.com/v1/chat/completions"
    
    async with session.post(url, json=data, headers=headers) as response:
        result = await response.json()
        if 'choices' in result and len(result['choices']) > 0:
            if 'message' in result['choices'][0]:
                return result['choices'][0]['message']['content'].strip()
            elif 'text' in result['choices'][0]:
                return result['choices'][0]['text'].strip()
        else:
            print(f"Unexpected response for {model_name}: {result}")
            return None  # or a default value

# Function to evaluate responses
def write_to_json(responses_gpt, responses_davinci, data):
    evaluated_data = []
    for r_gpt, r_davinci, d1, d2 in zip(responses_gpt, responses_davinci, data[::2], data[1::2]):  # Assuming data contains pairs of statements and explanations
        if r_gpt is not None and r_davinci is not None:  # Only include responses that are not None
            evaluated_data.append({
                "statement1": d1["statement"],
                "classification1": d1["classification label"],
                "explanation1": d1["explanation"],
                "true certainty1": d1["true certainty"],
                "statement2": d2["statement"],
                "classification2": d2["classification label"],
                "explanation2": d2["explanation"],
                "true certainty2": d2["true certainty"],
                "prediction_gpt": r_gpt,
                "prediction_davinci": r_davinci,
                "truth": '1' if d1["true certainty"] > d2["true certainty"] else '2'
            })
    with open('LIAR-Evaluated.json', 'w') as f:
        json.dump(evaluated_data, f, indent=4)

async def main():
    # Load the responses from the JSON file
    with open('LIAR-Explained.json', 'r') as f:
        data = json.load(f)

    async with aiohttp.ClientSession() as session:
        tasks_gpt = [ask_model(session, "gpt-3.5-turbo", d1['statement'], d1['explanation'], d2['statement'], d2['explanation']) for d1, d2 in zip(data[::2], data[1::2])]
        tasks_davinci = [ask_model(session, "text-davinci-002", d1['statement'], d1['explanation'], d2['statement'], d2['explanation']) for d1, d2 in zip(data[::2], data[1::2])]
        responses_gpt = await asyncio.gather(*tasks_gpt)
        responses_davinci = await asyncio.gather(*tasks_davinci)

    write_to_json(responses_gpt, responses_davinci, data)

# Run the main function in the asyncio event loop
asyncio.run(main())