import asyncio
import aiohttp
import json

import os

from dotenv import load_dotenv

# Load environment variables from the key.env file
load_dotenv('key.env')

# Your OpenAI API key
api_key = os.getenv('API_KEY')

# Adjust the prompt
prompt_template = "You will be given a short statement. Give your classification label for the statement by writing true if you are convinced and write false if you are not convinced, followed by a period. Give a certainty value from 0 to 100 representing the percentage likelihood of your statement being correct, followed by a period. Then, give an explanation for your label. Here is the text: {text}."

# Async function to ask GPT-3.5 the question
async def ask_gpt(session, statement):
    prompt = prompt_template.format(text=statement)
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
                if latest_message.count('. ') >= 2:
                    classification_label, true_certainty, explanation = latest_message.split('. ', 2)
                    return (classification_label, true_certainty, explanation)
            retries += 1
    return None  # Return None if no valid response is received after 5 retries

# New function to write results to a JSON file
def write_to_json(responses, statements):
    data = []
    for s, r in zip(statements, responses):
        if r is not None:  # Only include responses that are not None
            data.append({"statement": s, "classification label": r[0], "true certainty": r[1], "explanation": r[2]})
    with open('LIAR-Explained.json', 'w') as f:
        json.dump(data, f, indent=4)

async def main():
    # Read statements from the LIAR-New.jsonl file
    with open('LIAR-New.jsonl', 'r') as f:
        statements = [json.loads(line)['statement'] for line in f]

    async with aiohttp.ClientSession() as session:
        tasks = [ask_gpt(session, statement) for statement in statements]
        responses = await asyncio.gather(*tasks)

    write_to_json(responses, statements)

# Run the main function in the asyncio event loop
asyncio.run(main())