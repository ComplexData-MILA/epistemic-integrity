import pandas as pd
import asyncio
import aiohttp
import time
import hashlib
import json
import os
import argparse
from tqdm import tqdm

# Constants
MAX_REQUESTS_PER_MINUTE = 400
MAX_TOKENS_PER_MINUTE = 300000

# Assume each data point is less than 1250 tokens
AVERAGE_TOKENS_PER_REQUEST = 1250
MAX_CONCURRENT_REQUESTS = MAX_REQUESTS_PER_MINUTE

# Rate limit calculations
REQUESTS_INTERVAL = 60 / MAX_REQUESTS_PER_MINUTE  # Time between requests to stay within the limit
TOKENS_INTERVAL = 60 / (MAX_TOKENS_PER_MINUTE / AVERAGE_TOKENS_PER_REQUEST)

CACHE_FILE = "changemyview_pei/data/request_cache.json"

# Load the data
def get_data() -> pd.DataFrame:
    data = pd.read_csv("changemyview_pei/data/cleaned_changemyview_costefficient.csv")
    return data

def save_data(data: pd.DataFrame, path: str) -> None:
    data.to_csv(path, index=False)

# Load cache from file
def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as file:
            return json.load(file)
    return {}

# Save cache to file
def save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w") as file:
        json.dump(cache, file)

df = get_data()
data_points = df["full_text"].to_list()

# Load existing cache
cache = load_cache()

async def send_request(session, url, payload, retries=5):
    payload_str = json.dumps(payload, sort_keys=True)
    payload_hash = hashlib.md5(payload_str.encode()).hexdigest()

    if payload_hash in cache:
        # print(f"Reading from cache for payload: {payload_hash}")
        return cache[payload_hash]

    for attempt in range(retries):
        try:
            async with session.post(url, json=payload) as response:
                if response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", 1))
                    print(f"Rate limited. Retrying in {retry_after} seconds...")
                    await asyncio.sleep(retry_after)
                else:
                    response.raise_for_status()
                    response_data = await response.json()
                    cache[payload_hash] = response_data
                    # print(f"Saving to cache for payload: {payload_hash}")
                    save_cache(cache)  # Update the cache immediately after receiving the response
                    return response_data
        except aiohttp.ClientError as e:
            print(f"HTTP Client Error on attempt {attempt + 1}: {e}")
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            print(f"An error occurred on attempt {attempt + 1}: {e}")
            await asyncio.sleep(2 ** attempt)

    print(f"All retry attempts failed for payload: {payload_hash}")
    return {"error": "All retry attempts failed"}

async def process_data_point(session, url, model, message, max_tokens=20, temperature=0.1):
    payload = {
        "model": model,
        "messages": message,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    response = await send_request(session, url, payload)
    return response

async def main():
    # use arg parse to get model
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, default="8b")
    args = argparser.parse_args()
    if args.model == "8b":
        model = "bedrock/meta.llama3-8b-instruct-v1:0"
    elif args.model == "70b":
        model = "bedrock/meta.llama3-70b-instruct-v1:0"

    url = "http://0.0.0.0:4000/chat/completions"  
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = []
    results = []

    async with aiohttp.ClientSession() as session:
        start_time = time.time()

        with tqdm(total=len(data_points), desc="Processing data points") as pbar:
            for i, data_point in enumerate(data_points):
                # Add a task for each data point
                message = [{"role": "user", "content": data_point}]

                async with semaphore:
                    task = asyncio.create_task(process_data_point(session, url, model, message))
                    tasks.append(task)

                    # Respect the token rate limit
                    if (i + 1) * AVERAGE_TOKENS_PER_REQUEST > MAX_TOKENS_PER_MINUTE:
                        await asyncio.sleep(TOKENS_INTERVAL)

                    # Respect the request rate limit
                    if len(tasks) >= MAX_CONCURRENT_REQUESTS:
                        await asyncio.sleep(REQUESTS_INTERVAL)

                    # Limit the number of concurrent tasks
                    if len(tasks) >= MAX_CONCURRENT_REQUESTS:
                        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        tasks = list(pending)
                        for task in done:
                            result = task.result()
                            results.append(result)
                            pbar.update(1)

            # Gather any remaining results
            if tasks:
                results += await asyncio.gather(*tasks)
                pbar.update(len(tasks))  # Update the progress bar for the remaining tasks

        print(f"Processed {len(results)} data points in {time.time() - start_time:.2f} seconds")
        print("Saving the results to a CSV file...")
        relevance = []
        for result in results:
            if "error" in result:
                relevance.append(2)  # Add a 2 to the relevance column for errors
            elif "relevant" in result["choices"][0]["message"]["content"].lower():
                relevance.append(1)
            else:
                relevance.append(0)
        df["relevance"] = relevance
        save_data(df, "changemyview_pei/data/labelled_changemyview.csv")

if __name__ == "__main__":
    asyncio.run(main())

    # Display the updated DataFrame
    print("done labeling")
