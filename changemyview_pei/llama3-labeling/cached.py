"""A script to only process the cached data and label it based on the cache.
"""
import logging
import os
import pandas as pd
import json
import hashlib 

CACHE_FILE = "changemyview_pei/data/request_cache.json"
CSV_FILE = "changemyview_pei/data/cleaned_changemyview_costefficient.csv"
OUTPUT_FILE = "changemyview_pei/data/labelled_changemyview.csv"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as file:
            return json.load(file)
    return {}

def hash_payload(payload: dict) -> str:
    payload_str = json.dumps(payload, sort_keys=True)
    return hashlib.md5(payload_str.encode()).hexdigest()

# Save data to CSV
def save_data(data: pd.DataFrame, path: str) -> None:
    data.to_csv(path, index=False)

# Load the data
df = pd.read_csv(CSV_FILE)

# Load existing cache
cache = load_cache()

# Process the cache data
relevance = []
for i, row in df.iterrows():
    # Recreate the payload
    payload = {
        "model": "bedrock/meta.llama3-70b-instruct-v1:0",  # Use the model you used when generating the cache
        "messages": [{"role": "user", "content": row["full_text"]}],  # Use the data point as the content
        "max_tokens": 20,  # Use the max_tokens value you used when generating the cache
        "temperature": 0.1,  # Use the temperature value you used when generating the cache
    }
    key = hash_payload(payload)  # Generate a hash for the payload
    if key in cache:
        value = cache[key]
        if "error" in value:
            relevance.append(2)  # Add a 2 to the relevance column for errors
            logging.info(f'Row {i}: Error in cache')
        elif "irrelevant" in value["choices"][0]["message"]["content"].lower():
            relevance.append(0)
            logging.info(f'Row {i}: Irrelevant')
        else:
            relevance.append(1)
            logging.info(f'Row {i}: Relevant')
    else:
        relevance.append(None)  # Ignore data points that are not in the cache
        logging.info(f'Row {i}: Not in cache')

# Add the relevance to the DataFrame and save it
df["relevance"] = relevance
save_data(df, OUTPUT_FILE)

logging.info("Done labeling")