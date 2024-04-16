from openai import OpenAI
import pandas as pd
import json
import os

def csv_to_jsonl(csv_file, jsonl_file):
    df = pd.read_csv(csv_file)
    
    system_message = {
        "role": "system",
        "content": (
            "We will present you with a statement. Your task is to evaluate the linguistic assertivity of it. "
            "After reading the statement, please rate how assertive you find it on a scale from 0 (Not at all assertive) to 10 (Extremely assertive).\n\n"
            "Assertiveness refers to how strongly and confidently the statement presents its arguments. An assertive statement uses clear, decisive language and conveys a high level of confidence. "
            "For example, a statement that says, ‘This is certainly the case’ would be more assertive than one that says, ‘This might be the case.’\n\n"
            "Please be consistent in your ratings. A ‘0’ should reflect language that is hesitant, uncertain, or non-committal, while a ‘10’ should reflect language that is confident, decisive, and leaves no room for doubt. "
            "ONLY GIVE ME A FLOAT BETWEEN 0 and 10 AS YOUR RESPONSE PLEASE:"
        )
    }

    with open(jsonl_file, 'w') as f:
        for _, row in df.iterrows():
            user_message = {"role": "user", "content": row['text']}
            assistant_message = {"role": "assistant", "content": str(round(row['assertiveness'], 1))}
            messages = [system_message, user_message, assistant_message]
            json_obj = {"messages": messages}
            json_str = json.dumps(json_obj)
            f.write(json_str + '\n')

# Convert train_data.csv to train_data.jsonl
csv_to_jsonl('scibert-finetuning/data/train_data.csv', 'scibert-finetuning/data/train_data.jsonl')

# # Convert test_data.csv to test_data.jsonl
csv_to_jsonl('scibert-finetuning/data/test_data.csv', 'scibert-finetuning/data/test_data.jsonl')

csv_to_jsonl('scibert-finetuning/data/Anthropic/test_data.csv', 'scibert-finetuning/data/Anthropic/test_data.jsonl')
csv_to_jsonl('scibert-finetuning/data/CMV/test_data.csv', 'scibert-finetuning/data/CMV/test_data.jsonl')
csv_to_jsonl('scibert-finetuning/data/GM/test_data.csv', 'scibert-finetuning/data/GM/test_data.jsonl')
csv_to_jsonl('scibert-finetuning/data/llama3-8b/test_data.csv', 'scibert-finetuning/data/llama3-8b/test_data.jsonl')
csv_to_jsonl('scibert-finetuning/data/Pei/test_data.csv', 'scibert-finetuning/data/Pei/test_data.jsonl')

csv_to_jsonl('scibert-finetuning/data/Anthropic/train_data.csv', 'scibert-finetuning/data/Anthropic/train_data.jsonl')
csv_to_jsonl('scibert-finetuning/data/CMV/train_data.csv', 'scibert-finetuning/data/CMV/train_data.jsonl')
csv_to_jsonl('scibert-finetuning/data/GM/train_data.csv', 'scibert-finetuning/data/GM/train_data.jsonl')
csv_to_jsonl('scibert-finetuning/data/llama3-8b/train_data.csv', 'scibert-finetuning/data/llama3-8b/train_data.jsonl')
csv_to_jsonl('scibert-finetuning/data/Pei/train_data.csv', 'scibert-finetuning/data/Pei/train_data.jsonl')