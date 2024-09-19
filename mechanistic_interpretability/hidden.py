"""Script to get the penultimate representation of an LLM on
the second last token of the input sequence which is the concatenated
input text to the generated text.
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import re
from tqdm import tqdm

def load_model(model_id):
    """Load the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer

def single_greedy_generation(model, tokenizer, user_text: str, system_text=None):
    """Generate a single sequence."""
    if not system_text:
        messages = [
        {"role": "user", "content": user_text}]
    else:
        messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text}]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        return_tensors="pt"
    ).to(model.device) 

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        eos_token_id=terminators,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.01,    # low for deterministic output
        top_p=0.85,        # high for deterministic output
    )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return outputs, output_text

def batch_greedy_generation(model, tokenizer, user_texts: list[str], system_text=None, batch_size=64):
    """Generate a batch of sequences with efficient GPU utilization."""
    # Prepare batches
    batches = [user_texts[i:i + batch_size] for i in range(0, len(user_texts), batch_size)]
    all_output_texts = []
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    tokenizer.padding_side = "left"

    for batch in tqdm(batches, desc="Processing batches"):
        # Prepare messages
        messages = [[{"role": "user", "content": text}] for text in batch]
        if system_text:
            messages = [{"role": "system", "content": system_text}] + messages

        # Apply chat template without tokenization
        texts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # Tokenize with padding
        tokenizer.pad_token_id = tokenizer.eos_token_id  # Set a padding token
        inputs = tokenizer(texts, padding="longest", return_tensors="pt")
        inputs = {key: val.to(model.device) for key, val in inputs.items()}

        # Generate outputs
        outputs = model.generate(**inputs,
                                eos_token_id=terminators,
                                pad_token_id=tokenizer.pad_token_id,
                                do_sample=True,
                                temperature=0.01,    # low for deterministic output
                                top_p=0.85,        # high for deterministic output
                                )

        # Decode and collect outputs
        output_texts = [tokenizer.decode(output.cpu().numpy(), skip_special_tokens=True) for output in outputs]
        all_output_texts.extend(output_texts)

    return all_output_texts

def single_nondet_generation(model, tokenizer, user_text: str, system_text=None):
    """Generate a single sequence."""
    if not system_text:
        messages = [
        {"role": "user", "content": user_text}]
    else:
        messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text}]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device) 

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        eos_token_id=terminators,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.8,    # low for deterministic output
        top_p=0.75,        # high for deterministic output
    )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return outputs, output_text

def get_hidden_representation(model, tokenizer, user_text, model_layers_length=None):
    """
    Extracts the hidden representation of the second-to-last token from a given text using a specified model.

    This function hooks into a specific layer of the model to capture the output after processing the input text.
    The captured output is then used to extract the hidden representation of the second-to-last token in the input sequence.

    Parameters:
    - model (torch.nn.Module): The model to use for extracting hidden representations.
    - tokenizer (PreTrainedTokenizer): The tokenizer for preprocessing the input text.
    - user_text (str): The input text to process.
    - model_layers_length (int, optional): The total number of layers in the model. Defaults to 32.

    Returns:
    - numpy.ndarray: The hidden representation of the second-to-last token in the input text as a NumPy array.
    """
    if not model_layers_length:
        model_layers_length = get_model_layers_length(model)
    captured_outputs = {}

    for name, module in model.named_modules():
        if name == f"model.layers.{model_layers_length - 2}.post_attention_layernorm":
            target_module= module
            break

    # Define the hook function
    def capture_output(module, input, output):
        # Store the output in the external container
        captured_outputs['penultimate_layer_output'] = output

    hook = target_module.register_forward_hook(capture_output)
    inputs = tokenizer(user_text, return_tensors="pt")
    model(**inputs)
    hook.remove()

    # Access the captured output
    penultimate_layer_output = captured_outputs['penultimate_layer_output']
    second_last_token_representation = penultimate_layer_output[0, -2, :]

    # Convert the tensor to a supported dtype before calling .numpy()
    return second_last_token_representation.detach().cpu().to(dtype=torch.float32).numpy()


def get_batch_hidden_representations(model, tokenizer, texts, batch_size=64, model_layers_length=None):
    """
    Extracts the hidden representations of the second-to-last token for a batch of texts using a specified model.

    This function processes texts in batches, applying tokenization with padding and attention mask. It hooks into a
    specific layer of the model to capture the output after processing each batch. The captured outputs are used to
    extract the hidden representations of the second-to-last token in each input sequence.

    Parameters:
    - model (torch.nn.Module): The model to use for extracting hidden representations.
    - tokenizer (PreTrainedTokenizer): The tokenizer for preprocessing the input texts.
    - texts (list of str): The input texts to process.
    - batch_size (int, optional): The size of each batch for processing. Defaults to 32.
    - model_layers_length (int, optional): The total number of layers in the model. Defaults to 32.

    Returns:
    - numpy.ndarray: An array of hidden representations for the second-to-last token in each input text.
    """
    if not model_layers_length:
        model_layers_length = get_model_layers_length(model)
    # Prepare storage for all batch results
    all_representations = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        # Tokenize the batch of texts with padding and attention mask
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True)

        captured_outputs = {}
        for name, module in model.named_modules():
            if name == f"model.layers.{model_layers_length - 2}.post_attention_layernorm":
                target_module = module
                break

        # Define the hook function
        def capture_output(module, input, output):
            captured_outputs['penultimate_layer_output'] = output

        hook = target_module.register_forward_hook(capture_output)
        # Include attention mask in the model's forward pass
        model(**inputs)
        hook.remove()

        # Access the captured output
        penultimate_layer_output = captured_outputs['penultimate_layer_output']
        # Extract second-to-last token representations for all sequences in the batch
        batch_representations = penultimate_layer_output[:, -2, :].detach().cpu().to(dtype=torch.float32).numpy()
        all_representations.append(batch_representations)

    # Concatenate all batch results into a single NumPy array
    return np.concatenate(all_representations, axis=0)

def get_model_layers_length(model):
    """Get the number of layers in the model."""
    for name, module in model.named_modules():
        match = re.match(r'model\.layers\.(\d+)', name)
        if match:
            layer_number = int(match.group(1))  # Extract the layer number
    
    return layer_number + 1

def main():
    """Main function."""
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model, tokenizer = load_model(model_id)

    # system_text = "You are a data scientist"
    user_text = "would you prefer pandas over sql and why?"
    text_2 = "I am a data scientist"
    # outputs, output_text = single_greedy_generation(model, tokenizer, user_text, system_text)
    # print(output_text)    

    # get the hidden representations
    hidden = get_batch_hidden_representations(model, tokenizer, [user_text, text_2])
    print(hidden)
    print(hidden.shape)
    

if __name__ == "__main__":
    main()