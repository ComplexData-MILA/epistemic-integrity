import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

SCRATCH="$SCRATCH"

class AssertivenessDataset(Dataset):
    def __init__(self, texts, scores, tokenizer_name='pedropei/sentence-level-certainty', max_length=512):
        """
        Args:
            texts (list of str): List of input text data.
            scores (list of float): Corresponding assertiveness scores.
            tokenizer_name (str): The name of the tokenizer to use.
            max_length (int): Maximum length for tokenization.
        """
        self.texts = texts
        self.scores = scores
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=f'{SCRATCH}/assertiveness_model_cache', output_hidden_states=False, num_labels=1)
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        For a given index, returns the tokenized text and the corresponding score.
        """
        text = self.texts[idx]
        score = self.scores[idx]

        # Tokenize the input text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()  # type: ignore # Squeeze to remove extra dimensions
        attention_mask = encoding['attention_mask'].squeeze() # type: ignore

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'score': torch.tensor(score, dtype=torch.float)
        }

# Example usage:
if __name__ == "__main__":
    texts = ["Text 1", "Text 2", "Text 3"]
    scores = [3.5, 5.0, 2.1]
    dataset = AssertivenessDataset(texts, scores)
    dataset.__getitem__(0)
