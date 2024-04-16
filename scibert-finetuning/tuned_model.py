from functools import cache
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

SCRATCH="$SCRATCH"

class ModifiedAssertivenessModel(nn.Module):
    def __init__(self, model_name='pedropei/sentence-level-certainty', add_activation=False):
        """
        Args:
            model_name (str): The name of the pretrained model to load.
            add_activation (bool): Whether to add an activation function after the linear layer.
        """
        super(ModifiedAssertivenessModel, self).__init__()
        # Load the pretrained model
        self.base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, output_attentions=False, output_hidden_states=False, cache_dir=f'{SCRATCH}/assertiveness_model_cache')
        
        # Add a linear layer for adjusting the output
        self.linear = nn.Linear(1, 1)

        # Optionally add an activation function (e.g., Sigmoid for outputs between 0 and 1)
        self.add_activation = add_activation
        if self.add_activation:
            self.activation = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        # Get the base model output (logits)
        base_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = base_output.logits
        
        # Pass through the additional linear layer
        linear_output = self.linear(logits)
        
        # Apply activation if specified
        if self.add_activation:
            output = self.activation(linear_output)
        else:
            output = linear_output
        
        return output


# Example usage:    
if __name__ == "__main__":
    model = ModifiedAssertivenessModel(add_activation=False)
