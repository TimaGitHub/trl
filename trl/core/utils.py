import random
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM

class LengthSampler:
    def sample_length(self):
        return random.randint(self.min_samples, self.max_samples)

    def __init__(self, min_samples, max_samples):
        self.min_samples = min_samples
        self.max_samples = max_samples

    def __call__(self):
        return self.sample_length()

class AutoModelForCausalLMWithValueHead(nn.Module):

    def __init__(self, model_name: str,
                 **kwargs):
        super().__init__()

        self.model_name = model_name
        self.pretrained_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                                     **kwargs)

        self.hidden_size = self.pretrained_model.config.hidden_size
        self.value_head = nn.Linear(self.hidden_size, 1)

    @classmethod
    def from_pretrained(cls, model_name: str,
                        *args,
                        **kwargs):
        return AutoModelForCausalLMWithValueHead(model_name, **kwargs)

    def forward(self, input_ids, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.pretrained_model(input_ids=input_ids,
                                        output_hidden_states=True,
                                        **kwargs)

        # logits = outputs.logits

        last_hidden_state = outputs.hidden_states[-1]

        value = self.value_head(last_hidden_state)

        return outputs, value
