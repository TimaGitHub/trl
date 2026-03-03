import torch
import torch.nn as nn
import transformers

from dataclasses import dataclass, field

from transformers import AutoTokenizer

from typing import Optional, Callable, List, Union, Tuple, Dict, Any

from ... import AutoModelForCausalLMWithValueHead

from utils import filter_logits


@dataclass
class PPOConfig:
    model_name: str = field(
        default="gpt2",
        metadata={"help": "Название модели для загрузки"}
    )
    learning_rate: float = 1.41e-5
    batch_size: int = 16


class PPOTrainer:

    @staticmethod
    def load_data(dataset: torch.utils.data.dataset.Dataset,
                 batch_size: int = 16,
                 shuffle: bool = False,
                 num_workers: int = 0,
                 collate_fn: Optional[Callable] = None) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           num_workers=num_workers,
                                           collate_fn=collate_fn)

    def __init__(self,
                 config: Optional[PPOConfig] = None,
                 model = None,
                 ref_model = None,
                 tokenizer: Optional[transformers.tokenization_utils_base.PreTrainedTokenizerBase] = None,
                 dataset: Optional[torch.utils.data.dataset.Dataset] = None,
                 optimizer: Optional[torch.optim.optimizer.Optimizer] = None,
                 data_collator: Optional[Callable] = None,
    ):# Дополнить аннотацию
        """
        The PPOTrainer uses Proximal Policy Optimization to optimise language models.
        Note, this trainer is heavily inspired by the hugging-face framework Transformers Reinforcement Learning:
        https://github.com/huggingface/trl

        Attributes:
        config -- Configuration object for PPOTrainer.
        model -- Model to be optimized, Hugging Face transformer model with a value head.
        ref_model -- Reference model to be used for KL penalty.
        tokenizer -- Tokenizer to be used for encoding the data.
        dataset -- PyTorch dataset.
        optimizer -- Optimizer to be used for training.
        data_collator -- Data collator to be used for training and passed along the dataloader.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or PPOConfig()
        self.model = model or AutoModelForCausalLMWithValueHead.from_pretrained(self.config.model_name)
        self.model = self.model.to(self.device)
        self.ref_model = ref_model or AutoModelForCausalLMWithValueHead.from_pretrained(self.config.model_name)
        self.ref_model = self.ref_model.to(self.device)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(self.config.model_name, max_length=1024)
        self.dataset = dataset # !!!Продумать если не дадут dataset!!!
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.data_collator = data_collator # !!!Продумать если не дадут dataset!!!

        if self.dataset:
            self.dataloader = self.load_data(self.dataset, self.config.batch_size,
                                             shuffle=False, num_workers=0, collate_fn=self.data_collator)
        else:
            self.dataloader = None

        self.states = dict()

    def generate(self,
                 query_tensor: Union[torch.Tensor, List[torch.Tensor]],
                 length_sampler: Optional[int] = 10,
                 return_prompt: bool = False,
                 generate_ref_response: bool = False,
                 **generation_kwargs
                 ) -> torch.Tensor | Tuple[torch.Tensor] | Tuple[torch.Tensor, ...] | List[torch.Tensor]:
        """
        Generate response with the model given the query tensor.
        call the `generate` method of the model.
        Args:
        query_tensor (`torch.LongTensor`):
            A tensor of shape (`seq_len`) containing query tokens or a list of tensors of shape (`seq_len`).
        length_sampler (`Callable`, *optional*):
            Callable that returns the number of newly generated tokens.
        return_prompt (`bool`, *optional*):
            If set to `False` the prompt is not returned but only the newly generated tokens, defaults to `True`.
        generate_ref_response (`bool`, *optional*):
            If set to `True` the reference response is also generated, defaults to `False`.
        generation_kwargs (dict[str, Any]):
            Keyword arguments for generation.

        Returns:
            `torch.LongTensor`: A tensor of shape (`batch_size`, `gen_len`) containing response tokens.
        """

        generated = query_tensor
        values_list = []

        top_k = generation_kwargs.get('top_k', 0)
        top_p = generation_kwargs.get('top_p', 1.0)
        temperature = generation_kwargs.get('temperature', 1.0)
        pad_token = self.tokenizer.pad_token_id

        for token in range(length_sampler):
            with torch.no_grad():
                outputs, values = self.model(generated)
                next_token_logits = outputs.logits[:, -1, :]

            values_list.append(values)
            next_token_logits = next_token_logits / temperature
            filtered_logits = filter_logits(next_token_logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            new_next_token = pad_token * (generated[:, -1:] == pad_token) +  next_token * (generated[:, -1:] != pad_token)
            generated = torch.cat((generated, new_next_token), dim=1)

        if generate_ref_response:
            ref_generated = query_tensor
            for token in range(length_sampler):
                with torch.no_grad():
                    ref_outputs, _ = self.ref_model(ref_generated)
                    next_ref_token_logits = ref_outputs.logits[:, -1, :]

                next_ref_token_logits = next_ref_token_logits / temperature
                filtered_ref_logits = filter_logits(next_ref_token_logits, top_k=top_k, top_p=top_p)
                ref_probs = torch.softmax(filtered_ref_logits , dim=-1)
                next_ref_token = torch.multinomial(ref_probs, num_samples=1)
                new_next_ref_token = pad_token * (ref_generated[:, -1:] == pad_token) + next_ref_token * (ref_generated[:, -1:] != pad_token)
                ref_generated = torch.cat((ref_generated, new_next_ref_token), dim=1)


            if return_prompt:
                return generated.long(), values_list, ref_generated.long()
            else:
                return generated[:, query_tensor.shape[1]:].long(), values_list, ref_generated[:, query_tensor.shape[1]:].long()
        else:
            if return_prompt:
                return generated.long(), values_list
            else:
                return generated[:, query_tensor.shape[1]:].long(), values_list

    def step(self,
             query_tensor: torch.Tensor | List[torch.Tensor],
             responses: torch.Tensor | List[torch.Tensor],
             scores: torch.Tensor | List[torch.Tensor],
             response_masks: Optional[List[torch.LongTensor]] = None) -> Dict[str, Any]:
        if type(scores) == type(list):
            scores = torch.stack(scores)[:, None]


        if type(query_tensor) == type(list):
            for index, (query, response, score) in reversed(list(enumerate(zip(query_tensor, responses, scores)))):


