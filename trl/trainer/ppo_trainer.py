import torch
import torch.nn as nn
import transformers

from dataclasses import dataclass, field

from transformers import AutoTokenizer

from typing import Optional, Callable, List, Union, Tuple, Dict, Any

from trl.core import AutoModelForCausalLMWithValueHead

from .utils import filter_logits, collate_batch, collate_left_padding

import torch.nn.functional as F

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
                 model: AutoModelForCausalLMWithValueHead  = None,
                 ref_model: AutoModelForCausalLMWithValueHead = None,
                 tokenizer: Optional[transformers.tokenization_utils_base.PreTrainedTokenizerBase] = None,
                 dataset: Optional[torch.utils.data.dataset.Dataset] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
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
                 attention_mask: torch.Tensor = None,
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

        if type(query_tensor) in (type(list()), type(())):
            pad_token_id = generation_kwargs.get('pad_token_id', self.tokenizer.pad_token_id)
            query_tensor, attention_mask = collate_left_padding(query_tensor, pad_token_id=pad_token_id)
        else:
            if not attention_mask:
                attention_mask = torch.ones_like(query_tensor) ### Исправить на случай если сам пользователь заполнить паддингом перед подачей в self.generate()

        ref_attention_mask = attention_mask.clone()

        generated = query_tensor

        top_k = generation_kwargs.get('top_k', 0)
        top_p = generation_kwargs.get('top_p', 1.0)
        temperature = generation_kwargs.get('temperature', 1.0)
        pad_token = self.tokenizer.pad_token_id

        for token in range(length_sampler):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

            with torch.no_grad():
                outputs, __values = self.model(input_ids=generated,
                                               attention_mask=attention_mask,
                                               position_ids=position_ids)
                next_token_logits = outputs.logits[:, -1, :]

            next_token_logits = next_token_logits / temperature
            filtered_logits = filter_logits(next_token_logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            new_next_token = pad_token * (generated[:, -1:] == pad_token) +  next_token * (generated[:, -1:] != pad_token)
            generated = torch.cat((generated, new_next_token), dim=1)

            new_mask_token = (new_next_token != pad_token).long()
            attention_mask = torch.cat((attention_mask, new_mask_token), dim=1)

        if generate_ref_response:
            ref_generated = query_tensor
            for token in range(length_sampler):
                ref_position_ids = ref_attention_mask.long().cumsum(-1) - 1
                ref_position_ids.masked_fill_(ref_attention_mask == 0, 0)

                with torch.no_grad():
                    ref_outputs, __values = self.ref_model(input_ids=ref_generated,
                                                           attention_mask=ref_attention_mask,
                                                           position_ids=ref_position_ids)
                    next_ref_token_logits = ref_outputs.logits[:, -1, :]

                next_ref_token_logits = next_ref_token_logits / temperature
                filtered_ref_logits = filter_logits(next_ref_token_logits, top_k=top_k, top_p=top_p)
                ref_probs = torch.softmax(filtered_ref_logits , dim=-1)
                next_ref_token = torch.multinomial(ref_probs, num_samples=1)
                new_next_ref_token = pad_token * (ref_generated[:, -1:] == pad_token) + next_ref_token * (ref_generated[:, -1:] != pad_token)
                ref_generated = torch.cat((ref_generated, new_next_ref_token), dim=1)

                new_mask_ref_token = (new_next_ref_token != pad_token).long()
                ref_attention_mask = torch.cat((ref_attention_mask, new_mask_ref_token), dim=1)

            if return_prompt:
                return generated.long(), ref_generated.long()
            else:
                return generated[:, query_tensor.shape[1]:].long(), ref_generated[:, query_tensor.shape[1]:].long()
        else:
            if return_prompt:
                return generated.long()
            else:
                return generated[:, query_tensor.shape[1]:].long()

    @staticmethod
    def calculate_log_probs(model: AutoModelForCausalLMWithValueHead,
                            query: List[torch.Tensor],
                            response: torch.Tensor,
                            pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # работает с query = [torch.Tensor1, torch.Tensor2, ...]
        # response = torch.Tensor
        # настроить обработку, если query - Тензор, а не List

        query, attention_mask = collate_left_padding(query, pad_token_id=pad_token_id)
        full_response = torch.cat((query, response), dim=-1)
        full_response_attention_mask = (full_response != pad_token_id).long()
        position_ids = full_response_attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(full_response_attention_mask == 0, 0)

        outputs, values = model(input_ids=full_response,
                                attention_mask=full_response_attention_mask,
                                position_ids=position_ids)
        shift_logits = outputs.logits[:, query.shape[-1] - 1: -1, :].contiguous()
        shift_log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(shift_log_probs, dim=2, index=response[:, :, None]).squeeze(-1)

        return token_log_probs, values

    @staticmethod
    def calculate_kl_divergence(log_probs_1: torch.Tensor,
                                log_probs_2: torch.Tensor) -> torch.Tensor:
        return log_probs_1 - log_probs_2

    @staticmethod
    def calculate_ratio(log_probs_1: torch.Tensor,
                        log_probs_2: torch.Tensor) -> torch.Tensor:
        return torch.exp(log_probs_1 - log_probs_2)

    @staticmethod
    def clip_ratio(ratio: torch.Tensor,
                   eps: float =0.2) -> torch.Tensor:
        return torch.clamp(ratio, 1 - eps, 1 + eps)

    @staticmethod
    def calculate_advantages(rewards: List[torch.Tensor],
                             values: torch.Tensor,
                             kl_div: torch.Tensor,
                             query_length: int,
                             max_length: int,
                             beta: float = 0.1,
                             gamma: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        returns = []  # Накопленный доход
        last_token_reward = torch.Tensor(rewards, device=kl_div.device) - beta * kl_div[:, -1]
        returns.append(last_token_reward)

        for i in range(1, max_length):
            current_kl = kl_div[:, -1 - i]
            current_return = - beta * current_kl + gamma * returns[-1]
            returns.append(current_return)

        returns = returns[::-1]
        returns_tensor = torch.stack(returns, dim=1)
        advantages = returns_tensor - values[:, query_length - 1: -1, :].squeeze(-1)
        return returns_tensor, advantages

    @staticmethod
    def calculate_loss(ratio, clipped_ratio, advantages):
        return -torch.min(ratio * advantages, clipped_ratio * advantages).sum(dim=-1).mean(dim=0)

    def step(self,
             query_tensor: torch.Tensor | List[torch.Tensor],
             responses: torch.Tensor | List[torch.Tensor],
             scores: torch.Tensor | List[torch.Tensor],
             n_steps: int = 3,
             beta: float = 0.1,
             gamma: float = 0.95,
             eps: float = 0.2,
             value_loss_coef: float = 0.1

        ) -> Dict[str, Any]:

        # if type(scores) in (type(list()), type(())):
        #     scores = torch.stack(scores)[:, None]

        stats = dict()

        max_length = responses.shape[-1]
        query_length = collate_left_padding(query_tensor, pad_token_id=self.tokenizer.pad_token_id)[0].shape[-1]
        mse_loss = torch.nn.MSELoss()

        with torch.no_grad():
            old_log_probs, old_values = PPOTrainer.calculate_log_probs(self.model, query_tensor, responses, pad_token_id=self.tokenizer.pad_token_id)
            log_ref_probs, ref_values = PPOTrainer.calculate_log_probs(self.ref_model, query_tensor, responses, pad_token_id=self.tokenizer.pad_token_id)

        kl_div = PPOTrainer.calculate_kl_divergence(old_log_probs, log_ref_probs)
        returns, advantages = PPOTrainer.calculate_advantages(scores, old_values, kl_div, query_length=query_length, max_length=max_length, beta=beta,
                                                   gamma=gamma)

        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for i in range(n_steps):
            log_probs, values = PPOTrainer.calculate_log_probs(self.model, query_tensor, responses, pad_token_id=self.tokenizer.pad_token_id)
            ratio = PPOTrainer.calculate_ratio(log_probs, old_log_probs)
            clipped_ratio = PPOTrainer.clip_ratio(ratio, eps=eps)
            clipped_surrogate_objective_function = PPOTrainer.calculate_loss(ratio, clipped_ratio, advantages)
            value_loss = mse_loss(values[:, query_length - 1: -1, :].squeeze(-1), returns)
            total_loss = clipped_surrogate_objective_function + value_loss_coef * value_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        stats['kl_div'] = kl_div.detach().cpu().mean()
        stats['log_probs'] = log_probs.detach().cpu()
        stats['old_log_probs'] = old_log_probs.detach().cpu()
        stats['old_values'] = old_values.detach().cpu()
        stats['old_log_probs'] = old_log_probs.detach().cpu()
        stats['mean_scores'] = torch.stack(scores).mean().detach().cpu().item()
        stats['std_scores'] = torch.stack(scores).std().detach().cpu().item()
        stats['returns'] = returns.detach().cpu()
        stats['advantages'] = advantages.detach().cpu()
        stats['surrogate_objective_function'] = clipped_surrogate_objective_function.cpu().item()
        stats['value_loss'] = value_loss.detach().cpu().item()
        stats['total_loss'] = total_loss.detach().cpu().item()

        return stats

