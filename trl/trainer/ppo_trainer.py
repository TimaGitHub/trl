import wandb
import torch
import torch.nn.functional as F
import transformers
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Union, Tuple, Dict, Any

from transformers import AutoTokenizer
from trl.core import AutoModelForCausalLMWithValueHead

from .utils import filter_logits, collate_left_padding

@dataclass
class PPOConfig:
    model_name: str = field(
        default="gpt2",
        metadata={"help": "Название модели для загрузки"}
    )
    learning_rate: float = 1.41e-5
    batch_size: int = 16
    mini_batch_size: int = 4
    ppo_epochs: int = 4
    target_kl_coef = 0.05
    use_adaptive_kl: bool = True
    kl_coef: float = 0.1
    kl_ctl_update_rate: float = 0.1
    kl_coef_min: float = 0.001
    kl_coef_max: float = 10.0
    use_gae: bool = True
    gamma: float = 0.95
    vf_coef: float = 0.1
    entropy_coef: float = 0.01
    clip_range_value: float = 0.2
    clip_range: float = 0.2
    normalize_advantage: bool = True
    max_grad_norm: Optional[float] = 1.0
    lambda_coef: float = 0.95
    use_wandb: bool = False
    wandb_project: str = "custom-trl-project"

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
        self.dataset = dataset # !!! Продумать если не дадут dataset!!!
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.data_collator = data_collator # !!! Продумать если не дадут dataset!!!

        if self.dataset:
            self.dataloader = self.load_data(self.dataset, self.config.batch_size,
                                             shuffle=False, num_workers=0, collate_fn=self.data_collator)
        else:
            self.dataloader = None

        self.states = dict()

        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                config=vars(self.config),  # Логируем гиперпараметры
                name=f"ppo-{self.config.model_name}"
            )

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
            query_tensor - A tensor of shape (`seq_len`) containing query tokens or a list of tensors of shape (`seq_len`).
            length_sampler - Callable that returns the number of newly generated tokens.
            return_prompt - If set to `False` the prompt is not returned but only the newly generated tokens, defaults to `True`.
            generate_ref_response - If set to `True` the reference response is also generated, defaults to `False`.
            attention_mask -  Tensors containing masks of the response tokens.
            generation_kwargs - Keyword arguments for generation.

        Returns:
            A tensor of shape containing response tokens.
        """

        if type(query_tensor) == torch.Tensor:
            if query_tensor.ndim == 1:
                query_tensor = query_tensor.unsqueeze(0)

        if type(query_tensor) in (type(list()), type(())):
            pad_token_id = generation_kwargs.get('pad_token_id', self.tokenizer.pad_token_id)
            query_tensor, attention_mask = collate_left_padding(query_tensor, pad_token_id=pad_token_id, device=self.device)
        else:
            if not attention_mask:
                attention_mask = torch.ones_like(query_tensor, device=self.device)

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
                            query: torch.Tensor,
                            response: torch.Tensor,
                            pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:

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
                   eps: float = 0.2) -> torch.Tensor:
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
        last_token_reward = torch.Tensor(rewards).to(kl_div.device) - beta * kl_div[:, -1]
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
    def calculate_generalized_advantages(rewards: List[torch.Tensor],
                                         values: torch.Tensor,
                                         kl_div: torch.Tensor,
                                         query_length: int,
                                         beta: float = 0.1,
                                         gamma: float = 0.95,
                                         lambda_coef: float = 0.95
                                         ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, seq_len = kl_div.shape
        rewards_tensor = torch.stack(rewards).to(dtype=torch.float32, device=kl_div.device)[:, None]

        advantages = torch.zeros_like(kl_div)
        last_gae_lam = 0.0

        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_values = 0.0
                step_reward = rewards_tensor - beta * kl_div[:, t:t + 1]
            else:
                next_values = values[:, query_length - 1 + t + 1]
                step_reward = -beta * kl_div[:, t:t + 1]

            delta = step_reward + gamma * next_values - values[:, query_length - 1 + t]

            last_gae_lam = delta + gamma * lambda_coef * last_gae_lam
            advantages[:, t] = last_gae_lam.squeeze(-1)

        returns = advantages + values[:, query_length:, 0]

        return returns, advantages

    @staticmethod
    def calculate_loss(ratio, clipped_ratio, advantages):
        return -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    @staticmethod
    def calculate_entropy(log_probs: torch.Tensor) -> torch.Tensor:
        # probs = torch.exp(log_probs)
        # entropy = -torch.sum(probs * log_probs, dim=-1)
        # return entropy.mean()

        return -log_probs.mean()

    # GEMINI GENERATED
    @staticmethod
    def get_adaptive_beta(config: Optional[PPOConfig],
                          kl_div: torch.Tensor) -> float:
        # 1. Защита от NaN/Inf во входных данных
        if torch.isnan(kl_div).any() or torch.isinf(kl_div).any():
            return config.kl_coef  # Возвращаем текущий коэффициент без изменений

        # 2. Считаем среднее KL
        mean_kl_coef = kl_div.sum(dim=-1).mean().item()

        # 3. Добавляем epsilon для защиты от деления на ноль
        eps = 1e-8
        target = config.target_kl_coef if config.target_kl_coef > 0 else eps

        # 4. Считаем изменение (с ограничением шага изменения)
        # Используем clamp, чтобы коэффициент не прыгал слишком резко за один шаг
        diff_ratio = (mean_kl_coef - target) / target
        diff_ratio = max(-0.2, min(diff_ratio, 0.2))  # Ограничиваем изменение на 20% за раз

        kl_coef = config.kl_coef * (1.0 + config.kl_ctl_update_rate * diff_ratio)

        # 5. Жесткие границы из конфига
        kl_coef = max(config.kl_coef_min, min(kl_coef, config.kl_coef_max))

        return kl_coef

    def step(self,
             query_tensor: torch.Tensor | List[torch.Tensor],
             responses: torch.Tensor | List[torch.Tensor],
             scores: torch.Tensor | List[torch.Tensor],
             ) -> Dict[str, Any]:
        """
        Run a PPO optimisation step given a list of queries, model responses, and rewards.

        Args:
            query_tensor - List of tensors containing the encoded queries of shape (`query_length`)
            responses - List of tensors containing the encoded responses of shape (`response_length`)
            scores - List of tensors containing the scores.
        Returns:
            A summary of the training statistics
        """

        assert self.config.batch_size % self.config.mini_batch_size == 0, \
            f"batch_size ({self.config.batch_size}) must be divisible by mini_batch_size ({self.config.mini_batch_size})"

        if type(query_tensor) in (type(list()), type(())):
            query_tensor, query_attention_mask = collate_left_padding(query_tensor,
                                                                      pad_token_id=self.tokenizer.pad_token_id,
                                                                      device=self.device)
        # else:
        #     if not query_attention_mask:
        #         query_attention_mask = torch.ones_like(query_tensor, device=self.device)

        if type(responses) in (type(list()), type(())):
            responses, response_attention_mask = collate_left_padding(responses,
                                                                      pad_token_id=self.tokenizer.pad_token_id,
                                                                      device=self.device)
        # else:
        #     if not response_attention_mask:
        #         response_attention_mask = torch.ones_like(responses, device=self.device)

        full_response = torch.cat((query_tensor, responses), dim=-1)
        full_response_attention_mask = (full_response != self.tokenizer.pad_token_id).long()
        position_ids = full_response_attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(full_response_attention_mask == 0, 0)

        stats = dict()

        max_length = responses.shape[-1]
        query_length = query_tensor.shape[-1]
        mse_loss = torch.nn.MSELoss()

        with torch.no_grad():
            old_log_probs, old_values = PPOTrainer.calculate_log_probs(self.model, query_tensor, responses,
                                                                       pad_token_id=self.tokenizer.pad_token_id)
            log_ref_probs, ref_values = PPOTrainer.calculate_log_probs(self.ref_model, query_tensor, responses,
                                                                       pad_token_id=self.tokenizer.pad_token_id)

        kl_div = PPOTrainer.calculate_kl_divergence(old_log_probs, log_ref_probs)
        # Добавить обрезку value function self.config.clip_range_value

        if self.config.use_adaptive_kl:
            self.config.kl_coef = PPOTrainer.get_adaptive_beta(self.config, kl_div)

        if not self.config.use_gae:
            returns, advantages = PPOTrainer.calculate_advantages(scores, old_values, kl_div, query_length=query_length,
                                                                  max_length=max_length, beta=self.config.kl_coef,
                                                                  gamma=self.config.gamma)
        else:
            returns, advantages = PPOTrainer.calculate_generalized_advantages(scores, old_values, kl_div,
                                                                              query_length=query_length,
                                                                              beta=self.config.kl_coef,
                                                                              gamma=self.config.gamma,
                                                                              lambda_coef=self.config.lambda_coef)

            # CHECK IF IT WORKS (lambda_coef == 1  in GAE makes the advantage estimation equivalent to the Monte Carlo (MC) return)
            # returns, advantages = PPOTrainer.calculate_advantages(scores, old_values, kl_div, query_length=query_length,
            #                                                       max_length=max_length, beta=1,
            #                                                       gamma=self.config.gamma)


        if self.config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        running_stats = {
            'surrogate_objective_function': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': []
        }

        for ppo_epoch in range(self.config.ppo_epochs):

            query_split = torch.split(query_tensor, self.config.mini_batch_size)
            responses_split = torch.split(responses, self.config.mini_batch_size)
            old_log_probs_split = torch.split(old_log_probs, self.config.mini_batch_size)
            advantages_split = torch.split(advantages, self.config.mini_batch_size)
            returns_split = torch.split(returns, self.config.mini_batch_size)

            for query_mini, response_mini, old_log_probs_mini, adv_mini, ret_mini in zip(
                    query_split, responses_split, old_log_probs_split, advantages_split, returns_split
            ):

                log_probs, values = PPOTrainer.calculate_log_probs(self.model, query_mini, response_mini,
                                                                   pad_token_id=self.tokenizer.pad_token_id)
                ratio = PPOTrainer.calculate_ratio(log_probs, old_log_probs_mini)
                clipped_ratio = PPOTrainer.clip_ratio(ratio, eps=self.config.clip_range)
                clipped_surrogate_objective_function = PPOTrainer.calculate_loss(ratio, clipped_ratio, adv_mini)
                value_loss = mse_loss(values[:, query_length - 1: -1, :].squeeze(-1), ret_mini)

                entropy = PPOTrainer.calculate_entropy(log_probs)

                total_loss = clipped_surrogate_objective_function \
                             + self.config.vf_coef * value_loss \
                             - self.config.entropy_coef * entropy

                self.optimizer.zero_grad()
                total_loss.backward()
                if self.config.max_grad_norm != -1 or (self.config.max_grad_norm is not None):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                running_stats['surrogate_objective_function'].append(
                    clipped_surrogate_objective_function.detach().cpu().item())
                running_stats['value_loss'].append(value_loss.detach().cpu().item())
                running_stats['entropy'].append(entropy.detach().cpu().item())
                running_stats['total_loss'].append(total_loss.detach().cpu().item())

        stats['kl_div'] = kl_div.detach().cpu().mean().item()
        stats['old_log_probs'] = old_log_probs.detach().cpu()
        stats['old_values'] = old_values.detach().cpu()
        stats['mean_scores'] = torch.stack(scores).mean().detach().cpu().item()
        stats['std_scores'] = torch.stack(scores).std().detach().cpu().item()
        stats['returns'] = returns.detach().cpu()
        stats['advantages'] = advantages.detach().cpu()

        stats['surrogate_objective_function'] = sum(running_stats['surrogate_objective_function']) / len(
            running_stats['surrogate_objective_function'])
        stats['value_loss'] = sum(running_stats['value_loss']) / len(running_stats['value_loss'])
        stats['entropy'] = sum(running_stats['entropy']) / len(running_stats['entropy'])
        stats['total_loss'] = sum(running_stats['total_loss']) / len(running_stats['total_loss'])

        # GEMINI GENERATED
        if getattr(self.config, 'use_wandb', False) and wandb.run is not None:
            wandb_stats = {
                "objective/kl": stats['kl_div'],
                "env/reward_mean": stats['mean_scores'],
                "env/reward_std": stats['std_scores'],
                "ppo/loss/policy": stats['surrogate_objective_function'],
                "ppo/loss/value": stats['value_loss'],
                "ppo/loss/entropy": stats['entropy'],
                "ppo/loss/total": stats['total_loss'],
            }

            table = wandb.Table(columns=["query", "response", "reward"])

            queries_text = self.tokenizer.batch_decode(query_tensor, skip_special_tokens=True)
            responses_text = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

            for q, r, s in zip(queries_text, responses_text, scores):
                score_val = s.item() if isinstance(s, torch.Tensor) else s
                table.add_data(q, r, score_val)

            wandb_stats["game_log"] = table

            wandb.log(wandb_stats)


        return stats

