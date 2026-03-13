import wandb
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any, Tuple
import numpy as np

from transformers import AutoTokenizer

from .utils import filter_logits, collate_left_padding


@dataclass
class GRPOConfig:
    model_name: str = field(
        default="HuggingFaceTB/SmolLM-135M-Instruct",
        metadata={"help": "Название модели для загрузки"}
    )
    learning_rate: float = 1.41e-5
    batch_size: int = 4
    grpo_epochs: int = 4
    grpo_groups: int = 3
    temperature: float = 1.0
    max_completion_length: int = 20
    kl_coef: float = 0.05
    entropy_coef: float = 0.01
    clip_range: float = 0.2
    system_prompt: str = ''
    normalize_advantage: bool = True
    max_grad_norm: Optional[float] = 1.0
    shuffle: bool = True
    use_wandb: bool = False
    wandb_project: str = "custom-trl-grpo-project"


class GRPOTrainer:

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

    @staticmethod
    def process_tokens_to_string(token_tensor, tokenizer):
        B, G, T = token_tensor.shape
        flat_tokens = token_tensor.contiguous().view(B * G, T)
        texts = tokenizer.batch_decode(flat_tokens, skip_special_tokens=True)
        return texts

    @staticmethod
    def get_batch_log_probs(logits: torch.Tensor,
                            tokens: torch.Tensor,
                            masks: torch.Tensor,
                            prompt_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        shift_logits = logits[:, prompt_length - 1: -1, :].contiguous()
        shift_log_probs = F.log_softmax(shift_logits, dim=-1)

        completion_tokens = tokens[:, prompt_length:].contiguous()
        completion_masks = masks[:, prompt_length:].contiguous()

        log_probs_chosen = shift_log_probs.gather(
            dim=-1,
            index=completion_tokens.unsqueeze(-1)
        ).squeeze(-1)

        return log_probs_chosen * completion_masks, completion_masks

    def __init__(self,
                 config: Optional[GRPOConfig] = None,
                 model: AutoModelForCausalLM = None,
                 ref_model: AutoModelForCausalLM = None,
                 reward_funcs: List[Callable] | Callable = None,
                 tokenizer: Optional[transformers.tokenization_utils_base.PreTrainedTokenizerBase] = None,
                 dataset: Optional[torch.utils.data.dataset.Dataset] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 data_collator: Optional[Callable] = None,
                 ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or GRPOConfig()
        self.model = model or AutoModelForCausalLM.from_pretrained(self.config.model_name)
        self.model = self.model.to(self.device)
        self.ref_model = ref_model or AutoModelForCausalLM.from_pretrained(self.config.model_name)
        self.ref_model = self.ref_model.to(self.device)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(self.config.model_name, max_length=1024)
        self.dataset = dataset
        self.reward_funcs = reward_funcs
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.data_collator = data_collator

        if self.dataset:
            self.data_loader = self.load_data(self.dataset, self.config.batch_size,
                                              shuffle=self.config.shuffle, num_workers=0, collate_fn=self.data_collator)
        else:
            self.data_loader = None

        self.vec_score = np.vectorize(reward_funcs)

        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                config=vars(self.config),
                name=f"grpo-{self.config.model_name.split('/')[-1]}"
            )

    def step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        pad_token = self.tokenizer.pad_token_id
        messages = [
            [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": question}
            ]
            for question in batch['question']
        ]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)

        # Кодируем и переносим на устройство через collate_left_padding
        tokens = self.tokenizer.encode(input_text) if isinstance(input_text, str) else [self.tokenizer.encode(t) for t
                                                                                        in input_text]
        tokens = [torch.as_tensor(token) for token in tokens]
        prompts, masks = collate_left_padding(tokens, pad_token_id=pad_token, device=self.device)

        prompts_length = prompts.shape[-1]
        B, T = prompts.shape
        G = self.config.grpo_groups

        # Расширяем для групп
        prompts = prompts.unsqueeze(1).expand(-1, G, -1).clone()
        masks = masks.unsqueeze(1).expand(-1, G, -1).clone()

        # === 1. Фаза генерации ===
        for _ in range(self.config.max_completion_length):
            B, G, T = prompts.shape
            flat_prompts = prompts.contiguous().view(B * G, T)
            flat_masks = masks.contiguous().view(B * G, T)

            position_ids = flat_masks.long().cumsum(-1) - 1
            position_ids.masked_fill_(flat_masks == 0, 0)

            with torch.no_grad():
                outputs = self.model(input_ids=flat_prompts,
                                     attention_mask=flat_masks,
                                     position_ids=position_ids)

            next_token_logits = outputs.logits[:, -1, :] / self.config.temperature
            filtered_logits = filter_logits(next_token_logits, top_k=0, top_p=1.0)
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).view(B, G, 1)

            last_tokens = prompts[..., -1:]
            is_padded = (last_tokens == pad_token)
            new_next_token = pad_token * is_padded + next_token * (~is_padded)

            prompts = torch.cat((prompts, new_next_token), dim=-1)
            new_mask_token = (new_next_token != pad_token).long()
            masks = torch.cat((masks, new_mask_token), dim=-1)

        # === 2. Подсчет наград (Rewards) ===
        token_to_string = GRPOTrainer.process_tokens_to_string(
            prompts[..., -self.config.max_completion_length:].cpu(),
            self.tokenizer
        )
        vectorized_scores = self.vec_score(token_to_string)
        tensor_scores = torch.as_tensor(vectorized_scores, dtype=torch.float32, device=self.device).view(B, G)

        mean = tensor_scores.mean(dim=-1, keepdim=True)
        std = tensor_scores.std(dim=-1, keepdim=True, unbiased=False)
        advantages = (tensor_scores - mean) / (std + 1e-8)
        advantages_to_all_tokens = advantages.unsqueeze(-1).expand(-1, -1, self.config.max_completion_length)

        B, G, T = prompts.shape
        flat_prompts = prompts.contiguous().view(B * G, T)
        flat_masks = masks.contiguous().view(B * G, T)

        position_ids = flat_masks.long().cumsum(-1) - 1
        position_ids.masked_fill_(flat_masks == 0, 0)

        with torch.no_grad():
            old_outputs = self.model(input_ids=flat_prompts, attention_mask=flat_masks, position_ids=position_ids)
            ref_outputs = self.ref_model(input_ids=flat_prompts, attention_mask=flat_masks, position_ids=position_ids)

        old_log_probs_chosen, completion_masks_flat = GRPOTrainer.get_batch_log_probs(
            old_outputs.logits, flat_prompts, flat_masks, prompts_length
        )
        ref_log_probs_chosen, _ = GRPOTrainer.get_batch_log_probs(
            ref_outputs.logits, flat_prompts, flat_masks, prompts_length
        )

        old_log_probs_chosen = old_log_probs_chosen.view(B, G, -1)
        ref_log_probs_chosen = ref_log_probs_chosen.view(B, G, -1)
        completion_masks_3d = completion_masks_flat.view(B, G, -1)
        actual_seq_lens = completion_masks_3d.sum(dim=-1)

        running_stats = {'policy_loss': [], 'kl_div': [], 'total_loss': []}

        for epoch in range(self.config.grpo_epochs):
            outputs = self.model(input_ids=flat_prompts, attention_mask=flat_masks, position_ids=position_ids)

            log_probs_chosen, _ = GRPOTrainer.get_batch_log_probs(
                outputs.logits, flat_prompts, flat_masks, prompts_length
            )
            log_probs_chosen = log_probs_chosen.view(B, G, -1)

            ratio = torch.exp(log_probs_chosen - old_log_probs_chosen)
            d_kl = torch.exp(ref_log_probs_chosen - log_probs_chosen) - ref_log_probs_chosen + log_probs_chosen - 1

            surrogate1 = ratio * advantages_to_all_tokens
            surrogate2 = torch.clamp(ratio, 1.0 - self.config.clip_range,
                                     1.0 + self.config.clip_range) * advantages_to_all_tokens

            policy_loss = torch.min(surrogate1, surrogate2)
            per_token_loss = policy_loss - self.config.kl_coef * d_kl

            per_token_loss = per_token_loss * completion_masks_3d
            loss_per_seq = per_token_loss.sum(dim=-1) / actual_seq_lens
            loss = -loss_per_seq.mean()

            self.optimizer.zero_grad()
            loss.backward()
            if self.config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            with torch.no_grad():
                masked_kl = (d_kl * completion_masks_3d).sum(dim=-1) / actual_seq_lens
                masked_policy = (policy_loss * completion_masks_3d).sum(dim=-1) / actual_seq_lens
                running_stats['kl_div'].append(masked_kl.mean().item())
                running_stats['policy_loss'].append(masked_policy.mean().item())
                running_stats['total_loss'].append(loss.item())

        # === 5. Логирование ===
        stats = {
            'mean_scores': tensor_scores.mean().item(),
            'std_scores': tensor_scores.std(unbiased=False).item(),
            'policy_loss': sum(running_stats['policy_loss']) / len(running_stats['policy_loss']),
            'kl_div': sum(running_stats['kl_div']) / len(running_stats['kl_div']),
            'total_loss': sum(running_stats['total_loss']) / len(running_stats['total_loss'])
        }

        if getattr(self.config, 'use_wandb', False) and wandb.run is not None:
            wandb_stats = {
                "grpo/kl": stats['kl_div'],
                "env/reward_mean": stats['mean_scores'],
                "env/reward_std": stats['std_scores'],
                "grpo/loss/policy": stats['policy_loss'],
                "grpo/loss/total": stats['total_loss'],
            }

            table = wandb.Table(columns=["query", "response", "reward"])
            for b_idx in range(B):
                q_text = batch['question'][b_idx]
                for g_idx in range(G):
                    flat_idx = b_idx * G + g_idx
                    r_text = token_to_string[flat_idx]
                    score_val = vectorized_scores[flat_idx]
                    table.add_data(q_text, r_text, score_val)

            wandb_stats["game_log"] = table
            wandb.log(wandb_stats)

        return stats

    def train(self):
        if self.data_loader is None:
            raise ValueError("Dataloader не инициализирован. Передайте dataset при создании GRPOTrainer.")

        for batch in self.data_loader:
            stats = self.step(batch)