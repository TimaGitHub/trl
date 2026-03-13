import wandb
import torch
import torch.nn.functional as F
import transformers
from dataclasses import dataclass
from typing import Optional, List
from tqdm import tqdm

from .utils import collate_left_padding


@dataclass
class DPOConfig:
    model_name: str
    learning_rate: float = 1.41e-5
    batch_size: int = 16
    beta: float = 0.1
    shuffle: bool = False
    num_workers: int = 0
    dpo_epochs: int = 5
    use_wandb: bool = False
    wandb_project: str = "custom-trl-dpo-project"


class DPOTrainer:

    def __init__(self, config: DPOConfig,
                 #base_model: AutoModelForCausalLM,
                 base_model: torch.nn.Module,
                 # ref_model: AutoModelForCausalLM,
                 ref_model: torch.nn.Module,
                 train_dataset: torch.utils.data.dataset.Dataset,
                 eval_dataset: Optional[torch.utils.data.dataset.Dataset],
                 tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase
                 ):

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model = base_model
        self.base_model.to(self.device)
        self.ref_model = ref_model
        self.ref_model.to(self.device)
        self.ref_model.eval()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.Adam(self.base_model.parameters(), lr=self.config.learning_rate)

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                           batch_size=config.batch_size,
                           shuffle=config.shuffle,
                           num_workers=config.num_workers)

        self.eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                           batch_size=config.batch_size,
                           shuffle=config.shuffle,
                           num_workers=config.num_workers)

    @staticmethod
    def get_tokens(batch: List[str],
                   tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase):
        tokens = tokenizer.encode(batch)
        return tokens

    @staticmethod
    def get_batch_log_probs(logits: torch.Tensor,
                            labels: torch.Tensor,
                            loss_mask: torch.Tensor):
        """
        logits: [batch_size, seq_len, vocab_size]
        labels: [batch_size, seq_len]
        loss_mask: [batch_size, seq_len] 1 response, 0 pad and prompt
        """
        logits_shifted = logits[:, :-1, :]
        labels_shifted = labels[:, 1:].unsqueeze(-1)
        loss_mask_shifted = loss_mask[:, 1:]

        per_token_log_probs = torch.gather(logits_shifted, dim=-1, index=labels_shifted).squeeze(-1)

        per_token_log_probs = per_token_log_probs * loss_mask_shifted

        return per_token_log_probs.sum(dim=-1)

    def train(self):
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                config=vars(self.config),
                name=f"dpo-{self.config.model_name}"
            )

        for epoch in tqdm(range(self.config.dpo_epochs), desc='Номер эпохи обучения'):
            for index, batch in enumerate(tqdm(self.train_loader, leave=False, desc='Номер итерации загрузчика данных')):
                prompt_tokens = DPOTrainer.get_tokens(batch['prompt'], self.tokenizer)
                chosen_tokens = DPOTrainer.get_tokens(batch['chosen'], self.tokenizer)
                rejected_tokens = DPOTrainer.get_tokens(batch['rejected'], self.tokenizer)

                ##################################################################################################################

                prompt_chosen_tokens = [p + c for p, c in zip(prompt_tokens, chosen_tokens)]
                loss_mask_chosen = [[0] * len(p) + [1] * len(c) for p, c in zip(prompt_tokens, chosen_tokens)]

                prompt_rejected_tokens = [p + r for p, r in zip(prompt_tokens, rejected_tokens)]
                loss_mask_rejected = [[0] * len(p) + [1] * len(r) for p, r in zip(prompt_tokens, rejected_tokens)]

                ##################################################################################################################

                prompt_chosen_tokens = [torch.as_tensor(token) for token in prompt_chosen_tokens]
                loss_mask_chosen = [torch.as_tensor(mask) for mask in loss_mask_chosen]

                prompt_rejected_tokens = [torch.as_tensor(token) for token in prompt_rejected_tokens]
                loss_mask_rejected = [torch.as_tensor(mask) for mask in loss_mask_rejected]

                ##################################################################################################################

                prompt_chosen_token_tensor, prompt_chosen_attention_mask = collate_left_padding(prompt_chosen_tokens,
                                                                                                pad_token_id=self.tokenizer.pad_token_id,
                                                                                                device=self.device)

                loss_mask_chosen_tensor, _ = collate_left_padding(loss_mask_chosen, pad_token_id=0, device=self.device)

                prompt_rejected_token_tensor, prompt_rejected_attention_mask = collate_left_padding(
                    prompt_rejected_tokens, pad_token_id=self.tokenizer.pad_token_id, device=self.device)

                loss_mask_rejected_tensor, _ = collate_left_padding(loss_mask_rejected, pad_token_id=0, device=self.device)

                ##################################################################################################################

                prompt_chosen_position_ids = prompt_chosen_attention_mask.long().cumsum(-1) - 1
                prompt_chosen_position_ids.masked_fill_(prompt_chosen_attention_mask == 0, 0)

                prompt_rejected_position_ids = prompt_rejected_attention_mask.long().cumsum(-1) - 1
                prompt_rejected_position_ids.masked_fill_(prompt_rejected_attention_mask == 0, 0)

                ##################################################################################################################

                output_chosen = self.base_model(prompt_chosen_token_tensor.to(self.device),
                                            position_ids=prompt_chosen_position_ids,
                                            attention_mask=prompt_chosen_attention_mask)

                output_rejected = self.base_model(prompt_rejected_token_tensor.to(self.device),
                                              position_ids=prompt_rejected_position_ids,
                                              attention_mask=prompt_rejected_attention_mask)

                ##################################################################################################################

                with torch.no_grad():
                    ref_output_chosen = self.ref_model(prompt_chosen_token_tensor.to(self.device),
                                                    position_ids=prompt_chosen_position_ids,
                                                    attention_mask=prompt_chosen_attention_mask)

                    ref_output_rejected = self.ref_model(prompt_rejected_token_tensor.to(self.device),
                                                      position_ids=prompt_rejected_position_ids,
                                                      attention_mask=prompt_rejected_attention_mask)

                ##################################################################################################################

                log_probs_chosen_all = torch.log_softmax(output_chosen.logits, dim=-1)
                log_probs_rejected_all = torch.log_softmax(output_rejected.logits, dim=-1)

                ref_log_probs_chosen_all = torch.log_softmax(ref_output_chosen.logits, dim=-1)
                ref_log_probs_rejected_all = torch.log_softmax(ref_output_rejected.logits, dim=-1)

                ##################################################################################################################

                chosen_log_probs = DPOTrainer.get_batch_log_probs(log_probs_chosen_all, prompt_chosen_token_tensor,
                                                       loss_mask_chosen_tensor)
                rejected_log_probs = DPOTrainer.get_batch_log_probs(log_probs_rejected_all, prompt_rejected_token_tensor,
                                                         loss_mask_rejected_tensor)

                ##################################################################################################################

                ref_chosen_log_probs = DPOTrainer.get_batch_log_probs(ref_log_probs_chosen_all, prompt_chosen_token_tensor,
                                                       loss_mask_chosen_tensor)
                ref_rejected_log_probs = DPOTrainer.get_batch_log_probs(ref_log_probs_rejected_all, prompt_rejected_token_tensor,
                                                         loss_mask_rejected_tensor)

                ##################################################################################################################

                loss = - F.logsigmoid(self.config.beta * (chosen_log_probs - ref_chosen_log_probs - (rejected_log_probs - ref_rejected_log_probs))).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.config.use_wandb:
                    with torch.no_grad():
                        # Неявные награды (rewards)
                        chosen_rewards = self.config.beta * (chosen_log_probs - ref_chosen_log_probs)
                        rejected_rewards = self.config.beta * (rejected_log_probs - ref_rejected_log_probs)

                        reward_margin = chosen_rewards - rejected_rewards

                        accuracy = (reward_margin > 0).float().mean()

                    wandb.log({
                        "dpo/loss": loss.item(),
                        "dpo/rewards_chosen": chosen_rewards.mean().item(),
                        "dpo/rewards_rejected": rejected_rewards.mean().item(),
                        "dpo/reward_margin": reward_margin.mean().item(),
                        "dpo/accuracy": accuracy.item(),
                    })

            if self.eval_loader is not None:
                eval_metrics = self.evaluate()

                print(f"Epoch {epoch} | Eval Loss: {eval_metrics['eval/loss']:.4f} | "
                      f"Eval Acc: {eval_metrics['eval/accuracy']:.4f}")

                if wandb.run is not None: # if self.config.use_wandb:
                    wandb.log(eval_metrics)

    def evaluate(self):
        self.base_model.eval()

        total_loss = 0.0
        total_accuracy = 0.0
        total_margin = 0.0
        total_chosen_rewards = 0.0
        total_rejected_rewards = 0.0

        eval_steps = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc='Валидация', leave=False):
                prompt_tokens = self.get_tokens(batch['prompt'], self.tokenizer)
                chosen_tokens = self.get_tokens(batch['chosen'], self.tokenizer)
                rejected_tokens = self.get_tokens(batch['rejected'], self.tokenizer)

                prompt_chosen_tokens = [p + c for p, c in zip(prompt_tokens, chosen_tokens)]
                loss_mask_chosen = [[0] * len(p) + [1] * len(c) for p, c in
                                    zip(prompt_tokens, chosen_tokens)]

                prompt_rejected_tokens = [p + r for p, r in zip(prompt_tokens, rejected_tokens)]
                loss_mask_rejected = [[0] * len(p) + [1] * len(r) for p, r in
                                      zip(prompt_tokens, rejected_tokens)]

                prompt_chosen_tokens = [torch.as_tensor(token) for token in prompt_chosen_tokens]
                loss_mask_chosen = [torch.as_tensor(mask) for mask in loss_mask_chosen]

                prompt_rejected_tokens = [torch.as_tensor(token) for token in prompt_rejected_tokens]
                loss_mask_rejected = [torch.as_tensor(mask) for mask in loss_mask_rejected]

                prompt_chosen_token_tensor, prompt_chosen_attention_mask = collate_left_padding(
                    prompt_chosen_tokens, pad_token_id=self.tokenizer.pad_token_id, device=self.device)
                loss_mask_chosen_tensor, _ = collate_left_padding(
                    loss_mask_chosen, pad_token_id=0, device=self.device)

                prompt_rejected_token_tensor, prompt_rejected_attention_mask = collate_left_padding(
                    prompt_rejected_tokens, pad_token_id=self.tokenizer.pad_token_id, device=self.device)
                loss_mask_rejected_tensor, _ = collate_left_padding(
                    loss_mask_rejected, pad_token_id=0, device=self.device)

                prompt_chosen_position_ids = prompt_chosen_attention_mask.long().cumsum(-1) - 1
                prompt_chosen_position_ids.masked_fill_(prompt_chosen_attention_mask == 0, 0)

                prompt_rejected_position_ids = prompt_rejected_attention_mask.long().cumsum(-1) - 1
                prompt_rejected_position_ids.masked_fill_(prompt_rejected_attention_mask == 0, 0)

                # 2. Получаем выходы обучаемой модели
                output_chosen = self.base_model(prompt_chosen_token_tensor,
                                                position_ids=prompt_chosen_position_ids,
                                                attention_mask=prompt_chosen_attention_mask)
                output_rejected = self.base_model(prompt_rejected_token_tensor,
                                                  position_ids=prompt_rejected_position_ids,
                                                  attention_mask=prompt_rejected_attention_mask)

                log_probs_chosen_all = torch.log_softmax(output_chosen.logits, dim=-1)
                log_probs_rejected_all = torch.log_softmax(output_rejected.logits, dim=-1)

                chosen_log_probs = self.get_batch_log_probs(log_probs_chosen_all,
                                                            prompt_chosen_token_tensor,
                                                            loss_mask_chosen_tensor)
                rejected_log_probs = self.get_batch_log_probs(log_probs_rejected_all,
                                                              prompt_rejected_token_tensor,
                                                              loss_mask_rejected_tensor)

                # 3. Получаем выходы референсной модели
                ref_output_chosen = self.ref_model(prompt_chosen_token_tensor,
                                                   position_ids=prompt_chosen_position_ids,
                                                   attention_mask=prompt_chosen_attention_mask)
                ref_output_rejected = self.ref_model(prompt_rejected_token_tensor,
                                                     position_ids=prompt_rejected_position_ids,
                                                     attention_mask=prompt_rejected_attention_mask)

                ref_log_probs_chosen_all = torch.log_softmax(ref_output_chosen.logits, dim=-1)
                ref_log_probs_rejected_all = torch.log_softmax(ref_output_rejected.logits, dim=-1)

                ref_chosen_log_probs = self.get_batch_log_probs(ref_log_probs_chosen_all,
                                                                prompt_chosen_token_tensor,
                                                                loss_mask_chosen_tensor)
                ref_rejected_log_probs = self.get_batch_log_probs(ref_log_probs_rejected_all,
                                                                  prompt_rejected_token_tensor,
                                                                  loss_mask_rejected_tensor)

                # 4. Расчет Loss и наград
                loss = - F.logsigmoid(self.config.beta * (chosen_log_probs - ref_chosen_log_probs - (
                            rejected_log_probs - ref_rejected_log_probs))).mean()

                chosen_rewards = self.config.beta * (chosen_log_probs - ref_chosen_log_probs)
                rejected_rewards = self.config.beta * (rejected_log_probs - ref_rejected_log_probs)
                reward_margin = chosen_rewards - rejected_rewards
                accuracy = (reward_margin > 0).float().mean()

                total_loss += loss.item()
                total_accuracy += accuracy.item()
                total_margin += reward_margin.mean().item()
                total_chosen_rewards += chosen_rewards.mean().item()
                total_rejected_rewards += rejected_rewards.mean().item()
                eval_steps += 1

        self.base_model.train()

        eval_metrics = {
            "eval/loss": total_loss / eval_steps,
            "eval/accuracy": total_accuracy / eval_steps,
            "eval/reward_margin": total_margin / eval_steps,
            "eval/rewards_chosen": total_chosen_rewards / eval_steps,
            "eval/rewards_rejected": total_rejected_rewards / eval_steps
        }

        return eval_metrics
