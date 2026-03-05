import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from typing import Optional, Callable, List, Union, Tuple, Dict, Any

# GEMINI GENERATED
def filter_logits(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0):
    """
    Фильтрация логитов с использованием стратегий Top-K и Nucleus (Top-P) sampling.
    """
    logits = logits.clone()
    batch_size = logits.size(0)

    # 1. Top-K фильтрация
    if top_k > 0:
        # Находим значения k-го самого большого элемента
        top_k_values, _ = torch.topk(logits, top_k)
        min_values = top_k_values[..., -1, None]  # Берем последний (минимальный из топ-к)
        indices_to_remove = logits < min_values
        logits[indices_to_remove] = float('-inf')

    # 2. Top-P (Nucleus) фильтрация
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Создаем маску: удаляем всё, что идет ПОСЛЕ того, как сумма превысила top_p
        sorted_indices_to_remove = cumulative_probs > top_p

        # Сдвигаем маску вправо:
        # Мы хотим ОСТАВИТЬ первый токен, который превысил порог,
        # поэтому маска удаления для него должна быть False.
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Рассеиваем маску обратно на оригинальную форму тензора
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, sorted_indices,
                                                                                sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')

    return logits


# GEMINI GENERATED
def collate_batch(batch: List[torch.Tensor], pad_token_id: int):
    # Паддинг: превращаем список в тензор [batch_size, max_len]
    input_ids = pad_sequence(batch, batch_first=True, padding_value=pad_token_id)

    # Маска: 1 там, где есть данные, и 0 там, где pad_token_id
    attention_mask = (input_ids != pad_token_id).long()

    return input_ids, attention_mask


# GEMINI GENERATED
def collate_left_padding(batch: List[torch.Tensor], pad_token_id: int):
    max_len = max(len(x) for x in batch)
    batch_size = len(batch)

    # 1. Создаем тензор, сразу заполненный pad_token_id
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)

    # 2. Заполняем "хвосты" строк нашими данными
    for i, seq in enumerate(batch):
        input_ids[i, -len(seq):] = seq

    # 3. Маска создается так же
    attention_mask = (input_ids != pad_token_id).long()

    return input_ids, attention_mask