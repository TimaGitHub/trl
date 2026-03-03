import torch
import torch.nn.functional as F

# GEMINI GENERATED
def filter_logits(logits, top_k: int = 0, top_p: float = 1.0):
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