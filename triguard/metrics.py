import torch
import torch.nn.functional as F

def _ensure_single_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Приводит логиты к виду (V,) для batch=1.

    Поддерживаемые формы:
      - (V,)
      - (1, V)
      - (1, S, V)  (берём последний шаг)

    Для batch>1 — явная ошибка, чтобы не получить молча неверные метрики.
    """
    if logits.dim() == 3:
        # (B, S, V) -> (B, V) последнего шага
        logits = logits[:, -1, :]

    if logits.dim() == 2:
        if logits.size(0) != 1:
            raise ValueError(f"Expected batch=1 logits, got shape {tuple(logits.shape)}")
        return logits[0]

    if logits.dim() == 1:
        return logits

    raise ValueError(f"Unsupported logits shape {tuple(logits.shape)}")

def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет энтропию распределения следующего токена из логитов.
    Использует logsumexp для численной стабильности.
    
    Args:
        logits: (batch, V) или (V,)
    Returns:
        entropy: (batch,) или скаляр (в натах)
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    
    # H = log Z - sum(p_i * l_i)
    log_z = torch.logsumexp(logits, dim=-1)
    p = F.softmax(logits, dim=-1)
    exp_l = torch.sum(p * logits, dim=-1)
    
    entropy = log_z - exp_l
    return entropy.squeeze()

def normalized_entropy(logits: torch.Tensor) -> float:
    """
    Вычисляет нормированную энтропию H / log(V).
    """
    logits_1d = _ensure_single_logits(logits)
    v = logits_1d.size(-1)
    h = entropy_from_logits(logits_1d)
    denom = torch.log(torch.tensor(float(v), device=logits_1d.device, dtype=logits_1d.dtype))
    return (h / denom).item()

def jsd_from_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Jensen-Shannon Divergence.
    p, q: (batch, V) или (V,)
    """
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    
    kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)), dim=-1)
    kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)), dim=-1)
    
    return 0.5 * (kl_pm + kl_qm)

def tail_jsd_from_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Вычисляет JSD только для "хвоста" распределения (без top-1 токена).
    Полезно для обнаружения "уверенных галлюцинаций", где top-1 токен доминирует,
    но распределение альтернатив хаотично.
    """
    # 1. Находим и обнуляем top-1
    p_flat = p.clone()
    q_flat = q.clone()
    
    # Для p
    top_p_val, top_p_idx = torch.max(p_flat, dim=-1)
    p_flat[top_p_idx] = 0.0
    
    # Для q - обнуляем ТЕ ЖЕ индексы, что были top-1 в P (чтобы сравнивать тот же хвост)
    # Или лучше находить свой top-1?
    # Логичнее убрать доминирующий токен P, чтобы посмотреть, что под ним.
    q_flat[top_p_idx] = 0.0
    
    # 2. Перенормировка
    sum_p = p_flat.sum()
    sum_q = q_flat.sum()
    
    if sum_p < 1e-9 or sum_q < 1e-9:
        return 0.0 # Хвоста нет
        
    p_tail = p_flat / sum_p
    q_tail = q_flat / sum_q
    
    return jsd_from_probs(p_tail, q_tail).item()

def get_gtg_metrics(logits: torch.Tensor, t_guard: float = 1.0):
    """
    Базовые метрики Grey Token Guard (GTG).
    Теперь поддерживает температуру t_guard для "оживления" метрик на малых моделях.
    """
    logits_1d = _ensure_single_logits(logits)
    v = logits_1d.size(-1)

    # Применяем температуру
    p = torch.softmax(logits_1d / t_guard, dim=-1)

    # Энтропия тоже считается на этом распределении
    h = entropy_from_logits(logits_1d / t_guard)
    denom = torch.log(torch.tensor(float(v), device=logits_1d.device, dtype=logits_1d.dtype))
    h_norm = (h / denom).item()

    top_probs = torch.topk(p, k=2).values
    p_max = top_probs[0].item()
    margin = (top_probs[0] - top_probs[1]).item()

    return {"h_norm": h_norm, "p_max": p_max, "margin": margin}

def get_tri_guard_metrics(logits: torch.Tensor, t_cold: float = 0.2, t_hot: float = 1.0, t_metric: float = 1.0):
    """
    Возвращает расширенный набор метрик для Tri-Guard: 
    H_norm, p_max, margin, JSD, Tail-JSD.

    Args:
        t_cold: Temperature for the "anchor" distribution (JSD).
        t_hot: Temperature for the "perturbed" distribution (JSD).
        t_metric: Temperature for confidence metrics (H, P_max, Margin). 
                  Should typically be 1.0 to measure true uncertainty.
    """
    logits_1d = _ensure_single_logits(logits)
    v = logits_1d.size(-1)
    
    # 1. Metric Distribution (for H, Margin, P_max)
    # Using t_metric allows checking "true" confidence, not sharpened.
    p_metric = torch.softmax(logits_1d / t_metric, dim=-1)
    
    h = entropy_from_logits(logits_1d / t_metric)
    denom = torch.log(torch.tensor(float(v), device=logits_1d.device, dtype=logits_1d.dtype))
    h_norm = (h / denom).item()
    
    top_probs_m, _ = torch.topk(p_metric, k=2)
    p_max = top_probs_m[0].item()
    margin = (top_probs_m[0] - top_probs_m[1]).item()

    # 2. JSD Distributions (Split-Decoding)
    p_cold = torch.softmax(logits_1d / t_cold, dim=-1)
    p_hot = torch.softmax(logits_1d / t_hot, dim=-1)
    
    # Дивергенция
    jsd = jsd_from_probs(p_cold, p_hot).item()
    
    # Tail-JSD
    tail_jsd = tail_jsd_from_probs(p_cold, p_hot)
    
    return {
        "h_norm": h_norm,
        "p_max": p_max,
        "margin": margin,
        "jsd": jsd,
        "tail_jsd": tail_jsd
    }
