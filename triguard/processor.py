import torch
from transformers import LogitsProcessor, StoppingCriteria, LogitsProcessorList, StoppingCriteriaList
from .metrics import get_tri_guard_metrics
from .critic import get_critic_score, is_fact_atom

class TriGuardState:
    def __init__(self, fallback_ids, facts_mode=True):
        self.mode = "normal" # "normal" or "fallback"
        self.fallback_ids = fallback_ids
        self.fb_pos = 0
        self.trigger_step = None
        self.reason = None
        self.facts_mode = facts_mode
        self.triggered = False
        self.max_risk = 0.0
        self.max_risk_metrics = {}
        self.metrics_history = []

class TriGuardLogitsProcessor(LogitsProcessor):
    def __init__(self, 
                 tokenizer, 
                 state: TriGuardState,
                 alpha=0.06, beta=0.00, gamma=0.00, delta=0.75, eps=0.46, zeta=0.0,
                 theta=0.71,
                 t_cold=0.2, t_hot=1.0, t_metric=1.0):
        
        self.tokenizer = tokenizer
        self.state = state
        # Веса формулы R
        self.alpha = alpha   # H_norm
        self.beta = beta     # (1 - p_max)
        self.gamma = gamma   # (1 - margin_norm)
        self.delta = delta   # jsd_norm
        self.eps = eps       # critic_score
        self.zeta = zeta     # tail_jsd_norm
        
        self.theta = theta   # Порог отсечки
        self.t_cold = t_cold
        self.t_hot = t_hot
        self.t_metric = t_metric
        
        self.last_metrics = {}
        self.prompt_text = ""
        self.generated_text = ""
        self._last_input_len = None

    def set_prompt(self, prompt: str):
        self.prompt_text = prompt
        self.generated_text = ""
        self._last_input_len = None

    def _sync_generated_text(self, input_ids: torch.LongTensor):
        # input_ids содержит prompt + уже сгенерированные токены.
        # __call__ вызывается ДО выбора следующего токена, поэтому на каждом вызове
        # мы можем подтянуть токены, появившиеся с прошлого вызова.
        cur_len = int(input_ids.size(-1))
        if self._last_input_len is None:
            self._last_input_len = cur_len
            return
        if cur_len <= self._last_input_len:
            return

        new_ids = input_ids[0, self._last_input_len:cur_len].tolist()
        self.generated_text += self.tokenizer.decode(new_ids, skip_special_tokens=True)
        self._last_input_len = cur_len

    def _force_token(self, scores: torch.Tensor, token_id: int):
        new_scores = torch.full_like(scores, -float("inf"))
        new_scores[:, token_id] = 0
        return new_scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self._sync_generated_text(input_ids)

        # 1. Если уже в fallback-mode
        if self.state.mode == "fallback":
            if self.state.fb_pos < len(self.state.fallback_ids):
                target = self.state.fallback_ids[self.state.fb_pos]
                self.state.fb_pos += 1
                return self._force_token(scores, target)
            else:
                return self._force_token(scores, self.tokenizer.eos_token_id)

        # 2. Если фактологический режим выключен (или если это первый токен и мы еще не решили)
        if not self.state.facts_mode:
            return scores

        # 0. GATING STRATEGY:
        # Don't check risk on every token. Only check on "atoms" (numbers, entities, rare words).
        # This prevents "any-step veto" where random stopwords trigger the guard.
        
        # Calculate greedy candidate to check if it's an atom
        cold_next_id = torch.argmax(scores / self.t_cold, dim=-1).item()
        candidate_token = self.tokenizer.decode([cold_next_id])
        
        if not is_fact_atom(candidate_token):
            # Not a fact/entity -> skip guard, just return scores as is
            # (We might want to still log metrics, but for efficiency/stability we skip)
            return scores

        # 3. Получаем метрики (H_norm, p_max, margin, JSD)
        metrics = get_tri_guard_metrics(scores, self.t_cold, self.t_hot, self.t_metric)
        
        # 4. Получаем оценку критика
        
        critic = get_critic_score(self.prompt_text, self.generated_text, candidate_token)
        
        # 5. Нормализация для формулы
        m0 = 0.20
        j0 = 0.50
        margin_n = max(0.0, min(1.0, metrics["margin"] / m0))
        jsd_n = max(0.0, min(1.0, metrics["jsd"] / j0))
        
        # Tail-JSD
        tail_jsd = metrics.get("tail_jsd", 0.0)
        tail_jsd_n = max(0.0, min(1.0, tail_jsd / j0))
        
        # 6. Считаем итоговый риск R
        r = (
            self.alpha * metrics["h_norm"] +
            self.beta * (1.0 - metrics["p_max"]) +
            self.gamma * (1.0 - margin_n) +
            self.delta * jsd_n +
            self.zeta * tail_jsd_n +
            self.eps * critic
        )
        
        self.last_metrics = {**metrics, "critic": critic, "total_risk": r}
        self.state.metrics_history.append(self.last_metrics)
        if r > self.state.max_risk:
            self.state.max_risk = r
            self.state.max_risk_metrics = self.last_metrics
        
        # 7. Решение о переходе в fallback
        if r > self.theta:
            self.state.mode = "fallback"
            self.state.triggered = True
            self.state.trigger_step = input_ids.size(-1)
            self.state.reason = "risk>theta"
            
            target = self.state.fallback_ids[0]
            self.state.fb_pos = 1
            return self._force_token(scores, target)

        return scores

class StopOnFallbackDone(StoppingCriteria):
    def __init__(self, state: TriGuardState):
        self.state = state

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.state.mode == "fallback" and self.state.fb_pos >= len(self.state.fallback_ids):
            return True
        return False

def get_tri_guard_callbacks(tokenizer, fallback_text="Не уверен в фактах без источника. Дайте ссылку/контекст.", facts_mode=True, prompt=None, **kwargs):
    fallback_ids = tokenizer(fallback_text, add_special_tokens=False).input_ids
    state = TriGuardState(fallback_ids=fallback_ids, facts_mode=facts_mode)
    
    processor = TriGuardLogitsProcessor(tokenizer, state=state, **kwargs)
    
    if prompt:
        processor.set_prompt(prompt)
        
    stopping_criteria = StopOnFallbackDone(state)
    
    return LogitsProcessorList([processor]), StoppingCriteriaList([stopping_criteria]), processor
