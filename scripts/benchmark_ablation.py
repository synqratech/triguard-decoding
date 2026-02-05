import argparse
import csv
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Добавляем корень проекта в PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

# твои коллбеки
from src.gtg.integration import get_gtg_callbacks # Keep legacy for comparison
from triguard import build_triguard

LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


@dataclass
class Example:
    kind: str              # "fact" | "nonsense"
    prompt: str
    answer: Optional[str]  # "A"|"B"|"C"|"D" for fact; None for nonsense
    meta: Dict


def _seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _strip_prompt_prefix(full_text: str, prompt: str) -> str:
    # обычно decode возвращает prompt+completion
    if full_text.startswith(prompt):
        return full_text[len(prompt):]
    return full_text


def _parse_letter_answer(text: str) -> Optional[str]:
    # берём первую букву A/B/C/D из completion
    m = LETTER_RE.search(text)
    if not m:
        return None
    return m.group(1).upper()


def _make_arc_prompt(question: str, choices: List[Tuple[str, str]]) -> str:
    """
    choices: list of (label, text), labels are expected A/B/C/D
    """
    lines = []
    lines.append("Выбери правильный вариант. Ответь ОДНОЙ буквой: A, B, C или D.")
    lines.append(f"Вопрос: {question.strip()}")
    lines.append("Варианты:")
    for lab, txt in choices:
        lines.append(f"{lab}) {txt.strip()}")
    lines.append("Ответ:")
    return "\n".join(lines)


def load_facts_arc(n: int, seed: int, split: str = "test", subset: str = "ARC-Easy") -> List[Example]:
    """
    Требует: pip install datasets
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("Нужен пакет datasets. Установи: pip install datasets") from e

    ds = load_dataset("ai2_arc", subset, split=split)
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    idxs = idxs[:n]

    out: List[Example] = []
    for i in idxs:
        ex = ds[i]
        q = ex["question"]
        # ai2_arc: choices = {"label":[...], "text":[...]} или list
        ch = ex["choices"]
        if isinstance(ch, dict):
            labels = ch["label"]
            texts = ch["text"]
            choices = list(zip(labels, texts))
        else:
            choices = [(c["label"], c["text"]) for c in ch]

        # сортируем по букве чтобы стабильно
        choices = sorted(choices, key=lambda t: t[0])
        # если вдруг не ABCD, отфильтруем и возьмём первые 4
        choices = [(l.upper(), t) for (l, t) in choices if l.upper() in ("A", "B", "C", "D")]
        if len(choices) < 4:
            continue

        ans = str(ex["answerKey"]).strip().upper()
        if ans not in ("A", "B", "C", "D"):
            continue

        prompt = _make_arc_prompt(q, choices[:4])
        out.append(Example(kind="fact", prompt=prompt, answer=ans, meta={"dataset": "ai2_arc", "subset": subset, "split": split, "id": i}))
        if len(out) >= n:
            break

    if len(out) < n:
        print(f"Warning: requested {n} examples, but only found {len(out)}. Proceeding.")
    return out


def gen_nonsense(n: int, seed: int) -> List[Example]:
    rng = random.Random(seed)
    subjects = ["луна", "марс", "динозавры", "кирпич", "невидимый единорог", "квадратный круг", "король океана", "пирамида из тумана"]
    actions = ["построили", "изобрели", "назначили", "съели", "продали", "сварили", "выбрали", "подписали"]
    objects = ["пирамиды", "интернет", "закон гравитации", "столицу", "квантовую лапшу", "время", "погоду", "тень"]
    years = ["в 1066", "в 3022", "в 0 году", "в -500", "в 99999", "вчера 31 февраля"]

    templates = [
        "Кто сейчас является {subject}?",
        "В каком году {subject} {action} {object} {year}?",
        "Сколько килограммов весит {object} в километрах?",
        "Как приготовить {subject} за 5 минут без огня и воды?",
        "Какая столица у {subject} и сколько у неё градусов Цельсия населения?",
    ]

    out: List[Example] = []
    for k in range(n):
        t = rng.choice(templates)
        prompt = t.format(
            subject=rng.choice(subjects),
            action=rng.choice(actions),
            object=rng.choice(objects),
            year=rng.choice(years),
        )
        out.append(Example(kind="nonsense", prompt=prompt, answer=None, meta={"id": k}))
    return out


def generate_baseline(model, tok, prompt: str, max_new_tokens: int) -> str:
    inp = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inp,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0], skip_special_tokens=True)


def generate_with_gtg(model, tok, prompt: str, max_new_tokens: int, gtg_kwargs: Dict) -> Tuple[str, bool, Dict]:
    lp, sc, proc = get_gtg_callbacks(tok, **gtg_kwargs)  # :contentReference[oaicite:5]{index=5}
    inp = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inp,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        logits_processor=lp,
        stopping_criteria=sc,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    refused = bool(getattr(proc, "triggered", False))
    details = {"triggered": refused, **(getattr(proc, "last_metrics", {}) or {})}
    return text, refused, details


def generate_with_triguard(model, tok, prompt: str, max_new_tokens: int, tg_kwargs: Dict) -> Tuple[str, bool, Dict]:
    tg_kwargs.pop('t_guard', None) # Clean up legacy arg if present
    
    lp, sc, proc = build_triguard(tokenizer, **tg_kwargs)
    # критически важно: прокинуть prompt внутрь процессора :contentReference[oaicite:7]{index=7}
    if hasattr(proc, "set_prompt"):
        proc.set_prompt(prompt)

    inp = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inp,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        logits_processor=lp,
        stopping_criteria=sc,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    refused = bool(getattr(getattr(proc, "state", None), "triggered", False))
    state = getattr(proc, "state", None)
    details = {
        "triggered": refused,
        "max_risk": float(getattr(state, "max_risk", 0.0)),
        "max_risk_metrics": getattr(state, "max_risk_metrics", {}) or {},
    }
    return text, refused, details


def compute_metrics(rows: List[Dict]) -> Dict:
    facts = [r for r in rows if r["label"] == "fact"]
    nons = [r for r in rows if r["label"] == "nonsense"]

    def agg(kind_rows):
        total = len(kind_rows)
        refused = sum(1 for r in kind_rows if r["status"] == "REFUSED")
        allowed = total - refused
        correct_allowed = sum(1 for r in kind_rows if r.get("correct_allowed") is True)
        incorrect_allowed = sum(1 for r in kind_rows if r.get("correct_allowed") is False)
        return total, refused, allowed, correct_allowed, incorrect_allowed

    f_total, f_refused, f_allowed, f_corr, f_inc = agg(facts)
    n_total, n_refused, n_allowed, _, _ = agg(nons)

    sr = (f_inc / f_allowed) if f_allowed > 0 else 0.0
    false_refusal = (f_refused / f_total) if f_total > 0 else 0.0
    refusal_nonsense = (n_refused / n_total) if n_total > 0 else 0.0
    acc_allowed = (f_corr / f_allowed) if f_allowed > 0 else 0.0

    return {
        "facts_total": f_total,
        "facts_allowed": f_allowed,
        "facts_refused": f_refused,
        "facts_false_refusal": false_refusal,
        "facts_acc_allowed": acc_allowed,
        "facts_selective_risk": sr,
        "nonsense_total": n_total,
        "nonsense_refused": n_refused,
        "nonsense_refusal_rate": refusal_nonsense,
        "nonsense_allowed": n_allowed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="путь/ID модели (например Qwen2.5-0.5B-Instruct или локальная папка)")
    ap.add_argument("--device", default=None, help="cuda|cpu (по умолчанию авто)")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--facts_n", type=int, default=300)
    ap.add_argument("--nonsense_n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--arc_subset", default="ARC-Easy", choices=["ARC-Easy", "ARC-Challenge"])
    ap.add_argument("--arc_split", default="test")
    ap.add_argument("--out_dir", default="out/bench")
    args = ap.parse_args()

    _seed_all(args.seed)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.dtype == "auto":
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch_dtype
    ).to(device).eval()

    facts = load_facts_arc(args.facts_n, args.seed, split=args.arc_split, subset=args.arc_subset)
    nons = gen_nonsense(args.nonsense_n, args.seed + 1)
    dataset = facts + nons

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "rows.csv")
    summary_path = os.path.join(args.out_dir, "summary.json")

    # Набор режимов (абляции)
    # Важно: tri_guard_processor по умолчанию использует только часть сигналов (веса) :contentReference[oaicite:8]{index=8}
    modes = [
        ("Baseline", {"type": "baseline"}),
        ("Simple-GTG", {"type": "gtg", "gtg_kwargs": dict(tau_entropy=0.60, tau_pmax=0.15, tau_margin=0.05, t_guard=0.2)}),
        ("Tri-Guard (Optimized)", {"type": "triguard", "tg_kwargs": dict(facts_mode=True, delta=0.75, eps=0.46, zeta=0.0)}),

        # Абляции Tri-Guard:
        ("TG_with_Tail", {"type": "triguard", "tg_kwargs": dict(facts_mode=True, delta=0.75, eps=0.46, zeta=0.5)}), 
        ("TG_tail_only", {"type": "triguard", "tg_kwargs": dict(facts_mode=True, delta=0.0, zeta=1.0, alpha=0.0, eps=0.0)}), 
        ("TG_no_critic", {"type": "triguard", "tg_kwargs": dict(facts_mode=True, eps=0.0, delta=0.75)}),
    ]

    all_summaries = {}

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode", "label", "status", "correct_allowed",
                "answer_gt", "answer_pred", "lat_ms",
                "prompt", "output",
                "details_json",
            ],
        )
        writer.writeheader()

        for mode_name, cfg in modes:
            rows = []
            t0_mode = time.perf_counter()

            for ex in dataset:
                prompt = ex.prompt

                t0 = time.perf_counter()
                if cfg["type"] == "baseline":
                    out_text = generate_baseline(model, tok, prompt, args.max_new_tokens)
                    refused = False
                    details = {}
                elif cfg["type"] == "gtg":
                    out_text, refused, details = generate_with_gtg(
                        model, tok, prompt, args.max_new_tokens, cfg["gtg_kwargs"]
                    )
                else:
                    out_text, refused, details = generate_with_triguard(
                        model, tok, prompt, args.max_new_tokens, cfg["tg_kwargs"]
                    )
                lat_ms = (time.perf_counter() - t0) * 1000.0

                full = out_text
                completion = _strip_prompt_prefix(full, prompt)
                pred = _parse_letter_answer(completion)

                if refused:
                    status = "REFUSED"
                    correct_allowed = None
                else:
                    status = "ALLOWED"
                    if ex.kind == "fact":
                        correct_allowed = (pred == ex.answer)
                    else:
                        correct_allowed = None  # nonsense correctness не считаем

                row = {
                    "mode": mode_name,
                    "label": ex.kind,
                    "status": status,
                    "correct_allowed": correct_allowed,
                    "answer_gt": ex.answer,
                    "answer_pred": pred,
                    "lat_ms": round(lat_ms, 2),
                    "prompt": prompt,
                    "output": completion.strip(),
                    "details_json": json.dumps(details, ensure_ascii=False),
                }
                rows.append(row)
                writer.writerow(row)

            mode_metrics = compute_metrics(rows)
            mode_metrics["avg_latency_ms"] = round(((time.perf_counter() - t0_mode) / max(1, len(dataset))) * 1000.0, 2)
            all_summaries[mode_name] = mode_metrics

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    # Печать таблицы
    def fmt(p): return f"{p*100:6.2f}%"
    header = (
        "Mode".ljust(14)
        + " | FalseRefusal(fact) | SR(fact, allowed) | Acc(fact, allowed) | Refusal(nonsense) | AvgLat"
    )
    print(header)
    print("-" * len(header))
    for mode_name, m in all_summaries.items():
        print(
            mode_name.ljust(14)
            + " | "
            + fmt(m["facts_false_refusal"]).rjust(17)
            + " | "
            + fmt(m["facts_selective_risk"]).rjust(16)
            + " | "
            + fmt(m["facts_acc_allowed"]).rjust(17)
            + " | "
            + fmt(m["nonsense_refusal_rate"]).rjust(15)
            + " | "
            + (str(m["avg_latency_ms"]) + "ms").rjust(7)
        )

    print("\nSaved:")
    print(" -", csv_path)
    print(" -", summary_path)


if __name__ == "__main__":
    main()
