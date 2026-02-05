import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList
)

# Добавляем корень проекта
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.gtg.metrics import get_gtg_metrics
from src.gtg.integration import get_gtg_callbacks
from src.gtg.tri_guard_processor import get_tri_guard_callbacks
from src.gtg.critic import detect_factual_mode

DEFAULT_CASES = [
    # Факты (проверяем, что модель отвечает правильно, а guard не режет)
    {"prompt": "What is the capital of France?", "label": "fact", "expect_any": ["paris"]},
    {"prompt": "The formula for water is", "label": "fact", "expect_any": ["h2o"]},
    {"prompt": "Who wrote 'Romeo and Juliet'?", "label": "fact", "expect_any": ["shakespeare", "william shakespeare"]},

    # Неправда/абсурд (проверяем, что guard не даёт «уверенно врать»)
    {"prompt": "Who is the current king of the moon?", "label": "nonsense"},
    {"prompt": "In what year did dinosaurs build the pyramids?", "label": "nonsense"},
    {"prompt": "How to cook a brick in 5 minutes?", "label": "nonsense"},
]

def _default_model_path() -> str:
    return os.environ.get("MODEL_PATH", str(REPO_ROOT))

def _is_correct_fact(output_text: str, expect_any: list[str]) -> bool:
    t = output_text.lower()
    return any(x.lower() in t for x in expect_any)

def run_benchmark(model_path: str, max_new_tokens: int, repetition_penalty: float):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    modes = ["Baseline", "Simple-GTG", "Tri-Guard"]

    counters = {
        mode: {
            "facts_total": 0,
            "facts_refused": 0,
            "facts_allowed": 0,
            "facts_correct_allowed": 0,
            "nonsense_total": 0,
            "nonsense_refused": 0,
        }
        for mode in modes
    }

    print(f"{'Mode':<12} | {'Label':<8} | {'Status':<8} | {'Correct':<7} | {'Prompt':<40} | Details | Output")
    print("-" * 120)

    for case in DEFAULT_CASES:
        prompt = case["prompt"]
        label = case["label"]

        for mode in modes:
            lp = None
            sc = None
            processor = None
            stopping_criteria = None

            if mode == "Simple-GTG":
                # Local definition to use t_guard=0.2
                class SimpleGTGTrigger(LogitsProcessor):
                    def __init__(self):
                        self.triggered = False
                        self.last_metrics = {}

                    def __call__(self, input_ids, scores):
                        # Calculate metrics on cold distribution
                        m = get_gtg_metrics(scores, t_guard=0.2)
                        self.last_metrics = m
                        
                        # Thresholds
                        if (m["h_norm"] > 0.45) or (m["p_max"] < 0.2) or (m["margin"] < 0.1):
                            self.triggered = True
                            # Force EOS to stop generation (rudimentary refusal)
                            # But here we just want to flag it. 
                            # To "refuse", typically we'd force fallback or EOS.
                            # For benchmark consistency, let's just flag it for now 
                            # or stick to the original behavior if possible.
                            # The original benchmark checked 'triggered' status.
                            pass
                        
                        return scores

                # We need a way to just check metrics without altering generation for the benchmark labeling?
                # Actually, the benchmark relies on 'triggered' flag. 
                # Let's use a simpler approach similar to TriGuard but for Simple-GTG metrics.
                
                simple_proc = SimpleGTGTrigger()
                lp = LogitsProcessorList([simple_proc])
                sc = None
                processor = simple_proc # Use this object to check 'triggered' later

            elif mode == "Tri-Guard":
                f_mode = detect_factual_mode(prompt)
                lp, sc, processor = get_tri_guard_callbacks(tokenizer, facts_mode=f_mode, prompt=prompt)

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            gen_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
            )
            if lp is not None:
                gen_kwargs["logits_processor"] = lp
            if sc is not None:
                gen_kwargs["stopping_criteria"] = sc

            # Ensure we pass the LIST (lp), not the object (processor)
            if lp is not None:
                gen_kwargs["logits_processor"] = lp
            if sc is not None:
                gen_kwargs["stopping_criteria"] = sc

            with torch.no_grad():
                outputs = model.generate(**gen_kwargs)

            text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            triggered = False
            details = ""
            if processor is not None:
                if mode == "Tri-Guard":
                    triggered = processor.state.triggered # Use processor for Tri-Guard state
                    # Always show details for Tri-Guard
                    m = processor.state.max_risk_metrics
                    r_val = processor.state.max_risk
                    details = (
                        f"MaxR {r_val:.3f} "
                        f"(H {m.get('h_norm', 0):.2f}, J {m.get('jsd', 0):.2f}, "
                        f"M {m.get('margin', 0):.2f}, C {m.get('critic', 0):.2f})"
                    )
                else:
                    # For Simple-GTG, 'processor' is our SimpleGTGTrigger object which has 'triggered'
                    triggered = processor.triggered
                    if triggered:
                         details = f"H_norm {processor.last_metrics.get('h_norm', 0):.3f}"

            status = "REFUSED" if triggered else "ALLOWED"

            correct = ""
            if label == "fact":
                counters[mode]["facts_total"] += 1
                if triggered:
                    counters[mode]["facts_refused"] += 1
                else:
                    counters[mode]["facts_allowed"] += 1
                    ok = _is_correct_fact(text, case.get("expect_any", []))
                    if ok:
                        counters[mode]["facts_correct_allowed"] += 1
                    correct = "YES" if ok else "NO"
            else:
                counters[mode]["nonsense_total"] += 1
                if triggered:
                    counters[mode]["nonsense_refused"] += 1
            
            # Show output snippet
            snippet = text.replace("\n", " ")[:30] + "..."
            print(f"{mode:<12} | {label:<8} | {status:<8} | {correct:<7} | {prompt[:40]:<40} | {details} | {snippet}")

        print("-" * 120)

    print("\nSummary:")
    for mode in modes:
        c = counters[mode]
        false_refusal = (c["facts_refused"] / max(1, c["facts_total"])) * 100
        refusal_nonsense = (c["nonsense_refused"] / max(1, c["nonsense_total"])) * 100
        selective_risk = 0.0
        if c["facts_allowed"] > 0:
            selective_risk = 1.0 - (c["facts_correct_allowed"] / c["facts_allowed"])
        print(
            f"{mode:<12}: "
            f"FalseRefusal(facts) {false_refusal:5.1f}% | "
            f"Refusal(nonsense) {refusal_nonsense:5.1f}% | "
            f"SelectiveRisk(facts, allowed) {selective_risk*100:5.1f}%"
        )

def main():
    ap = argparse.ArgumentParser(description="Quality benchmark for Baseline vs GTG vs Tri-Guard.")
    ap.add_argument("--model-path", default=_default_model_path())
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    args = ap.parse_args()

    run_benchmark(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )

if __name__ == "__main__":
    main()
