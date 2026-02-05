# Tri-Guard: Inference-Time Abstention for Transformers

> **One-Liner:** Inference-time abstention guard for `transformers.generate()` that prevents confident nonsense via controlled fallback.

Tri-Guard serves as a "safety fuse" for small Language Models (SLMs), preventing them from hallucinating wildly when they don't know the answer. It uses a combination of **Tail-JSD**, **Entropy**, and **Internal Critics** to detect when a model is "making things up" and forces it to confusingly admit ignorance instead of lying confidently.

## 🚀 Features

* **Does:** Reduces the risk of confident hallucinations by checking generation stability (Split-Decoding) and token probability.
* **Does NOT:** Verify factual truth against a database or search engine. It relies on the model's own uncertainty signals.
* **Zero-Overhead:** Uses `Tail-JSD` signal which incurs minimal latency overhead compared to full sampling.

## ⚡ Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from triguard import build_triguard

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# 1. Build the Guard (using the 'tail_only' SOTA preset)
lp, sc, _ = build_triguard(
    tokenizer, 
    preset="tail_only",
    fallback_text="I am not sure about this fact."
)

# 2. Generate with protection
prompt = "Who is the King of Mars?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

out = model.generate(
    **inputs,
    max_new_tokens=50,
    logits_processor=lp,
    stopping_criteria=sc,
    do_sample=False # works best with greedy/low-temp
)

print(tokenizer.decode(out[0], skip_special_tokens=True))
# Output: "I am not sure about this fact." (instead of hallucinating a name)
```

## 🎛️ Presets

| Preset | Description | When to use |
| :--- | :--- | :--- |
| **`tail_only`** | **SOTA (Recommended).** Uses only `Tail-JSD` (divergence in the tail of distribution). Ignores noise between top candidates. | Best for Q&A, ARC, and reasoning tasks where the model might validly hesitate between two good options. |
| **`balanced`** | Uses JSD + Internal Critic. | Good starting point if `tail_only` is too aggressive. |
| **`strict`** | High penalty for any uncertainty. | Use for critical extractions where any doubt should trigger fallback. |

## 📊 Benchmarks (Qwen 0.5B)

Tested on **ARC-Easy** (Exact Match) and **Nonsense Questions** (Hallucination Trigger).

| Method | False Refusal (Facts) | True Refusal (Nonsense) |
| :--- | :--- | :--- |
| Baseline | 0% | 0% (Hallucinates 100%) |
| Simple-GTG | 51.4% | 39.5% |
| **Tri-Guard (Tail-Only)** | **18.1%** | **52.0%** |

*Running benchmarks:*

```bash
python scripts/benchmark_ablation.py --model "Qwen/Qwen2.5-0.5B-Instruct" --facts_n 300 --nonsense_n 200
```

*Results will be saved to `out/bench/`.*

## ⚠️ Limitations

1. **Small Models**: Designed and tuned for <3B parameter models. Larger models might require different thresholds ($\theta$).
2. **Late Refusal**: The guard triggers *during* generation. It stops the model after it sees the risk, which might be a few tokens in.
3. **Not a Fact Checker**: If the model is confidently wrong (low entropy, low JSD), Tri-Guard will **NOT** catch it.

## 📦 Installation

```bash
pip install -e .
```

## License

Apache 2.0
