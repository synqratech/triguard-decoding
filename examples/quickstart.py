import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from triguard import build_triguard

def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)

    # 1. Build the Guard
    # "tail_only" is the recommended preset for general Q&A
    print("Initializing Tri-Guard (Tail-Only preset)...")
    lp, sc, _ = build_triguard(
        tokenizer, 
        preset="tail_only",
        fallback_text=" [Tri-Guard: Uncertainty detected. Falling back.]"
    )

    # 2. Test Cases
    questions = [
        "What is the capital of France?",      # Fact (Should pass)
        "Who is the King of the Moon?",        # Nonsense (Should be blocked)
        "What is the formula for water?",      # Fact (Should pass)
    ]

    print("\n--- Running Inference ---\n")
    for q in questions:
        inputs = tokenizer(q, return_tensors="pt").to(model.device)
        
        # Standard generation + Tri-Guard
        out = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            logits_processor=lp,
            stopping_criteria=sc,
            pad_token_id=tokenizer.eos_token_id
        )
        
        answer = tokenizer.decode(out[0], skip_special_tokens=True)
        # Strip prompt for cleaner output
        answer_only = answer[len(q):].strip()
        
        print(f"Q: {q}")
        print(f"A: {answer_only}")
        print("-" * 30)

if __name__ == "__main__":
    main()
