import os
import sys

# Добавляем корень проекта в путь поиска модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.gtg.integration import get_gtg_callbacks

def calibrate():
    model_path = os.environ.get("MODEL_PATH", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device).eval()

    # Датасет для калибровки
    # GOOD: факты, которые модель должна знать
    # BAD: галлюцинации или абсурдные вопросы
    dataset = [
        {"prompt": "What is the capital of France?", "type": "GOOD"},
        {"prompt": "2 + 2 =", "type": "GOOD"},
        {"prompt": "Who is the current king of the moon?", "type": "BAD"},
        {"prompt": "The formula for water is", "type": "GOOD"},
        {"prompt": "In what year did humans land on Mars?", "type": "BAD"},
    ]

    # Сетки порогов для поиска
    entropy_grids = [0.5, 0.6, 0.7, 0.8]
    margin_grids = [0.01, 0.03, 0.05, 0.1]

    print(f"{'Entropy':<10} | {'Margin':<10} | {'Refusal (GOOD)':<15} | {'Refusal (BAD)':<15}")
    print("-" * 60)

    for eth in entropy_grids:
        for mth in margin_grids:
            refused_good = 0
            refused_bad = 0
            
            for item in dataset:
                lp, sc, proc = get_gtg_callbacks(
                    tokenizer, 
                    tau_entropy=eth, 
                    tau_margin=mth,
                    tau_pmax=0.0 # Отключим pmax временно для упрощения
                )
                
                inputs = tokenizer(item["prompt"], return_tensors="pt").to(device)
                
                # Отключаем сэмплирование и предупреждения
                model.generate(
                    **inputs,
                    max_new_tokens=20,
                    logits_processor=lp,
                    stopping_criteria=sc,
                    do_sample=False,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                if proc.triggered:
                    if item["type"] == "GOOD": refused_good += 1
                    else: refused_bad += 1
            
            print(f"{eth:<10} | {mth:<10} | {refused_good:<15} | {refused_bad:<15}")

if __name__ == "__main__":
    calibrate()
