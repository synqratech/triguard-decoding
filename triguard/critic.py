import re

SOURCE_CUES = [
    "по данным", "источник", "согласно", "википед", "doi", "arxiv", 
    "http", "https", "цитата", "документ", "reference", "source"
]

FACT_CUES = [
    "кто", "когда", "сколько", "в каком году", "столица", "население",
    "дата", "год", "capital", "population", "when", "how many", "who is",
    "year", "formula", "located", "is a"
]

def has_any_cue(text: str, cues: list) -> bool:
    text_lower = text.lower()
    return any(cue in text_lower for cue in cues)

def is_fact_atom(token_str: str) -> bool:
    """
    Проверяет, похож ли токен на часть факта (число, имя, единица измерения).
    Игнорирует общие стоп-слова, которые часто пишутся с большой буквы (например, в начале предложения).
    """
    # Убираем пробелы и знаки препинания для проверки
    clean_token = token_str.strip().strip(".,!?:;\"'")
    if not clean_token:
        return False
        
    # Стоп-слова (английские и русские), которые часто в начале предложения
    STOP_WORDS = {
        "The", "A", "An", "In", "On", "At", "To", "For", "With", "By",
        "This", "That", "It", "They", "He", "She", "We", "You",
        "Moscow", "Paris", "London" # Подождите, эти НЕ должны быть в стопе!
    }
    # Реальный список стоп-слов, которые НЕ являются "атомами" фактов сами по себе
    COMMON_NON_ATOM = {
        "The", "A", "An", "In", "On", "At", "By", "With", "From", "To",
        "This", "That", "It", "They", "We", "You", "He", "She",
        "And", "Or", "But", "Wait", "So", "Then",
        "Что", "Это", "Он", "Она", "Они", "Мы", "Вы", "Но", "А", "И", "В", "На"
    }
    
    if clean_token in COMMON_NON_ATOM:
        return False

    # Содержит цифры
    if re.search(r"\d", clean_token):
        return True
    
    # Содержит заглавные буквы (имена, названия)
    if re.search(r"[A-ZА-ЯЁ]", clean_token):
        return True
    
    # Специфические единицы измерения
    units = ["кг", "км", "млн", "млрд", "км/ч", "°", "%", "год", "лет", "year"]
    if clean_token.lower() in units:
        return True
        
    return False

def detect_factual_mode(prompt: str) -> bool:
    """
    Определяет, является ли вопрос фактологическим (Уровень 1 по tri-guard.md).
    """
    t = prompt.lower()
    score = 0

    # Вопросительный знак - сильный сигнал
    if "?" in t:
        score += 2
    
    # Поиск годов (19xx, 20xx)
    if re.search(r"\b(19\d{2}|20\d{2})\b", t):
        score += 2
        
    # Ключевые слова (объединенные)
    if any(c in t for c in FACT_CUES):
        score += 2
        
    # Наличие цифр
    if re.search(r"\d", t):
        score += 1

    # Порог снижен до 2 для большей чувствительности
    return score >= 2

def get_critic_score(prompt: str, generated_so_far: str, candidate_token: str) -> float:
    """
    Вычисляет оценку критика C в диапазоне [0, 1].
    C выше, если мы пытаемся выдать факт без опоры на источник.
    """
    full_context = prompt + " " + generated_so_far
    
    # Есть ли источники в контексте (промпте ИЛИ уже сгенерированном)?
    # FIX: Смотрим на full_context, так как модель могла придумать источник только что
    has_source_in_context = has_any_cue(full_context, SOURCE_CUES)
    
    # Является ли вопрос фактологическим?
    # FIX: Используем унифицированную функцию
    is_factual_question = detect_factual_mode(prompt)
    
    # Похож ли токен на конкретный факт?
    candidate_is_atom = is_fact_atom(candidate_token)
    
    # Новая логика "Zero-shot Tolerance":
    # Если в промпте нет источников (обычный вопрос), то мы не требуем их в ответе.
    # Но если источники ЕСТЬ, а модель выдает факт без ссылок - это высокий риск.
    
    if not is_factual_question:
        return 0.0

    if has_source_in_context:
        # Режим RAG: строгий контроль
        w_atom = 0.8
        w_base = 0.3
        score = w_atom if candidate_is_atom else w_base
    else:
        # Режим Zero-shot: практически не штрафуем за отсутствие источников,
        # так как их и не должно быть в контексте.
        # FIX: Поднимаем штраф, чтобы критик хоть как-то реагировал
        w_atom = 0.25  
        w_base = 0.0
        score = w_atom if candidate_is_atom else w_base
    
    return float(max(0.0, min(1.0, score)))
