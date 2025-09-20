from src.models.llm.ai_provider import AiProvider
ai = AiProvider()

def action_translate_eng(text: str) -> str:
    text = [p.strip() for p in text.split("\n") if p.strip()]
    translation = ai.translate_rus_to_eng(text)
    translation = [p.strip() for p in translation if p.strip()]
    return "\n".join(translation)

def action_translate_rus(text: str) -> str:
    return ai.translate_eng_to_rus(text)
