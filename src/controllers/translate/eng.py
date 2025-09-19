from src.models.llm.ai_provider import AiProvider

ai = AiProvider()

def action_translate_eng(text: str) -> str:
    return ai.translate_rus_to_eng(text)

def action_translate_rus(text: str) -> str:
    return ai.translate_eng_to_rus(text)
