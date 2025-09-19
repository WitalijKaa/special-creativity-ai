from src.models.llm.ai_provider import AiProvider

def action_translate_eng(text: str) -> str:
    ai = AiProvider()
    return ai.translate_rus_to_eng(text)
