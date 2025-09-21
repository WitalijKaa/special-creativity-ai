from src.models.llm.ai_provider import AiProvider
ai = AiProvider()

def raw_text_to_llm(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n") if p.strip()]
def translation_to_raw_text(translation: list[str]) -> str:
    translation = [p.strip() for p in translation if p.strip()]
    return "\n".join(translation)

def action_translate_eng(text: str) -> str:
    translation = ai.translate_rus_to_eng(raw_text_to_llm(text))
    return translation_to_raw_text(translation)

def action_translate_rus(text: str) -> str:
    translation = ai.translate_eng_to_rus(raw_text_to_llm(text))
    return translation_to_raw_text(translation)
