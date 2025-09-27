from src.models.llm.ai_provider import AiProvider
from src.models.poetry.chapter import Chapter

ai = AiProvider()

def raw_text_to_llm(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n") if p.strip()]
def translation_to_raw_text(translation: list[str]) -> str:
    translation = [p.strip() for p in translation if p.strip()]
    return "\n".join(translation)

def action_translate_eng(chapter: Chapter) -> str:
    translation = ai.with_special_words(chapter.special_words).with_names(chapter.names).translate_rus_to_eng(raw_text_to_llm(chapter.text))
    return translation_to_raw_text(translation)

def action_translate_rus(chapter: Chapter) -> str:
    translation = ai.with_special_words(chapter.special_words).with_names(chapter.names).translate_eng_to_rus(raw_text_to_llm(chapter.text))
    return translation_to_raw_text(translation)

def action_translate_ukr(chapter: Chapter) -> str:
    translation = ai.with_special_words(chapter.special_words).with_names(chapter.names).translate_eng_to_ukr(raw_text_to_llm(chapter.text))
    return translation_to_raw_text(translation)

def action_translate_srb(chapter: Chapter) -> str:
    translation = ai.with_special_words(chapter.special_words).with_names(chapter.names).translate_eng_to_srb(raw_text_to_llm(chapter.text))
    return translation_to_raw_text(translation)
