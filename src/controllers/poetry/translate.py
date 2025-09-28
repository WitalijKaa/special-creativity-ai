from src.models.llm.llm_poetry_provider import LlmProvider
from src.models.poetry.chapter import Chapter


def action_translate_eng(chapter: Chapter) -> str:
    translation = LlmProvider().fresh().with_special_words(chapter.special_words).with_names(chapter.names).translate_rus_to_eng(Chapter.raw_text_to_paragraphs(chapter.text))
    return Chapter.paragraphs_to_raw_text(translation)

def action_translate_rus(chapter: Chapter) -> str:
    translation = LlmProvider().fresh().with_special_words(chapter.special_words).with_names(chapter.names).translate_eng_to_rus(Chapter.raw_text_to_paragraphs(chapter.text))
    return Chapter.paragraphs_to_raw_text(translation)

def action_translate_ukr(chapter: Chapter) -> str:
    translation = LlmProvider().fresh().translate_eng_to_ukr(Chapter.raw_text_to_paragraphs(chapter.text))
    return Chapter.paragraphs_to_raw_text(translation)

def action_translate_srb(chapter: Chapter) -> str:
    translation = LlmProvider().fresh().translate_eng_to_srb(Chapter.raw_text_to_paragraphs(chapter.text))
    return Chapter.paragraphs_to_raw_text(translation)
