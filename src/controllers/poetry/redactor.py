from src.models.llm.llm_factory import LlmFactory
from src.models.poetry.chapter import Chapter
from src.models.poetry.final_parts import FinalParts


def action_make_text_better(chapter: Chapter, separator: str) -> str:
    llm = LlmFactory.llm().with_special_words(chapter.special_words).with_names(chapter.names)
    llm.make_text_better(chapter)
    return chapter.get_redacted_as_text(separator)

def action_make_final_text(parts: FinalParts) -> str:
    llm = LlmFactory.llm().with_special_words(parts.special_words).with_names(parts.names)
    return llm.make_final_text(parts)