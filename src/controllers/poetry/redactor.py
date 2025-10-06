from src.models.llm.llm_factory import LlmFactory
from src.models.poetry.chapter import Chapter

def action_make_text_better(chapter: Chapter, separator: str) -> str:
    llm = LlmFactory.llm().with_special_words(chapter.special_words).with_names(chapter.names)
    llm.make_text_better(chapter)
    return chapter.get_redacted(separator)