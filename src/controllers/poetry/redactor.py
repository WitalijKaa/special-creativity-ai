from src.models.llm.llm_factory import LlmFactory
from src.models.poetry.chapter import Chapter

def action_make_text_better(chapter: Chapter, separator: str) -> str:
    llm = LlmFactory.llm().with_special_words(chapter.special_words).with_names(chapter.names)
    while len(chapter.portion_of_text()) > 0:
        paragraphs = chapter.portion_of_text()
        redacted = llm.make_text_better(paragraphs, chapter.next_portion())
        chapter.add_redacted(redacted)
    return chapter.get_redacted(separator)
