from src.middleware.pipe import LlmPipeMiddleware
from src.models.basic_logger import aLog
from src.models.llm.llm_poetry_core import LlmPoetryCore
from src.models.llm.llm_video_card_base import LlmVideoCardBase
from src.models.poetry.chapter import Chapter
from src.models.services.prompt.abstract_prompt import AbstractPromptService
from src.models.services.prompt.eng_to_slavic import PromptServiceEngToSlavic
from src.models.services.prompt.make_text_better import PromptServiceMakeTextBetter
from src.models.services.prompt.slavic_to_eng import PromptServiceSlavicToEng


class LlmVideoCard(LlmPoetryCore, LlmVideoCardBase):
    prompter: AbstractPromptService | None

    def make_text_better(self, chapter: Chapter):
        self.init_llm()
        chapter.portion_params(4, 4)
        while len(chapter.portion_of_text()) > 0:
            redacted = self.make_text_portion_better(chapter.portion_of_text(), chapter.next_portion())
            chapter.add_redacted(redacted)

    def make_text_portion_better(self, text: list[str], text_next: list[str]) -> list[str]:
        self.prompter = PromptServiceMakeTextBetter()
        self.prompter.with_next_context(text_next)
        chunk = self.improve_paragraphs(text)
        chunk = self.correct_chunk_paragraphs(chunk, len(text))
        return chunk

    def translate_rus_to_eng(self, chapter: Chapter) -> list[str]:
        self.init_llm()
        self.prompter = PromptServiceSlavicToEng()
        return self.translate_paragraphs_one_by_one(chapter)

    def translate_eng_to_rus(self, chapter: Chapter) -> list[str]:
        self.init_llm()
        self.prompter = PromptServiceEngToSlavic()
        return self.translate_paragraphs_one_by_one(chapter)

    def translate_eng_to_ukr(self, chapter: Chapter) -> list[str]:
        self.init_llm()
        self.prompter = PromptServiceEngToSlavic('Ukrainian')
        return self.translate_paragraphs_one_by_one(chapter)

    def translate_eng_to_srb(self, chapter: Chapter) -> list[str]:
        self.init_llm()
        self.prompter = PromptServiceEngToSlavic('Serbian')
        return self.translate_paragraphs_one_by_one(chapter)

    def translate_paragraphs_one_by_one(self, chapter: Chapter) -> list[str]:
        chapter.portion_params(1, 1)
        aLog.debug(f'LLM-local Translation one-by-one {len(Chapter.raw_text_to_paragraphs(chapter.text))} Start')
        while len(chapter.portion_of_text()) > 0:
            paragraph = ' '.join(chapter.portion_of_text())
            prompt = self.prompter.prompt(paragraph, self.special_words, self.names)
            piping = LlmPipeMiddleware.configurator()
            piping.calculate_tokens(self.count_tokens_of_text(paragraph), *self.prompter.min_max_multiplicator())
            chunk = self.answer_vs_prompt(prompt, piping.config())
            aLog.debug(f'CHUNK {chunk.split('\n')}')
            translated_paragraph = ' '.join(self.split_chunk_to_paragraphs(chunk))
            chapter.add_redacted([translated_paragraph])
        return chapter.get_redacted()