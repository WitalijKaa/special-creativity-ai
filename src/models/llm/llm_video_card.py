from src.models.llm.llm_poetry_core import LlmPoetryCore
from src.models.llm.llm_video_card_base import LlmVideoCardBase
from src.models.services.prompt.abstract_prompt import AbstractPromptService
from src.models.services.prompt.eng_to_slavic import PromptServiceEngToSlavic
from src.models.services.prompt.make_text_better import PromptServiceMakeTextBetter
from src.models.services.prompt.slavic_to_eng import PromptServiceSlavicToEng


class LlmVideoCard(LlmPoetryCore, LlmVideoCardBase):
    prompter: AbstractPromptService | None

    def make_text_better(self, text: list[str], text_next: list[str]) -> list[str]:
        self.init_llm()
        self.prompter = PromptServiceMakeTextBetter()
        self.prompter.with_next_context(text_next)
        chunk = self.improve_paragraphs(text)
        chunk = self.correct_chunk_paragraphs(chunk, len(text))
        return chunk

    def translate_rus_to_eng(self, text: list[str]) -> list[str]:
        self.init_llm()
        self.prompter = PromptServiceSlavicToEng()
        return self.translate_paragraphs(text)

    def translate_eng_to_rus(self, text: list[str]) -> list[str]:
        self.init_llm()
        self.prompter = PromptServiceEngToSlavic()
        return self.translate_paragraphs(text)

    def translate_eng_to_ukr(self, text: list[str]) -> list[str]:
        self.init_llm()
        self.prompter = PromptServiceEngToSlavic('Ukrainian')
        return self.translate_paragraphs(text)

    def translate_eng_to_srb(self, text: list[str]) -> list[str]:
        self.init_llm()
        self.prompter = PromptServiceEngToSlavic('Serbian')
        return self.translate_paragraphs(text)
