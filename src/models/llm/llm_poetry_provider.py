from src.models.llm.llm_provider_base import LlmProviderBase
from src.models.poetry.poetry_word import PoetryWord
from src.models.services.prompt.eng_to_slavic import PromptServiceEngToSlavic
from src.models.services.prompt.make_text_better import PromptServiceMakeTextBetter
from src.models.services.prompt.slavic_to_eng import PromptServiceSlavicToEng


class LlmProvider(LlmProviderBase):
    def with_special_words(self, words: list[PoetryWord]):
        self.special_words = [word.translation() for word in words]
        return self

    def with_names(self, names: list[str]):
        self.names = names
        return self

    def fresh(self):
        self.special_words = []
        self.names = []
        return self

    def make_text_better(self, text: list[str], text_next: list[str]) -> list[str]:
        self.init_llm()
        self.prompter = PromptServiceMakeTextBetter()
        self.prompter.with_next_context(text_next)
        return self.improve_paragraphs(text)

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

    _singleton = None
    def __new__(cls):
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
            cls._singleton.init_provider()
        return cls._singleton
