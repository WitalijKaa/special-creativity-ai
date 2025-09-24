from src.models.llm.ai_provider_base import AiProviderBase
from src.models.poetry.poetry_word_llm import PoetryWordLlm
from src.models.services.prompt.eng_to_slavic import PromptServiceEngToSlavic
from src.models.services.prompt.slavic_to_eng import PromptServiceSlavicToEng


class AiProvider(AiProviderBase):
    def with_special_words(self, words: list[PoetryWordLlm]):
        self.special_words = [word.translation() for word in words]
        return self

    def with_names(self, names: list[str]):
        self.names = names
        return self

    def translate_rus_to_eng(self, text: list[str]) -> list[str]:
        self.init_ai()
        self.prompter = PromptServiceSlavicToEng
        return self.translate_paragraphs(text)

    def translate_eng_to_rus(self, text: list[str]) -> list[str]:
        self.init_ai()
        self.prompter = PromptServiceEngToSlavic
        return self.translate_paragraphs(text)
