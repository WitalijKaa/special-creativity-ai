from src.models.llm.ai_provider_base import AiProviderBase
from src.models.services.prompt.eng_to_slavic import PromptServiceEngToSlavic
from src.models.services.prompt.slavic_to_eng import PromptServiceSlavicToEng


class AiProvider(AiProviderBase):
    def translate_rus_to_eng(self, text: list[str]) -> list[str]:
        self.init_ai()
        self.prompter = PromptServiceSlavicToEng
        return self.translate_paragraphs(text)

    def translate_eng_to_rus(self, text: list[str]) -> list[str]:
        self.init_ai()
        self.prompter = PromptServiceEngToSlavic
        return self.translate_paragraphs(text)
