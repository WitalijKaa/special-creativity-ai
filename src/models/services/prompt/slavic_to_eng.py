from src.models.poetry.poetry_word import PoetryWord
from src.models.services.prompt.abstract_prompt import AbstractPromptService


class PromptServiceSlavicToEng(AbstractPromptService):

    def prompt(self, text: str, special_words: list[PoetryWord], names: list[str]) -> list[dict]:
        return self.prompt_structure(self.system_prompt(special_words, names), self.user_prompt(text))

    @classmethod
    def system_prompt(cls, special_words: list[PoetryWord], names: list[str]) -> str:
        return (
            'You are a professional translator of science-fiction from Slavic languages into English.\n' + '\n' +
            cls.rule_translation_quality() + '\n' +
            cls.rule_special_words(special_words, 'The text contains new Slavic words; translate them strictly according to this list:') + '\n' +
            cls.rule_names(names) + '\n' + '\n' +
            'Translate from Russian into English.\n'
            'Answer only in English, and only translated text.'
        )

    @classmethod
    def user_prompt(cls, text: str) -> str:
        return 'Translate one paragraph:\n' + text

    @staticmethod
    def min_max_multiplicator() -> tuple[float, float]:
        return 0.5, 1.5
