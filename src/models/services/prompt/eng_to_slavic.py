from src.models.poetry.poetry_word import PoetryWord
from src.models.services.prompt.abstract_prompt import AbstractPromptService


class PromptServiceEngToSlavic(AbstractPromptService):
    def __init__(self, slavic_lang: str = 'Russian'):
        self.slavic_lang = slavic_lang

    def prompt(self, text: str, special_words: list[PoetryWord], names: list[str]) -> list[dict]:
        return self.prompt_structure(self.system_prompt(special_words, names, self.slavic_lang), self.user_prompt(text))

    @classmethod
    def system_prompt(cls, special_words: list[PoetryWord], names: list[str], slavic_lang: str = 'Russian') -> str:
        return (
            f'You are a professional translator of science-fiction from English into {slavic_lang}.\n' + '\n' +
            cls.rule_translation_quality() + '\n' +
            cls.rule_special_words(special_words, 'The text contains special concepts; translate them strictly according to this list:') + '\n' +
            cls.rule_names(names) + '\n' + '\n' +
            f'Translate from English into {slavic_lang}.\n'
            f'Answer only in {slavic_lang}, and only translated text, no additional comments allowed.'
        )

    @classmethod
    def user_prompt(cls, text: str) -> str:
        return 'Translate one paragraph:\n' + text

    @staticmethod
    def min_max_multiplicator() -> tuple[float, float]:
        return 0.2, 2.5