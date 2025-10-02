from src.models.poetry.poetry_word import PoetryWord
from src.models.services.prompt.abstract_prompt import AbstractPromptService


class PromptServiceEngToSlavic(AbstractPromptService):
    def __init__(self, slavic_lang: str = 'Russian'):
        self.slavic_lang = slavic_lang

    def prompt(self, text: str, special_words: list[PoetryWord], names: list[str]) -> list[dict]:
        return self.prompt_by_lang(text, special_words, names, self.slavic_lang)

    @classmethod
    def prompt_by_lang(cls, text: str, special_words: list[PoetryWord], names: list[str], slavic_lang: str = 'Russian') -> list[dict]:
        single_paragraph = cls.is_single_paragraph(text)
        system_content = (
            f'You are a professional translator from English into {slavic_lang}.\n' +
            cls.rule_separators(single_paragraph) +
            cls.rule_improve(single_paragraph) +
            cls.rule_special_words(special_words, 'The text contains special context words; when translating them, take into account singular and plural forms, but translate them strictly according to this list:') +
            cls.rule_names(names) +
            f'Translate from English into {slavic_lang}.\n'
            f'Answer only in {slavic_lang}, and only translated text.'
        )
        user_content = 'Translate:\n' + text

        return cls.prompt_structure(system_content, user_content)

    @staticmethod
    def min_max_multiplicator() -> tuple[float, float]:
        return 0.2, 1.8