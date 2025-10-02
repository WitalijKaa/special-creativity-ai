from src.models.poetry.poetry_word import PoetryWord
from src.models.services.prompt.abstract_prompt import AbstractPromptService


class PromptServiceSlavicToEng(AbstractPromptService):
    def prompt(self, text: str, special_words: list[PoetryWord], names: list[str]) -> list[dict]:
        return self.prompt_by_lang(text, special_words, names)

    @classmethod
    def system_prompt(cls, text: str, special_words: list[PoetryWord], names: list[str]) -> str:
        single_paragraph = cls.is_single_paragraph(text)
        return (
            'You are a professional translator from Slavic languages into English.\n' +
            cls.rule_separators(single_paragraph) +
            cls.rule_improve(single_paragraph) +
            cls.rule_special_words(special_words, 'The text contains new Slavic words; when translating them, take into account singular and plural forms, but translate them strictly according to this list:') +
            cls.rule_names(names) +
            'Translate from Russian into English.\n'
            'Answer only in English, and only translated text.'
        )

    @classmethod
    def user_prompt(cls, text: str) -> str:
        return 'Translate:\n' + text

    @classmethod
    def prompt_by_lang(cls, text: str, special_words: list[PoetryWord], names: list[str]) -> list[dict]:
        return cls.prompt_structure(cls.system_prompt(text, special_words, names), cls.user_prompt(text))

    @staticmethod
    def min_max_multiplicator() -> tuple[float, float]:
        return 0.5, 1.5
