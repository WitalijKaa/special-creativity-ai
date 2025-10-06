from src.models.poetry.poetry_word import PoetryWord
from src.models.services.prompt.abstract_prompt import AbstractPromptService


class PromptServiceMakeTextBetter(AbstractPromptService):
    _next_context: list[str]

    def with_next_context(self, text: list[str]):
        self._next_context = text

    def prompt(self, text: str, special_words: list[PoetryWord], names: list[str]) -> list[dict]:
        return self.prompt_of_redactor(text, names, self._next_context)

    @classmethod
    def prompt_of_redactor(cls, text: str, names: list[str], next_context: list[str]) -> list[dict]:
        return cls.prompt_structure(cls.system_prompt(text, names, next_context), cls.user_prompt(text))

    @classmethod
    def system_prompt(cls, text: str, names: list[str], next_context: list[str]) -> str:
        single_paragraph = cls.is_single_paragraph(text)
        return (
            'You are a professional redactor of text. Help a aspiring writer by editing and rewriting their text as if it were written by the great writer Jack London. Maintain a realistic and naturalistic style. Change anything you deem necessary to make great story, but keep the idea of the original script.\n' +
            cls.rule_separators_soft(single_paragraph) +
            cls.rule_names(names) +
            cls.rule_next_context(next_context) +
            'Answer only in English.'
        )

    @classmethod
    def user_prompt(cls, text: str) -> str:
        return 'Translate:\n' + text

    @staticmethod
    def min_max_multiplicator() -> tuple[float, float]:
        return 0.7, 3.5
