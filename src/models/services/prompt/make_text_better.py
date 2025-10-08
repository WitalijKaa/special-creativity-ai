from src.models.poetry.poetry_word import PoetryWord
from src.models.services.prompt.abstract_prompt import AbstractPromptService


class PromptServiceMakeTextBetter(AbstractPromptService):

    def prompt(self, text: str, special_words: list[PoetryWord], names: list[str]) -> list[dict]:
        return self.prompt_structure(self.system_prompt(text, special_words, names, self._next_context), self.user_prompt(text))

    @classmethod
    def system_prompt(cls, text: str, special_words: list[PoetryWord], names: list[str], next_context: list[str]) -> str:
        return (
            'You are a professional redactor of text. Help a aspiring writer by editing and rewriting their text as if it were written by the great writer Jack London. Maintain a realistic and naturalistic style. Change anything you deem necessary to make great story, but keep the idea of the original script.\n' + '\n' +
            cls.rule_separators(text) + '\n' +
            cls.rule_names(names) + '\n' +
            cls.rule_special_words_list(special_words, 'The science-fiction contains special concepts; do not change it:') + '\n' +
            cls.rule_next_context(next_context) + '\n' + '\n' +
            'Answer only in English and only with improved text, no additional comments allowed.'
        )

    @classmethod
    def user_prompt(cls, text: str) -> str:
        return 'Improve my script:\n' + text

    @staticmethod
    def min_max_multiplicator() -> tuple[float, float]:
        return 0.7, 3.5
