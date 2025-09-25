from src.models.services.prompt.abstract_prompt import AbstractTranslateService


class PromptServiceEngToSlavic(AbstractTranslateService):
    @classmethod
    def prompt(cls, text: str, special_words: list[tuple[str, str]], names: list[str]) -> list[dict]:
        single_paragraph = cls.is_single_paragraph(text)
        system_content = (
            'You are a professional translator from English into Russian.\n' +
            cls.rule_separators(single_paragraph) +
            cls.rule_improve(single_paragraph) +
            cls.rule_special_words(special_words, 'The text contains special context words; when translating them, take into account singular and plural forms, but translate them strictly according to this list:') +
            cls.rule_names(names) +
            'Translate from English into Russian.\n'
            'Answer only in Russian, and only translated text.'
        )
        user_content = 'Translate:\n' + text

        return cls.prompt_structure(system_content, user_content)

    @staticmethod
    def min_max_multiplicator() -> tuple[float, float]:
        return 0.5, 1.8