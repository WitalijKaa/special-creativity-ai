from src.models.services.prompt.abstract_prompt import AbstractTranslateService


class PromptServiceSlavicToEng(AbstractTranslateService):
    @staticmethod
    def prompt(text: str, special_words: list[tuple[str, str]], names: list[str]) -> list[dict]:
        single_paragraph = AbstractTranslateService.is_single_paragraph(text)
        return [
            {
                'role': 'user',
                'content': (
                    'You are a professional translator from Slavic languages into English.\n' +
                    AbstractTranslateService.rule_separators(single_paragraph) +
                    AbstractTranslateService.rule_improve(single_paragraph) +
                    AbstractTranslateService.rule_special_words(special_words, 'The text contains new Slavic words; when translating them, take into account singular and plural forms, but translate them strictly according to this list:') +
                    AbstractTranslateService.rule_names(names) +
                    # 'Do not modify any names written using English letters.\n'
                    'Translate from Russian into English.\n'
                    'Answer only in English, and only translated text.\n'
                    'Translate:\n' + text
                ),
            },
        ]

    @staticmethod
    def min_max_multiplicator() -> tuple[float, float]:
        return 0.5, 1.8
