# return (min_answer_vs_question_length, max_answer_vs_question_length, [
# min_length, max_length, prompt =

from src.middleware.llm import LlmAiMiddleware
from src.models.services.prompt.abstract_prompt import AbstractTranslateService


class PromptServiceEngToSlavic(AbstractTranslateService):
    @staticmethod
    def prompt(text: str, special_words: list[tuple[str, str]], names: list[str]) -> tuple[float, float, list[dict]]:
        single_paragraph = AbstractTranslateService.is_single_paragraph(text)
        return (0.5, 1.8, [
            {
                'role': 'user',
                'content': (
                    'You are a professional translator from English into Russian.\n' +
                    AbstractTranslateService.rule_separators(single_paragraph) +
                    AbstractTranslateService.rule_improve(single_paragraph) +
                    AbstractTranslateService.rule_special_words(special_words, 'The text contains special context words; when translating them, take into account singular and plural forms, but translate them strictly according to this list:') +
                    AbstractTranslateService.rule_names(names) +
                    'Translate from English into Russian.\n'
                    'Answer only in Russian, and only translated text.\n'
                    'Translate:\n' + text
                ),
            },
        ])
