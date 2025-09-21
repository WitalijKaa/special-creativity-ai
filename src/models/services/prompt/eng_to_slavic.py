# return (min_answer_vs_question_length, max_answer_vs_question_length, [
# min_length, max_length, prompt =

from src.middleware.llm import LlmAiMiddleware
from src.models.services.prompt.abstract_prompt import AbstractTranslateService


class PromptServiceEngToSlavic(AbstractTranslateService):
    @staticmethod
    def prompt(text: str) -> tuple[float, float, list[dict]]:
        single_paragraph = AbstractTranslateService.is_single_paragraph(text)
        return (0.5, 1.8, [
            {
                'role': 'user',
                'content': (
                    'You are a professional translator from English into Russian.\n' +
                    AbstractTranslateService.rule_separators(single_paragraph) +
                    AbstractTranslateService.rule_improve(single_paragraph) +
                    'Translate from English into Russian.\n'
                    'Answer only in Russian, and only translated text.\n'
                    'Translate:\n' + text
                ),
            },
        ])
