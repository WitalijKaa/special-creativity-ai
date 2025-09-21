# return (min_answer_vs_question_length, max_answer_vs_question_length, [
# min_length, max_length, prompt =

from src.middleware.llm import LlmAiMiddleware
from src.models.services.prompt.abstract_prompt import AbstractTranslateService


class PromptServiceSlavicToEng(AbstractTranslateService):
    @staticmethod
    def prompt(text: str) -> tuple[float, float, list[dict]]:
        single_paragraph = AbstractTranslateService.is_single_paragraph(text)
        return (0.5, 1.8, [
            {
                'role': 'user',
                'content': (
                    'You are a professional translator from Slavic languages into English.\n' +
                    AbstractTranslateService.rule_separators(single_paragraph) +
                    AbstractTranslateService.rule_improve(single_paragraph) +
                    'Do not modify any names written using English letters.\n'
                    'Translate from Russian into English.\n'
                    'Answer only in English, and only translated text.\n'
                    'Translate:\n' + text
                ),
            },
        ])
