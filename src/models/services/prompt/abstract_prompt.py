from src.middleware.llm import LlmModelMiddleware
from src.models.poetry.poetry_word import PoetryWord


class AbstractPromptService:
    _next_context: list[str] = []

    def with_next_context(self, context: list[str]):
        self._next_context = context

    def prompt(self, text: str, special_words: list[PoetryWord], names: list[str]) -> list[dict]|dict:
        raise NotImplementedError('AbstractTranslateService must implement prompt()')

    @staticmethod
    def min_max_multiplicator() -> tuple[float, float]:
        raise NotImplementedError('AbstractTranslateService must implement min_max_multiplicator()')

    @staticmethod
    def must_use_api_structure() -> bool:
        return LlmModelMiddleware.is_api_llm()

    @staticmethod
    def must_use_only_user_role() -> bool:
        return LlmModelMiddleware.must_use_only_user_role()

    @staticmethod
    def is_single_paragraph(text: str) -> bool:
        return 2 > len(text.split(LlmModelMiddleware.text_separ()))

    @staticmethod
    def rule_separators(text: str) -> str:
        if not AbstractPromptService.is_single_paragraph(text):
            return 'Strictly keep every separator ' + LlmModelMiddleware.text_separ() + ' in the translation exactly as in the source text.\n'
        return ''

    @staticmethod
    def rule_translation_quality() -> str:
        return ('Translate in a literary manner, using the style of realism and naturalism. '
                'Stay maximally close to the original text, but revise and improve any parts that are clearly weak or poorly written.\n')

    @staticmethod
    def rule_special_words(special_words: list[PoetryWord], prefix: str) -> str:
        if not len(special_words):
            return ''
        return prefix + '\n' + '\n'.join([f"{word.slavic} -> {word.english}" for word in special_words]) + '\n'

    @staticmethod
    def rule_special_words_slavic(special_words: list[PoetryWord], prefix: str) -> str:
        if not len(special_words):
            return ''
        return prefix + '\n' + '\n'.join([f"{word.english} -> {word.slavic}" for word in special_words]) + '\n'

    @staticmethod
    def rule_special_words_list(special_words: list[PoetryWord], prefix: str) -> str:
        if not len(special_words):
            return ''
        return prefix + '\n' + '\n'.join([f"{word.english}" for word in special_words]) + '\n'

    @staticmethod
    def rule_special_words_meaning(special_words: list[PoetryWord], prefix: str) -> str:
        if not len(special_words):
            return ''
        return prefix + '\n' + '\n'.join([f"{word.english} - {word.definition}" for word in special_words]) + '\n'

    @staticmethod
    def rule_names(names: list[str]) -> str:
        if not len(names):
            return ''
        return ('These names appear in the text; consider it as just names:\n'
                + '\n'.join([f"{name}" for name in names]) + '\n')

    @staticmethod
    def rule_next_context(paragraphs: list[str]) -> str:
        if not len(paragraphs):
            return ''
        return ('Here is what happens after the part of the story you need to improve. Just use it as context, do not improve it and do not respond with this text:\n'
                + '\n'.join([f"{text}" for text in paragraphs]) + '\n')

    @classmethod
    def prompt_structure(cls, system_content: str, user_content: str) -> list[dict]|dict:
        if cls.must_use_api_structure():
            return {
                'instructions': system_content,
                'input': user_content,
            }
        elif cls.must_use_only_user_role():
            return [
                {
                    'role': 'user',
                    'content': system_content + '\n\n' + user_content,
                },
            ]
        else:
            return [
                {
                    'role': 'system',
                    'content': system_content,
                },
                {
                    'role': 'user',
                    'content': user_content,
                },
            ]