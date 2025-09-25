from src.middleware.llm import LlmModelMiddleware


class AbstractTranslateService:
    @staticmethod
    def prompt(text: str, special_words: list[tuple[str, str]], names: list[str]) -> list[dict]:
        raise NotImplementedError('AbstractTranslateService must implement prompt()')

    @staticmethod
    def min_max_multiplicator() -> tuple[float, float]:
        raise NotImplementedError('AbstractTranslateService must implement min_max_multiplicator()')

    @staticmethod
    def must_use_only_user_role() -> bool:
        return LlmModelMiddleware.must_use_only_user_role()

    @staticmethod
    def is_single_paragraph(text: str) -> bool:
        return 2 > len(text.split(LlmModelMiddleware.text_separ()))

    @staticmethod
    def rule_separators(single_paragraph: bool) -> str:
        if not single_paragraph:
            return ('Strictly keep every separator ' + LlmModelMiddleware.text_separ() + ' in the translation exactly as in the source text. '
                    'Never change the separator itself.\n')
        return ''

    @staticmethod
    def rule_improve(single_paragraph: bool) -> str:
        if single_paragraph:
            return ('Translate in a literary manner, using the style of realism and naturalism. '
                    'Stay maximally close to the original text, but revise and improve any parts that are clearly weak or poorly written.\n')
        return ('Translate in a literary manner, using the style of realism and naturalism. '
                'Stay maximally close to the original text, but revise and improve any parts that are clearly weak or poorly written, '
                'strictly doing so only within each block delimited by separator ' + LlmModelMiddleware.text_separ() + ' and never mixing text across different blocks (even small blocks). '
                'Never break the rule for separator ' + LlmModelMiddleware.text_separ() + ' during the process of improving the text.\n')

    @staticmethod
    def rule_special_words(special_words: list[tuple[str, str]], prefix: str) -> str:
        if not len(special_words):
            return ''
        return prefix + '\n' + '\n'.join([f"{from_word} -> {to_word}" for from_word, to_word in special_words]) + '\n'

    @staticmethod
    def rule_names(names: list[str]) -> str:
        if not len(names):
            return ''
        return ('These names appear in the text; consider it as just names:\n'
                + '\n'.join([f"{name}" for name in names]) + '\n')

    @classmethod
    def prompt_structure(cls, system_content: str, user_content: str) -> list[dict]:
        if cls.must_use_only_user_role():
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