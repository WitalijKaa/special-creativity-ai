from src.middleware.llm import LlmAiMiddleware


class AbstractTranslateService:
    @staticmethod
    def prompt(text: str) -> tuple[float, float, list[dict]]:
        raise NotImplementedError('AbstractTranslateService must implement prompt()')

    @staticmethod
    def is_single_paragraph(text: str) -> bool:
        return 2 > len(text.split(LlmAiMiddleware.text_separ()))

    @staticmethod
    def rule_separators(single_paragraph: bool) -> str:
        if not single_paragraph:
            return ('Strictly keep every separator ' + LlmAiMiddleware.text_separ() + ' in the translation exactly as in the source text. '
                    'Never change the separator itself.\n')
        return ''

    @staticmethod
    def rule_improve(single_paragraph: bool) -> str:
        if single_paragraph:
            return ('Translate in a literary manner, using the style of realism and naturalism. '
                    'Stay maximally close to the original text, but revise and improve any parts that are clearly weak or poorly written.\n')
        return ('Translate in a literary manner, using the style of realism and naturalism. '
                'Stay maximally close to the original text, but revise and improve any parts that are clearly weak or poorly written, '
                'strictly doing so only within each block delimited by separator ' + LlmAiMiddleware.text_separ() + ' and never mixing text across different blocks (even small blocks). '
                'Never break the rule for separator ' + LlmAiMiddleware.text_separ() + ' during the process of improving the text.\n')
