# return (min_answer_vs_question_length, max_answer_vs_question_length, [
# min_length, max_length, prompt =

from src.middleware.llm import LlmAiMiddleware

class PromptService:
    @staticmethod
    def prompt_translate_slavic_to_english(text: str) -> tuple[float, float, list[dict]]:
        single_paragraph = 2 > len(text.split(LlmAiMiddleware.text_separ()))

        rule_separator = '' if single_paragraph else 'Strictly keep every separator ' + LlmAiMiddleware.text_separ() + ' in the translation exactly as in the source text. Never change the separator itself.\n'
        rule_improve = 'Translate in a literary manner, using the style of realism and naturalism. Stay maximally close to the original text, but revise and improve any parts that are clearly weak or poorly written.\n' if single_paragraph else 'Translate in a literary manner, using the style of realism and naturalism. Stay maximally close to the original text, but revise and improve any parts that are clearly weak or poorly written, strictly doing so only within each block delimited by separator ' + LlmAiMiddleware.text_separ() + ' and never mixing text across different blocks (even small blocks). Never break the rule for separator ' + LlmAiMiddleware.text_separ() + ' during the process of improving the text.\n'
        return (0.4, 1.8, [
            {
                'role': 'user',
                'content': (
                    'You are a professional translator from Slavic languages into English.\n'
                    + rule_separator + rule_improve +
                    'Translate from Russian into English.\n'
                    'Answer only in English, and only translated text.\n'
                    'Translate:\n' + text
                ),
            },
        ])

    @staticmethod
    def prompt_translate_english_to_russian(text: str) -> tuple[float, float, list[dict]]:
        return (0.4, 1.5, [
            {
                'role': 'user',
                'content': (
                    'You are a professional translator from English into Russian.'
                    'Translate in a literary manner, using the style of realism and naturalism. Stay maximally close to the original text, but revise and improve any parts that are clearly weak or poorly written.'
                    'Strictly preserve the paragraph structure of the original text. Do not merge paragraphs and do not create new ones.'
                    'Translate from English into Russian.'
                    'Answer only in Russian.'
                    'Translate: ' + text
                ),
            },
        ])
