# return (min_answer_vs_question_length, max_answer_vs_question_length, [
# min_length, max_length, prompt =

class PromptService:
    @staticmethod
    def prompt_translate_slavic_to_english(text: str) -> tuple[float, float, list[dict]]:
        return (0.65, 1.5, [
            {
                'role': 'user',
                'content': (
                    'You are a professional translator from Slavic languages into English.'
                    'Translate in a literary manner, using the style of realism and naturalism. Stay maximally close to the original text, but revise and improve any parts that are clearly weak or poorly written.'
                    'Translate from Russian into English.'
                    'Answer only in English.'
                    'Translate: ' + text
                ),
            },
        ])

    @staticmethod
    def prompt_translate_english_to_russian(text: str) -> tuple[float, float, list[dict]]:
        return (0.65, 1.5, [
            {
                'role': 'user',
                'content': (
                    'You are a professional translator from English into Russian.'
                    'Translate in a literary manner, using the style of realism and naturalism. Stay maximally close to the original text, but revise and improve any parts that are clearly weak or poorly written.'
                    'Translate from English into Russian.'
                    'Answer only in Russian.'
                    'Translate: ' + text
                ),
            },
        ])
