from typing import Any


class PromptService:
    @staticmethod
    def prompt_translate_slavic_to_english(text: str) -> list[dict]:
        return [
            {
                'role': 'user',
                'content': (
                    'You are a professional translator from Slavic languages into English.'
                    'You translate strictly according to the text and do not make anything up.'
                    'You must translate from Russian into English.'
                    'You answer only in English.'
                    'Text for translation: ' + text
                ),
            },
        ]
