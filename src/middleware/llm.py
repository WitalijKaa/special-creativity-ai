import config

ai_models = {
    'meta': {
        'id': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'max_tokens': 1950,
    },
    'china': {
        'id': 'Qwen/Qwen2.5-7B-Instruct',
        'max_tokens': 2050,
    },
    'france': {
        'id': 'mistralai/Mistral-7B-Instruct-v0.3',
        'max_tokens': 2600,
    },
    'google': {
        'id': 'google/gemma-7b-it',
        'max_tokens': 1800,
        'only_user_role': True,
    },
    'microsoft': {
        'id': 'microsoft/Phi-3.5-mini-instruct',
        'max_tokens': 1800,
    },
}

class LlmModelMiddleware:
    @staticmethod
    def handle(llm_id: str):
        config.llm_id = llm_id

    @staticmethod
    def get_config() -> str:
        return ai_models[config.llm_id]['id']

    @staticmethod
    def text_separ() -> str:
        return '#'

    @staticmethod
    def max_tokens() -> str:
        return ai_models[config.llm_id]['max_tokens']

    @staticmethod
    def must_use_only_user_role() -> bool:
        return ai_models[config.llm_id].get('only_user_role') is True