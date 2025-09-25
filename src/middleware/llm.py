import config

ai_models = {
    'google': {
        'id': 'google/gemma-7b-it',
        'max_tokens': 1100,
    },
    'meta': {
        'id': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'max_tokens': 1800,
    },
    'microsoft': {
        'id': 'microsoft/Phi-3.5-mini-instruct',
        'max_tokens': 1100,
    },
    'france': {
        'id': 'mistralai/Mistral-7B-Instruct-v0.3',
        'max_tokens': 1100,
    },
    'china': {
        'id': 'Qwen/Qwen2.5-7B-Instruct',
        'max_tokens': 1100,
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