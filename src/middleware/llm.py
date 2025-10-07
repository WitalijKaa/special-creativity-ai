import config
from src.models.llm.llm_model_nick import LlmModelNick

ai_models = {
    LlmModelNick.gpt_best: {
        'vs_api': True,
        'id': 'gpt-5',
    },
    LlmModelNick.gpt_mini: {
        'vs_api': True,
        'id': 'gpt-5-mini',
    },
    LlmModelNick.gpt_nano: {
        'vs_api': True,
        'id': 'gpt-5-nano',
    },

#gpt-5-nano
    LlmModelNick.meta: {
        'id': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'max_tokens': 1950,
    },
    LlmModelNick.china: {
        'id': 'Qwen/Qwen2.5-7B-Instruct',
        'max_tokens': 2050,
    },
    LlmModelNick.france: {
        'id': 'mistralai/Mistral-7B-Instruct-v0.3',
        'max_tokens': 2600,
    },
    LlmModelNick.google: {
        'id': 'google/gemma-7b-it',
        'max_tokens': 1800,
        'only_user_role': True,
    },
    LlmModelNick.microsoft: {
        'id': 'microsoft/Phi-3.5-mini-instruct',
        'max_tokens': 1800,
    },
    LlmModelNick.britain: { # NOT WORKING !!!! # https://huggingface.co/stabilityai/StableBeluga-7B
        'id': 'stabilityai/StableBeluga-7B',
        'max_tokens': 1800,

        # stabilityai/StableBeluga-13B
    },
    LlmModelNick.russia: { # https://huggingface.co/t-tech/T-lite-it-1.0
        'id': 't-tech/T-lite-it-1.0',
        'max_tokens': 1800,
    },
    LlmModelNick.europa: { # NOT WORKING !!!! # https://huggingface.co/utter-project/EuroLLM-9B-Instruct
        'id': 'utter-project/EuroLLM-9B-Instruct',
        'max_tokens': 1800,
    },
    LlmModelNick.slavic: {  # https://huggingface.co/IlyaGusev/saiga_llama3_8b
        'id': 'IlyaGusev/saiga_llama3_8b',
        'max_tokens': 1800,
    },
}

class LlmModelMiddleware:
    @staticmethod
    def handle(llm_id: LlmModelNick):
        config.llm_id = llm_id

    @staticmethod
    def is_api_llm() -> bool:
        return ai_models[config.llm_id].get('vs_api') is True

    @staticmethod
    def get_config() -> str:
        return ai_models[config.llm_id]['id']

    @staticmethod
    def text_separ() -> str:
        return '#'

    @staticmethod
    def max_tokens() -> int:
        return ai_models[config.llm_id]['max_tokens']

    @staticmethod
    def must_use_only_user_role() -> bool:
        return ai_models[config.llm_id].get('only_user_role') is True