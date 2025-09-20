import config

ai_models = {
    'google': ['google/gemma-7b-it', 800],
    'meta': ['meta-llama/Meta-Llama-3.1-8B-Instruct', 1800],
    'microsoft': ['microsoft/Phi-3.5-mini-instruct', 800],
    'france': ['mistralai/Mistral-7B-Instruct-v0.3', 800],
    'china': ['Qwen/Qwen2.5-7B-Instruct', 800],
}

class LlmAiMiddleware:
    @staticmethod
    def handle(llm_id: str):
        config.llm_id = llm_id

    @staticmethod
    def get_config() -> str:
        return ai_models[config.llm_id][0]

    @staticmethod
    def text_separ() -> str:
        return '#'

    @staticmethod
    def max_tokens() -> str:
        return ai_models[config.llm_id][1]