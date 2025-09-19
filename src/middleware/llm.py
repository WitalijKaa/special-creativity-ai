import config

ai_models = {
    'google': 'google/gemma-7b-it',
    'meta': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'microsoft': 'microsoft/Phi-3.5-mini-instruct',
    'france': 'mistralai/Mistral-7B-Instruct-v0.3',
    'china': 'Qwen/Qwen2.5-7B-Instruct',
}

class LlmAiMiddleware:
    @staticmethod
    def handle(llm_id: str):
        config.llm_id = llm_id

    @staticmethod
    def get_config() -> str:
        return ai_models[config.llm_id]