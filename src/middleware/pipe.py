import config

class LlmPipeMiddleware:
    @staticmethod
    def handle():
        config.llm_mode = 'strict'

    @staticmethod
    def get_config() -> str:
        return config.llm_mode