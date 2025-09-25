import config
from src.models.services.pipe.pipe_params import LlmPipeParams


class LlmPipeMiddleware:
    @staticmethod
    def handle(pipe_config: str):
        config.llm_mode = pipe_config

    @staticmethod
    def get_config() -> LlmPipeParams:
        model = LlmPipeParams()
        model.by_mode(config.llm_mode)
        return model