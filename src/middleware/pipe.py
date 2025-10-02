import config
from src.middleware.llm import LlmModelMiddleware
from src.models.services.pipe.pipe_gpt_params import LlmPipeGptParams
from src.models.services.pipe.pipe_params import LlmPipeParams


class LlmPipeMiddleware:
    @staticmethod
    def handle(pipe_config: str):
        config.llm_mode = pipe_config

    @staticmethod
    def get_config() -> LlmPipeParams | LlmPipeGptParams:
        if LlmModelMiddleware.is_api_llm():
            model = LlmPipeGptParams()
        else:
            model = LlmPipeParams()
        model.by_mode(config.llm_mode)
        return model