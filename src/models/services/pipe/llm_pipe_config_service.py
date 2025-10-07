from src.middleware.pipe import LlmPipeMiddleware
from src.models.services.pipe.pipe_params import LlmPipeParams

# TOT USED !!!!

class LlmPipeConfigService:
    @staticmethod
    def get_config() -> LlmPipeParams:
        return LlmPipeMiddleware.configurator()
