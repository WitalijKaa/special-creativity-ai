from src.middleware.pipe import LlmPipeMiddleware

class LlmPipeConfigService:
    @staticmethod
    def get_config() -> dict:
        the_config = getattr(LlmPipeConfigService, LlmPipeMiddleware.get_config())
        return the_config()
