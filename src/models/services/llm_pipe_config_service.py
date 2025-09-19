from src.middleware.pipe import LlmPipeMiddleware

class LlmPipeConfigService:
    @staticmethod
    def get_config() -> dict:
        the_config = getattr(LlmPipeConfigService, LlmPipeMiddleware.get_config())
        return the_config()

    @staticmethod
    def strict(calculate_level: int = 1, no_repeat_size: int = 3) -> dict:
        return {
            'temperature': None,
            'top_p': None,
            'top_k': None,
            'do_sample': False,
            'num_beams': calculate_level,
            'no_repeat_ngram_size': no_repeat_size,
        }