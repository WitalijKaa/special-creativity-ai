from src.middleware.llm import LlmModelMiddleware
from src.models.llm.llm_gpt import LlmGpt
from src.models.llm.llm_video_card import LlmVideoCard


class LlmFactory:
    _singleton_local: LlmVideoCard | None = None
    _singleton_api: LlmGpt | None = None

    @classmethod
    def llm(cls) -> LlmGpt | LlmVideoCard:
        if LlmModelMiddleware.is_api_llm():
            if cls._singleton_api is None:
                cls._singleton_api = LlmGpt()
                cls._singleton_api.init_provider()
            return cls._singleton_api.fresh()

        if cls._singleton_local is None:
            cls._singleton_local = LlmVideoCard()
            cls._singleton_local.init_provider()
        return cls._singleton_local.fresh()