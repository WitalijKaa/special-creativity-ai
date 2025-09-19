from src.middleware.llm import LlmAiMiddleware
from src.middleware.pipe import LlmPipeMiddleware

def web_llm_middleware():
    LlmPipeMiddleware.handle()
    LlmAiMiddleware.handle()