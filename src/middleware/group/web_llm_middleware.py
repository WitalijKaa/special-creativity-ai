from src.controllers.requests.request_ai import RequestAi
from src.middleware.llm import LlmAiMiddleware
from src.middleware.pipe import LlmPipeMiddleware

def web_llm_middleware(request: RequestAi):
    LlmPipeMiddleware.handle()
    LlmAiMiddleware.handle(request.ai)