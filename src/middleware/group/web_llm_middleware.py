from src.requests.request_ai import RequestAi
from src.middleware.llm import LlmModelMiddleware
from src.middleware.pipe import LlmPipeMiddleware

def web_llm_middleware(request: RequestAi):
    LlmModelMiddleware.handle(request.llm)
    LlmPipeMiddleware.handle(request.pipe)
