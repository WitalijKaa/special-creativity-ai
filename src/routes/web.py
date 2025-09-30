from fastapi import FastAPI

from src.controllers.poetry.redactor import action_make_text_better
from src.requests.request_improve import RequestImprove
from src.middleware.group.web_llm_middleware import web_llm_middleware
from src.controllers.poetry.translate import action_translate_rus, action_translate_eng, action_translate_ukr, action_translate_srb
from src.requests.request_translate import RequestTranslate
from src.responses.response_paragraphs import ResponseParagraphs

from src.models.basic_logger import aLog
def web_routes_init(app: FastAPI):

    # :( need to put config to headers or similar
    # @app.middleware("http")
    # async def middleware(request: RequestAi, call_next):
    #     web_llm_middleware(request)
    #     return await call_next(request)

    @app.post("/translate_eng")
    async def route_translate_eng(request: RequestTranslate) -> ResponseParagraphs:
        web_llm_middleware(request)
        return ResponseParagraphs(response=action_translate_eng(request.content))

    @app.post("/translate_rus")
    async def route_translate_rus(request: RequestTranslate) -> ResponseParagraphs:
        web_llm_middleware(request)
        return ResponseParagraphs(response=action_translate_rus(request.content))

    @app.post("/translate_ukr")
    async def route_translate_rus(request: RequestTranslate) -> ResponseParagraphs:
        web_llm_middleware(request)
        return ResponseParagraphs(response=action_translate_ukr(request.content))

    @app.post("/translate_srb")
    async def route_translate_rus(request: RequestTranslate) -> ResponseParagraphs:
        web_llm_middleware(request)
        return ResponseParagraphs(response=action_translate_srb(request.content))

    @app.post("/make_text_better")
    async def route_translate_eng(request: RequestImprove) -> ResponseParagraphs:
        web_llm_middleware(request)
        return ResponseParagraphs(response=action_make_text_better(request.content, request.separator))
