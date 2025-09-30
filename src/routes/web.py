from typing import Annotated
from fastapi import FastAPI, Header, Request, HTTPException
from src.controllers.poetry.redactor import action_make_text_better
from src.middleware.llm import LlmModelMiddleware
from src.middleware.pipe import LlmPipeMiddleware
from src.models.llm.llm_local_models import LlmModelNick
from src.requests.headers_llm import HeadersLlm
from src.requests.request_improve import RequestImprove
from src.controllers.poetry.translate import action_translate_rus, action_translate_eng, action_translate_ukr, action_translate_srb
from src.requests.request_translate import RequestTranslate
from src.responses.response_paragraphs import ResponseParagraphs

def web_routes_init(app: FastAPI):

    @app.middleware("http")
    async def middleware(request: Request, call_next):
        if request.headers.get('Llm-Nick') is not None:
            LlmModelMiddleware.handle(LlmModelNick(request.headers.get('Llm-Nick')))
            LlmPipeMiddleware.handle(request.headers.get('Llm-Pipe'))
        return await call_next(request)

    @app.post("/translate_eng")
    async def route_translate_eng(request: RequestTranslate, headers: Annotated[HeadersLlm, Header()]) -> ResponseParagraphs:
        return ResponseParagraphs(response=action_translate_eng(request.content))

    @app.post("/translate_rus")
    async def route_translate_rus(request: RequestTranslate, headers: Annotated[HeadersLlm, Header()]) -> ResponseParagraphs:
        return ResponseParagraphs(response=action_translate_rus(request.content))

    @app.post("/translate_ukr")
    async def route_translate_rus(request: RequestTranslate, headers: Annotated[HeadersLlm, Header()]) -> ResponseParagraphs:
        return ResponseParagraphs(response=action_translate_ukr(request.content))

    @app.post("/translate_srb")
    async def route_translate_rus(request: RequestTranslate, headers: Annotated[HeadersLlm, Header()]) -> ResponseParagraphs:
        return ResponseParagraphs(response=action_translate_srb(request.content))

    @app.post("/make_text_better")
    async def route_translate_eng(request: RequestImprove, headers: Annotated[HeadersLlm, Header()]) -> ResponseParagraphs:
        return ResponseParagraphs(response=action_make_text_better(request.content, request.separator))
