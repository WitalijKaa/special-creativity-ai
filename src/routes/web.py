from fastapi import FastAPI

from src.controllers.poetry.redactor import action_make_text_better
from src.controllers.requests.request_improve import RequestImprove
from src.middleware.group.web_llm_middleware import web_llm_middleware
from src.controllers.poetry.translate import action_translate_rus, action_translate_eng, action_translate_ukr, action_translate_srb
from src.controllers.requests.request_translate import RequestTranslate

def web_routes_init(app: FastAPI):
    @app.post("/translate_eng")
    async def route_translate_eng(request: RequestTranslate):
        web_llm_middleware(request)
        return {"response": action_translate_eng(request.content)}

    @app.post("/translate_rus")
    async def route_translate_rus(request: RequestTranslate):
        web_llm_middleware(request)
        return {"response": action_translate_rus(request.content)}

    @app.post("/translate_ukr")
    async def route_translate_rus(request: RequestTranslate):
        web_llm_middleware(request)
        return {"response": action_translate_ukr(request.content)}

    @app.post("/translate_srb")
    async def route_translate_rus(request: RequestTranslate):
        web_llm_middleware(request)
        return {"response": action_translate_srb(request.content)}

    @app.post("/make_text_better")
    async def route_translate_eng(request: RequestImprove):
        web_llm_middleware(request)
        return {"response": action_make_text_better(request.content, request.separator)}
