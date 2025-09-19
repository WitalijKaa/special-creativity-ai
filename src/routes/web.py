from fastapi import FastAPI

from src.middleware.group.web_llm_middleware import web_llm_middleware

from src.controllers.translate.eng import *
from controllers.requests.request_translate import RequestTranslate

def web_routes_init(app: FastAPI):
    @app.post("/translate_eng")
    async def route_translate_eng(body: RequestTranslate):
        web_llm_middleware()
        return {"response": action_translate_eng(body.text)}

    @app.post("/translate_rus")
    async def route_translate_rus(body: RequestTranslate):
        web_llm_middleware()
        return {"response": action_translate_rus(body.text)}