from fastapi import FastAPI

from src.middleware.group.web_llm_middleware import web_llm_middleware

from src.controllers.translate.eng import action_translate_eng
from src.controllers.requests.request_translate_eng import RequestTranslateEng

def web_routes_init(app: FastAPI):
    @app.post("/translate_eng")
    async def route_translate_eng(body: RequestTranslateEng):
        web_llm_middleware()
        return action_translate_eng(body.text)
