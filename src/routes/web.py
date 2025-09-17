from fastapi import FastAPI
from pydantic import BaseModel

from src.controllers.translate.eng import *
from src.controllers.requests.request_translate_eng import RequestTranslateEng

def web_routes_init(app: FastAPI):

    @app.post("/translate_eng")
    async def route_translate_eng(body: RequestTranslateEng):
        return action_translate_eng(body.text)
