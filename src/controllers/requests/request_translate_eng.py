from pydantic import BaseModel

class RequestTranslateEng(BaseModel):
    text: str