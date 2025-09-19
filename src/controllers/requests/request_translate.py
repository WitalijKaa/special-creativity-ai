from pydantic import BaseModel

class RequestTranslate(BaseModel):
    text: str