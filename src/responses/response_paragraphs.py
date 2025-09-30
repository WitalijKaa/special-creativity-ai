from pydantic import BaseModel


class ResponseParagraphs(BaseModel):
    response: str
