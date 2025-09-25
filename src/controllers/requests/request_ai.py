from pydantic import BaseModel

class RequestAi(BaseModel):
    llm: str
    pipe: str