from pydantic import BaseModel
from src.models.llm.llm_local_models import LlmModelNick


class RequestAi(BaseModel):
    llm: LlmModelNick
    pipe: str