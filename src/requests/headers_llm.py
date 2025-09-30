from pydantic import BaseModel
from src.models.llm.llm_local_models import LlmModelNick


class HeadersLlm(BaseModel):
    llm_nick: LlmModelNick
    llm_pipe: str