from pydantic import BaseModel
from src.models.llm.llm_model_nick import LlmModelNick


class HeadersLlm(BaseModel):
    llm_nick: LlmModelNick
    llm_pipe: str