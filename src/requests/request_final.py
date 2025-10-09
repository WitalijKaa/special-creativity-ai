from pydantic import BaseModel
from src.models.poetry.final_parts import FinalParts


class RequestFinal(BaseModel):
    content: FinalParts
