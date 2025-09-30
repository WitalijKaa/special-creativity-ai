from pydantic import BaseModel
from src.models.poetry.chapter import Chapter


class RequestImprove(BaseModel):
    content: Chapter
    separator: str
