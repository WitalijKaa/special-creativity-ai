from pydantic import BaseModel
from src.models.poetry.chapter import Chapter


class RequestTranslate(BaseModel):
    content: Chapter
