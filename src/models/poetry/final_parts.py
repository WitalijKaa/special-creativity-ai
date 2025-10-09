from pydantic import BaseModel
from src.models.poetry.poetry_word import PoetryWord


class FinalParts(BaseModel):
    part_alpha: str
    part_beta: str
    part_emotional: str | None
    part_original: str
    special_words: list[PoetryWord]
    names: list[str]
