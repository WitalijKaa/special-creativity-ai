from pydantic import BaseModel
from src.models.poetry.poetry_word_llm import PoetryWordLlm


class Chapter(BaseModel):
    text: str
    special_words: list[PoetryWordLlm]
    names: list[str]