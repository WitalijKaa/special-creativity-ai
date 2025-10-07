from pydantic import BaseModel
import config
from src.models.poetry.poetry_word import PoetryWord


class Chapter(BaseModel):
    text: str
    special_words: list[PoetryWord]
    names: list[str]

    _portion_len: int = 1
    _portion_next_len: int = 4
    _portion_origin: None | list[str] = None
    _portion_redacted: list[str] = []

    def portion_params(self, portion_len: int, portion_next_len: int):
        self._portion_len = portion_len
        self._portion_next_len = portion_next_len

    def portion_of_text(self) -> list[str]:
        if self._portion_origin is None:
            self._portion_origin = self.raw_text_to_paragraphs(self.text)
        portion_ix = len(self._portion_redacted)
        return self._portion_origin[portion_ix:portion_ix + self._portion_len]

    def next_portion(self) -> list[str]:
        portion_ix = len(self._portion_redacted)
        return self._portion_origin[portion_ix + self._portion_len:portion_ix + self._portion_len + self._portion_len]

    def add_redacted(self, text: list[str]):
        self._portion_redacted.extend(text)

    def get_redacted_as_text(self, separator: str) -> str:
        return separator.join([' '.join(p.splitlines()).strip() for p in self._portion_redacted if p.strip()])

    def get_redacted(self) -> list[str]:
        return self._portion_redacted

    @staticmethod
    def raw_text_to_paragraphs(text: str) -> list[str]:
        return [p.strip() for p in text.split("\n") if p.strip()]

    @staticmethod
    def paragraphs_to_raw_text(translation: list[str]) -> str:
        translation = [' '.join(p.splitlines()).strip() for p in translation if p.strip()]
        return "\n".join(translation)