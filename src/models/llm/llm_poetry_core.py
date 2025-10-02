from src.models.poetry.poetry_word import PoetryWord
import re


class LlmPoetryCore:
    special_words: list[PoetryWord]
    names: list[str]
    text_separ: str

    def with_special_words(self, words: list[PoetryWord]):
        self.special_words = words
        return self

    def with_names(self, names: list[str]):
        self.names = names
        return self

    def fresh(self):
        self.special_words = []
        self.names = []
        return self

    def split_chunk_to_paragraphs(self, chunk: str) -> list[str]:
        separ_sign = self.text_separ[0]
        if separ_sign in chunk:
            chunk = re.sub(re.escape(separ_sign) + '+', separ_sign, chunk)
            return [paragraph.strip() for paragraph in chunk.split(separ_sign) if paragraph.strip()]
        else:
            return [paragraph.strip() for paragraph in chunk.splitlines() if paragraph.strip()]

    @staticmethod
    def correct_chunk_paragraphs(chunk: list[str], correct: int) -> list[str]:
        if len(chunk) > correct:
            joined = chunk[:correct - 1]
            joined.append(' '.join(chunk[correct - 1:]))
            chunk = joined
        elif len(chunk) < correct:
            chunk.extend(['PARAGRAPH REQUIRES MANUAL EDITING FROM THE TOP'] * (correct - len(chunk)))
        return chunk