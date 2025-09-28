from pydantic import BaseModel


class PoetryWord(BaseModel):
    slavic: str
    english: str
    definition: str

    def translation(self) -> tuple[str, str]:
        return self.slavic, self.english