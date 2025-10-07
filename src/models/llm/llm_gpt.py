from src.middleware.auth import AuthMiddleware
from src.middleware.llm import LlmModelMiddleware
from openai import OpenAI

from src.middleware.pipe import LlmPipeMiddleware
from src.models.basic_logger import aLog
from src.models.llm.llm_poetry_core import LlmPoetryCore
from src.models.poetry.chapter import Chapter
from src.models.services.prompt.abstract_prompt import AbstractPromptService
from src.models.services.prompt.eng_to_slavic import PromptServiceEngToSlavic
from src.models.services.prompt.make_text_better import PromptServiceMakeTextBetter
from src.models.services.prompt.slavic_to_eng import PromptServiceSlavicToEng


from src.helpers.php_wk import *

class LlmGpt(LlmPoetryCore):
    model_id: str
    token: str
    client_gpt: OpenAI

    prompter: AbstractPromptService | None

    text_separ: str

    def init_provider(self):
        self.token = AuthMiddleware.open_ai_token()
        self.client_gpt = OpenAI(api_key=self.token)
        self.model_id = LlmModelMiddleware.get_config()
        self.text_separ = LlmModelMiddleware.text_separ() * 4

        self.special_words = []
        self.names = []

    def make_text_better(self, chapter: Chapter):
        self.prompter = PromptServiceMakeTextBetter()
        redacted = self.request_paragraphs(Chapter.raw_text_to_paragraphs(chapter.text))
        chapter.add_redacted(redacted)

    def translate_rus_to_eng(self, text: list[str]) -> list[str]:
        self.prompter = PromptServiceSlavicToEng()
        return self.request_paragraphs(text)

    def translate_eng_to_rus(self, text: list[str]) -> list[str]:
        self.prompter = PromptServiceEngToSlavic()
        return self.request_paragraphs(text)

    def translate_eng_to_ukr(self, text: list[str]) -> list[str]:
        self.prompter = PromptServiceEngToSlavic('Ukrainian')
        return self.request_paragraphs(text)

    def translate_eng_to_srb(self, text: list[str]) -> list[str]:
        self.prompter = PromptServiceEngToSlavic('Serbian')
        return self.request_paragraphs(text)

    def request_config(self) -> dict:
        piping = LlmPipeMiddleware.configurator()
        config = {
            'model': self.model_id,
            'service_tier': 'flex',
            'timeout': 1500.0,
        }
        config.update(piping.config())
        return config # text={"verbosity": "low"}, # how long the answer should be

    def request_paragraphs(self, text: list[str]) -> list[str]:
        aLog.debug(f'GPT paragraphs {len(text)} Start')
        config = self.request_config()
        config.update(self.prompter.prompt(self.text_separ.join(text), self.special_words, self.names))
        response = self.client_gpt.responses.create(**config)
        aLog.debug(f'GPT RESPONSE {response}')
        chunk = self.split_chunk_to_paragraphs(response.output_text)
        aLog.debug(f'GPT RESPONSE chunk ( {len(text)} -> {len(chunk)} ) {chunk}')
        chunk = self.correct_chunk_paragraphs(chunk, len(text))
        return chunk
