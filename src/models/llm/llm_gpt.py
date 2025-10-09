from src.middleware.auth import AuthMiddleware
from src.middleware.llm import LlmModelMiddleware
from openai import OpenAI

from src.middleware.pipe import LlmPipeMiddleware
from src.models.basic_logger import aLog
from src.models.llm.llm_poetry_core import LlmPoetryCore
from src.models.poetry.chapter import Chapter
from src.models.poetry.final_parts import FinalParts
from src.models.services.prompt.abstract_prompt import AbstractPromptService
from src.models.services.prompt.eng_to_slavic import PromptServiceEngToSlavic
from src.models.services.prompt.make_text_better import PromptServiceMakeTextBetter
from src.models.services.prompt.slavic_final_text import PromptServiceFinalSlavicText
from src.models.services.prompt.slavic_to_eng import PromptServiceSlavicToEng


from src.helpers.php_wk import *

class LlmGpt(LlmPoetryCore):
    model_id: str
    token: str
    client_gpt: OpenAI

    prompter: PromptServiceFinalSlavicText | PromptServiceMakeTextBetter | PromptServiceSlavicToEng | PromptServiceEngToSlavic

    text_separ: str

    def init_provider(self):
        self.token = AuthMiddleware.open_ai_token()
        self.client_gpt = OpenAI(api_key=self.token)
        self.model_id = LlmModelMiddleware.get_config()
        self.text_separ = LlmModelMiddleware.text_separ() * 4

        self.special_words = []
        self.names = []

    def make_final_text(self, parts: FinalParts) -> str:
        self.prompter = PromptServiceFinalSlavicText().special_words(self.special_words).names(self.names)
        aLog.debug(f'GPT parts combination for {[parts.part_original]}')
        prompt = self.prompter.prompt(parts.part_alpha, parts.part_beta, parts.part_emotional, parts.part_original)
        final = self.request_api(prompt, 1)
        return ' '.join(final)

    def make_text_better(self, chapter: Chapter):
        self.prompter = PromptServiceMakeTextBetter()
        text = Chapter.raw_text_to_paragraphs(chapter.text)
        aLog.debug(f'GPT improve text {len(text)} Start')
        prompt = self.prompter.prompt(self.prompter.prompt(self.text_separ.join(text), self.special_words, self.names))
        redacted = self.request_api(prompt, len(text))
        chapter.add_redacted(redacted)

    def translate_rus_to_eng(self, text: list[str]) -> list[str]:
        self.prompter = PromptServiceSlavicToEng()
        aLog.debug(f'GPT Translate to eng {len(text)} Start')
        prompt = self.prompter.prompt(self.prompter.prompt(self.text_separ.join(text), self.special_words, self.names))
        return self.request_api(prompt, len(text))

    def translate_eng_to_rus(self, text: list[str]) -> list[str]:
        self.prompter = PromptServiceEngToSlavic()
        aLog.debug(f'GPT Translate to rus {len(text)} Start')
        prompt = self.prompter.prompt(self.prompter.prompt(self.text_separ.join(text), self.special_words, self.names))
        return self.request_api(prompt, len(text))

    def translate_eng_to_ukr(self, text: list[str]) -> list[str]:
        self.prompter = PromptServiceEngToSlavic('Ukrainian')
        prompt = self.prompter.prompt(self.prompter.prompt(self.text_separ.join(text), self.special_words, self.names))
        return self.request_api(prompt, len(text))

    def translate_eng_to_srb(self, text: list[str]) -> list[str]:
        self.prompter = PromptServiceEngToSlavic('Serbian')
        prompt = self.prompter.prompt(self.prompter.prompt(self.text_separ.join(text), self.special_words, self.names))
        return self.request_api(prompt, len(text))

    def request_config(self) -> dict:
        piping = LlmPipeMiddleware.configurator()
        config = {
            'model': self.model_id,
            'service_tier': 'flex',
            'timeout': 3000.0,
        }
        config.update(piping.config())
        return config

    def request_api(self, prompt: dict, correct_length: int) -> list[str]:
        config = self.request_config()
        config.update(prompt)
        response = self.client_gpt.responses.create(**config)
        aLog.debug(f'GPT RESPONSE PARTS {response}')
        chunk = self.split_chunk_to_paragraphs(response.output_text)
        chunk = self.correct_chunk_paragraphs(chunk, correct_length)
        return chunk
