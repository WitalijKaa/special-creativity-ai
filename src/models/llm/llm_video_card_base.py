from src.helpers.php_wk import *
from src.middleware.pipe import LlmPipeMiddleware
from src.models.basic_logger import aLog
import gc
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, pipeline
from transformers import PreTrainedTokenizerFast
from transformers import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel

from src.middleware.auth import AuthMiddleware
from src.middleware.llm import LlmModelMiddleware
from src.models.services.prompt.abstract_prompt import AbstractPromptService


class LlmVideoCardBase:
    hf_auth: str
    model_id: str
    llm: PreTrainedModel | None
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None
    prompter: AbstractPromptService | None
    text_separ: str
    tokens_small_paragraph_max: int
    tokens_prefix: int
    tokens_maxima: int
    special_words: list[tuple[str, str]]
    names: list[str]

    def init_provider(self):
        self.hf_auth = AuthMiddleware.hugging_face_token()
        self.model_id = LlmModelMiddleware.get_config()
        self.llm = None
        self.tokenizer = None
        self.prompter = None
        self.text_separ = LlmModelMiddleware.text_separ() * 4
        self.tokens_small_paragraph_max = 20

        self.tokens_prefix = 0
        self.tokens_maxima = 42

        self.special_words = []
        self.names = []

    def improve_paragraphs(self, text: list[str]) -> list[str]:
        self.tokens_prefix = self.prompt_max_length()
        sum_tokens = sum([self.count_tokens_of_text(paragraph) for paragraph in text])
        aLog.debug(f'Improve chunk {len(text)} Start tokens({self.tokens_prefix}+{sum_tokens}={sum_tokens + self.tokens_prefix})')
        piping = LlmPipeMiddleware.get_config().calculate_tokens(sum_tokens, *self.prompter.min_max_multiplicator())
        prompt = self.prompter.prompt(self.text_separ.join(text), self.special_words, self.names)
        chunk = self.answer_vs_prompt(prompt, piping.config())
        sum_tokens = self.count_tokens_of_text(chunk)
        chunk = self.split_chunk_to_paragraphs(chunk)
        aLog.debug(f'Improve chunk ( {len(text)} -> {len(chunk)} ) Done tokens({sum_tokens})')
        return chunk

    def translate_paragraphs(self, text: list[str]) -> list[str]:
        self.tokens_prefix = self.prompt_max_length()
        tokens_vs_paragraphs = [self.count_tokens_of_text(paragraph) for paragraph in text]
        response = []
        aLog.debug(f'Translation paragraphs {len(text)} Start')
        while len(text) > 0:
            chunk, next_paragraphs_count = self.translate_paragraphs_chunk(tokens_vs_paragraphs, text)
            del text[:next_paragraphs_count]
            del tokens_vs_paragraphs[:next_paragraphs_count]
            response.extend(chunk)
        return response

    def translate_paragraphs_chunk(self, tokens_vs_paragraphs: list[int], paragraphs: list[str], *, separation_strategy: int = 1) -> tuple[list[str], int]:
        piping = LlmPipeMiddleware.get_config()
        next_paragraphs_count_original = LlmVideoCardBase.count_paragraphs_vs_tokens(tokens_vs_paragraphs, piping.calculate_tokens_maxima(self.tokens_maxima) - self.tokens_prefix)
        next_paragraphs_count = LlmVideoCardBase.use_separation_strategy(separation_strategy, next_paragraphs_count_original)
        next_text = paragraphs[:next_paragraphs_count]
        sum_tokens = sum(tokens_vs_paragraphs[:next_paragraphs_count])
        aLog.debug(f'Translation chunk {next_paragraphs_count} Start tokens({self.tokens_prefix}+{sum_tokens}={sum_tokens + self.tokens_prefix})')
        piping.calculate_tokens(sum_tokens, *self.prompter.min_max_multiplicator())
        prompt = self.prompter.prompt(self.text_separ.join(next_text), self.special_words, self.names)
        chunk = self.answer_vs_prompt(prompt, piping.config())
        sum_tokens = self.count_tokens_of_text(chunk)
        chunk = self.split_chunk_to_paragraphs(chunk)
        aLog.debug(f'Translation chunk ( {next_paragraphs_count} -> {len(chunk)} ) Done tokens({sum_tokens}) {chunk}')
        if len(chunk) != next_paragraphs_count and next_paragraphs_count > 1:
            separation_strategy += 1 if separation_strategy < 3 else 2
            aLog.debug(f'Translation chunk was problematic, separating paragraphs {next_paragraphs_count} -> {int(next_paragraphs_count_original / separation_strategy)}')
            return self.translate_paragraphs_chunk(tokens_vs_paragraphs, paragraphs, separation_strategy=separation_strategy)
        if len(chunk) != next_paragraphs_count:
            chunk = [' '.join(chunk)]
        return chunk, next_paragraphs_count

    def split_chunk_to_paragraphs(self, chunk: str) -> list[str]:
        raise NotImplementedError('Use split_chunk_to_paragraphs() from LlmPoetryCore or implement')

    @staticmethod
    def count_paragraphs_vs_tokens(tokens_vs_paragraphs: list[int], tokens_limit: int) -> int:
        tokens_sum = 0
        for ix, no in enumerate(tokens_vs_paragraphs):
            tokens_sum += no
            if tokens_limit < tokens_sum:
                return 1 if ix < 1 else ix
        return len(tokens_vs_paragraphs)

    @staticmethod
    def use_separation_strategy(separation_strategy: int, paragraphs_count: int) -> int:
        if separation_strategy > 1 and paragraphs_count > 1:
            next_paragraphs_count = int(paragraphs_count / separation_strategy)
            return 1 if 1 > next_paragraphs_count else next_paragraphs_count
        return paragraphs_count

    def prompt_max_length(self) -> int:
        prompt_empty = self.prompter.prompt(''.join(['', '']), self.special_words, self.names)
        prompt_llm_empty = self.tokenizer.apply_chat_template(prompt_empty, tokenize=False, add_generation_prompt=True)
        return self.count_tokens_of_text(prompt_llm_empty)

    def count_tokens_of_text(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def answer_vs_prompt(self, prompt: list[dict], pipe_config: dict) -> str:
        prompt_llm = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        pipe = pipeline('text-generation', self.llm, tokenizer=self.tokenizer, **pipe_config)
        answer = pipe(prompt_llm, return_full_text=False)[0]['generated_text']
        return answer

    def init_llm(self):
        if self.llm and self.model_id == LlmModelMiddleware.get_config():
            return
        if self.llm:
            self.llm = None
            self.tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()
        self.model_id = LlmModelMiddleware.get_config()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.hf_auth, use_fast=True)
        self.llm = self.create_llm()
        self.llm.to(DEVICE_CUDA)
        self.tokens_maxima = LlmModelMiddleware.max_tokens()

    def create_llm(self):
        wages_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=wages_config,
            low_cpu_mem_usage=True,
            token=self.hf_auth,
        )
