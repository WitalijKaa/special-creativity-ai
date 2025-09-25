from src.helpers.php_wk import *
from src.middleware.pipe import LlmPipeMiddleware
from src.models.basic_logger import aLog
import gc
import re
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, pipeline
from transformers import PreTrainedTokenizerFast
from transformers import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel

from src.middleware.auth import AuthMiddleware
from src.middleware.llm import LlmModelMiddleware
from src.models.services.pipe.pipe_params import LlmPipeParams
from src.models.services.prompt.abstract_prompt import AbstractTranslateService
from src.models.services.pipe.llm_pipe_config_service import LlmPipeConfigService


class AiProviderBase:
    def __init__(self):
        self.hf_auth: str = AuthMiddleware.hugging_face_token()
        self.model_id: str = LlmModelMiddleware.get_config()
        self.llm: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None
        self.prompter: AbstractTranslateService | None = None
        self.max_tokens_per_chunk = 42
        self.text_separ = LlmModelMiddleware.text_separ() * 4
        self.small_paragraphs_max_tokens = 20

        self.special_words: list[tuple[str, str]] = []
        self.names: list[str] = []

    def translate_paragraphs(self, text: list[str]) -> list[str]:
        tokens_vs_paragraphs = [len(self.tokenizer.encode(paragraph, add_special_tokens=False)) for paragraph in text]
        response = []
        while len(text) > 0:
            chunk, next_paragraphs_count = self.translate_paragraphs_chunk(tokens_vs_paragraphs, text)
            del text[:next_paragraphs_count]
            del tokens_vs_paragraphs[:next_paragraphs_count]
            response.extend(chunk)
        return response

    def translate_paragraphs_chunk(self, tokens_vs_paragraphs: list[int], paragraphs: list[str], *, separate_tiny: int = 0) -> tuple[list[str], int]:
        next_paragraphs_count = AiProviderBase.count_paragraphs_vs_tokens(tokens_vs_paragraphs, self.max_tokens_per_chunk, separate_tiny)
        next_text = paragraphs[:next_paragraphs_count]
        sum_tokens = sum(tokens_vs_paragraphs[:next_paragraphs_count])
        min_length, max_length, prompt = self.prompter.prompt(self.text_separ.join(next_text), self.special_words, self.names)
        pipe_customization = LlmPipeMiddleware.get_config().calculate_tokens(sum_tokens, min_length, max_length)
        chunk = self.answer_vs_prompt(prompt, pipe_customization.config())
        chunk = self.split_chunk_to_paragraphs(chunk)
        aLog.debug(f"Translation chunk p{next_paragraphs_count} -> p{len(chunk)} tokens({sum_tokens}) {chunk}")
        if len(chunk) != next_paragraphs_count and separate_tiny < self.small_paragraphs_max_tokens:
            separate_tiny += 10
            aLog.debug(f"Translation chunk was problematic, so separate tiny paragraph vs_tokens_size({separate_tiny})")
            return self.translate_paragraphs_chunk(tokens_vs_paragraphs, paragraphs, separate_tiny=separate_tiny)
        return chunk, next_paragraphs_count

    @staticmethod
    def count_paragraphs_vs_tokens(elements: list[int], limit: int, separate_tiny: int) -> int:
        no_sum = 0
        for ix, no in enumerate(elements):
            no_sum += no
            if limit < no_sum or (separate_tiny > 0 and no <= separate_tiny):
                return 1 if ix < 1 else ix
        return len(elements)

    def split_chunk_to_paragraphs(self, chunk: str) -> list[str]:
        separ_sign = self.text_separ[0]
        chunk = re.sub(separ_sign + '+', separ_sign, chunk)
        return chunk.split(separ_sign)

    def answer_vs_prompt(self, prompt: list[dict], pipe_config: dict) -> str:
        prompt_llm = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        pipe = pipeline('text-generation', self.llm, tokenizer=self.tokenizer, **pipe_config)
        answer = pipe(prompt_llm, return_full_text=False)[0]['generated_text']
        return answer

    def init_ai(self):
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
        self.max_tokens_per_chunk = LlmModelMiddleware.max_tokens()

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
