from src.helpers.php_wk import *
import gc
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, pipeline
from transformers import PreTrainedTokenizerFast
from transformers import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel

from src.middleware.auth import AuthMiddleware
from src.middleware.llm import LlmAiMiddleware
from src.models.services.prompt_service import PromptService
from src.models.services.llm_pipe_config_service import LlmPipeConfigService


class AiProvider:
    def __init__(self):
        self.hf_auth: str = AuthMiddleware.hugging_face_token()
        self.model_id: str = LlmAiMiddleware.get_config()
        self.llm: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None

    def translate_rus_to_eng(self, text: str) -> str:
        self.init_ai()
        text_tokens_count = len(self.tokenizer.encode(text, add_special_tokens=False))
        min_length, max_length, prompt = PromptService.prompt_translate_slavic_to_english(text)
        return self.answer_vs_prompt(prompt, int(text_tokens_count * min_length), int(text_tokens_count * max_length))

    def translate_eng_to_rus(self, text: str) -> str:
        self.init_ai()
        text_tokens_count = len(self.tokenizer.encode(text, add_special_tokens=False))
        min_length, max_length, prompt = PromptService.prompt_translate_english_to_russian(text)
        return self.answer_vs_prompt(prompt, int(text_tokens_count * min_length), int(text_tokens_count * max_length))

    def answer_vs_prompt(self, prompt: list[dict], min_answer_length: int, max_answer_length: int) -> str:
        pipe_config = LlmPipeConfigService.get_config()
        pipe_config['min_new_tokens'] = min_answer_length
        pipe_config['max_new_tokens'] = max_answer_length

        prompt_llm = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        pipe = pipeline("text-generation", self.llm, tokenizer=self.tokenizer, **pipe_config)
        answer = pipe(prompt_llm, return_full_text=False)[0]['generated_text']
        return answer

    def init_ai(self):
        if self.llm and self.model_id == LlmAiMiddleware.get_config():
            return
        if self.llm:
            self.llm = None
            self.tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()
        self.model_id = LlmAiMiddleware.get_config()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.hf_auth, use_fast=True)
        self.llm = self.create_llm()
        self.llm.to(DEVICE_CUDA)

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
