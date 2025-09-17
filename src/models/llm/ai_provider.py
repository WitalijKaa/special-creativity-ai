import os, torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from transformers import pipeline

class AiProvider:
    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self.token = os.getenv("HF_TOKEN")
        self.model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct'

    def answer_vs_prompt(self, prompt: list[dict]) -> str:
        pipe_config = {
            'max_new_tokens': 2048,
            'min_new_tokens': 256,
            'temperature': None,
            'top_p': None,
            'top_k': None,
            'do_sample': False,
            'num_beams': 1,
            'no_repeat_ngram_size': 4,
        }

        self.init_ai()
        prompt_llm = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        pipe = pipeline("text-generation", self.llm, tokenizer=self.tokenizer, **pipe_config)
        answer = pipe(prompt_llm, return_full_text=False)[0]['generated_text']

        return answer

    def init_ai(self):
        if self.llm:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.token, use_fast=True)
        self.llm = self.create_llm()

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
            token=self.token,
        )
