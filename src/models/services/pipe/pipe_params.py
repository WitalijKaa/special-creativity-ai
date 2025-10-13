from src.models.services.pipe.pipe_base import PipeBase


class LlmPipeParams(PipeBase):
    min_new_tokens: int | None
    max_new_tokens: int | None
    calculation_branches: int
    repetition_penalty: float
    long_answer: float
    t_variants: int | None
    t_threshold: float | None

    def calculate_tokens_maxima(self, video_card_maxima: int) -> int:
        if 1 == self.calculation_branches:
            return video_card_maxima
        return int(video_card_maxima / self.calculation_branches * 1.35)

    def calculate_tokens(self, tokens: int, min_mult: float, max_mult: float ):
        max_mult *= self.long_answer
        self.min_new_tokens = int(tokens * min_mult)
        self.max_new_tokens = int(tokens * max_mult)
        return self


    def start_config(self):
        self.min_new_tokens = None
        self.max_new_tokens = None

        self.calculation_branches = 5  # num_beams
        self.temperature = None
        self.long_answer = 1.0  # length_penalty (if more 1.0 -> longer answer)

        self.repetition_penalty = 1.0  # (if more 1.0 -> reduces probability for repetitive word) (if less 1.0 -> high-up probability for repetitive word)
        # self.no_repeat_phrase_size = 0 # no_repeat_ngram_size

        self.t_variants = None  # top_k (count top variants and use only it)
        # if both: first will get t_variants after will reduce via t_threshold
        self.t_threshold = None  # top_p (sum top variants until this threshold and use them) (1.0 is big, take more words, 0.5 is tiny, take few words)


    # TEMPERATURE

    def temperature_secured(self):
        self.calculation_branches = 5
        self.temperature = None
        self.long_answer = 1.0
        self.repetition_penalty = 1.0

    def temperature_normal(self):
        self.calculation_branches = 1
        self.temperature = 0.71
        self.t_variants = 18
        self.t_threshold = 0.85
        self.long_answer = 1.0
        self.repetition_penalty = 1.0

    def temperature_medium(self):
        if self.temperature is not None:
            self.calculation_branches = 1
            self.temperature = 0.88
            self.t_variants = 18
            self.t_threshold = 0.91
            self.long_answer = 1.8
            self.repetition_penalty = 1.2

    def temperature_high(self):
        if self.temperature is not None:
            self.calculation_branches = 1
            self.temperature = 0.98
            self.t_variants = 24
            self.t_threshold = 0.95
            self.long_answer = 2.5
            self.repetition_penalty = 1.5

    # QUALITY

    def quality_medium(self):
        if self.temperature is not None:
            self.calculation_branches = 1
            if self.long_answer > 1.0:
                self.long_answer *= 0.8
            self.repetition_penalty *= 1.2
        else:
            self.calculation_branches = 7

    def quality_high(self):
        if self.temperature is not None:
            self.calculation_branches = 2
            if self.long_answer > 1.0:
                self.long_answer *= 0.65
            self.repetition_penalty *= 1.42
        else:
            self.calculation_branches = 11


    # OTHER

    def make_text_longer(self):
        self.long_answer = 1.65


    # BASE FUNCTIONS

    def config(self) -> dict:
        config = {
            'num_beams': self.calculation_branches,
            'do_sample': self.temperature is not None,
            'length_penalty': self.long_answer,
            'repetition_penalty': self.repetition_penalty,

            'renormalize_logits': True,
            'use_cache': True,
        }
        config.update(self.temperature_config())
        if self.min_new_tokens is not None and self.max_new_tokens is not None:
            config.update({'min_new_tokens': self.min_new_tokens, 'max_new_tokens': self.max_new_tokens})
        return config

    def temperature_config(self) -> dict:
        if self.temperature is None:
            return { }
        return {
            'temperature': self.temperature,
            'top_k': self.t_variants,
            'top_p': self.t_threshold,
            'min_p': 0.2,
            'typical_p': 0.95,
        }
