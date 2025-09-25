MODE_STRICT = 'strict'
MODE_CREATIVELY = 'creatively'

MODE_QUALITY_OK = 'ok'
MODE_QUALITY_NICE = 'nice'
MODE_QUALITY_MEGA = 'mega'

MODE_CREATIVELY_MORE = 'more'
MODE_CREATIVELY_INSANE = 'insane'


class LlmPipeParams:
    def __init__(self):
        self.min_new_tokens = None
        self.max_new_tokens = None

        self.calculation_branches = 1 # num_beams
        self.repetition_penalty = 1.0 # (if more 1.0 -> reduces probability for repetitive word) (if less 1.0 -> high-up probability for repetitive word)
        self.long_answer = 1.0 # length_penalty (if more 1.0 -> longer answer)
        # self.no_repeat_phrase_size = 0 # no_repeat_ngram_size

        # !! self.vs_temperature = True # do_sample
        # !! self.vs_temperature = False # do_sample (DEFAULT)

        self.temperature = None

        self.t_variants = None # top_k (count top variants and use only it)
        # if both: first will get t_variants after will reduce via t_threshold
        self.t_threshold = None # top_p (sum top variants until this threshold and use them) (1.0 is big, take more words, 0.5 is tiny, take few words)
        # typical_p maybe...

    def calculate_tokens_maxima(self, video_card_maxima: int) -> int:
        if 1 == self.calculation_branches:
            return video_card_maxima
        return int(video_card_maxima / self.calculation_branches * 1.2)

    def calculate_tokens(self, tokens: int, min_mult: float, max_mult: float ):
        self.min_new_tokens = int(tokens * min_mult)
        self.max_new_tokens = int(tokens * max_mult)
        return self

    def by_mode(self, mode: str):
        self.start_config()
        mode = mode.split('.')
        mode_provided = False
        for mode_param in mode:
            if mode_param in (MODE_STRICT, MODE_CREATIVELY):
                if mode_param == MODE_STRICT:
                    self.temperature_zero()
                    mode_provided = True
                elif mode_param == MODE_CREATIVELY:
                    self.temperature_low()
                    mode_provided = True

        if not mode_provided:
            raise NotImplementedError('Unexpected pipe mode')

        for mode_param in mode:
            if mode_param in (MODE_QUALITY_OK, MODE_QUALITY_NICE, MODE_QUALITY_MEGA):
                if mode_param == MODE_QUALITY_NICE:
                    self.quality_medium()
                elif mode_param == MODE_QUALITY_MEGA:
                    self.quality_high()
        for mode_param in mode:
            if self.temperature is not None and mode_param in (MODE_CREATIVELY_MORE, MODE_CREATIVELY_INSANE):
                if mode_param == MODE_QUALITY_NICE:
                    self.temperature_medium()
                elif mode_param == MODE_QUALITY_MEGA:
                    self.temperature_high()

    def start_config(self):
        self.min_new_tokens = None
        self.max_new_tokens = None
        self.calculation_branches = 1
        self.repetition_penalty = 1.0
        self.long_answer = 1.0
        self.temperature = None


    # TEMPERATURE

    def temperature_zero(self):
        self.temperature = None

    def temperature_low(self):
        self.temperature = 0.9
        self.t_variants = 25
        self.t_threshold = 0.85
        self.repetition_penalty = 1.05

    def temperature_medium(self):
        self.temperature = 0.8
        self.t_variants = 25
        self.t_threshold = 0.75
        self.repetition_penalty = 1.1

    def temperature_high(self):
        self.temperature = 0.65
        self.t_variants = 35
        self.t_threshold = 0.95
        self.repetition_penalty = 1.1


    # QUALITY

    def quality_medium(self):
        self.calculation_branches = 2
        self.long_answer = 1.1
        self.repetition_penalty = 1.05

    def quality_high(self):
        self.calculation_branches = 3
        self.long_answer = 1.2
        self.repetition_penalty = 1.05


    # BASE FUNCTIONS

    def config(self) -> dict:
        config = {
            'do_sample': self.temperature is not None,
            'repetition_penalty': self.repetition_penalty,
            'length_penalty': self.long_answer,
            'num_beams': self.calculation_branches,
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
        }
