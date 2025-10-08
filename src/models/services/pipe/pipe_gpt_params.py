from src.models.services.pipe.pipe_base import PipeBase


class LlmPipeGptParams(PipeBase):
    quality_reasoning: str
    long_answer: str

    def start_config(self):
        self.temperature_secured()
        self.quality_reasoning = 'low'
        self.long_answer = 'low'

    # TEMPERATURE

    def temperature_secured(self):
        self.long_answer = 'low'

    def temperature_normal(self):
        self.long_answer = 'low'

    def temperature_medium(self):
        self.long_answer = 'medium'

    def temperature_high(self):
        self.long_answer = 'high'

    # QUALITY

    def quality_medium(self):
        self.quality_reasoning = 'medium'

    def quality_high(self):
        self.quality_reasoning = 'high'


    # BASE FUNCTIONS

    def config(self) -> dict:
        config = {
            'reasoning': {"effort": self.quality_reasoning},
            'text': {'verbosity': self.long_answer},
        }
        return config
