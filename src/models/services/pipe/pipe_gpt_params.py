from src.models.services.pipe.pipe_base import PipeBase


class LlmPipeGptParams(PipeBase):
    quality_reasoning: str

    def start_config(self):
        self.temperature_secured()
        self.quality_reasoning = 'low'

    # TEMPERATURE

    def temperature_secured(self):
        self.temperature = 0.0

    def temperature_normal(self):
        self.temperature = 0.36

    def temperature_medium(self):
        if self.temperature > 0.1:
            self.temperature = 0.88

    def temperature_high(self):
        if self.temperature > 0.1:
            self.temperature = 1.42

    # QUALITY

    def quality_medium(self):
        self.quality_reasoning = 'medium'

    def quality_high(self):
        self.quality_reasoning = 'high'


    # BASE FUNCTIONS

    def config(self) -> dict:
        config = {
            'reasoning': {"effort": self.quality_reasoning},
            # GPT-5 cant do it 'temperature': self.temperature,
        }
        return config
