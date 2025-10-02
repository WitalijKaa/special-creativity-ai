from src.models.services.pipe.pipe_base import PipeBase


class LlmPipeGptParams(PipeBase):
    quality_reasoning: str

    def start_config(self):
        self.temperature_secured()
        self.quality_reasoning = 'low'

    # TEMPERATURE

    def temperature_secured(self):
        self.temperature = 0.8

    def temperature_normal(self):
        self.temperature = 1.0

    def temperature_medium(self):
        print('temperature_medium')
        if self.temperature > 0.9995:
            self.temperature = 2.0

    def temperature_high(self):
        print('temperature_high')
        if self.temperature > 0.9995:
            self.temperature = 5.0

    # QUALITY

    def quality_medium(self):
        self.quality_reasoning = 'medium'

    def quality_high(self):
        self.quality_reasoning = 'high'


    # BASE FUNCTIONS

    def config(self) -> dict:
        config = {
            'reasoning': {"effort": self.quality_reasoning},
            'temperature': self.temperature,
        }
        return config
