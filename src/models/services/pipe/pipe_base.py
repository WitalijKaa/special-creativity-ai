MODE_STRICT = 'strict'
MODE_CREATIVELY = 'creatively'

MODE_QUALITY_OK = 'ok'
MODE_QUALITY_NICE = 'nice'
MODE_QUALITY_MEGA = 'mega'

MODE_CREATIVELY_MORE = 'more'
MODE_CREATIVELY_INSANE = 'insane'

class PipeBase:
    temperature: float | None

    def __init__(self):
        self.start_config()

    def by_mode(self, mode: str):
        self.start_config()
        mode = mode.split('.')
        mode_provided = False
        for mode_param in mode:
            if mode_param in (MODE_STRICT, MODE_CREATIVELY):
                if mode_param == MODE_STRICT:
                    self.temperature_secured()
                    mode_provided = True
                elif mode_param == MODE_CREATIVELY:
                    self.temperature_normal()
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
            if mode_param in (MODE_CREATIVELY_MORE, MODE_CREATIVELY_INSANE):
                if mode_param == MODE_CREATIVELY_MORE:
                    self.temperature_medium()
                elif mode_param == MODE_CREATIVELY_INSANE:
                    self.temperature_high()

    def start_config(self):
        raise NotImplementedError('Pipe class must implement start_config()')

    def temperature_secured(self):
        raise NotImplementedError('Pipe class must implement temperature_secured()')

    def temperature_normal(self):
        raise NotImplementedError('Pipe class must implement temperature_normal()')

    def temperature_medium(self):
        raise NotImplementedError('Pipe class must implement temperature_medium()')

    def temperature_high(self):
        raise NotImplementedError('Pipe class must implement temperature_high()')

    def quality_medium(self):
        raise NotImplementedError('Pipe class must implement quality_medium()')

    def quality_high(self):
        raise NotImplementedError('Pipe class must implement quality_high()')

    def config(self) -> dict:
        raise NotImplementedError('Pipe class must implement config()')
