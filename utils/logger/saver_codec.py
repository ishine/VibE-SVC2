from .base import BaseSaver


class Codec_Saver(BaseSaver):
    def __init__(self, cfg, initial_global_step):
        super(Codec_Saver, self).__init__(cfg, initial_global_step=initial_global_step)
