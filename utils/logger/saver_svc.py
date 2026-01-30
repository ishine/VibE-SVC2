import matplotlib.pyplot as plt
import torch

from .base import BaseSaver


class Saver(BaseSaver):
    def log_spec(self, name, spec, spec_out, vmin=-14, vmax=3.5):
        spec_cat = torch.cat([(spec_out - spec).abs() + vmin, spec, spec_out], -1)
        spec = spec_cat[0]
        if isinstance(spec, torch.Tensor):
            spec = spec.cpu().numpy()
        fig = plt.figure(figsize=(12, 9))
        plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
        plt.tight_layout()
        self.writer.add_figure(name, fig, self.global_step)

    def log_audio(self, dict):
        for k, v in dict.items():
            self.writer.add_audio(
                k, v, global_step=self.global_step, sample_rate=self.sample_rate
            )
