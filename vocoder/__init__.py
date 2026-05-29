import json
import os

import torch
from torchaudio.transforms import Resample

from vocoder.bigvgan_v2.bigvgan import BigVGAN as BigVGanGenerator
from vocoder.bigvgan_v2.env import AttrDict
from vocoder.bigvgan_v2.meldataset import get_mel_spectrogram


class Vocoder:
    def __init__(self, vocoder_type, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if vocoder_type == "bigvgan_v2":
            self.vocoder = BigVGAN_V2(device=device)
        else:
            raise ValueError(f" [x] Unknown vocoder: {vocoder_type}")

        self.resample_kernel = {}
        self.vocoder_sample_rate = self.vocoder.sample_rate()
        self.vocoder_hop_size = self.vocoder.hop_size()
        self.dimension = self.vocoder.dimension()

    def extract(self, audio, sample_rate, key_shift=0):
        # resample
        if sample_rate == self.vocoder_sample_rate:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(
                    sample_rate, self.vocoder_sample_rate, lowpass_filter_width=128
                )
            audio_res = self.resample_kernel[key_str](audio)

        mel = self.vocoder.extract(audio_res)
        return mel  # [batch, n_frames, n_mel_bins]

    def infer(self, mel):
        audio = self.vocoder(torch.transpose(mel, 1, 2))
        return audio


def load_hparams_from_json(path) -> AttrDict:
    with open(path) as f:
        data = f.read()
    return AttrDict(json.loads(data))


class BigVGAN_V2(torch.nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device
        # load config
        config_file = os.path.join("./vocoder", "bigvgan_v2", "config.json")
        self.h = load_hparams_from_json(config_file)
        self.model = None

    def sample_rate(self):
        return self.h.sampling_rate

    def hop_size(self):
        return self.h.hop_size

    def dimension(self):
        return self.h.num_mels

    def extract(self, audio):
        mel = get_mel_spectrogram(audio, self.h).cuda(self.device)
        return mel

    def forward(self, mel, **kwargs):

        # Set vocoder
        if self.model is None:
            self.model = BigVGanGenerator.from_pretrained(
                "nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False
            )
            self.model.remove_weight_norm()
        self.model = self.model.eval().cuda(self.device)
        with torch.inference_mode():
            wav_gen = self.model(
                mel
            )  # wav_gen is FloatTensor with shape [B(1), 1, T_time] and values in [-1, 1]
        wav_gen_float = wav_gen.squeeze(0).cpu()
        return wav_gen_float
