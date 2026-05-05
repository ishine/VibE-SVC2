import torch
import torchaudio
import yaml

from models.model import Unit2Mel
from style_encoder.style_encoder import TechConverter
from utils.commons.encoder_utils import get_speech_encoder
from utils.commons.energy_utils import Volume_Extractor
from utils.commons.f0_utils import get_f0_predictor
from utils.commons.model_utils import load_config, load_model
from utils.commons.utils import repeat_expand_2d
from vocoder import Vocoder


class BaseSVC:
    def __init__(
        self,
        svc_model_path=None,
        svc_config_path=None,
        pretrained_path_config=None,
        f0_predictor=None,
        pitch_zeroshot=False,
        infer_mode=None,
        device=None,
        **kwargs
    ):
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)

        self.svc_config = load_config(svc_config_path)
        self.spk_stat_config = kwargs.get("spk_stat_config")

        self.dtype = torch.float32
        self.eps = kwargs.get("eps")
        self.eps_torch = torch.FloatTensor([kwargs.get("eps")]).to(self.dev)

        self.target_sample = self.svc_config.data.sampling_rate
        self.hop_size = self.svc_config.data.hop_size
        self.spk2id = self.svc_config.spk
        self.speech_encoder = self.svc_config.data.encoder
        self.unit_interpolate_mode = self.svc_config.data.encoder_unit_interpolate_mode

        self.wavelet_func = "db10"
        self.wavelet_cutoff = 4
        self.pitch_zeroshot = pitch_zeroshot
        self.vocal_fry_enforcement = kwargs.get("vocal_fry_enforcement")
        self.infer_mode = infer_mode

        # Load feature extractors
        self.hubert_model = get_speech_encoder(self.speech_encoder, device=self.dev)
        self.energy_extractor = Volume_Extractor(self.hop_size)
        self.f0_extractor = get_f0_predictor(
            f0_predictor,
            hop_length=self.hop_size,
            sampling_rate=self.target_sample,
            device=self.dev,
            threshold=0.05,
        )

        self.p_style2id = self.svc_config["singing_techniques"]["pitch_tech"]
        self.t_style2id = self.svc_config["singing_techniques"]["timbre_tech"]

        self.n_pitch_style = (
            self.svc_config.model.LUT.n_p_tech if infer_mode != "joint" else 2
        )
        self.n_timbre_style = self.svc_config.model.LUT.n_t_tech

        # Define models
        self.vocoder = Vocoder(self.svc_config.model.vocoder.type, device=self.dev)
        self.model = Unit2Mel(
            self.svc_config.data.encoder_out_channels,
            self.svc_config.model.LUT.n_spk,
            self.n_timbre_style,
            self.vocoder.dimension,
            self.svc_config.model.diffusion.n_layers,
            self.svc_config.model.diffusion.n_chans,
            self.svc_config.model.diffusion.n_hidden,
            self.svc_config.model.diffusion.timesteps,
            self.svc_config.model.diffusion.k_step_max,
        )
        # Load diffusion decoder checkpoint
        self.model = load_model(self.model, svc_model_path, self.dev)

        # Load style converter configs
        if infer_mode != "timbre":
            if not pitch_zeroshot:
                self.pitch_style_converter_config = load_config(
                    pretrained_path_config["style_converter"]["pitch_ID"]["config"]
                )
                self.pitch_style_converter_model_config = load_config(
                    pretrained_path_config["style_converter"]["pitch_ID"]["encoder_config"]
                )
                self.pitch_style_converter_model_type = pretrained_path_config["style_converter"]["pitch_ID"]["encoder_type"]
            else:
                self.pitch_style_converter_config = load_config(
                    pretrained_path_config["style_converter"]["pitch_ZS"]["config"]
                )
                self.pitch_style_converter_model_config = load_config(
                    pretrained_path_config["style_converter"]["pitch_ZS"]["encoder_config"]
                )
                self.pitch_style_converter_model_type = pretrained_path_config["style_converter"]["pitch_ZS"]["encoder_type"]
            self.pitch_style_converter = TechConverter(
                # self.pitch_style_converter_config["style_enc"],
                self.pitch_style_converter_model_config[self.pitch_style_converter_model_type],
                self.n_pitch_style,
                zero_shot=pitch_zeroshot,
            )
            if not pitch_zeroshot:
                self.pitch_style_converter = load_model(
                    self.pitch_style_converter,
                    pretrained_path_config["style_converter"]["pitch_ID"]["model"],
                    self.dev,
                )
            else:
                self.pitch_style_converter = load_model(
                    self.pitch_style_converter,
                    pretrained_path_config["style_converter"]["pitch_ZS"]["model"],
                    self.dev,
                )
        else:
            self.pitch_style_converter = None
            self.pitch_style_converter_config = None

        if infer_mode != "timbre":
            if not pitch_zeroshot:
                self.energy_style_converter_config = load_config(
                    pretrained_path_config["style_converter"]["energy_ID"]["config"]
                )
                self.energy_style_converter_model_config = load_config(
                    pretrained_path_config["style_converter"]["energy_ID"]["encoder_config"]
                )
                self.energy_style_converter_model_type = pretrained_path_config["style_converter"]["energy_ID"]["encoder_type"]
            else:
                self.energy_style_converter_config = load_config(
                    pretrained_path_config["style_converter"]["energy_ZS"]["config"]
                )
                self.energy_style_converter_model_config = load_config(
                    pretrained_path_config["style_converter"]["energy_ZS"]["encoder_config"]
                )
                self.energy_style_converter_model_type = pretrained_path_config["style_converter"]["energy_ZS"]["encoder_type"]

            self.energy_style_converter = TechConverter(
                # self.energy_style_converter_config["style_enc"],
                self.energy_style_converter_model_config[self.energy_style_converter_model_type],
                self.n_pitch_style,
                zero_shot=pitch_zeroshot,
            )
            if not pitch_zeroshot:
                self.energy_style_converter = load_model(
                    self.energy_style_converter,
                    pretrained_path_config["style_converter"]["energy_ID"]["model"],
                    self.dev,
                )
            else:
                self.energy_style_converter = load_model(
                    self.energy_style_converter,
                    pretrained_path_config["style_converter"]["energy_ZS"]["model"],
                    self.dev,
                )
        else:
            self.energy_style_converter = None
            self.energy_style_converter_config = None

    def get_unit_f0(
        self,
        wav,
    ):
        """
        Extracting input features.

        Args:
            wav (np.array)          : Input audio. Shape of (T, ).

        Returns:
            units(torch.FloatTensor)    : SSL features for content information. Shape of (B, T, C)
            f0(np.array)                : F0 contour for pitch. Shape of (T, ).
            uv(np.array)                : Voiced flag sequence. Shape of (T, ).
            energy(torch.FloatTensor)   : Energy contour sequence. Shape of (T, )
        """

        f0, uv = self.f0_extractor.compute_f0_uv(wav)

        units = self.hubert_model.encoder(self.resample_wav16k(wav))
        units = repeat_expand_2d(
            units.squeeze(0), len(f0), self.unit_interpolate_mode
        ).unsqueeze(0)
        units = units.transpose(-1, -2)
        units = units.to(self.dtype)

        energy = self.energy_extractor.extract(wav[None, :])
        return units, f0, uv, energy

    def get_energy(self, wav):
        return self.energy_extractor.extract(wav)

    def resample_wav16k(self, wav):
        wav = wav.to(self.dev)
        if not hasattr(self, "audio16k_resample_transform"):
            self.audio16k_resample_transform = torchaudio.transforms.Resample(
                self.target_sample, 16000
            ).to(self.dev)
        wav16k = self.audio16k_resample_transform(wav[None, :])[0]

        return wav16k

    def resample_wav(self, wav, orig_sr, target_sr):
        if (
            not hasattr(self, "audio_resample_transform")
            or self.audio_resample_transform.orig_freq != orig_sr
        ):
            self.audio_resample_transform = torchaudio.transforms.Resample(
                orig_sr, target_sr
            )

        wav = self.audio_resample_transform(wav)[0]
        return wav

    def vocoded(self, raw_path):
        wav, _ = torchaudio.load(raw_path)

        with torch.inference_mode():
            gt_audio = torch.FloatTensor(wav).to(self.dev)
            gt_mel = self.vocoder.extract(gt_audio[None:], self.target_sample)
            audio = self.vocoder.infer(gt_mel)

        return audio

    def infer(self):
        pass

    def clear_empty(self):
        torch.cuda.empty_cache()
