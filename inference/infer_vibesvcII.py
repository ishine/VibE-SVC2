import numpy as np
import torch
import torchaudio
import yaml

from utils.commons.dwt import separate_signal_dwt
from utils.commons.shc import subharmonics_correction

from .base import BaseSVC


class Svc(BaseSVC):
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
        super().__init__(
            svc_model_path,
            svc_config_path,
            pretrained_path_config,
            f0_predictor,
            pitch_zeroshot,
            infer_mode,
            device,
            **kwargs,
        )

    def extent_scaler(self, high_f0, scale_type, scale=None):
        frame_rate = self.target_sample / self.hop_size
        if scale_type == "global":
            extent_scaler = torch.FloatTensor([scale])
        elif scale_type == "inc_linear":
            extent_scaler = torch.arange(0, high_f0.shape[1]) / frame_rate
        elif scale_type == "dec_linear":
            extent_scaler = torch.arange(high_f0.shape[1], 0, -1) / frame_rate
        elif scale_type == "sinusodial":
            extent_scaler = (
                torch.sin(torch.arange(0, high_f0.shape[1]) / frame_rate) + 1
            )
        scale_high_f0 = high_f0 * extent_scaler.to(self.dev)
        return scale_high_f0

    def get_technique_f0(
        self,
        f0,
        uv,
        target_pitch_style_id,
        extent_scale=1.0,
        extent_scale_type="global",
        rate_scale=1.0,
        rate_scale_type="global",
    ):

        # Separate signal
        lf0 = np.log2(f0 + 1e-6)
        low_lf0, high_lf0 = separate_signal_dwt(
            lf0, self.wavelet_func, self.wavelet_cutoff
        )
        low_f0 = 2**low_lf0 - 1e-6

        # Numpy to torch
        low_f0 = torch.FloatTensor(low_f0).to(self.dev).to(self.dtype)
        uv_np = uv
        uv = torch.FloatTensor(uv).to(self.dev).to(self.dtype)
        high_lf0 = (
            torch.FloatTensor(high_lf0).to(self.dev).to(self.dtype)[None, :, None]
        )

        low_f0 = low_f0.view(1, 1, -1)
        uv = uv.view(1, 1, -1)

        # interpolate smooth f0
        orig_len = low_f0.shape[1]

        # squeeze or stretch for rate control
        if rate_scale != 1.0:
            upsampler = torch.nn.Upsample(size=(int(low_f0.shape[2] * rate_scale),))
            low_f0 = upsampler(low_f0).view(1, -1, 1)
            uv = upsampler(uv).view(1, -1, 1)
        else:
            low_f0 = low_f0.transpose(1, 2)
            uv = uv.transpose(1, 2)

        # predict high-frequency F0 contour
        low_lf0 = torch.log2(low_f0 + self.eps_torch)
        _, pred_high_lf0 = self.pitch_style_converter(
            low_lf0,
            uv,
            target_pitch_style_id,
            high_lf0 if self.pitch_zeroshot else None,
        )

        # control vibrato extent
        pred_high_f0 = (2**pred_high_lf0 - self.eps_torch) - 1
        scale_pred_high_f0 = (
            self.extent_scaler(pred_high_f0, extent_scale_type, extent_scale) + 1
        )

        scale_pred_high_lf0 = torch.log2(scale_pred_high_f0 + self.eps_torch)

        pred_lf0 = low_lf0 + scale_pred_high_lf0
        pred_f0 = 2**pred_lf0 - self.eps_torch

        # squeeze or stretch back to original time resolution
        if rate_scale != 1.0:
            downsampler = torch.nn.Upsample(size=(orig_len,))
            pred_f0 = downsampler(pred_f0.transpose(1, 2)).squeeze(1)
        else:
            pred_f0 = pred_f0.squeeze(-1)

        # interpolate F0 at unvoiced region
        pred_f0 = pred_f0.detach().cpu().numpy().squeeze(0)  # [,Length]
        pred_f0[uv_np == 0.0] = 0.0

        pred_f0, _ = self.f0_extractor.interpolate_f0(pred_f0)
        pred_f0 = (
            torch.from_numpy(pred_f0).float()[None, :, None].to(self.dev).to(self.dtype)
        )

        return pred_f0, uv

    def get_technique_energy(self, energy, uv, target_pitch_style_id):
        if self.energy_style_converter is not None:
            # predict technique energy contour
            low_energy, high_energy = separate_signal_dwt(
                energy.numpy(), self.wavelet_func, self.wavelet_cutoff
            )
            low_energy = torch.from_numpy(low_energy[None, :, None]).to(self.dev)
            high_energy = torch.from_numpy(high_energy[None, :, None]).to(self.dev)

            pred_energy, _ = self.energy_style_converter(
                low_energy,
                uv.view(1, -1, 1),
                target_pitch_style_id,
                high_energy if self.pitch_zeroshot else None,
            )

            energy = pred_energy.view(1, -1, 1)

        else:
            # utilize original energy contour
            energy = energy.view(1, -1, 1).to(self.dev)
        return energy

    def infer_style_encoders(
        self,
        f0_np,
        uv_np,
        energy,
        source_tech,
        target_pitch_style_id,
        f0_shift,
        extent_scale,
        extent_scale_type,
        rate_scale,
        vocal_fry_enforcement,
    ):
        # get F0 contour
        if self.n_pitch_style != 1:
            f0, uv = self.get_technique_f0(
                f0_np,
                uv_np,
                target_pitch_style_id,
                extent_scale=extent_scale,
                extent_scale_type=extent_scale_type,
                rate_scale=rate_scale,
            )
        else:
            if source_tech == "vocal_fry":
                f0_np, _ = subharmonics_correction(f0_np, uv_np)
            f0 = torch.from_numpy(f0_np).to(self.dev).to(self.dtype)[None, :, None]

        # F0 shifting based on speaker statistics
        if f0_shift != None:
            if self.n_timbre_style != 1:
                # f0 should be shifted based on f0 due to subharmonic characteristics
                f0 = (f0_shift / torch.mean(f0.view(-1))) * f0
            else:
                f0 = f0_shift * f0

        # only available in timbre technique conversion mode
        if self.n_pitch_style == 1 and vocal_fry_enforcement:
            f0_np /= 2

        # get energy contour
        if self.n_pitch_style != 1:
            energy = self.get_technique_energy(energy, uv, target_pitch_style_id)

        else:
            energy = energy.view(1, -1, 1).to(self.dev)
            energy = (energy - torch.mean(energy, dim=1, keepdim=True)) / torch.std(
                energy, dim=1, keepdim=True
            )
        return f0, energy

    def infer_mel(
        self,
        c,
        f0,
        energy,
        target_spk_id,
        target_timbre_style_id,
        k_step,
        return_audio=False,
    ):
        # generate Mel-Spectrogram through diffusion decoder
        mel = self.model(
            c,
            f0,
            energy,
            spk_id=target_spk_id,
            timbre_style_id=(
                target_timbre_style_id if self.n_timbre_style != 1 else None
            ),
            gt_spec=None,
            infer=True,
            infer_speedup=self.svc_config.model.diffusion.infer_speedup,
            method=self.svc_config.model.diffusion.infer_method,
            k_step=k_step,
            use_tqdm=False,
        )
        if return_audio:
            return self.vocoder.infer(mel)
        else:
            return mel

    def infer(
        self,
        source_spk,
        target_spk,
        source_tech,
        target_pitch_style,
        target_timbre_style,
        raw_path,
        k_step=1000,
        f0_shift=0,
        extent_scale=1.0,
        extent_scale_type="global",
        rate_scale=1.0,
        vocal_fry_enforcement=False,
    ):
        # load waveform
        wav, sr = torchaudio.load(raw_path)

        # resample to target sr
        wav = self.resample_wav(wav, sr, self.target_sample)

        # get speaker & technique IDs
        target_spk_id = int(self.spk2id.get(target_spk))
        target_spk_id = torch.LongTensor([target_spk_id]).to(self.dev).unsqueeze(0)

        source_spk_id = int(self.spk2id.get(source_spk))
        source_spk_id = torch.LongTensor([source_spk_id]).to(self.dev).unsqueeze(0)

        if self.n_pitch_style != 1:
            target_pitch_style_id = int(self.p_style2id.get(target_pitch_style))
            target_pitch_style_id = (
                torch.LongTensor([target_pitch_style_id]).to(self.dev).unsqueeze(0)
            )
        else:
            target_pitch_style_id = None

        if self.n_timbre_style != 1:
            target_timbre_style_id = int(self.t_style2id.get(target_timbre_style))
            target_timbre_style_id = (
                torch.LongTensor([target_timbre_style_id]).to(self.dev).unsqueeze(0)
            )
        else:
            target_timbre_style_id = None

        # get unit

        # get f0 and energy
        if self.infer_mode != "timbre":
            c, f0_np, uv_np, energy = self.get_unit_f0(
                wav,
            )
            f0, energy = self.infer_style_encoders(
                f0_np,
                uv_np,
                energy,
                source_tech,
                target_pitch_style_id,
                f0_shift,
                extent_scale,
                extent_scale_type,
                rate_scale,
                vocal_fry_enforcement,
            )
        else:  # infer_mode == timbre or joint
            c, f0_np, uv_np, energy = self.get_unit_f0_shc(
                wav,
                target_spk,
                source_tech,
                target_pitch_style_id,
                target_timbre_style,
            )
            f0 = (
                torch.from_numpy(f0_np)[None, :, None].float().to(self.dev)
            )  # [B, T, 1]
            energy = energy[None, :, None].to(self.dev)  # [B, T, 1]

        # generate Mel-Spectrogram through decoder
        audio = self.infer_mel(
            c,
            f0,
            energy,
            target_spk_id,
            target_timbre_style_id,
            k_step,
            return_audio=True,
        )

        return np.array(audio).squeeze(0)

    def get_unit_f0_shc(
        self,
        wav,
        target_spk,
        src_tech,
        target_pitch_style_id,
        target_timbre_style,
        extent_scale=1.0,
        extent_scale_type="global",
        rate_scale=1.0,
        vocal_fry_forcing=False,
        cr_threshold=0.05,
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

        units, f0_np, uv_np, energy = self.get_unit_f0(wav)

        if src_tech == "vocal_fry":
            f0_np, _ = subharmonics_correction(f0_np, uv_np)

        with open("./configs/spk_stats_timbre.yaml", "r") as config:
            spk_stat_config = yaml.safe_load(config)
        f0_shift = spk_stat_config[target_spk] / np.mean(f0_np)
        f0_np = f0_shift * f0_np

        if target_pitch_style_id is not None:
            # pitch style conversion
            pred_f0 = self.get_technique_f0(
                f0_np,
                uv_np,
                target_pitch_style_id,
                extent_scale=extent_scale,
                extent_scale_type=extent_scale_type,
                rate_scale=rate_scale,
            )
            pred_f0 = pred_f0.detach().cpu().numpy().squeeze(0)
            pred_f0[uv_np == 0.0] = 0.0

            f0, _ = self.f0_predictor_object.interpolate_f0(pred_f0)

            # energy style conversion
            energy = self.get_technique_energy(energy, uv_np, target_pitch_style_id)

        else:
            f0 = f0_np

        # Enable vocal fry forcing
        if self.vocal_fry_enforcement and target_timbre_style == "vocal_fry":
            f0 /= 2

        return units, f0, uv_np, energy
