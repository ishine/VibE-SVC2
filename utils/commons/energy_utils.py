import librosa
import torch
import torch.nn.functional as F


def change_rms(
    data1, sr1, data2, sr2, rate
):  # 1是输入音频，2是输出音频,rate是2的占比 from RVC
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # 每半秒一个点
    rms2 = librosa.feature.rms(
        y=data2.detach().cpu().numpy(), frame_length=sr2 // 2 * 2, hop_length=sr2 // 2
    )
    rms1 = torch.from_numpy(rms1).to(data2.device)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.from_numpy(rms2).to(data2.device)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= torch.pow(rms1, torch.tensor(1 - rate)) * torch.pow(
        rms2, torch.tensor(rate - 1)
    )
    return data2


class Volume_Extractor:
    def __init__(self, hop_size=512):
        self.hop_size = hop_size
        self.window_size = hop_size * 4

    def extract(self, audio):  # audio: 2d tensor array

        if not isinstance(audio, torch.Tensor):
            audio = torch.Tensor(audio)
        n_frames = int(audio.size(-1) // self.hop_size)
        audio2 = audio**2
        audio2 = F.pad(
            audio2,
            (int(self.window_size // 2), int((self.window_size + 1) // 2)),
            mode="reflect",
        )
        volume = F.unfold(
            audio2[:, None, None, :], (1, self.window_size), stride=self.hop_size
        )[:, :, :n_frames].mean(dim=1)[0]
        volume = torch.sqrt(volume)
        return volume
