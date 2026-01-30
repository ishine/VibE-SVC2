import numpy as np
import torch

f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def normalize_f0(f0, x_mask, uv, random_scale=True):
    # calculate means based on x_mask
    uv_sum = torch.sum(uv, dim=1, keepdim=True)
    uv_sum[uv_sum == 0] = 9999
    means = torch.sum(f0[:, 0, :] * uv, dim=1, keepdim=True) / uv_sum

    if random_scale:
        factor = torch.Tensor(f0.shape[0], 1).uniform_(0.8, 1.2).to(f0.device)
    else:
        factor = torch.ones(f0.shape[0], 1).to(f0.device)
    # normalize f0 based on means and factor
    f0_norm = (f0 - means.unsqueeze(-1)) * factor.unsqueeze(-1)
    if torch.isnan(f0_norm).any():
        exit(0)
    return f0_norm * x_mask


def f0_to_coarse(f0):
    f0_mel = 1127 * (1 + f0 / 700).log()
    a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
    b = f0_mel_min * a - 1.0
    f0_mel = torch.where(f0_mel > 0, f0_mel * a - b, f0_mel)
    # torch.clip_(f0_mel, min=1., max=float(f0_bin - 1))
    f0_coarse = torch.round(f0_mel).long()
    f0_coarse = f0_coarse * (f0_coarse > 0)
    f0_coarse = f0_coarse + ((f0_coarse < 1) * 1)
    f0_coarse = f0_coarse * (f0_coarse < f0_bin)
    f0_coarse = f0_coarse + ((f0_coarse >= f0_bin) * (f0_bin - 1))
    return f0_coarse


def get_f0_predictor(f0_predictor, hop_length, sampling_rate, **kargs):
    if f0_predictor == "crepe":
        from modules.F0Predictor.CrepeF0Predictor import CrepeF0Predictor

        f0_predictor_object = CrepeF0Predictor(
            hop_length=hop_length,
            sampling_rate=sampling_rate,
            device=kargs["device"],
            threshold=kargs["threshold"],
        )
    elif f0_predictor == "dio":
        from modules.F0Predictor.DioF0Predictor import DioF0Predictor

        f0_predictor_object = DioF0Predictor(
            hop_length=hop_length, sampling_rate=sampling_rate
        )
    elif f0_predictor == "rmvpe":
        from modules.F0Predictor.RMVPEF0Predictor import RMVPEF0Predictor

        f0_predictor_object = RMVPEF0Predictor(
            hop_length=hop_length,
            sampling_rate=sampling_rate,
            dtype=torch.float32,
            device=kargs["device"],
            threshold=kargs["threshold"],
        )
    else:
        raise Exception("Unknown f0 predictor")
    return f0_predictor_object


EPS = 1e-6
eps_torch = torch.FloatTensor([EPS])


def const_to_log_trans_np(signal):
    return np.log2(signal + EPS)


def log_to_constant_trans_np(signal):
    return 2**signal - EPS


def const_to_log_trans_torch(signal):
    return torch.log2(signal + eps_torch.to(signal.device))


def log_to_constant_trans_torch(signal):
    return 2**signal - eps_torch.to(signal.device)


def const_signal_masking_torch(signal, mask):
    return torch.where(mask, signal, eps_torch.to(signal.device))


def log_signal_masking_torch(signal, mask):
    return torch.where(mask, signal, eps_torch.to(signal.device))
