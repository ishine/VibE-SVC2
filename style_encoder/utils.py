import numpy as np
import torch

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
