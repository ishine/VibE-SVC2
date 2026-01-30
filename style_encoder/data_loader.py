import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from style_encoder.utils import const_to_log_trans_np
from utils.commons.dwt import separate_signal_dwt


def get_dataloader(args, task, feature_type):
    if feature_type == "f0":
        dataset = F0Dataset(
            args,
            task,
            fp16=args.train.cache.cache_fp16 if task == "train" else False,
            device=args.train.cache.cache_device,
        )
    elif feature_type == "energy":
        dataset = EnergyDataset(
            args,
            task,
            fp16=args.train.cache.cache_fp16 if task == "train" else False,
            device=args.train.cache.cache_device,
        )
    data_sampler = DistributedSampler(dataset=dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=(int(args.train.d_loader.batch_size) if task == "train" else 1),
        shuffle=False,
        num_workers=(
            int(args.train.d_loader.num_workers)
            if args.train.cache.cache_device == "cpu"
            else 0
        ),
        persistent_workers=(
            (args.train.d_loader.num_workers > 0)
            if args.train.cache.cache_device == "cpu"
            else False
        ),
        sampler=data_sampler if task == "train" else None,
        pin_memory=True if args.train.cache.cache_device == "cpu" else False,
    )
    return data_loader, data_sampler


class F0Dataset(Dataset):
    def __init__(
        self,
        args,
        task,
        fp16=False,
        device="cpu",
    ):
        super().__init__()
        self.waveform_sec = args.data.duration
        self.sample_rate = args.data.sampling_rate
        self.hop_size = args.data.hop_size

        self.seg_frame_len = int(self.waveform_sec * self.sample_rate / self.hop_size)

        spk_config = args.spk
        tech_config = args.singing_techniques.pitch_tech

        self.device = device
        self.task = task

        self.task = task
        self.use_low_aug = False
        filelists = f"./filelists/{args.env.exp}/{args.data.flist_train}"

        self.paths = []

        with open(filelists, "r") as f:
            flist = f.read().splitlines()

        # load filelist from the director
        for name_ext in tqdm(flist, total=len(flist)):

            # Load F0
            path_f0 = name_ext + ".f0.npy"
            f0, vuv = np.load(path_f0, allow_pickle=True)
            f0 = np.nan_to_num(np.array(f0, dtype=float))
            vuv = np.nan_to_num(np.array(vuv, dtype=float))

            lf0 = const_to_log_trans_np(f0)

            # Decompose F0 to Low- and High-frequency contour
            low_lf0, high_lf0 = separate_signal_dwt(
                lf0,
                args.data.dwt_type,
                args.data.dwt_level,
            )

            low_lf0 = np.nan_to_num(np.array(low_lf0, dtype=float))
            high_lf0 = np.nan_to_num(np.array(high_lf0, dtype=float))

            # Cast type to torch
            lf0 = torch.from_numpy(lf0).float()  # [T, ]
            low_lf0 = torch.from_numpy(low_lf0).float()  # [T, ]
            high_lf0 = torch.from_numpy(high_lf0).float()  # [T, ]
            vuv = torch.from_numpy(vuv).float()  # [T, ]

            # Load speaker ID
            spk_name = name_ext.split("/")[-2]
            spk_id = torch.LongTensor([spk_config[spk_name]])

            # Load style ID
            style_type = name_ext.split("/")[-1].split("#")[2]
            style_id = torch.LongTensor([tech_config[style_type]])

            sample = {
                "name": os.path.splitext(name_ext)[0],
                "audio_path": name_ext,
                "style_id": style_id,
                "spk_id": spk_id,
                "f0": lf0,
                "low_f0": low_lf0,
                "uv": vuv,
                "high_f0": high_lf0,
            }
            self.paths.append(sample)

    def __getitem__(self, index):
        items = self.paths[index]
        f0_len = items["f0"].shape[-1]

        sample = {}
        if self.task == "train":
            start_frame_idx = int(random.uniform(0, f0_len - self.seg_frame_len - 1))
            end_frame_idx = start_frame_idx + self.seg_frame_len

            f0 = items["f0"][start_frame_idx:end_frame_idx]
            low_f0 = items["low_f0"][start_frame_idx:end_frame_idx]
            high_f0 = items["high_f0"][start_frame_idx:end_frame_idx]
            uv = items["uv"][start_frame_idx:end_frame_idx]

            if self.use_low_aug:
                low_f0 = items["low_f0"] * random.uniform(0.5, 2)
        else:
            f0 = items["f0"]
            low_f0 = items["low_f0"]
            high_f0 = items["high_f0"]
            uv = items["uv"]

        sample = {
            "idx": index,
            "spk_id": items["spk_id"],
            "style_id": items["style_id"],
            "f0": f0,
            "low_f0": low_f0,
            "high_f0": high_f0,
            "uv": uv,
        }

        return sample

    def __len__(self):
        return len(self.paths)


class EnergyDataset(Dataset):
    def __init__(
        self,
        args,
        task,
        fp16=False,
        device="cpu",
    ):
        super().__init__()
        self.waveform_sec = args.data.duration
        self.sample_rate = args.data.sampling_rate
        self.hop_size = args.data.hop_size

        self.seg_frame_len = int(self.waveform_sec * self.sample_rate / self.hop_size)

        spk_config = args.spk
        tech_config = args.singing_techniques.pitch_tech

        self.device = device
        self.task = task

        self.task = task
        self.use_low_aug = False
        filelists = f"./filelists/{args.env.exp}/{args.data.flist_train}"

        self.paths = []

        with open(filelists, "r") as f:
            flist = f.read().splitlines()

        # load filelist from the director
        for name_ext in tqdm(flist, total=len(flist)):

            # Load F0
            path_f0 = name_ext + ".f0.npy"
            _, vuv = np.load(path_f0, allow_pickle=True)

            vuv = np.nan_to_num(np.array(vuv, dtype=float))

            # Load energy
            path_energy = name_ext + ".vol.npy"
            vol = np.load(path_energy)

            # Decompose F0 to Low- and High-frequency contour
            low_vol, high_vol = separate_signal_dwt(
                vol,
                args.data.dwt_type,
                args.data.dwt_level,
            )

            low_vol = np.nan_to_num(np.array(low_vol, dtype=float))
            high_vol = np.nan_to_num(np.array(high_vol, dtype=float))

            # Cast type to torch
            vol = torch.from_numpy(vol).float()  # [T, ]
            low_vol = torch.from_numpy(low_vol).float()  # [T, ]
            high_vol = torch.from_numpy(high_vol).float()  # [T, ]
            vuv = torch.from_numpy(vuv).float()  # [T, ]

            # Load speaker ID
            spk_name = name_ext.split("/")[-2]
            spk_id = torch.LongTensor([spk_config[spk_name]])

            # Load style ID
            style_type = name_ext.split("/")[-1].split("#")[2]
            style_id = torch.LongTensor([tech_config[style_type]])

            sample = {
                "name": os.path.splitext(name_ext)[0],
                "audio_path": name_ext,
                "style_id": style_id,
                "spk_id": spk_id,
                "vol": vol,
                "low_vol": low_vol,
                "uv": vuv,
                "high_vol": high_vol,
            }
            self.paths.append(sample)

    def __getitem__(self, index):
        items = self.paths[index]
        vol_len = items["vol"].shape[-1]

        sample = {}
        if self.task == "train":
            start_frame_idx = int(random.uniform(0, vol_len - self.seg_frame_len - 1))
            end_frame_idx = start_frame_idx + self.seg_frame_len

            vol = items["vol"][start_frame_idx:end_frame_idx]
            low_vol = items["low_vol"][start_frame_idx:end_frame_idx]
            high_vol = items["high_vol"][start_frame_idx:end_frame_idx]
            uv = items["uv"][start_frame_idx:end_frame_idx]

            if self.use_low_aug:
                low_vol = items["low_vol"] * random.uniform(0.5, 2)
        else:
            vol = items["vol"]
            low_vol = items["low_vol"]
            high_vol = items["high_vol"]
            uv = items["uv"]

        sample = {
            "idx": index,
            "spk_id": items["spk_id"],
            "style_id": items["style_id"],
            "vol": vol,
            "low_vol": low_vol,
            "high_vol": high_vol,
            "uv": uv,
        }

        return sample

    def __len__(self):
        return len(self.paths)
