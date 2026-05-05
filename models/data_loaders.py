import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils.commons.utils import repeat_expand_2d


# Add Distributed sampler
def get_dataloader(args, n_gpus, task):
    dataset = AudioDataset(
        args,
        task,
        fp16=args.train.cache.cache_fp16 if task == "train" else False,
        use_aug=args.model.aug_type.volume_aug if task == "train" else False,
        device=args.train.cache.cache_device,
    )
    data_sampler = DistributedSampler(dataset=dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=(
            int(args.train.d_loader.batch_size / n_gpus) if task == "train" else 1
        ),
        shuffle=False,
        num_workers=(
            int(args.train.d_loader.num_workers / n_gpus)
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


class AudioDataset(Dataset):
    def __init__(
        self,
        args,
        task,
        fp16=False,
        use_aug=False,
        device="cpu",
    ):
        super().__init__()
        self.waveform_sec = args.data.duration
        self.sample_rate = args.data.sampling_rate
        self.hop_size = args.data.hop_size

        spk_config = args.spk
        if args.model.LUT.n_t_tech == 1:
            tech_config = args.singing_techniques.pitch_tech
        else:
            tech_config = args.singing_techniques.timbre_tech

        self.task = task
        self.use_aug = use_aug
        unit_interpolate_mode = args.data.encoder_unit_interpolate_mode
        filelists = f"./filelists/{args.env.exp}/{args.data.flist_train}"

        self.data_buffer = {}
        self.paths = []

        # Load filelist from .txt
        with open(filelists, "r") as f:
            self.paths = f.read().splitlines()

        for name_ext in tqdm(self.paths, total=len(self.paths)):
            # Load F0
            path_f0 = name_ext + ".f0.npy"
            f0, _ = np.load(path_f0, allow_pickle=True)
            f0 = (
                torch.from_numpy(np.array(f0, dtype=float))
                .float()
                .unsqueeze(-1)
                .to(device)
            )

            # Load volume contour
            path_volume = name_ext + ".vol.npy"
            volume = np.load(path_volume)
            volume = torch.from_numpy(volume).float().unsqueeze(-1).to(device)

            # Load spk ID
            spk_name = name_ext.split("/")[-2]
            spk_id = spk_config[spk_name] if spk_name in spk_config else 0
            spk_id = torch.LongTensor(np.array([spk_id])).to(device)

            # Load Tech ID
            tech_type = name_ext.split("/")[-1].split("#")[2]
            tech_id = tech_config[tech_type]
            tech_id = torch.LongTensor(np.array([tech_id])).to(device)

            # Load Mel
            path_mel = name_ext + ".mel.npy"
            mel = np.load(path_mel)
            mel = torch.from_numpy(mel).to(device)

            # Augmentation
            if self.use_aug:
                path_augvol = name_ext + ".aug_vol.npy"
                aug_vol = np.load(path_augvol)
                aug_vol = torch.from_numpy(aug_vol).float().unsqueeze(-1).to(device)

                path_augmel = name_ext + ".aug_mel.npy"
                aug_mel = np.load(path_augmel, allow_pickle=True)
                aug_mel = np.array(aug_mel, dtype=float)
                aug_mel = torch.from_numpy(aug_mel).to(device)

            # Load unit
            path_units = name_ext + ".soft.pt"
            units = torch.load(path_units).to(device)
            units = repeat_expand_2d(
                units[0], f0.size(0), unit_interpolate_mode
            ).transpose(0, 1)

            # FP 16 setting
            if fp16:
                mel = mel.half()
                units = units.half()

                if self.use_aug:
                    aug_mel = aug_mel.half()

            self.data_buffer[name_ext] = {
                "mel": mel,
                "units": units,
                "f0": f0,
                "volume": volume,
                "spk_id": spk_id,
                "tech_id": tech_id,
            }
            if self.use_aug:
                self.data_buffer[name_ext].update(
                    {"aug_mel": aug_mel, "aug_vol": aug_vol}
                )

    def __getitem__(self, file_idx):
        name_ext = self.paths[file_idx]
        data_buffer = self.data_buffer[name_ext]
        return self.get_data(name_ext, data_buffer)

    def get_data(self, name_ext, data_buffer):
        name = os.path.splitext(name_ext)[0]
        mel_frame_len = data_buffer["mel"].size(1)
        seg_frame_len = int(self.waveform_sec * self.sample_rate / self.hop_size)

        start_frame = (
            int(random.uniform(0, mel_frame_len - seg_frame_len - 1))
            if self.task == "train"
            else 0
        )
        units_frame_len = seg_frame_len if self.task == "train" else mel_frame_len

        aug_flag = random.choice([True, False]) and self.use_aug

        # Crop features
        mel_key = "aug_mel" if aug_flag else "mel"
        mel = data_buffer.get(mel_key).transpose(0, 1)
        mel = mel[start_frame : start_frame + units_frame_len]

        f0 = data_buffer.get("f0")
        f0_frames = f0[start_frame : start_frame + units_frame_len]

        units = data_buffer.get("units")
        units = units[start_frame : start_frame + units_frame_len]

        vol_key = "aug_vol" if aug_flag else "volume"
        volume = data_buffer.get(vol_key)
        volume_frames = volume[start_frame : start_frame + units_frame_len]

        spk_id = data_buffer.get("spk_id")
        tech_id = data_buffer.get("tech_id")

        return dict(
            mel=mel,
            f0=f0_frames,
            volume=volume_frames,
            units=units,
            spk_id=spk_id,
            tech_id=tech_id,
            name=name,
            name_ext=name_ext,
        )

    def __len__(self):
        return len(self.paths)
