import os
import random
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from style_encoder.dwt import smoothing_dwt
from style_encoder.utils import const_to_log_trans_np


def get_data_loader(audio_path, config, device, task, n_gpus):
    dataset = AudioDataset(
        audio_path=audio_path,
        config=config,
        device=device,
        task=task,
    )
    sampler = DistributedSampler(
        dataset=dataset, shuffle=(True if task == "train" else False)
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=(
            int(config.optimizer.style_enc.batch_size / n_gpus)
            if task == "train"
            else 1
        ),
        shuffle=False,
        num_workers=0,
        sampler=(sampler if task == "train" else None),
        pin_memory=False,
    )
    return data_loader, sampler


class AudioDataset(Dataset):
    def __init__(
        self,
        audio_path,
        config,
        device="cpu",
        task=None,
    ):
        super().__init__()
        self.audio_path = audio_path
        self.config = config
        self.waveform_sec = 2
        self.device = device
        self.task = task

        # data buffer
        self.data_buffer = {}
        self.paths = []
        f_list_path = []

        self.hop_size = self.config.audio.hop_size
        self.sample_rate = self.config.audio.sampling_rate

        # Types of Vocalset techinques for training the model

        tech2id = config.tech

        with open(
            os.path.join(config.path.flist.dir_flist, config.path.flist[task]),
            "r",
        ) as f:
            f_list_path = f.readlines()

        # add another language
        if task == "train":
            other_lang = glob("./dataset/EN_gts/*/*.wav")

            new_lang = []
            for filename in other_lang:
                if filename.split("/")[-2].split("-")[0] == "EN":
                    continue
                else:
                    new_lang.append(filename)

            f_list_path += new_lang
        # load filelist from the director
        for audio in tqdm(f_list_path):
            audio_fname = audio.strip()
            name_ext = os.path.splitext(os.path.basename(audio_fname))[0]

            split_audio_name = name_ext.split("#")
            tech = split_audio_name[2]
            spk = audio_fname.split("/")[-2]
            group = split_audio_name[3]

            if group != "Vibrato":
                continue

            # skip another techniques
            if tech not in tech2id.keys():
                continue

            # Load F0
            f0, vuv = np.load(f"{audio_fname}.f0.npy", allow_pickle=True)
            f0 = np.nan_to_num(np.array(f0, dtype=float))
            vuv = np.nan_to_num(np.array(vuv, dtype=float))
            # seg_frame_len = int(5 * self.sample_rate / self.hop_size)

            # if f0.shape[0] < seg_frame_len:
            #     continue

            lf0 = const_to_log_trans_np(f0)

            # Decompose F0 to Low- and High-frequency contour
            low_lf0, high_lf0 = smoothing_dwt(
                lf0,
                config.dwt.wavelet,
                config.dwt.factor,
            )

            low_lf0 = np.nan_to_num(np.array(low_lf0, dtype=float))
            high_lf0 = np.nan_to_num(np.array(high_lf0, dtype=float))

            # Cast type to torch
            lf0 = torch.from_numpy(lf0).float().unsqueeze(-1).to(self.device)
            low_lf0 = torch.from_numpy(low_lf0).float().unsqueeze(-1).to(self.device)
            high_lf0 = torch.from_numpy(high_lf0).float().unsqueeze(-1).to(self.device)
            vuv = torch.from_numpy(vuv).float().unsqueeze(-1).to(self.device)
            self.paths.append(name_ext)

            # Spaker & Style ID
            # spk_id = torch.Tensor([spk2id[spk]]).int().to(device)
            style_feat = torch.load(f"{audio_fname}.mert.pt").to(device)
            if config.train.cache_fp16:
                style_feat = style_feat.half()
            self.data_buffer[name_ext] = {
                "name": name_ext,
                "audio_path": audio_fname,
                "style_feat": style_feat,
                # "spk_id": spk_id,
                "lf0": lf0,
                "low_lf0": low_lf0,
                "vuv": vuv,
                "high_lf0": high_lf0,
            }
        print(f"Total {len(self.data_buffer)} audio samples are detected!\n")

    def __getitem__(self, file_idx):
        name_ext = self.paths[file_idx]
        return self.get_data(self.data_buffer[name_ext])

    def get_data(self, data_buffer):
        f0_frame_len = data_buffer["lf0"].size(0)
        seg_frame_len = int(self.waveform_sec * self.sample_rate / self.hop_size)

        start_frame = (
            0
            if self.task == "valid"
            else int(random.uniform(0, f0_frame_len - seg_frame_len - 1))
        )
        segment_size = (
            data_buffer["lf0"].size(0)
            if self.task == "valid"
            else int(self.waveform_sec * self.sample_rate / self.hop_size)
        )

        end_frame = start_frame + segment_size

        # segmentation
        low_lf0 = data_buffer["low_lf0"][start_frame:end_frame]
        lf0 = data_buffer["lf0"][start_frame:end_frame]
        vuv = data_buffer["vuv"][start_frame:end_frame]
        high_lf0 = data_buffer["high_lf0"][start_frame:end_frame]

        # augmentation
        if self.task == "train":
            aug_shift = torch.FloatTensor([random.uniform(0.5, 2)]).to(self.device)
            lf0 *= aug_shift

        return dict(
            name=data_buffer["name"],
            audio_path=data_buffer["audio_path"],
            # spk_id=data_buffer["spk_id"],
            style_feat=data_buffer["style_feat"],
            orig_f0=lf0,
            smooth_f0=low_lf0,
            uv_vector=vuv,
            detail_pitch=high_lf0,
        )

    def __len__(self):
        return len(self.paths)
