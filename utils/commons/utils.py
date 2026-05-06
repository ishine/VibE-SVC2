import logging
import os
import random
import subprocess
import sys
from collections import deque

import numpy as np
import soundfile as sf
import torch
import yaml
from scipy.io.wavfile import read
from torch.nn import functional as F

logging.basicConfig(stream=sys.stdout, level=logging.WARN)


class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __hash__(self):
        # Convert to a frozenset of sorted items to make it hashable
        try:
            return hash(frozenset(self._flatten().items()))
        except TypeError:
            raise TypeError("Unhashable value inside DotDict")


def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=22050,
):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warn(
            "{} is not a git repository, therefore hash value comparison will be ignored.".format(
                source_dir
            )
        )
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warn(
                "git hash values are different. {}(saved) != {}(current)".format(
                    saved_hash[:8], cur_hash[:8]
                )
            )
    else:
        open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


def repeat_expand_2d(content, target_len, mode="left"):
    # content : [h, t]
    return (
        repeat_expand_2d_left(content, target_len)
        if mode == "left"
        else repeat_expand_2d_other(content, target_len, mode)
    )


def repeat_expand_2d_left(content, target_len):
    # content : [h, t]

    src_len = content.shape[-1]
    target = torch.zeros([content.shape[0], target_len], dtype=torch.float).to(
        content.device
    )
    temp = torch.arange(src_len + 1) * target_len / src_len
    current_pos = 0
    for i in range(target_len):
        if i < temp[current_pos + 1]:
            target[:, i] = content[:, current_pos]
        else:
            current_pos += 1
            target[:, i] = content[:, current_pos]

    return target


# mode : 'nearest'| 'linear'| 'bilinear'| 'bicubic'| 'trilinear'| 'area'
def repeat_expand_2d_other(content, target_len, mode="nearest"):
    # content : [h, t]
    content = content[None, :, :]
    target = F.interpolate(content, size=target_len, mode=mode)[0]
    return target


def load_config(path_config):
    with open(path_config, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    return args


def save_config(path_config, config):
    config = dict(config)
    with open(path_config, "w") as f:
        yaml.dump(config, f)


def split_list_per_gpus(n_gpus, flist):
    """
    Split list for multi-gpu inference mode.
    """
    len_chunk = len(flist) // n_gpus
    list_per_gpu = [flist[len_chunk * i : len_chunk * (i + 1)] for i in range(n_gpus)]
    list_per_gpu[-1] += flist[len_chunk * (n_gpus) :]

    # Shape of list_per_gpu : [n_gpus, len_chunk]
    return list_per_gpu


def list_to_chunk(lst, n):
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def get_ref_list(audio_list, mp=False):
    ref_list = []
    style_dict = {"straight": deque(), "vibrato": deque(), "Control_Group" : deque(), "Vibrato_Group" : deque()}
    new_audio_list = []

    for fname in audio_list:
        fname = fname.strip()
        basename = fname.split("/")[-1]

        # Condition 1a: _0.wav 샘플만 포함
        if not basename.split("#")[-1].endswith("_0.wav"):
            continue

        # Condition 1b: 10초 미만 샘플만 포함
        if sf.info(fname).duration >= 10.0:
            continue

        ftech = basename.split("#")[2]
        if ftech not in style_dict:
            continue
        if len(style_dict[ftech]) < 35:
            style_dict[ftech].append(fname)
            new_audio_list.append(fname)

    # Condition 2: 각 쌍의 개수를 작은 쪽에 맞춰 균형을 맞춤
    # straight <-> vibrato
    n_straight = len(style_dict["straight"])
    n_vibrato_vc = len(style_dict["vibrato"])
    if n_straight != n_vibrato_vc:
        n_min = min(n_straight, n_vibrato_vc)
        for key in ("straight", "vibrato"):
            excess = set(list(style_dict[key])[n_min:])
            style_dict[key] = deque(list(style_dict[key])[:n_min])
            new_audio_list = [f for f in new_audio_list if f not in excess]

    # Control_Group <-> Vibrato_Group
    n_vibrato = len(style_dict["Vibrato_Group"])
    if len(style_dict["Control_Group"]) > n_vibrato:
        excess = set(list(style_dict["Control_Group"])[n_vibrato:])
        style_dict["Control_Group"] = deque(list(style_dict["Control_Group"])[:n_vibrato])
        new_audio_list = [f for f in new_audio_list if f not in excess]

    random.seed(42)
    random.shuffle(new_audio_list)

    for fname in new_audio_list:
        ftech = fname.split("/")[-1].split("#")[2]

        if ftech == "straight":
            target_tech = "vibrato"
        elif ftech == "vibrato":
            target_tech = "straight"
        elif ftech == "Control_Group":
            target_tech = "Vibrato_Group"
        elif ftech == "Vibrato_Group":
            target_tech = "Control_Group"
        else:
            continue

        ref_list.append(style_dict[target_tech].popleft())

    return ref_list, new_audio_list


def get_style_lists(audio_list, target_tech, clip_filter=True, duration_filter=True, shuffle=True):
    """
    target_tech 기반으로 ref_list와 source_list를 분리 반환.

    Args:
        audio_list      : filelist 줄 목록 (strip 미적용 상태여도 무방)
        target_tech     : 변환 대상 style ('straight'|'vibrato'|'Control_Group'|'Vibrato_Group')
        clip_filter     : True이면 _0.wav 파일만 포함 (GTSinger 기본값)
        duration_filter : True이면 10초 미만 파일만 포함 (GTSinger 기본값)
        shuffle         : True이면 ref/source 순서를 랜덤 셔플 (GTSinger용, seed=42 고정)

    Returns:
        (ref_list, source_list)
        ref_list    — target_tech style 파일들
        source_list — 반대 style 파일들 (source → target 변환에 사용)
        두 리스트 길이는 min(len_ref, len_source)로 균형 맞춤.
    """
    tech_pair = {
        "straight":     "vibrato",
        "vibrato":      "straight",
        "Control_Group":  "Vibrato_Group",
        "Vibrato_Group":  "Control_Group",
    }
    source_tech = tech_pair[target_tech]

    ref_bucket    = []
    source_bucket = []

    for fname in audio_list:
        fname = fname.strip()
        if not fname:
            continue
        basename = fname.split("/")[-1]

        if clip_filter and not basename.split("#")[-1].endswith("_0.wav"):
            continue
        if duration_filter and sf.info(fname).duration >= 10.0:
            continue

        ftech = basename.split("#")[2]
        if ftech == target_tech:
            ref_bucket.append(fname)
        elif ftech == source_tech:
            source_bucket.append(fname)

    if shuffle:
        random.seed(42)
        random.shuffle(ref_bucket)
        random.seed(42)
        random.shuffle(source_bucket)

    n = min(len(ref_bucket), len(source_bucket))
    return ref_bucket[:n], source_bucket[:n]
