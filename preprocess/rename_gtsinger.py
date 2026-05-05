import argparse
import os
import shutil
from glob import glob

from tqdm import tqdm

DATASET_NAME = "GTSinger"


def main(args):
    raw_dir = args.raw_dir
    source_dir = args.data_dir
    wav_list = glob(f"{raw_dir}/English/*/*/*/*/*.wav")

    for wav_path in tqdm(wav_list, leave=True):
        task, lang, spk, style, song, ctrl_group, basename = wav_path.split("/")[-7:]
        target_fname = (
            f"{source_dir}/{DATASET_NAME}#{spk}/{spk}#{lang}#{ctrl_group}#{style}#{song}#{basename}"
        )

        os.makedirs(os.path.dirname(target_fname), exist_ok=True)
        shutil.copy(wav_path, target_fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir", type=str, default="./dataset/GTSinger", help="path to source dir"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset/gtsinger_style",
        help="path to source dir",
    )
    args = parser.parse_args()
    main(args)
