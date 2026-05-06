import argparse
import json
import os
import re
import shutil
from glob import glob

from tqdm import tqdm

DATASET_NAME = "GTSinger"
TAG_RE = re.compile(r"<[^>]+>")


def extract_transcription(json_path):
    with open(json_path) as f:
        data = json.load(f)
    words = [d["word"] for d in data if "word" in d]
    text = " ".join(words)
    text = TAG_RE.sub("", text)
    return " ".join(text.split())


def main(args):
    raw_dir = args.raw_dir
    source_dir = args.data_dir
    wav_list = glob(f"{raw_dir}/English/*/*/*/*/*.wav")

    skipped = 0
    for wav_path in tqdm(wav_list, leave=True):
        task, lang, spk, style, song, ctrl_group, basename = wav_path.split("/")[-7:]
        target_fname = (
            f"{source_dir}/{spk}/{spk}#{lang}#{ctrl_group}#{style}#{song}#{basename}"
        )

        os.makedirs(os.path.dirname(target_fname), exist_ok=True)
        shutil.copy(wav_path, target_fname)

        json_path = wav_path[:-4] + ".json"
        if not os.path.exists(json_path):
            print(f"WARNING: missing transcription JSON, skip txt: {wav_path}")
            skipped += 1
            continue
        target_txt = target_fname[:-4] + ".txt"
        with open(target_txt, "w") as f:
            f.write(extract_transcription(json_path))

    if skipped:
        print(f"Total skipped (no JSON): {skipped}")


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
