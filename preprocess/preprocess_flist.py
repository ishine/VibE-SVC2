import argparse
import os
from random import seed, shuffle

import librosa
from loguru import logger
from tqdm import tqdm

from utils.commons.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="./dataset/vocalset", help="path to source dir"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="/configs/diffusion.yaml",
        help="set yaml path",
    )
    parser.add_argument("--style_type", type=str, default="pitch")
    return parser.parse_args()


def main(args):
    seed(1234)

    train = []
    val = []
    test = []

    skip_file_cnt = 0
    total_file_cnt = 0
    config = load_config(args.config_path)
    if args.style_type == "pitch":
        style_config = config.singing_techniques.pitch_tech
    elif args.style_type == "timbre":
        style_config = config.singing_techniques.timbre_tech

    for speaker in tqdm(sorted(os.listdir(args.data_dir))):
        dataset_name = speaker.split("#")[0]

        tech_dict = {}
        wav_list = sorted(
            f for f in os.listdir(f"{args.data_dir}/{speaker}") if f.endswith(".wav")
        )
        total_file_cnt += len(wav_list)
        for fname in tqdm(wav_list):
            file_name = f"{args.data_dir}/{speaker}/{fname}"

            
            technique_type = file_name.split("/")[-1].split("#")[2]
           
            if technique_type not in style_config.keys():
                continue

            wav, sr = librosa.load(file_name)
            file_duration = librosa.get_duration(y=wav, sr=sr)
            if file_duration < config.data.duration + 0.1:
                # logger.info("Skip too short audio: " + file_name)
                skip_file_cnt += 1
                continue

            # Create technique dict
            if not tech_dict.get(technique_type):
                tech_dict[technique_type] = [file_name]
            else:
                tech_dict[technique_type].append(file_name)
            
     
        # Eval-only mode: all entries go to test, no train/val split
        if getattr(config.data, "eval_only", False):
            for tech in tech_dict.keys():
                tech_wavs = tech_dict[tech]
                shuffle(tech_wavs)
                test += tech_wavs
            continue
        # Split val & test set for 2 samples for each speaker and technique
        if dataset_name == "VocalSet" or dataset_name == "vocalset":
            for tech in tech_dict.keys():
                tech_wavs = tech_dict[tech]
                shuffle(tech_wavs)

                train += tech_wavs[4:]
                val += tech_wavs[2:4]
                test += tech_wavs[:2]
        # Split filelist for the gtsinger dataset
        elif dataset_name == "GTSinger" or dataset_name == "gtsinger":
            for tech in tech_dict.keys():

                tech_wavs = tech_dict[tech]
                shuffle(tech_wavs)
                test_size = int(len(tech_dict[tech]) * 0.1)

                train += tech_wavs[test_size * 2 :]
                val += tech_wavs[test_size : test_size * 2]
                test += tech_wavs[:test_size]
        else:
            logger.info("Unavailable Dataset!")

    shuffle(train)
    shuffle(val)
    shuffle(test)

    # filelist environment setting
    flist_env = f"./filelists/{config.env.exp}"
    os.makedirs(flist_env, exist_ok=True)

    train_list_path = f"{flist_env}/{config.data.flist_train}"
    val_list_path = f"{flist_env}/{config.data.flist_valid}"
    test_list_path = f"{flist_env}/{config.data.flist_test}"

    logger.info("Skip audio shorter than " + str(config.data.duration))
    logger.info(f"Total : {total_file_cnt} files, Skip : {skip_file_cnt} files.")
    logger.info(f"Filelist environment : {train_list_path}" + "\n")

    if not getattr(config.data, "eval_only", False):
        with open(train_list_path, "w") as f:
            for fname in tqdm(train):
                f.write(str(fname) + "\n")
        with open(val_list_path, "w") as f:
            for fname in tqdm(val):
                f.write(str(fname) + "\n")
    with open(test_list_path, "w") as f:
        for fname in tqdm(test):
            f.write(str(fname) + "\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
