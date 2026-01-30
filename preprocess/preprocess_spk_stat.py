import argparse
import os
from glob import glob

import numpy as np
import yaml
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset/vocalset_style",
        help="path to source dir",
    )

    parser.add_argument(
        "--target_stats_dir",
        type=str,
        default="configs/stats",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="vocalset",
    )
    return parser.parse_args()


def main(args):
    # select style annotation for each dataset
    if args.dataset == "vocalset":
        styles = {
            "pitch": ["straight", "vibrato"],
            "timbre": ["straight", "belt", "breathy"],
        }
    elif args.dataset == "gtsinger":
        styles = {"pitch": ["Control_Group", "Vibrato_Group"]}

    # get statistics of F0 contour for each dataset and each style types
    for style_type in styles.keys():
        spk_stat_dict = {}
        os.makedirs(args.target_stats_dir, exist_ok=True)
        print(f"\nProcessig for {style_type} statistics")

        #
        for spk_folder in os.listdir(args.data_dir):
            # Load F0 from preprocessed feature
            filenames = glob(f"{args.data_dir}/{spk_folder}/*.f0.npy")

            # aggregate the mean value of F0 contour for each utterances
            spk_mean_arr = []
            for f0_file in tqdm(filenames):
                style = os.path.basename(f0_file).split("#")[2]
                if style not in styles[style_type]:
                    continue
                f0_contour, _ = np.load(f0_file, allow_pickle=True)
                spk_mean_arr.append(np.mean(f0_contour))
            mean_spk = np.mean(spk_mean_arr)
            spk_stat_dict[spk_folder] = float(mean_spk)
        with open(f"{args.target_stats_dir}/f0_{style_type}.yaml", "w") as f:
            yaml.dump(spk_stat_dict, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
