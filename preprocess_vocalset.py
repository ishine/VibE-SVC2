import argparse

from preprocess import *


def main(args):
    cur_stage = args.start_stage
    # Stage 0 : Data Preparation
    if cur_stage == 0:
        rename_vocalset(args)
        preprocess_resample(args)
        cur_stage += 1
        print()

    # Stage 1 : Prepare Configureation
    if cur_stage < 2:
        # preprocessing for pitch style conversion
        print("Generating filelists ...")
        args.config_path = "configs/vibe_pitch.yaml"
        args.style_type = "pitch"
        preprocess_flist(args)

        # preprocessing for timbre style conversion
        args.config_path = "configs/vibe_timbre.yaml"
        args.style_type = "timbre"
        preprocess_flist(args)
        cur_stage += 1
        print()

    # Stage 2 : Extract feature set
    if cur_stage < 3:
        preprocess_feature_extraction(args)
        cur_stage += 1

    # Stage 3: Extract speaker statistic of F0
    if cur_stage < 4:
        preprocesss_spk_statistics(args)


def parse_args():
    parser = argparse.ArgumentParser()

    # model configuration
    parser.add_argument(
        "--start_stage",
        type=int,
        default=0,
    )

    # environment configureation
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="./dataset/vocalset_raw",
        help="path to source dir",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset/vocalset_style",
        help="path to source dir",
    )

    # preprocess_resample.py
    parser.add_argument(
        "--target_sr", type=int, default=24000, help="Target sampling rate"
    )
    parser.add_argument(
        "--skip_loudnorm",
        action="store_true",
        help="Skip loudness matching if you have done it",
    )

    # preprocess_flist
    parser.add_argument(
        "--config_path",
        type=str,
        default="/configs/diffusion.yaml",
        help="set yaml path",
    )
    parser.add_argument("--style_type", type=str, default="pitch")

    # preprocess_hubert_f0
    parser.add_argument(
        "--vocoder",
        type=str,
        default="bigvgan_v2",
        help="Available Vocoder| 1. BigVGAN_v2",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="You are advised to set the number of processes to the same as the number of CPU cores",
    )
    parser.add_argument("-d", "--device", type=str, default=None)

    # preprocess spk stats
    parser.add_argument(
        "--target_stats_dir",
        type=str,
        default="configs/stats",
        help="set yaml path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="vocalset",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
