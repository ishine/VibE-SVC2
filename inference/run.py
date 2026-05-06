import argparse
import os
import warnings

import torch
import torch.multiprocessing as mp

from utils.commons.utils import get_style_lists, split_list_per_gpus

warnings.filterwarnings("ignore")


def main():
    # Check Environment
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6115"
    n_gpus = torch.cuda.device_count()

    print(f" > MASTER_ADDR : {os.environ['MASTER_ADDR']}")
    print(f" > MASTER_PORT : {os.environ['MASTER_PORT']}")
    print(f" > n_gpus : {n_gpus}")
    cfg = parse_args()

    assert torch.cuda.is_available(), "CPU generation is not allowed."

    # parse arguments

    if cfg.infer_mode == "pitch":
        if not cfg.pitch_zeroshot:
            from inference.run_vibesvcII import run
            tech_list = ["straight", "vibrato"]
        else:

            from inference.run_zsvibesvcII import run
            tech_list = ['Control_Group','Vibrato_Group']

        # tech_list = ["straight", "vibrato"]

    elif cfg.infer_mode == "timbre":
        from inference.run_vibesvcII import run

        tech_list = ["straight", "belt", "breathy", "vocal_fry"]
    elif cfg.infer_mode == "joint":
        if cfg.pitch_zeroshot:
            from inference.run_zsvibesvcII import run
        else:
            from inference.run_vibesvcII import run

        tech_list = ["straight", "belt", "breathy", "vocal_fry"]
    else:
        print("Unkown task is entered!")
        exit(0)

    # Load filelists for inference
    with open(f"{cfg.eval_filelist}", "r") as f:
        audio_pairs = f.readlines()

    audio_list = split_list_per_gpus(n_gpus, audio_pairs)
    if cfg.pitch_zeroshot:
        first_tech = audio_pairs[0].strip().split("/")[-1].split("#")[2]
        if first_tech in ("Control_Group", "Vibrato_Group", "Glissando_Group"):
            ref_ctrl, src_vib = get_style_lists(audio_pairs, "Control_Group")
            balanced = ref_ctrl + src_vib
            audio_list = split_list_per_gpus(n_gpus, balanced)
            ref_list = balanced
        else:
            ref_list = [f.strip() for f in audio_pairs if f.strip()]

    if cfg.multi_infer:
        base_exp_dir = cfg.exp_dir
        for style in tech_list:
            cfg.exp_dir = f"{base_exp_dir}/{cfg.task}/to_{style}"
            if cfg.infer_mode == "pitch":
                cfg.target_pitch_style = style
                cfg.target_timbre_style = None
            else:
                cfg.target_pitch_style = None
                cfg.target_timbre_style = style
            print(cfg.exp_dir)
            mp.spawn(
                run,
                nprocs=n_gpus,
                args=(n_gpus, cfg, audio_list),
            )
    else:
        if not cfg.pitch_zeroshot:
            mp.spawn(
                run,
                nprocs=n_gpus,
                args=(n_gpus, cfg, audio_list),
            )
        else:
            mp.spawn(
                run,
                nprocs=n_gpus,
                args=(n_gpus, cfg, audio_list, ref_list),
            )


def parse_args():
    parser = argparse.ArgumentParser()
    # Environment settings
    parser.add_argument(
        "-f", "--eval_filelist", type=str, default="filelists/vocalset_final/test.txt"
    )
    parser.add_argument(
        "-e",
        "--exp_dir",
        type=str,
        default="exp",
        help="Samples are generated in 'results/\{exp\}' directory.",
    )

    # feature settings
    parser.add_argument("-f0p", "--f0_predictor", type=str, default="rmvpe")
    parser.add_argument("--task", type=str, default="pitch_style_conversion")

    # model settings
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="logs/vocalset_final_vib_pretrain/diffusion/model_200000.pt",
        help="Pretrain SVC model path.",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default="logs/vocalset_final_vib_pretrain/diffusion/config.yaml",
    )
    parser.add_argument("-ks", "--k_step", type=int, default=1000)
    parser.add_argument("--pitch_zeroshot", action="store_true", default=False)
    parser.add_argument("--infer_mode", type=str, default=None, required=True)

    parser.add_argument(
        "-s",
        "--stats_path",
        type=str,
        default="configs/stats/f0_pitch.yaml",
    )

    # Task configuration
    parser.add_argument(
        "-rcs",
        "--recon_spk",
        action="store_true",
        default=False,
        help="Reconstruction speaker setting for svc",
    )
    parser.add_argument(
        "-rct",
        "--recon_tech",
        action="store_true",
        default=False,
        help="Reconstruction pitch technique setting for svc",
    )
    parser.add_argument(
        "--vocoded",
        action="store_true",
        default=False,
        help="Generate vocoded samples",
    )
    parser.add_argument(
        "--multi_infer", action="store_true", default=False, help="Multiple inference"
    )

    # Style conversion
    parser.add_argument(
        "--target_spk",
        type=str,
        default=None,
        help="Select target speaker to transfer",
    )
    parser.add_argument(
        "--target_timbre_style",
        type=str,
        default=None,
        help="Select timbre technique to transfer",
    )
    parser.add_argument(
        "--target_pitch_style",
        type=str,
        default=None,
        help="Select pitch technique to transfer",
    )
    parser.add_argument(
        "--extent_scale",
        type=float,
        default=1.0,
        help="Value for global extent scaling",
    )
    parser.add_argument(
        "--extent_scale_type",
        type=str,
        default="global",
        help="Options : global, inc_linear, dec_linear, sinusoidal",
    )
    parser.add_argument(
        "--rate_scale", type=float, default=1.0, help="Value for global rate scaling"
    )
    parser.add_argument(
        "--extent_scale_energy",
        type=float,
        default=1.0,
        help="Value for energy extent scaling",
    )
    parser.add_argument(
        "--rate_scale_energy",
        type=float,
        default=1.0,
        help="Value for energy rate scaling",
    )
    parser.add_argument(
        "-vfe",
        "--vocal_fry_enforcement",
        type=bool,
        default=False,
        help="Enable vocal fry enforcement",
    )

    return parser.parse_args()
