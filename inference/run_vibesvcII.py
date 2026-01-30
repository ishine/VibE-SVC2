import os
import shutil
import warnings

import soundfile
import torch
import torch.distributed as dist
from tqdm import tqdm

from inference.infer_vibesvcII import Svc
from utils.commons.utils import load_config

warnings.filterwarnings("ignore")


def run(rank, n_gpus, args, audio_list):
    # Multiprocessing setting
    dist.init_process_group(
        backend="gloo" if os.name == "nt" else "nccl",
        init_method="env://",
        world_size=n_gpus,
        rank=rank,
    )
    torch.cuda.set_device(rank)

    # Load arguments
    exp_dir = args.exp_dir
    f0p = args.f0_predictor

    # model parameters
    model_path = args.model_path
    config_path = args.config_path
    k_step = args.k_step
    pitch_zeroshot = args.pitch_zeroshot

    # task settings
    recon_spk = args.recon_spk
    vocoded = args.vocoded
    infer_mode = args.infer_mode
    recon_tech = args.recon_tech

    # inference paramters
    target_spk = args.target_spk
    target_pitch_style = args.target_pitch_style
    target_timbre_style = args.target_timbre_style

    extent_scale = args.extent_scale
    extent_scale_type = args.extent_scale_type
    rate_scale = args.rate_scale

    vocal_fry_enforcement = args.vocal_fry_enforcement

    # Load configuration files
    pretrain_path_config = load_config("./configs/paths.yaml")
    spk_stat_path = args.stats_path

    d_config = load_config(config_path)
    spk_stat_config = load_config(spk_stat_path)

    svc_model = Svc(
        model_path,
        config_path,
        pretrain_path_config,
        f0p,
        pitch_zeroshot,
        infer_mode,
        rank,
        eps=float(d_config.data.eps),
        vocal_fry_enforcement=vocal_fry_enforcement,
    )

    # check task
    if target_timbre_style is not None and svc_model.n_timbre_style == 1:
        if rank == 0:
            print("Checkpoint is not available to control timbre technique.")
        return

    # Set output directory
    gt_dir = f"./results/{exp_dir}/gt"
    gen_dir = f"./results/{exp_dir}/gen"
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)

    d_config["env"]["expdir"] = f"./results/{exp_dir}"
    spk_list = d_config["spk"]

    # Allocate sample idx for multi-gpu inference
    prev_rank_idx = 0
    if rank != 0:
        for r in range(rank):
            prev_rank_idx += len(audio_list[r])

    # Main stream
    for idx, src in enumerate(tqdm(audio_list[rank])):
        src = src.strip()
        src_spk = src.split("/")[3]
        src_style = src.split("/")[4].split("#")[2]

        # copy GT & Reference audio from raw path
        gt_audio_path = f"{gt_dir}/{prev_rank_idx + idx}_{src_spk}#{src_style}#{src.split('/')[4].split('#')[-1]}"
        if not os.path.exists(gt_audio_path):
            shutil.copyfile(src, gt_audio_path)

        if infer_mode == "pitch" and src_style == target_pitch_style:
            continue
        elif infer_mode == "timbre" and src_style == target_timbre_style:
            continue

        # Generate vocoded samples
        if vocoded:
            gen_audio = svc_model.vocoded(gt_audio_path)
            gen_audio_path = f"{gen_dir}/{prev_rank_idx + idx}_{src_spk}#{src_style}#{src.split('/')[4].split('#')[-1]}"
            soundfile.write(
                gen_audio_path, gen_audio, svc_model.target_sample, format="wav"
            )
            continue

        # Generate converted samples
        for spk in spk_list:
            # Skip reconstruction settings
            if recon_spk and spk != src_spk:
                continue
            elif not recon_spk and spk == src_spk:
                if src_spk == target_spk:
                    print("This is style reconstruction setting with sample speaker")
                continue

            # Single speaker generation setting
            if target_spk is not None and spk != target_spk:
                continue

            # Shifting f0 contour by a scaler factor based on precalculate statistics
            if spk != src_spk:
                if svc_model.n_timbre_style != 1:
                    f0_shift = spk_stat_config[spk]
                else:
                    f0_shift = spk_stat_config[spk] / spk_stat_config[src_spk]
            else:
                f0_shift = None

            # Info : Try torch.no_grad when an error occurred by inplace computation
            with torch.inference_mode():
                gen_audio = svc_model.infer(
                    source_spk=src_spk,
                    target_spk=spk,
                    source_tech=src_style,
                    target_pitch_style=target_pitch_style,
                    target_timbre_style=target_timbre_style,
                    raw_path=gt_audio_path,
                    k_step=k_step,
                    f0_shift=f0_shift,
                    extent_scale=extent_scale,
                    extent_scale_type=extent_scale_type,
                    rate_scale=rate_scale,
                    vocal_fry_enforcement=vocal_fry_enforcement,
                )

            # Save the generated audio
            gen_audio_path = f"{gen_dir}/{prev_rank_idx + idx}_{spk}#{src_style}#{src.split('/')[4].split('#')[-1]}"
            soundfile.write(
                gen_audio_path, gen_audio, svc_model.target_sample, format="wav"
            )
            svc_model.clear_empty()
